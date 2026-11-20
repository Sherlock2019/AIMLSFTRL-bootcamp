"""Real SFT (TRL SFTTrainer) and real GRPO (TRL GRPOTrainer) with LoRA. Run as: python -m lab.train <run_id>

Each run reads its config from the registry, loads the selected parent model (never silently the base),
streams metrics to JSONL, saves a new LoRA adapter, and registers the new model version with its lineage.
"""
import dataclasses
import json
import sys
import time
import traceback
from pathlib import Path

from . import data, hardware, model_loader, paths, registry, rewards

OOM_HELP = ("Out of memory. Try, in order: lower max sequence length / max answer length, batch size 1 with more "
            "gradient accumulation, fewer generations per prompt (RL), fewer LoRA target modules, or close other "
            "GPU programs (the dashboard's loaded inference model can be unloaded on Settings & Hardware).")


def _filter(cfg_cls, kw, log):
    names = {f.name for f in dataclasses.fields(cfg_cls)}
    dropped = [k for k in kw if k not in names]
    if dropped:
        log(f"[note] this TRL version has no {dropped}; skipped")
    return cfg_cls(**{k: v for k, v in kw.items() if k in names})


def _make_callback(run_id, metrics_path, t0, step_holder):
    from transformers import TrainerCallback

    cancel_flag = paths.sub("runs") / f"{run_id}.cancel"

    class LabCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kw):
            step_holder["step"] = state.global_step + 1

        def on_log(self, args, state, control, logs=None, **kw):
            rec = {"step": state.global_step, "max_steps": state.max_steps, "epoch": state.epoch,
                   "elapsed_s": round(time.time() - t0, 1), "gpu_mem_gb": hardware.gpu_mem_used_gb(),
                   "ram_used_gb": round(__import__("psutil").Process().memory_info().rss / 1e9, 2)}
            rec.update({k: v for k, v in (logs or {}).items() if isinstance(v, (int, float))})
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

        def on_step_end(self, args, state, control, **kw):
            if cancel_flag.exists():
                control.should_training_stop = True

    return LabCallback(), cancel_flag


def _common_args(c, device, out_dir):
    return dict(output_dir=str(out_dir / "checkpoints"), learning_rate=float(c["learning_rate"]),
                per_device_train_batch_size=int(c["batch_size"]),
                gradient_accumulation_steps=int(c["grad_accum"]), max_steps=int(c.get("max_steps") or -1),
                num_train_epochs=float(c.get("epochs") or 1), logging_steps=1, seed=int(c["seed"]),
                report_to="none", save_strategy="steps" if int(c.get("save_steps") or 0) > 0 else "no",
                save_steps=int(c.get("save_steps") or 500), bf16=False, fp16=False,
                use_cpu=(device == "cpu"), dataloader_num_workers=0, lr_scheduler_type="constant")


def run_sft(r, log):
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    c = r["config"]
    model, base_name, rev = model_loader.load_for_training(r["parent_model_id"], _lora(c))
    tok = model_loader.load_tokenizer(base_name)
    device = model_loader.pick_device()
    log(f"Loaded parent {r['parent_model_id']} on base {base_name}@{rev} (device {device})")
    model.print_trainable_parameters()

    def to_pc(rows):
        return Dataset.from_list([{"prompt": model_loader.format_prompt(tok, x["prompt"]),
                                   "completion": x["completion"] + tok.eos_token} for x in rows])
    train = to_pc(data.load(r["train_ds"]))
    val = to_pc(data.load(r["val_ds"])) if r["val_ds"] else None
    out_dir = paths.sub("models") / r["output_model_id"]
    kw = _common_args(c, device, out_dir)
    kw.update(max_length=int(c["max_length"]), max_seq_length=int(c["max_length"]), completion_only_loss=True,
              eval_strategy="steps" if val is not None else "no", eval_steps=int(c.get("eval_steps") or 5),
              per_device_eval_batch_size=int(c["batch_size"]), packing=False, dataset_text_field=None)
    kw = {k: v for k, v in kw.items() if v is not None}
    args = _filter(SFTConfig, kw, log)
    cb, cancel = _make_callback(r["id"], r["metrics_path"], time.time(), {})
    trainer = SFTTrainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                         processing_class=tok, callbacks=[cb])
    trainer.train()
    if val is not None:
        ev = trainer.evaluate()
        with open(r["metrics_path"], "a") as f:
            f.write(json.dumps({"step": trainer.state.global_step, "final": True, **ev}) + "\n")
    return _finish(r, trainer.model, tok, base_name, rev, out_dir, cancel, log, val_rows=data.load(r["val_ds"]) if r["val_ds"] else [])


def run_rm(r, log):
    """Train a reward model on human preferences (TRL RewardTrainer, LoRA on a 1-output score head)."""
    from peft import LoraConfig
    from transformers import AutoModelForSequenceClassification
    from trl import RewardConfig, RewardTrainer

    from . import reward_model
    c = r["config"]
    pm = registry.model(r["parent_model_id"])
    base_name = pm["base_model"]
    if pm["kind"] != "base":
        log(f"[note] reward models start from the base weights of {base_name} (a new score head is added).")
    tok = model_loader.load_tokenizer(base_name)
    tok.padding_side = "right"
    device = model_loader.pick_device()
    rows = data.load(r["train_ds"])
    train_rows, val_rows = reward_model.split(rows, float(c.get("val_frac", 0.2)), int(c["seed"]))
    log(f"{len(train_rows)} training pairs, {len(val_rows)} held-out pairs")
    model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=1,
                                                               dtype=model_loader.pick_dtype(device))
    model.config.pad_token_id = tok.pad_token_id
    rev = model_loader.base_revision(model)
    lora = _lora(c)
    peft_cfg = LoraConfig(task_type="SEQ_CLS", **lora)
    out_dir = paths.sub("models") / r["output_model_id"]
    kw = _common_args(c, device, out_dir)
    kw.update(max_length=int(c["max_length"]), disable_dropout=True,
              center_rewards_coefficient=float(c.get("center_rewards_coefficient") or 0.01),
              eval_strategy="steps" if val_rows else "no", eval_steps=int(c.get("eval_steps") or 5),
              per_device_eval_batch_size=int(c["batch_size"]))
    args = _filter(RewardConfig, kw, log)
    cb, cancel = _make_callback(r["id"], r["metrics_path"], time.time(), {})
    trainer = RewardTrainer(model=model, args=args, train_dataset=reward_model.to_dataset(tok, train_rows),
                            eval_dataset=reward_model.to_dataset(tok, val_rows) if val_rows else None,
                            processing_class=tok, peft_config=peft_cfg, callbacks=[cb])
    trainer.train()
    if cancel.exists():
        cancel.unlink()
        raise KeyboardInterrupt("cancelled by user")
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    log(f"Saved reward model adapter to {out_dir}")
    # held-out accuracy: does the reward model prefer the human-chosen answer?
    import torch
    trainer.model.eval()

    def score(texts):
        with torch.no_grad():
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=int(c["max_length"]))
            enc = {k: v.to(trainer.model.device) for k, v in enc.items()}
            return trainer.model(**enc).logits[:, 0].float().tolist()
    eval_rows = val_rows or train_rows
    sc = score([reward_model.text(tok, x["prompt"], x["chosen"]) for x in eval_rows])
    sr = score([reward_model.text(tok, x["prompt"], x["rejected"]) for x in eval_rows])
    acc, margin = reward_model.accuracy(sc, sr)
    tr_acc, _ = reward_model.accuracy(score([reward_model.text(tok, x["prompt"], x["chosen"]) for x in train_rows]),
                                      score([reward_model.text(tok, x["prompt"], x["rejected"]) for x in train_rows]))
    log(f"Held-out preference accuracy: {acc:.2f} (train {tr_acc:.2f}), mean margin {margin:.2f}")
    with open(r["metrics_path"], "a") as f:
        f.write(json.dumps({"step": trainer.state.global_step, "final": True, "rm_val_accuracy": acc,
                            "rm_train_accuracy": tr_acc, "rm_val_margin": margin}) + "\n")
    reference = {"val_accuracy": acc, "train_accuracy": tr_acc, "val_margin": margin, "n_train": len(train_rows),
                 "n_val": len(val_rows), "val_on": "held-out pairs" if val_rows else "training pairs (too few to hold out)",
                 "pairs": [{"prompt": x["prompt"], "chosen": x["chosen"], "rejected": x["rejected"],
                            "chosen_score": round(a, 3), "rejected_score": round(b, 3)}
                           for x, a, b in zip(eval_rows, sc, sr)]}
    (Path(out_dir) / "lineage.json").write_text(json.dumps(
        {"model_id": r["output_model_id"], "parent": r["parent_model_id"], "base_model": base_name, "base_revision": rev,
         "run": r["id"], "method": "reward model", "train_ds": r["train_ds"], "config": c}, indent=2))
    registry.add_model("rm", r["parent_model_id"], base_name, rev, out_dir, r["id"], r["train_ds"], None,
                       size_bytes=model_loader.adapter_size(out_dir), reference=reference, mid=r["output_model_id"])
    return r["output_model_id"]


def run_grpo(r, log):
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    c = r["config"]
    source = c.get("reward_source", "rules")  # rules (RL) | rm (RLHF) | rules+rm
    rw = registry.reward(r["reward_id"]) if r.get("reward_id") else None
    if source in ("rules", "rules+rm") and (not rw or not rw["approved"]):
        raise RuntimeError(f"Reward {r['reward_id']} is not approved. Approve it in Reward Designer first.")
    scorer = None
    if source in ("rm", "rules+rm"):
        from . import reward_model
        scorer = reward_model.Scorer(c["rm_id"])
        log(f"Using reward model {c['rm_id']} as the reward (RLHF)")
    model, base_name, rev = model_loader.load_for_training(r["parent_model_id"], _lora(c))
    tok = model_loader.load_tokenizer(base_name)
    tok.padding_side = "left"
    device = model_loader.pick_device()
    log(f"Loaded parent {r['parent_model_id']} on base {base_name}@{rev} (device {device})")
    rows = data.load(r["train_ds"])
    train = Dataset.from_list([{"prompt": model_loader.format_prompt(tok, x["prompt"]),
                                "required_checks": x.get("required_checks") or [], "avoid": x.get("avoid") or [],
                                "question": x["prompt"]} for x in rows])
    step = {"step": 0}
    samples_path = r["samples_path"]

    def hook(prompt, answer, s):
        q = prompt.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0] if "<|im_start|>user" in prompt else prompt[-200:]
        with open(samples_path, "a") as f:
            f.write(json.dumps({"step": step["step"], "question": q, "answer": answer, "total": s["total"],
                                "breakdown": s["breakdown"]}) + "\n")

    use_rules = source in ("rules", "rules+rm")

    def reward_fn(prompts, completions, **kw):
        texts = [x if isinstance(x, str) else (x[-1]["content"] if x else "") for x in completions]
        n = len(texts)
        rm_scores = scorer.score_texts([p + rewards.clean_answer(t) + tok.eos_token for p, t in zip(prompts, texts)]) \
            if scorer else [0.0] * n
        out = []
        for i in range(n):
            bd, rule_total = [], 0.0
            if use_rules:
                s = rewards.score(texts[i], rw["rules"], {k: (v[i] if isinstance(v, list) and len(v) == n else None)
                                                          for k, v in kw.items()})
                bd, rule_total = s["breakdown"], s["total"]
            if scorer:
                bd.append({"rule": f"reward model {c['rm_id']}", "contribution": round(rm_scores[i], 3)})
            total = rule_total + rm_scores[i]
            out.append(total)
            hook(prompts[i], rewards.clean_answer(texts[i]), {"total": round(total, 3), "breakdown": bd})
        return out
    reward_fn.__name__ = {"rules": "rule_reward", "rm": "reward_model", "rules+rm": "rules_plus_reward_model"}[source]
    out_dir = paths.sub("models") / r["output_model_id"]
    kw = _common_args(c, device, out_dir)
    ng = int(c["num_generations"])
    kw.update(num_generations=ng, max_completion_length=int(c["max_completion_length"]),
              max_prompt_length=int(c.get("max_prompt_length") or 256), temperature=float(c.get("temperature") or 1.0),
              beta=float(c.get("beta") or 0.0), use_vllm=False, log_completions=False,
              generation_batch_size=int(c["batch_size"]) * int(c["grad_accum"]))
    if kw["generation_batch_size"] % ng:
        raise RuntimeError(f"batch size x grad accumulation ({kw['generation_batch_size']}) must be divisible by "
                           f"the number of generations ({ng}).")
    args = _filter(GRPOConfig, kw, log)
    cb, cancel = _make_callback(r["id"], r["metrics_path"], time.time(), step)
    trainer = GRPOTrainer(model=model, reward_funcs=reward_fn, args=args, train_dataset=train,
                          processing_class=tok, callbacks=[cb])
    trainer.train()
    val_rows = data.load(r["val_ds"]) if r["val_ds"] else []
    return _finish(r, trainer.model, tok, base_name, rev, out_dir, cancel, log, val_rows=val_rows,
                   rules=rw["rules"] if rw and source != "rm" else None, scorer=scorer)


def _lora(c):
    return dict(r=int(c.get("lora_r", 8)), lora_alpha=int(c.get("lora_alpha", 16)),
                lora_dropout=float(c.get("lora_dropout", 0.05)),
                target_modules=c.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"])


def _greedy(model, tok, questions, max_new_tokens=64):
    import torch
    device = next(model.parameters()).device
    tok.padding_side = "left"
    model.eval()
    outs = []
    with torch.no_grad():
        for q in questions:
            enc = tok([model_loader.format_prompt(tok, q)], return_tensors="pt").to(device)
            g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            outs.append(rewards.clean_answer(tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)))
    return outs


def _finish(r, model, tok, base_name, rev, out_dir, cancel, log, val_rows, rules=None, scorer=None):
    if cancel.exists():
        cancel.unlink()
        raise KeyboardInterrupt("cancelled by user")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    log(f"Saved adapter to {out_dir}")
    ref_q = [x["prompt"] for x in val_rows[:2]] or ["Can I promise a drill tomorrow?"]
    reference = {"questions": ref_q, "answers": _greedy(model, tok, ref_q), "max_new_tokens": 64}
    if (rules or scorer) and val_rows:  # RL validation: greedy answers on the validation set, scored with the same reward
        answers = _greedy(model, tok, [x["prompt"] for x in val_rows], 64)
        rm_scores = scorer.score([x["prompt"] for x in val_rows], answers) if scorer else [0.0] * len(answers)
        results = []
        for row, a, rs in zip(val_rows, answers, rm_scores):
            s = rewards.score(a, rules, row) if rules else {"total": 0.0, "breakdown": []}
            bd = s["breakdown"] + ([{"rule": f"reward model {scorer.rm_id}", "contribution": round(rs, 3)}] if scorer else [])
            results.append({"id": row["id"], "question": row["prompt"], "answer": a, "reward": round(s["total"] + rs, 3),
                            "breakdown": bd, "approved": row.get("completion", "")})
        mean = sum(x["reward"] for x in results) / len(results)
        label = " + ".join(([r["reward_id"]] if rules else []) + ([scorer.rm_id] if scorer else []))
        registry.add_eval(r["output_model_id"], r["val_ds"], label,
                          {"max_new_tokens": 64, "temperature": 0, "source": f"end of {r['id']}"}, results, mean)
        log(f"Validation mean reward ({label}): {mean:.3f}")
    lineage = {"model_id": r["output_model_id"], "parent": r["parent_model_id"], "base_model": base_name,
               "base_revision": rev, "run": r["id"], "method": r["kind"], "train_ds": r["train_ds"],
               "val_ds": r["val_ds"], "reward": r.get("reward_id"), "config": r["config"]}
    (Path(out_dir) / "lineage.json").write_text(json.dumps(lineage, indent=2))
    registry.add_model(r["kind"], r["parent_model_id"], base_name, rev, out_dir, r["id"], r["train_ds"], r["val_ds"],
                       reward_id=r.get("reward_id"), size_bytes=model_loader.adapter_size(out_dir),
                       reference=reference, mid=r["output_model_id"])
    return r["output_model_id"]


def main(run_id):
    r = registry.run(run_id)
    log_f = open(r["log_path"], "a", buffering=1)

    def log(msg):
        log_f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    registry.update("runs", run_id, status="running", started=time.time(),
                    hardware_json=hardware.check(), libs_json=hardware.libs())
    log(f"Starting {r['kind'].upper()} run {run_id}: parent={r['parent_model_id']} -> {r['output_model_id']}")
    try:
        {"sft": run_sft, "rm": run_rm}.get(r["kind"], run_grpo)(r, log)  # rl / rlhf -> GRPO
        registry.update("runs", run_id, status="completed", finished=time.time())
        log("Run completed.")
    except KeyboardInterrupt as e:
        registry.update("runs", run_id, status="cancelled", finished=time.time(), error=str(e))
        log("Run cancelled - no model version was registered.")
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "out of memory" in msg.lower():
            msg = f"{OOM_HELP}\n\nOriginal error: {msg}"
        registry.update("runs", run_id, status="failed", finished=time.time(), error=msg)
        log("FAILED: " + msg)
        log(traceback.format_exc())


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main(sys.argv[1])
