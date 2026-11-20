import json

import pandas as pd
import plotly.express as px
import streamlit as st

import run_panel
import ui_common as ui
from lab import jobs, registry

st.title("🔁 RL Training — GRPO (rules = RL, reward model = RLHF)")
ui.explain("Reinforcement fine-tuning. With **reward rules** it is RL (track B). With a **reward model trained on "
           "human preferences** it is RLHF (track C).",
           "A starting model (normally your SFT output), RL questions, and either an APPROVED reward version or a reward model.",
           "For each question it generates several answers; the reward rules score each one; GRPO pushes the model "
           "towards the answers that scored above their group's average.",
           "Average reward over time, the reward distribution, which rules fire, and the validation reward at the end.",
           "A new LoRA adapter (continuing the parent's adapter), sampled answers with reward breakdowns, metrics, "
           "validation results and lineage.")
st.info("The model generates answers. The reward function scores them. GRPO uses those scores to update the model.")
st.warning("RL is much slower and more memory-hungry than SFT: every step generates *number of generations* answers "
           "per question. Start with the Tiny preset.")

tab_new, tab_runs = st.tabs(["▶️ New run", "📈 Runs, samples & rewards"])
with tab_new:
    parent = ui.model_picker("Starting model", "rl_parent", default_kind="sft",
                             help="Normally the SFT model you just evaluated.")
    st.caption("Lineage of the starting model: " + " ← ".join(m["id"] for m in registry.lineage(parent)))
    pm = registry.model(parent)
    if pm["kind"] == "base":
        st.warning("You picked the base model. RL usually starts from an SFT model; this is allowed as an explicit branch.")
    source = st.radio("Where does the reward come from?", ["rules", "rm", "rules+rm"], horizontal=True,
                      format_func={"rules": "Reward rules → RL", "rm": "Reward model (human preferences) → RLHF",
                                   "rules+rm": "Both (rules + reward model) → RLHF"}.get)
    approved = [r for r in registry.rewards() if r["approved"]]
    rms = [m for m in registry.models() if m["kind"] == "rm"]
    c1, c2, c3 = st.columns(3)
    with c1:
        train_ds = ui.dataset_picker("RL training dataset (questions)", "rl_train", "rl_tr")
    with c2:
        val_ds = ui.dataset_picker("RL validation dataset", "rl_val", "rl_va", allow_none=True)
    with c3:
        reward_id = rm_id = None
        if source in ("rules", "rules+rm"):
            reward_id = st.selectbox("Reward version (approved only)", [r["id"] for r in approved]) if approved else None
        if source in ("rm", "rules+rm"):
            rm_id = st.selectbox("Reward model", [m["id"] for m in rms],
                                 format_func=lambda i: f"{i} · held-out accuracy "
                                 f"{(json.loads(registry.model(i)['reference_json'] or '{}').get('val_accuracy') or 0):.0%}") \
                if rms else None
    if source in ("rules", "rules+rm") and not approved:
        st.error("No approved reward version. Go to Reward Designer, test the rules, and approve a version.")
    if source in ("rm", "rules+rm") and not rms:
        st.error("No reward model yet. Collect preferences on Human Feedback, then train one on Reward Model.")
    kind = "rl" if source == "rules" else "rlhf"
    if kind == "rlhf":
        st.info("Watch for **reward hacking**: a reward model trained on cautious 'chosen' answers can learn to love "
                "refusals or vague answers. Check the sampled answers below — if they drift that way, add preference "
                "pairs where a helpful answer beats a refusal, retrain the reward model, or use 'Both (rules + reward model)'.")
    mode = st.radio("Mode", ["Beginner", "Advanced"], horizontal=True, key="rl_mode")
    preset = st.selectbox("Preset", list(jobs.RL_PRESETS), key="rl_preset")
    cfg = dict(jobs.RL_PRESETS[preset])
    if pm["kind"] != "base":
        ac = json.loads(open(f"{pm['adapter_path']}/adapter_config.json").read())
        cfg.update(lora_r=ac["r"], lora_alpha=ac["lora_alpha"])
        st.caption(f"LoRA shape inherited from {parent} (r={ac['r']}). Training continues the parent's adapter.")
    if mode == "Advanced":
        a, b, c = st.columns(3)
        with a:
            cfg["num_generations"] = st.number_input("Answers generated per question", 2, 16, cfg["num_generations"],
                                                     help="GRPO compares answers within a group. More = better signal, slower.")
            cfg["max_completion_length"] = st.number_input("Max answer length (tokens)", 16, 512, cfg["max_completion_length"],
                                                           help="Shorter = faster and less memory.")
            cfg["temperature"] = st.number_input("Sampling temperature", 0.1, 2.0, cfg["temperature"],
                                                 help="Higher = more varied answers to compare.")
        with b:
            cfg["learning_rate"] = st.number_input("Learning rate", value=cfg["learning_rate"], format="%.1e")
            cfg["batch_size"] = st.number_input("Answers per update (batch)", 2, 64, cfg["batch_size"],
                                                help="Must be divisible by answers per question (with grad accumulation).")
            cfg["grad_accum"] = st.number_input("Gradient accumulation", 1, 16, cfg["grad_accum"])
        with c:
            cfg["max_steps"] = st.number_input("Max training steps", 1, 2000, cfg["max_steps"])
            cfg["beta"] = st.number_input("KL penalty beta", 0.0, 1.0, cfg["beta"],
                                          help="Keeps the model close to the start. 0 = off (TRL default, uses less memory).")
            cfg["seed"] = st.number_input("Random seed", 0, 10**6, cfg["seed"], key="rl_seed")
    else:
        st.write(f"Beginner defaults: {cfg['num_generations']} answers per question, max {cfg['max_completion_length']} "
                 f"tokens, {cfg['max_steps']} steps, lr {cfg['learning_rate']}, seed {cfg['seed']}.")
    if (cfg["batch_size"] * cfg["grad_accum"]) % cfg["num_generations"]:
        st.error("Batch size × gradient accumulation must be divisible by answers per question.")
    active = jobs.active_run()
    if active:
        st.warning(f"{active['id']} is running — one training run at a time.")
    cfg.update(reward_source=source, rm_id=rm_id)
    missing = (source != "rm" and not reward_id) or (source != "rules" and not rm_id)
    if st.button(f"🚀 Start real {'GRPO (RL)' if kind == 'rl' else 'RLHF'} training", type="primary",
                 disabled=bool(active) or missing or not train_ds):
        try:
            ui.unload_engine()
            rid = jobs.start(kind, parent, train_ds, val_ds, cfg, reward_id=reward_id)
            st.success(f"Started {rid}. Open the 'Runs, samples & rewards' tab.")
        except Exception as e:  # noqa: BLE001
            st.error(str(e))

def rl_details(r):
    rid = r["id"]
    if True:
        samples = jobs.read_jsonl(r["samples_path"])
        if samples:
            st.markdown("#### What happened during RL " + ui.tag("reward"), unsafe_allow_html=True)
            sdf = pd.DataFrame(samples)
            a, b = st.columns(2)
            with a:
                fig = px.histogram(sdf, x="total", nbins=20, title="Reward distribution (all sampled answers)",
                                   template="plotly_dark")
                fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with b:
                contrib = [{"step": s["step"], "rule": x["rule"], "contribution": x["contribution"]}
                           for s in samples for x in s["breakdown"]]
                cdf = pd.DataFrame(contrib).groupby(["step", "rule"], as_index=False)["contribution"].mean()
                fig = px.line(cdf, x="step", y="contribution", color="rule", title="Average contribution per rule",
                              template="plotly_dark")
                fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10), legend=dict(font=dict(size=9)))
                st.plotly_chart(fig, use_container_width=True)
            steps = sorted(int(s) for s in sdf["step"].unique())
            step = st.selectbox("Show sampled answers from step", steps, index=len(steps) - 1, key=f"rlstep_{rid}")
            view = [{"Question": s["question"], "Generated answer": s["answer"],
                     "Reward breakdown": ", ".join(f"{x['contribution']:+g} {x['rule']}" for x in s["breakdown"] if x["contribution"]),
                     "Total reward": s["total"]} for s in samples if s["step"] == step]
            st.dataframe(pd.DataFrame(view), use_container_width=True, hide_index=True)
        ev = [e for e in registry.evals(r["output_model_id"]) if e["settings"].get("source") == f"end of {rid}"]
        if ev:
            e = ev[0]
            st.markdown(f"#### Validation after training {ui.tag('val')}{ui.tag('reward')}", unsafe_allow_html=True)
            st.write(f"Greedy answers on **{e['dataset_id']}**, scored with **{e['reward_id']}**: "
                     f"mean reward **{e['mean_reward']:.2f}**")
            st.dataframe(pd.DataFrame([{k: x[k] for k in ("question", "answer", "reward")} for x in e["results"]]),
                         use_container_width=True, hide_index=True)
        elif r["status"] == "completed" and not r["val_ds"]:
            st.info("No RL validation dataset was selected for this run, so no validation results were produced.")


with tab_runs:
    run_panel.show(("rl", "rlhf"), extra=rl_details)
