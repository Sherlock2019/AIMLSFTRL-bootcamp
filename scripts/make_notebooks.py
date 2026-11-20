"""Generate notebooks/NN_<step>.ipynb — one per lab step, prefilled with the Python that does what the page does.

    .venv/bin/python scripts/make_notebooks.py

Notebooks use the SAME registry as the dashboard (artifacts/), so models and datasets you create in code show up in
the UI and vice versa. Cells that would overwrite your working data are behind SAVE = False.
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"

SETUP = '''# ── Setup (run first) ────────────────────────────────────────────────────────────
# Uses the same lab registry as the dashboard. To experiment in a sandbox instead:
#   os.environ["LAB_HOME"] = "/tmp/my_lab"   (before importing lab)
import os, sys
ROOT = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # models are already cached locally
import pandas as pd
from lab import data, registry, paths, rewards
data.bootstrap_samples()
BASE = registry.ensure_base(paths.DEFAULT_BASE_MODEL)
if not registry.rewards():
    registry.add_reward(rewards.DEFAULT_RULES, note="Default draft rules")

def latest(kind):
    """Newest model id of a kind: base | sft | rl | rlhf | rm (or None)."""
    ms = [m for m in registry.models() if m["kind"] == kind]
    return ms[-1]["id"] if ms else None

def latest_ds(role):
    d = registry.latest_dataset(role)
    return d["id"] if d else None

print("Lab home:", paths.home(), "| base model:", BASE)'''

SAVE = "SAVE = False   # set True to write new versions into your lab (they appear in the dashboard)"


def nb(title, intro, cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    n.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}\n\n*Same as the dashboard page, done in code.*"),
               nbf.v4.new_code_cell(SETUP)]
    for kind, src in cells:
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src.strip("\n")))
    return n


md = lambda s: ("md", s)  # noqa: E731
py = lambda s: ("py", s)  # noqa: E731

NOTEBOOKS = {
    # ------------------------------------------------------------------ 1 · Setup
    "00_setup_hardware": ("views/settings_hardware.py", nb("Step 0 · Check your hardware",
        "What can this machine train? GPU, memory, PyTorch device and the recommended mode.", [
        py('''from lab import hardware
info = hardware.check()
pd.Series({k: v for k, v in info.items() if k != "recommended"})'''),
        py('''mode, advice = info["recommended"]
print(mode, "→", advice)'''),
        py('''pd.Series(hardware.libs())   # library versions recorded with every training run'''),
    ])),
    "01_model_hub": ("views/model_hub.py", nb("Step 1 · Import a model from Hugging Face",
        "Register an open model as a new base model you can train, evaluate and compare.", [
        py('''from lab import hub
pd.DataFrame([{"repo": r, "why": why, "cached": hub.is_cached(r)} for r, why in hub.SUGGESTED])'''),
        md("Check size, licence and GPU fit before downloading (needs internet; `hub.info` lifts offline mode just for this call)."),
        py('''REPO = "Qwen/Qwen2.5-0.5B-Instruct"
try:
    meta = hub.info(REPO); print(meta)
    print(hub.fit_advice(meta["size_gb"]))
except Exception as e:
    print("Offline or not reachable:", type(e).__name__, e)'''),
        py('''# Download once (if needed) and register as a base model
mid = hub.import_model(REPO)
print("registered:", mid)
pd.DataFrame([m for m in registry.models() if m["kind"] == "base"])[["id", "base_model"]]'''),
    ])),
    # ------------------------------------------------------------------ 2 · Supervised
    "02_data_studio": ("views/data_studio.py", nb("Step 2 · Prepare supervised data",
        "Versioned datasets: inspect, check quality and leakage, load a pack, edit an example.", [
        py(SAVE),
        py('''pd.DataFrame(registry.datasets())[["id", "role", "n_rows", "source", "note"]]'''),
        py('''rows = data.load(latest_ds("sft_train"))
print(len(rows), "examples"); pd.DataFrame(rows).head()'''),
        py('''# Quality and leakage (a question in train AND validation/test makes scores look better than they are)
print(data.quality_warnings("sft_train", rows))
splits = {latest_ds(r): data.load(latest_ds(r)) for r in data.ROLES if latest_ds(r)}
print("leakage:", data.leakage(splits) or "none")'''),
        py('''# The 10 bundled supervised packs
pd.DataFrame(data.list_packs())[["slug", "title", "n_train", "n_val"]]'''),
        py('''# Edit one example → a new immutable version (old versions are kept)
new_rows = [dict(r) for r in rows]
new_rows[0]["completion"] = new_rows[0]["completion"].rstrip(".") + ", and tell the customer it is an estimate."
added, removed, changed = data.diff(rows, new_rows)
print(f"{len(changed)} changed")
if SAVE:
    print("saved", data.save_version("sft_train", new_rows, parent_id=latest_ds("sft_train"), note="notebook edit"))
    # data.load_pack("05_bank_fraud", mode="replace")   # load a pack as new sft_train + sft_val versions'''),
    ])),
    "03_sft_training": ("views/sft_training.py", nb("Step 3 · Supervised fine-tuning (SFT)",
        "Train a LoRA adapter on question → approved answer pairs with TRL's SFTTrainer.", [
        py('''from lab import jobs
cfg = dict(jobs.SFT_PRESETS["Tiny (smoke test)"])
cfg'''),
        py('''# Runs `python -m lab.train <run>` in the background and waits for it (about 15 s on a GTX 1080 Ti)
rid = jobs.start("sft", BASE, latest_ds("sft_train"), latest_ds("sft_val"), cfg, wait=True)
run = registry.run(rid)
print(run["status"], "→ new model", run["output_model_id"], run["error"] or "")'''),
        py('''m = pd.DataFrame(jobs.read_jsonl(run["metrics_path"]))
m[[c for c in ("step", "loss", "eval_loss") if c in m]].dropna(how="all", subset=[c for c in ("loss", "eval_loss") if c in m])'''),
        md("What the trainer actually does (the lab's own code):"),
        py('''import inspect
from lab import train
print(inspect.getsource(train.run_sft))'''),
        py('''from lab import engines
SFT = run["output_model_id"]
engines.ask(SFT, "Can I promise a customer their ladder arrives tomorrow?")'''),
    ])),
    "04_evaluation": ("views/evaluation.py", nb("Step 4 · Evaluate",
        "Run a model on validation questions it never trained on and score every answer.", [
        py('''from lab import engines
MODEL = latest("sft") or BASE
DS = latest_ds("sft_val")
rows = data.load(DS)
rid = registry.rewards()[0]["id"]; rules = registry.reward(rid)["rules"]
answers = engines.get(registry.model(MODEL)["base_model"]).generate(MODEL, [r["prompt"] for r in rows], max_new_tokens=96)'''),
        py('''results = []
for r, a in zip(rows, answers):
    s = rewards.score(a, rules, r)
    results.append({"id": r["id"], "question": r["prompt"], "answer": a, "approved": r.get("completion", ""),
                    "reward": s["total"], "breakdown": s["breakdown"], "review": ""})
mean = sum(x["reward"] for x in results) / len(results)
EVAL = registry.add_eval(MODEL, DS, rid, {"max_new_tokens": 96, "temperature": 0, "source": "notebook"}, results, mean)
print(f"evaluation #{EVAL}: mean rule score {mean:.2f}")
pd.DataFrame(results)[["question", "answer", "approved", "reward"]]'''),
    ])),
    "05_evals_workbench": ("views/evals_workbench.py", nb("Step 5 · Error analysis & judges",
        "Label answers, count failure modes, measure reviewer agreement, validate a judge, gate a release.", [
        py(SAVE),
        py('''from lab import evals, judge
e = registry.evals()[0] if registry.evals() else None
assert e, "Run notebook 04 (or Evaluation & Scoring) first"
R = e["results"]; print(f"evaluation #{e['id']} · {len(R)} answers")'''),
        py('''# Label answers (here: a toy rule so the cell runs — in real life you read each answer)
for x in R:
    if not x.get("review"):
        x["review"] = "correct" if "check" in x["answer"].lower() else "incorrect"
        x["failure_mode"] = "" if x["review"] == "correct" else "no check before promising"
pd.DataFrame(evals.failure_table(R), columns=["failure mode", "count", "share"])'''),
        py('''# Agreement between two reviewers (Cohen's kappa corrects for chance)
me    = [x["review"] for x in R]
other = [("incorrect" if i % 4 == 0 else lab) for i, lab in enumerate(me)]
k, n = evals.cohen_kappa(me, other); print(f"kappa {k:.2f} ({evals.kappa_meaning(k)}) on {n} answers")'''),
        py('''# Can the rule score be trusted as a judge? TPR/TNR + bias-corrected pass rate with a bootstrap CI
rep = evals.judge_report(R, threshold=0.5)
{k: rep[k] for k in ("tp", "fn", "tn", "fp", "tpr", "tnr", "judge_pass_rate", "corrected_pass_rate", "ci95")}'''),
        py('''# LLM-as-judge on one answer (validate it against your labels before trusting it)
x = R[0]
judge.grade(BASE, x["question"], x["answer"], x.get("approved", ""))'''),
        py('''ok, msgs = evals.gate(R, e["mean_reward"], min_score=0.5, min_human=0.7)
print("GATE", "PASSED" if ok else "FAILED", msgs)
if SAVE:
    import json; registry.update("evals", e["id"], results_json=json.dumps(R))'''),
    ])),
    "06_human_review": ("views/human_review.py", nb("Step 6 · Human review loop",
        "Rate answers to unseen test questions, write corrections, turn them into the next SFT dataset.", [
        py(SAVE),
        py('''from lab import engines
MODEL = latest("rlhf") or latest("rl") or latest("sft") or BASE
test = data.active_test_rows(data.load(latest_ds("test")))
answers = engines.get(registry.model(MODEL)["base_model"]).generate(MODEL, [t["prompt"] for t in test], max_new_tokens=96)
pd.DataFrame({"question": [t["prompt"] for t in test], "answer": answers})'''),
        py('''# A review = rating + optional correction (the correction becomes a training example)
t, a = test[0], answers[0]
review = dict(model_id=MODEL, test_ds=latest_ds("test"), question_id=t["id"], question=t["prompt"], answer=a,
              rating="partly correct", correction="Check warehouse stock and delivery availability before confirming.")
if SAVE:
    registry.add_review(**review, approved=0)
review'''),
        py('''# Corrections → new SFT version; the used test question is retired (it is no longer unseen)
if SAVE:
    base = latest_ds("sft_train")
    new_sft = data.save_version("sft_train", data.load(base) + [{"id": f"corr-{t['id']}", "prompt": t["prompt"],
                                "completion": review["correction"]}], parent_id=base, source="corrections")
    trows = [dict(x, retired=True) if x["id"] == t["id"] else x for x in data.load(latest_ds("test"))]
    print(new_sft, data.save_version("test", trows, parent_id=latest_ds("test"), source="retire"))'''),
    ])),
    # ------------------------------------------------------------------ 3 · RL
    "07_reward_designer": ("views/reward_designer.py", nb("Step 7 · Design reward rules",
        "Turn 'what a good answer looks like' into scoring rules — then try to fool them.", [
        py(SAVE),
        py('''rules = rewards.DEFAULT_RULES
pd.DataFrame(rules)[["name", "type", "weight"]]'''),
        py('''# Test bench: good, bad and tricky answers. Which ones fool the rules?
pd.DataFrame([{"case": t["label"], "answer": t["answer"], "score": rewards.score(t["answer"], rules, t["row"])["total"]}
              for t in rewards.TEST_BENCH])'''),
        py('''pd.DataFrame(rewards.score("Don't guarantee anything; check warehouse stock.", rules)["breakdown"])'''),
        py('''# Save + approve a version (RL only accepts approved reward versions)
if SAVE:
    rid = registry.add_reward(rules, note="from notebook"); registry.update("rewards", rid, approved=1); print(rid)
pd.DataFrame(registry.rewards())[["id", "approved", "note"]]'''),
    ])),
    "08_rl_rlhf_training": ("views/rl_training.py", nb("Step 8 · RL (rules) and RLHF (reward model) with GRPO",
        "The model writes several answers per question, a reward scores them, GRPO moves toward the better ones. "
        "Reward = rules → RL. Reward = a model trained on human preferences → RLHF.", [
        py('''from lab import jobs
APPROVE_IF_NEEDED = True   # RL needs an approved reward version; approves the newest draft only if none is approved
approved = [r["id"] for r in registry.rewards() if r["approved"]]
if not approved and APPROVE_IF_NEEDED:
    registry.update("rewards", registry.rewards()[0]["id"], approved=1)
    approved = [registry.rewards()[0]["id"]]
PARENT = latest("sft") or BASE
cfg = dict(jobs.RL_PRESETS["Tiny (smoke test)"], reward_source="rules")
print("RL from", PARENT, "with", approved[0], cfg)'''),
        py('''rid = jobs.start("rl", PARENT, latest_ds("rl_train"), latest_ds("rl_val"), cfg, reward_id=approved[0], wait=True)
run = registry.run(rid); print(run["status"], "→", run["output_model_id"], run["error"] or "")
pd.DataFrame(jobs.read_jsonl(run["samples_path"]))[["step", "question", "answer", "total"]].tail(8)'''),
        md("### RLHF — same GRPO, reward from the reward model (train one in notebook 10 first)"),
        py('''RM = latest("rm")
if RM:
    cfg = dict(jobs.RL_PRESETS["Tiny (smoke test)"], reward_source="rm", rm_id=RM)
    rid = jobs.start("rlhf", PARENT, latest_ds("rl_train"), latest_ds("rl_val"), cfg, wait=True)
    run = registry.run(rid); print(run["status"], "→", run["output_model_id"], run["error"] or "")
    display(pd.DataFrame(jobs.read_jsonl(run["samples_path"]))[["question", "answer", "total"]].tail(8))
else:
    print("No reward model yet — run notebooks 09 and 10, then come back.")'''),
    ])),
    # ------------------------------------------------------------------ 4 · RLHF
    "09_human_feedback": ("views/human_feedback.py", nb("Step 9 · Collect human preferences",
        "Two answers to the same question → a person picks the better one → (question, chosen, rejected).", [
        py(SAVE),
        py('''from lab import engines
MODEL = latest("sft") or BASE
eng = engines.get(registry.model(MODEL)["base_model"])
questions = [r["prompt"] for r in data.load(latest_ds("rl_train"))][:3]
pairs = []
for i, q in enumerate(questions):
    a = eng.generate(MODEL, [q], temperature=0.9, seed=11 + i)[0]
    b = eng.generate(MODEL, [q], temperature=0.9, seed=17 + i)[0]
    pairs.append({"prompt": q, "A": a, "B": b})
pd.DataFrame(pairs)'''),
        py('''# YOUR choice goes here. (This demo picks the answer that mentions checking — replace with your judgement.)
choices = ["A" if "check" in p["A"].lower() else "B" for p in pairs]
prefs = [{"prompt": p["prompt"], "chosen": p[c], "rejected": p["B" if c == "A" else "A"]} for p, c in zip(pairs, choices)]
pd.DataFrame(prefs)'''),
        py('''if SAVE:
    base = latest_ds("pref"); old = data.load(base)
    rows = old + [dict(p, id=f"pref-nb{len(old) + i + 1:03d}") for i, p in enumerate(prefs)]
    print(data.save_version("pref", rows, parent_id=base, source="human feedback", note="from notebook"))
print("PREF dataset:", latest_ds("pref"), registry.latest_dataset("pref")["n_rows"], "pairs")'''),
    ])),
    "10_reward_model": ("views/reward_model.py", nb("Step 10 · Train a reward model",
        "A score head on the base model learns score(chosen) > score(rejected) (TRL RewardTrainer + LoRA).", [
        py('''from lab import jobs, reward_model
cfg = dict(jobs.RM_PRESETS["Tiny (smoke test)"])
rid = jobs.start("rm", BASE, latest_ds("pref"), None, cfg, wait=True)
run = registry.run(rid); RM = run["output_model_id"]; print(run["status"], "→", RM, run["error"] or "")'''),
        py('''ref = reward_model.info(RM)
print(f"held-out accuracy {ref['val_accuracy']:.0%} (train {ref['train_accuracy']:.0%}) on {ref['n_val']} pairs — 50% = guessing")
pd.DataFrame(ref["pairs"])'''),
        py('''# Score any two answers
scorer = reward_model.Scorer(RM)
q = "Can I promise a customer their lawn mower arrives tomorrow?"
scorer.score([q, q], ["I will check warehouse stock and delivery slots before confirming a date.",
                      "Yes, it will definitely arrive tomorrow."])'''),
        py('''del scorer   # free GPU memory before RLHF training (notebook 08)'''),
    ])),
    "11_compare_models": ("views/compare_models.py", nb("Step 11 · Finale: SFT vs RL vs RLHF",
        "Same questions, same settings, side by side.", [
        py('''from lab import engines
models = [m for m in (BASE, latest("sft"), latest("rl"), latest("rlhf")) if m]
qs = [t["prompt"] for t in data.active_test_rows(data.load(latest_ds("test")))][:4]
table = {"question": qs}
for mid in models:
    table[mid] = engines.get(registry.model(mid)["base_model"]).generate(mid, qs, max_new_tokens=80)
pd.set_option("display.max_colwidth", 120)
pd.DataFrame(table)'''),
        py('''rules = registry.rewards()[0]["rules"]
pd.Series({mid: sum(rewards.score(a, rules)["total"] for a in table[mid]) / len(qs) for mid in models}, name="mean rule score")'''),
    ])),
    # ------------------------------------------------------------------ 5 · Build apps
    "12_api_langchain_flowise": ("views/api_integrations.py", nb("Step 12 · Serve with FastAPI, call from LangChain",
        "An OpenAI-compatible endpoint for your models — usable from curl, LangChain, Flowise.", [
        py('''import time, httpx
from lab import services
services.start_api()
for _ in range(60):
    if services.api_running(): break
    time.sleep(1)
API = f"http://localhost:{services.API_PORT}"
print("API running:", services.api_running(), API)'''),
        py('''MODEL = latest("sft") or BASE
r = httpx.post(f"{API}/v1/chat/completions", timeout=300,
               json={"model": MODEL, "messages": [{"role": "user", "content": "Can I give a 20% discount?"}], "max_tokens": 80})
r.json()["choices"][0]["message"]["content"]'''),
        py('''from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
llm = ChatOpenAI(base_url=f"{API}/v1", api_key="not-needed", model=MODEL, max_tokens=80)
(ChatPromptTemplate.from_template("Answer for store staff: {q}") | llm).invoke({"q": "Do we price match marketplaces?"}).content'''),
        md(f"**Flowise** (Docker): `services.start_flowise()`, then in a chatflow use *ChatOpenAI Custom* with "
           f"BasePath `http://host.docker.internal:8811/v1`."),
        py('''services.stop("api")   # free GPU memory'''),
    ])),
    "13_rag": ("views/rag_lab.py", nb("Step 13 · Retrieval-augmented generation (RAG)",
        "Chunk documents, retrieve with BM25 / embeddings, answer with a LangChain chain, measure retrieval.", [
        py('''from lab import rag
if "tool_store" not in rag.list_kbs():
    rag.build_sample(embed=True)
pd.DataFrame(rag.retrieve("tool_store", "Who approves a discount for an upset shopper?", k=3, method="embedding"))[["doc", "score", "text"]]'''),
        py('''rows = []
for m in ("bm25", "embedding", "hybrid"):
    _, mean = rag.eval_retrieval("tool_store", rag.sample_eval(), k=3, method=m)
    rows.append({"method": m, **{k: round(v, 3) for k, v in mean.items()}})
pd.DataFrame(rows)'''),
        py('''chain = rag.build_chain(latest("sft") or BASE)      # prompt | LabChatModel | StrOutputParser
q = "How long is the warranty on a drill battery?"
hits = rag.retrieve("tool_store", q, k=3, method="hybrid")
chain.invoke({"context": rag.format_context(hits), "question": q})'''),
    ])),
    "14_agents_langgraph": ("views/agent_lab.py", nb("Step 14 · Agents with LangGraph",
        "agent → tools → agent … until it answers. Every step is traced; scenarios test tool use and safety.", [
        py('''from lab import agent, agent_evals
r = agent.run(BASE, "A customer buys 3 ladders at 89.90 each. What is the total?")
print(r["answer"]); pd.DataFrame(r["steps"])'''),
        py('''# Multi-turn: pass the conversation back in
r1 = agent.run(BASE, "How many cordless drills do we have at store 210?")
r2 = agent.run(BASE, "And at the warehouse?", history=r1["messages"])
print(r1["answer"], "|", r2["answer"])'''),
        py('''scs = [s for s in agent_evals.scenarios() if s["id"] in ("tool-04", "sec-01")]
eid, results, summary = agent_evals.evaluate(BASE, scs)
print(summary); pd.DataFrame(results)[["id", "success", "called", "first_failure"]]'''),
    ])),
    "15_traces_monitoring": ("views/traces.py", nb("Step 15 · Monitor in production",
        "Every RAG / agent / API call is a trace: latency, failures, turn bad traces into tests.", [
        py('''tr = pd.DataFrame(registry.traces(limit=1000))
if tr.empty:
    print("No traces yet — run notebooks 12–14.")
else:
    display(tr.groupby("kind")["latency"].describe()[["count", "50%", "max"]])
    display(tr[["id", "kind", "source", "model_id", "input", "output", "latency"]].head(10))'''),
        py('''# Flag a bad trace and turn it into an agent test (uncomment to write)
# registry.update("traces", int(tr.id[0]), flag="bad")
# from lab import agent_evals
# agent_evals.add_scenario({"type": "tool", "turns": [tr.input[0]], "turn_tools": [["policy_search"]], "must_not_include": ["50%"]})'''),
    ])),
    "16_model_registry": ("views/model_registry.py", nb("Step 16 · Model registry & lineage",
        "Every model version, its parent chain and what produced it.", [
        py('''pd.DataFrame(registry.models())[["id", "kind", "parent_id", "base_model", "train_ds", "reward_id"]]'''),
        py('''mid = registry.models()[-1]["id"]
print(" ← ".join(m["id"] for m in registry.lineage(mid)))
run = registry.run(registry.model(mid)["run_id"]) if registry.model(mid)["run_id"] else None
run and {k: run[k] for k in ("id", "kind", "status", "train_ds", "val_ds", "reward_id")} | {"config": run["config"]}'''),
    ])),
}

PAGE_TO_NOTEBOOK = {page: f"{name}.ipynb" for name, (page, _) in NOTEBOOKS.items()}


def main(only=None):
    OUT.mkdir(exist_ok=True)
    for name, (_, book) in NOTEBOOKS.items():
        if only and name not in only:
            continue
        nbf.write(book, OUT / f"{name}.ipynb")
    print(f"wrote {len(NOTEBOOKS)} notebooks to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
