"""The training path: modules → hands-on tasks. Most tasks tick themselves by looking at what you actually did
in the lab (registry state); a few are ticked by you (manual) because the lab cannot see them."""
from . import registry


def _state():
    ds = registry.datasets()
    evs = registry.evals()
    tr = registry.traces(limit=2000)
    models = registry.models()
    prog = registry.progress()
    try:
        from . import agent_evals, rag, services
        kbs, user_sc, fl = rag.list_kbs(), agent_evals.user_file().exists(), services.flowise_running()
    except Exception:  # noqa: BLE001
        kbs, user_sc, fl = [], False, False
    res = [x for e in evs for x in e["results"]]
    return dict(
        ds_sources={d["source"] for d in ds}, models=models, kinds={m["kind"] for m in models},
        bases={m["base_model"] for m in models if m["kind"] == "base"}, evs=evs, res=res,
        reviewed=sum(bool(x.get("review")) for x in res),
        tagged=sum(bool(x.get("failure_mode")) for x in res),
        two_reviewers=any(len(x.get("reviews") or {}) >= 2 for x in res),
        judged=any(x.get("judge") for x in res),
        approved_reward=any(r["approved"] for r in registry.rewards()),
        reviews=registry.reviews(), tr_kinds={t["kind"] for t in tr},
        multi_turn=any(t["kind"] == "agent" and (t["meta"] or {}).get("history_len", 0) > 0 for t in tr),
        flagged=any(t["flag"] for t in tr), agent_evals=registry.agent_evals(), kbs=kbs, user_sc=user_sc,
        flowise=fl, prog=prog,
        eval_bases={registry.model(e["model_id"])["base_model"] for e in evs if registry.model(e["model_id"])},
        n_sft=sum(m["kind"] == "sft" for m in models),
        rl_eval=any((registry.model(e["model_id"]) or {}).get("kind") == "rl" for e in evs),
        cmp_kinds={(registry.model(e["model_id"]) or {}).get("kind") for e in evs if e["settings"].get("source") == "compare"},
        rlhf_reviewed=any((registry.model(r["model_id"]) or {}).get("kind") == "rlhf" for r in registry.reviews()),
    )


TRACKS = {
    "setup": ("Setup", "Your machine and your models."),
    "A": ("Track A · Supervised learning (SFT), A to Z", "Data → train → evaluate → analyse errors → human corrections → retrain."),
    "B": ("Track B · Reinforcement learning (RL)", "Write reward rules → GRPO → check the model didn't just game the rules."),
    "C": ("Track C · RLHF = SFT + human feedback + RL", "Collect human preferences → train a reward model → RL with it → compare all three."),
    "D": ("Track D · Build apps on your model (optional)", "Serve it, RAG, agents, Flowise, monitoring."),
}

MODULES = [
    {"id": "m1", "track": "setup", "title": "Setup & your hardware", "concept": "Know what your machine can train before you start.",
     "tasks": [("m1-hw", "Open Settings & Hardware and read the recommended mode", "views/settings_hardware.py", None)]},
    {"id": "m8", "track": "setup", "title": "Import models from Hugging Face", "concept": "Compare model sizes and families on the same tasks.",
     "tasks": [("m8-imp", "Import a second base model on Model Hub", "views/model_hub.py", lambda s: len(s["bases"]) >= 2),
               ("m8-cmp", "Evaluate the imported model on the same validation set", "views/evaluation.py", lambda s: len(s["eval_bases"]) >= 2)]},
    {"id": "m2", "track": "A", "title": "Supervised data", "concept": "A model learns what its examples show. Good data beats clever training.",
     "tasks": [("m2-pack", "Load one of the 10 SFT packs", "views/data_studio.py", lambda s: "pack" in s["ds_sources"]),
               ("m2-edit", "Edit or add one example in 📝 Edit one example", "views/data_studio.py", lambda s: "edit" in s["ds_sources"]),
               ("m2-leak", "Read the Quality & leakage tab and explain why leakage matters", "views/data_studio.py", None)]},
    {"id": "m3", "track": "A", "title": "Supervised fine-tuning (SFT)", "concept": "Teach the model your answers with a LoRA adapter.",
     "tasks": [("m3-sft", "Train an SFT model (Tiny preset)", "views/sft_training.py", lambda s: "sft" in s["kinds"])]},
    {"id": "m4", "track": "A", "title": "Evaluation & error analysis", "concept": "Read real answers, label them, and count *why* they fail.",
     "tasks": [("m4-eval", "Run an evaluation", "views/evaluation.py", lambda s: bool(s["evs"])),
               ("m4-review", "Label at least 10 answers in Workbench ①", "views/evals_workbench.py", lambda s: s["reviewed"] >= 10),
               ("m4-tag", "Tag at least 3 failures and read the chart in ②", "views/evals_workbench.py", lambda s: s["tagged"] >= 3)]},
    {"id": "m5", "track": "A", "title": "Collaborative evaluation & judges", "concept": "Agree on 'good' as a team, then automate it — but validate the judge.",
     "tasks": [("m5-two", "Have a second reviewer label the same answers (③ agreement)", "views/evals_workbench.py", lambda s: s["two_reviewers"]),
               ("m5-llmj", "Run an LLM judge and check its TPR/TNR in ④", "views/evals_workbench.py", lambda s: s["judged"])]},
    {"id": "m7", "track": "A", "title": "Human review loop", "concept": "Corrections from people become the next training data.",
     "tasks": [("m7-rev", "Rate test answers and write a correction", "views/human_review.py", lambda s: bool(s["reviews"])),
               ("m7-ds", "Create an SFT dataset from corrections", "views/human_review.py", lambda s: "corrections" in s["ds_sources"]),
               ("m7-re", "Retrain SFT on the corrected dataset (2nd SFT model)", "views/sft_training.py", lambda s: s["n_sft"] >= 2)]},
    {"id": "m6", "track": "B", "title": "Reward rules", "concept": "Turn 'what good looks like' into scoring rules — and try to fool them.",
     "tasks": [("m6-rew", "Run the test bench and approve a reward version", "views/reward_designer.py", lambda s: s["approved_reward"])]},
    {"id": "mB2", "track": "B", "title": "RL training (GRPO)", "concept": "The model writes several answers, the rules score them, GRPO moves toward the better ones.",
     "tasks": [("m6-rl", "Train an RL model from your SFT model (reward source: rules)", "views/rl_training.py", lambda s: "rl" in s["kinds"]),
               ("mB2-ev", "Read the RL validation reward and sampled answers", "views/rl_training.py", lambda s: s["rl_eval"])]},
    {"id": "mB3", "track": "B", "title": "Did RL help, or game the rules?", "concept": "A higher reward is not automatically a better answer.",
     "tasks": [("mB3-cmp", "Compare SFT vs RL answers on the same questions", "views/compare_models.py", lambda s: s["cmp_kinds"] >= {"sft", "rl"}),
               ("mB3-hack", "Find one answer that scores high but is wrong (Workbench ④ false positives)", "views/evals_workbench.py", None)]},
    {"id": "mC1", "track": "C", "title": "Collect human preferences", "concept": "People compare two answers; their choices are the human feedback.",
     "tasks": [("mC1-pref", "Pick the better answer for 10+ pairs and save a PREF version", "views/human_feedback.py",
                lambda s: "human feedback" in s["ds_sources"])]},
    {"id": "mC2", "track": "C", "title": "Train a reward model", "concept": "A model learns to predict which answer people prefer.",
     "tasks": [("mC2-rm", "Train a reward model", "views/reward_model.py", lambda s: "rm" in s["kinds"]),
               ("mC2-acc", "Check held-out accuracy and try it on two answers", "views/reward_model.py", None)]},
    {"id": "mC3", "track": "C", "title": "RL with human feedback (RLHF)", "concept": "Same GRPO, but the reward now comes from the reward model.",
     "tasks": [("mC3-rlhf", "Train an RLHF model (reward source: reward model)", "views/rl_training.py", lambda s: "rlhf" in s["kinds"])]},
    {"id": "mC4", "track": "C", "title": "Finale: SFT vs RL vs RLHF", "concept": "Same questions, same settings — which workflow gave the best answers?",
     "tasks": [("mC4-cmp", "Compare SFT, RL and RLHF side by side", "views/compare_models.py", lambda s: s["cmp_kinds"] >= {"sft", "rl", "rlhf"}),
               ("mC4-rev", "Review the RLHF model's test answers in Human Review", "views/human_review.py", lambda s: s["rlhf_reviewed"])]},
    {"id": "m9", "track": "D", "title": "Serve your model with FastAPI", "concept": "An OpenAI-compatible endpoint lets any app use your trained model.",
     "tasks": [("m9-api", "Start the API and send a request (Try it panel or curl)", "views/api_integrations.py", lambda s: "api" in s["tr_kinds"]),
               ("m9-lc", "Call it from LangChain with the ChatOpenAI snippet", "views/api_integrations.py", None)]},
    {"id": "m10", "track": "D", "title": "Retrieval-augmented generation (RAG)", "concept": "Give the model the right documents instead of hoping it memorised them.",
     "tasks": [("m10-kb", "Build a knowledge base", "views/rag_lab.py", lambda s: bool(s["kbs"])),
               ("m10-ask", "Ask a question and check the cited sources", "views/rag_lab.py", lambda s: "rag" in s["tr_kinds"]),
               ("m10-ir", "Measure retrieval (recall@k, MRR, nDCG) and compare BM25 vs embeddings", "views/rag_lab.py",
                lambda s: s["prog"].get("m10-ir", False))]},
    {"id": "m11", "track": "D", "title": "Agents & tool use (LangGraph)", "concept": "Let the model call tools in a loop — and trace every step.",
     "tasks": [("m11-run", "Run the agent and open its step-by-step trace", "views/agent_lab.py", lambda s: "agent" in s["tr_kinds"]),
               ("m11-multi", "Hold a multi-turn conversation (2+ turns)", "views/agent_lab.py", lambda s: s["multi_turn"]),
               ("m11-eval", "Run the agent eval (tool use, multi-turn, security)", "views/agent_lab.py", lambda s: bool(s["agent_evals"]))]},
    {"id": "m12", "track": "D", "title": "Visual flows with Flowise", "concept": "Build a no-code chatflow on top of your own model.",
     "tasks": [("m12-run", "Install and start Flowise", "views/api_integrations.py", lambda s: s["flowise"]),
               ("m12-flow", "Build a chatflow that answers through your model's API", "views/api_integrations.py", None)]},
    {"id": "m13", "track": "D", "title": "CI/CD & monitoring", "concept": "Gate releases on evals; turn production failures into tests.",
     "tasks": [("m13-flag", "Flag a bad trace in Traces & Monitoring", "views/traces.py", lambda s: s["flagged"]),
               ("m13-test", "Turn a trace into a new agent test case", "views/traces.py", lambda s: s["user_sc"]),
               ("m13-gate", "Run scripts/eval_gate.py (command in Workbench ⑥)", "views/evals_workbench.py", None)]},
    {"id": "m14", "track": "D", "title": "Improve with evals", "concept": "Use evals to choose fixes, models and trade-offs (quality vs speed).",
     "tasks": [("m14-fix", "Turn evaluation failures into training rows (Workbench ⑥)", "views/evals_workbench.py", lambda s: "eval-fixes" in s["ds_sources"]),
               ("m14-cmp", "Run the agent eval on two different models and compare", "views/agent_lab.py",
                lambda s: len({e["model_id"] for e in s["agent_evals"]}) >= 2)]},
]


def status():
    """[(module, [(task_id, title, page, auto, done)])], overall done, total."""
    s = _state()
    out, done, total = [], 0, 0
    for m in MODULES:
        rows = []
        for tid, title, page, check in m["tasks"]:
            try:
                ok = bool(check(s)) if check else s["prog"].get(tid, False)
            except Exception:  # noqa: BLE001
                ok = False
            rows.append((tid, title, page, check is not None, ok))
            done += ok
            total += 1
        out.append((m, rows))
    return out, done, total
