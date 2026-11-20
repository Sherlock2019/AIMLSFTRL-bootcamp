"""Pipeline status and 'next recommended action', computed only from completed, recorded steps."""
from . import registry

STAGES = ["Data", "SFT", "Evaluation", "Reward Design", "RL", "Review", "Next Cycle"]


def pipeline():
    ms = registry.models()
    sfts = [m for m in ms if m["kind"] == "sft"]
    rls = [m for m in ms if m["kind"] == "rl"]
    latest_sft = sfts[-1] if sfts else None
    rl_of_sft = [m for m in rls if latest_sft and m["parent_id"] == latest_sft["id"]]
    latest_rl = rl_of_sft[-1] if rl_of_sft else None
    evals = registry.evals(latest_sft["id"]) if latest_sft else []
    approved = [r for r in registry.rewards() if r["approved"]]
    revs = registry.reviews(latest_rl["id"]) if latest_rl else []
    corr = [d for d in registry.datasets("sft_train") if d["source"] == "corrections"
            and latest_rl and d["created"] > latest_rl["created"]]
    done = {
        "Data": bool(registry.datasets()),
        "SFT": bool(latest_sft),
        "Evaluation": bool(evals),
        "Reward Design": bool(approved),
        "RL": bool(latest_rl),
        "Review": bool(revs),
        "Next Cycle": bool(corr),
    }
    return {"done": done, "cycle": max(1, len(sfts)), "latest_sft": latest_sft, "latest_rl": latest_rl,
            "approved_rewards": approved, "next": next_action(done)}


def next_action(done):
    tips = {
        "Data": ("Data Studio", "Inspect the sample datasets."),
        "SFT": ("SFT Training", "Run a tiny SFT job from the base model."),
        "Evaluation": ("Evaluation & Scoring", "Run the SFT model on validation questions and review its answers."),
        "Reward Design": ("Reward Designer", "Test the reward rules on good and bad answers, then approve a version."),
        "RL": ("RL Training", "Run a tiny GRPO job starting from your latest SFT model."),
        "Review": ("Human Review", "Rate the RL model's answers on the test questions and write corrections."),
        "Next Cycle": ("Human Review", "Turn approved corrections into a new SFT dataset version, then train SFT "
                                       "again from the RL model."),
    }
    for s in STAGES:
        if not done[s]:
            return tips[s]
    return ("SFT Training", "Cycle complete. Start the next SFT stage from the RL model using the new dataset version.")
