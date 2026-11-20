"""CI gate for the lab: exit 0 if an evaluation meets the thresholds, exit 1 if not.

  # check an existing evaluation record (no GPU needed)
  .venv/bin/python scripts/eval_gate.py --eval-id 3 --min-score 0.5 --min-human 0.7
  .venv/bin/python scripts/eval_gate.py --latest --min-score 0.5
  # run a fresh evaluation first, then gate it (loads the model)
  .venv/bin/python scripts/eval_gate.py --model sft-001 --dataset rl_val-v1 --reward reward-v1 --min-score 0.5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lab import data, evals, registry, rewards  # noqa: E402


def run_eval(model_id, dataset_id, reward_id, max_new_tokens=96):
    from lab.model_loader import Engine
    rows = data.load(dataset_id)
    rules = registry.reward(reward_id)["rules"]
    answers = Engine(registry.model(model_id)["base_model"]).generate(model_id, [r["prompt"] for r in rows],
                                                                       max_new_tokens=max_new_tokens)
    results = []
    for r, a in zip(rows, answers):
        s = rewards.score(a, rules, r)
        results.append({"id": r["id"], "question": r["prompt"], "answer": a, "approved": r.get("completion", ""),
                        "reward": s["total"], "breakdown": s["breakdown"], "review": ""})
    mean = sum(x["reward"] for x in results) / max(1, len(results))
    return registry.add_eval(model_id, dataset_id, reward_id,
                             {"max_new_tokens": max_new_tokens, "temperature": 0, "source": "eval_gate.py"}, results, mean)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-id", type=int)
    ap.add_argument("--latest", action="store_true", help="use the newest evaluation record")
    ap.add_argument("--model"), ap.add_argument("--dataset"), ap.add_argument("--reward")
    ap.add_argument("--min-score", type=float, help="minimum mean automated rule score")
    ap.add_argument("--min-human", type=float, help="minimum share of reviewed answers rated correct")
    a = ap.parse_args(argv)
    if a.model:
        if not (a.dataset and a.reward):
            ap.error("--model needs --dataset and --reward")
        a.eval_id = run_eval(a.model, a.dataset, a.reward)
    evs = registry.evals()
    e = evs[0] if a.latest and evs else next((x for x in evs if x["id"] == a.eval_id), None)
    if not e:
        print("No matching evaluation record.")
        return 2
    ok, msgs = evals.gate(e["results"], e["mean_reward"], a.min_score, a.min_human)
    print(f"Evaluation #{e['id']} · {e['model_id']} on {e['dataset_id']} · {e['reward_id']}")
    print("\n".join("  " + m for m in msgs) or "  (no thresholds given)")
    print("GATE PASSED" if ok else "GATE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
