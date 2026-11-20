"""Full real-training smoke test of the cycle: base -> SFT v1 -> RL v1 -> SFT v2 (from RL v1).
Usage: LAB_HOME=/tmp/labsmoke .venv/bin/python scripts/smoke_cycle.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab import data, jobs, model_loader, paths, registry, rewards  # noqa: E402

data.bootstrap_samples()
base = registry.ensure_base(paths.DEFAULT_BASE_MODEL)
rid = registry.reward("reward-v1") and "reward-v1" or registry.add_reward(rewards.DEFAULT_RULES, "default")
registry.update("rewards", rid, approved=1)


def go(kind, parent, train, val, preset, reward=None):
    cfg = dict((jobs.SFT_PRESETS if kind == "sft" else jobs.RL_PRESETS)[preset])
    run_id = jobs.start(kind, parent, train, val, cfg, reward_id=reward, wait=True)
    r = registry.run(run_id)
    print(run_id, r["status"], r["output_model_id"], (r["error"] or "")[:500])
    if r["status"] != "completed":
        print(jobs.tail(r["log_path"], 40))
        sys.exit(1)
    m = jobs.read_jsonl(r["metrics_path"])
    print("  metrics rows:", len(m), "last:", {k: v for k, v in m[-1].items() if k in ("step", "loss", "eval_loss", "reward")})
    return r["output_model_id"]


sft1 = go("sft", base, "sft_train-v1", "sft_val-v1", "Tiny (smoke test)")
rl1 = go("rl", sft1, "rl_train-v1", "rl_val-v1", "Tiny (smoke test)", reward=rid)
sft2 = go("sft", rl1, "sft_train-v1", "sft_val-v1", "Tiny (smoke test)")
print("lineage:", [m["id"] for m in registry.lineage(sft2)])

eng = model_loader.Engine(paths.DEFAULT_BASE_MODEL)
for mid in (base, sft1, rl1, sft2):
    m = registry.model(mid)
    ref = json.loads(m["reference_json"] or "{}")
    q = ref.get("questions") or ["Can I promise a drill tomorrow?"]
    ans = eng.generate(mid, q, max_new_tokens=ref.get("max_new_tokens", 64))
    same = ans == ref.get("answers") if ref else None
    print(f"{mid}: reload matches saved reference answers: {same}\n   {ans[0][:160]}")
