"""Real (tiny) training: SFT from base, then SFT continuing from that adapter; save + reload reproduces answers.
Skipped automatically when the base model is not downloaded. Uses a 3-step config to stay fast."""
import json

import pytest

from lab import data, hardware, jobs, model_loader, paths, registry, rewards

pytestmark = pytest.mark.skipif(not hardware.model_cached(paths.DEFAULT_BASE_MODEL),
                                reason="base model not downloaded")


def _cfg(kind):
    c = dict((jobs.SFT_PRESETS if kind == "sft" else jobs.RL_PRESETS)["Tiny (smoke test)"])
    c["max_steps"] = 3 if kind == "sft" else 2
    return c


def test_sft_rl_sft_chain_and_reload(lab_home):
    data.bootstrap_samples()
    base = registry.ensure_base(paths.DEFAULT_BASE_MODEL)
    rid = registry.add_reward(rewards.DEFAULT_RULES)
    registry.update("rewards", rid, approved=1)
    out = []
    for kind, parent, tr, va in [("sft", base, "sft_train-v1", "sft_val-v1"),
                                 ("rl", None, "rl_train-v1", "rl_val-v1"),
                                 ("sft", None, "sft_train-v1", "sft_val-v1")]:
        parent = parent or out[-1]
        run_id = jobs.start(kind, parent, tr, va, _cfg(kind), reward_id=rid if kind == "rl" else None, wait=True)
        r = registry.run(run_id)
        assert r["status"] == "completed", jobs.tail(r["log_path"])
        m = registry.model(r["output_model_id"])
        assert m["parent_id"] == parent  # each stage starts from the selected previous output
        assert any("loss" in x for x in jobs.read_jsonl(r["metrics_path"]))
        out.append(m["id"])
    assert [m["id"] for m in registry.lineage(out[-1])] == ["sft-002", "rl-001", "sft-001", "base-001"]
    # child adapter weights differ from the parent's: training actually continued
    from safetensors.torch import load_file
    w1 = load_file(f"{registry.model(out[1])['adapter_path']}/adapter_model.safetensors")
    w2 = load_file(f"{registry.model(out[2])['adapter_path']}/adapter_model.safetensors")
    assert w1.keys() == w2.keys() and any((w1[k] != w2[k]).any() for k in w1)
    eng = model_loader.Engine(paths.DEFAULT_BASE_MODEL)
    for mid in out:
        ref = json.loads(registry.model(mid)["reference_json"])
        assert eng.generate(mid, ref["questions"], max_new_tokens=ref["max_new_tokens"]) == ref["answers"]
