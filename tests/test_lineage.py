import json

import pytest

from lab import jobs, model_loader, registry, status


def _fake_adapter(path, base):
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base, "r": 8, "lora_alpha": 16}))
    return path


def test_lineage_chain_and_parent_resolution(lab_home):
    base = registry.ensure_base("Qwen/Qwen3-0.6B")
    assert registry.ensure_base("Qwen/Qwen3-0.6B") == base
    sft1 = registry.add_model("sft", base, "Qwen/Qwen3-0.6B", "rev", _fake_adapter(lab_home / "s1", "Qwen/Qwen3-0.6B"),
                              "run-0001", "sft_train-v1", "sft_val-v1")
    rl1 = registry.add_model("rl", sft1, "Qwen/Qwen3-0.6B", "rev", _fake_adapter(lab_home / "r1", "Qwen/Qwen3-0.6B"),
                             "run-0002", "rl_train-v1", "rl_val-v1", reward_id="reward-v1")
    sft2 = registry.add_model("sft", rl1, "Qwen/Qwen3-0.6B", "rev", _fake_adapter(lab_home / "s2", "Qwen/Qwen3-0.6B"),
                              "run-0003", "sft_train-v2", "sft_val-v1")
    assert [m["id"] for m in registry.lineage(sft2)] == ["sft-002", "rl-001", "sft-001", "base-001"]
    # the selected parent's own adapter is resolved - never the base model
    assert model_loader.resolve(rl1) == ("Qwen/Qwen3-0.6B", str(lab_home / "r1"))
    assert model_loader.resolve(base) == ("Qwen/Qwen3-0.6B", None)


def test_missing_or_mismatched_adapter_fails_loudly(lab_home):
    base = registry.ensure_base("Qwen/Qwen3-0.6B")
    gone = registry.add_model("sft", base, "Qwen/Qwen3-0.6B", "rev", lab_home / "nope", "run-1", "a", "b")
    with pytest.raises(model_loader.LineageError):
        model_loader.resolve(gone)
    wrong = registry.add_model("sft", base, "Qwen/Qwen3-0.6B", "rev", _fake_adapter(lab_home / "w", "other/model"),
                               "run-2", "a", "b")
    with pytest.raises(model_loader.LineageError):
        model_loader.resolve(wrong)


def test_run_records_selected_parent_and_unique_output_ids(lab_home, monkeypatch):
    base = registry.ensure_base("Qwen/Qwen3-0.6B")
    sft1 = registry.add_model("sft", base, "Qwen/Qwen3-0.6B", "rev", _fake_adapter(lab_home / "s1", "Qwen/Qwen3-0.6B"),
                              "run-0001", "sft_train-v1", "sft_val-v1")

    class P:
        pid = 999999

        def wait(self):
            pass
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: P())
    rid = jobs.start("rl", sft1, "rl_train-v1", "rl_val-v1", dict(jobs.RL_PRESETS["Tiny (smoke test)"]),
                     reward_id="reward-v1")
    r = registry.run(rid)
    assert r["parent_model_id"] == sft1 and r["output_model_id"] == "rl-001" and r["kind"] == "rl"
    with pytest.raises(RuntimeError):
        jobs.start("rl", "does-not-exist", "x", None, {})


def test_pipeline_next_action(lab_home):
    p = status.pipeline()
    assert p["done"]["Data"] is False and p["next"][0] == "Data Studio"
