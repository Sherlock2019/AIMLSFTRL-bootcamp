import pytest

from lab import data, evals, registry


def test_ten_sft_packs_are_valid_and_leak_free():
    packs = data.list_packs()
    assert len(packs) == 10
    seen = {}
    for p in packs:
        _, train, val = data.pack_rows(p["slug"])
        assert len(train) >= 10 and len(val) >= 3
        for r in train + val:
            assert r["prompt"].strip() and r["completion"].strip()
            key = data.normalize(r["prompt"])
            assert key not in seen, f"duplicate question in {p['slug']} and {seen.get(key)}"
            seen[key] = p["slug"]
    # no pack question collides with the bundled tool-store sample
    for role in data.ROLES:
        for r in data.read_jsonl(data.paths.SAMPLE_DIR / f"{role}.jsonl"):
            assert data.normalize(r["prompt"]) not in seen


def test_load_pack_replace_and_append(lab_home):
    data.bootstrap_samples()
    n0 = registry.latest_dataset("sft_train")["n_rows"]
    tr, va = data.load_pack("01_it_helpdesk", "replace")
    assert registry.dataset(tr)["n_rows"] == 12 and registry.dataset(va)["n_rows"] == 4
    assert registry.dataset(tr)["source"] == "pack"
    tr2, _ = data.load_pack("02_hr_policy", "append")
    assert registry.dataset(tr2)["n_rows"] == 24 and registry.dataset(tr2)["parent_id"] == tr
    assert n0 > 0 and data.load(tr2)[0]["id"].startswith("it_helpdes")


def test_find_replace():
    rows = [{"id": "a", "prompt": "q", "completion": "Ask a Manager."}, {"id": "b", "prompt": "q2", "completion": "ok"}]
    out, n = data.find_replace(rows, "completion", "manager", "supervisor")
    assert n == 1 and out[0]["completion"] == "Ask a supervisor." and rows[0]["completion"] == "Ask a Manager."


def _res(reward, review, **kw):
    return dict(id=kw.pop("id", "x"), question="q", answer="a", approved="good", reward=reward, review=review, **kw)


def test_kappa():
    assert evals.cohen_kappa(["correct", "incorrect"] * 5, ["correct", "incorrect"] * 5)[0] == pytest.approx(1.0)
    k, n = evals.cohen_kappa(["correct", "correct", "incorrect", "incorrect"], ["correct", "incorrect", "correct", "incorrect"])
    assert n == 4 and k == pytest.approx(0.0)


def test_judge_report_and_correction():
    R = [_res(1.0, "correct")] * 8 + [_res(0.0, "correct")] * 2 + [_res(0.0, "incorrect")] * 8 + [_res(1.0, "incorrect")] * 2
    rep = evals.judge_report(R, 0.5)
    assert (rep["tp"], rep["fn"], rep["tn"], rep["fp"]) == (8, 2, 8, 2)
    assert rep["tpr"] == pytest.approx(0.8) and rep["tnr"] == pytest.approx(0.8)
    assert rep["corrected_pass_rate"] == pytest.approx(0.5) and rep["ci95"][0] <= 0.5 <= rep["ci95"][1]


def test_error_analysis_sampling_fixes_and_gate():
    R = [_res(0.9, "incorrect", id="1", failure_mode="promised a date", cause="data gap", correction="Check first."),
         _res(0.1, "partly correct", id="2", failure_mode="Promised a date"),
         _res(0.5, "correct", id="3"), _res(0.2, "", id="4")]
    assert evals.failure_table(R)[0][:2] == ("promised a date", 2)
    assert evals.order_for_review(R, "Not reviewed yet") == [3]
    assert evals.order_for_review(R, "Judge vs human disagree") == [0]
    rows = evals.failures_to_sft_rows(R, 7)
    assert [r["completion"] for r in rows] == ["Check first.", "good"] and rows[0]["id"] == "fix-e7-1"
    ok, _ = evals.gate(R, 0.42, min_score=0.4, min_human=0.5)
    assert not ok  # 1/3 reviewed correct
    assert evals.gate(R, 0.42, min_score=0.4)[0]
