"""RLHF pieces: preference data, reward-model helpers, leakage rules, course tracks (no model needed)."""
from lab import course, data, registry, reward_model


def test_pref_role_bootstraps_and_does_not_leak(lab_home):
    data.bootstrap_samples()
    d = registry.latest_dataset("pref")
    rows = data.load(d["id"])
    assert d["n_rows"] == 24 and set(data.FIELDS["pref"]) <= set(rows[0])
    assert data.quality_warnings("pref", rows) == []
    splits = {registry.latest_dataset(r)["id"]: data.load(registry.latest_dataset(r)["id"]) for r in data.ROLES}
    assert data.leakage(splits) == []


def test_pref_quality_flags_identical_pair():
    rows = data.clean_rows("pref", [{"prompt": "q", "chosen": "Same.", "rejected": "same"}] * 1)
    assert any("identical" in w for w in data.quality_warnings("pref", rows))


def test_leakage_only_counts_val_or_test_overlap():
    q = [{"id": "1", "prompt": "Can I promise a drill?"}]
    assert data.leakage({"pref-v1": q, "rl_train-v1": q}) == []        # two training sets: fine
    assert len(data.leakage({"pref-v1": q, "rl_val-v1": q})) == 1      # training vs validation: leakage


def test_reward_model_split_and_accuracy():
    rows = [{"id": str(i), "prompt": f"q{i}", "chosen": "c", "rejected": "r"} for i in range(10)]
    tr, va = reward_model.split(rows, 0.2, seed=1)
    assert len(va) == 2 and len(tr) == 8 and not ({r["id"] for r in tr} & {r["id"] for r in va})
    assert reward_model.split(rows[:3], 0.2)[1] == []                 # too few to hold out
    acc, margin = reward_model.accuracy([2.0, 1.0, 0.0], [1.0, 2.0, -1.0])
    assert round(acc, 3) == 0.667 and round(margin, 3) == 0.333


def test_course_has_three_workflows_in_order(lab_home):
    tracks = [m["track"] for m in course.MODULES]
    order = [t for i, t in enumerate(tracks) if t not in tracks[:i]]
    assert order == ["setup", "A", "B", "C", "D"]
    ids = [t[0] for m in course.MODULES for t in m["tasks"]]
    assert len(ids) == len(set(ids))
    data.bootstrap_samples()
    mods, done, total = course.status()
    assert total >= 35 and done == 0
