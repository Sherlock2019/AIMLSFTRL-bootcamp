from lab import data, registry


def test_bootstrap_registers_all_roles(lab_home):
    data.bootstrap_samples()
    for role in data.ROLES:
        d = registry.latest_dataset(role)
        assert d and d["version"] == 1 and d["n_rows"] > 0
    data.bootstrap_samples()  # idempotent
    assert registry.latest_dataset("sft_train")["version"] == 1


def test_sample_splits_do_not_leak(lab_home):
    data.bootstrap_samples()
    splits = {registry.latest_dataset(r)["id"]: data.load(registry.latest_dataset(r)["id"]) for r in data.ROLES}
    assert data.leakage(splits) == []


def test_leakage_and_duplicates_detected():
    a = [{"id": "1", "prompt": "Can I promise a drill tomorrow?"}]
    b = [{"id": "x", "prompt": "can i promise a DRILL tomorrow"}]
    assert len(data.leakage({"sft_train-v1": a, "test-v1": b})) == 1
    # retired test questions are no longer counted as leakage
    assert data.leakage({"sft_train-v1": a, "test-v2": [dict(b[0], retired=True)]}) == []
    warns = data.quality_warnings("sft_train", data.clean_rows("sft_train", a + a))
    assert any("duplicate" in w.lower() for w in warns)


def test_versioning_hash_and_diff(lab_home):
    rows = [{"id": "a", "prompt": "q1", "completion": "c1"}]
    v1 = data.save_version("sft_train", rows, source="edit")
    assert data.save_version("sft_train", rows, source="edit") == v1  # identical content -> no new version
    v2 = data.save_version("sft_train", rows + [{"id": "b", "prompt": "q2", "completion": "c2"}], parent_id=v1)
    assert v2 == "sft_train-v2" and registry.dataset(v2)["parent_id"] == v1
    added, removed, changed = data.diff(data.load(v1), data.load(v2))
    assert [r["id"] for r in added] == ["b"] and not removed and not changed


def test_csv_list_fields_roundtrip():
    rows = data.clean_rows("rl_train", [{"prompt": "q", "required_checks": "warehouse stock; delivery availability"}])
    assert rows[0]["required_checks"] == ["warehouse stock", "delivery availability"]
    back = data.parse_upload("x.csv", data.to_csv(rows).encode())
    assert data.clean_rows("rl_train", back)[0]["required_checks"] == rows[0]["required_checks"]
