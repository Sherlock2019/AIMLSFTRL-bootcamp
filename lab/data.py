"""Dataset versioning, import/export, quality checks and split-leakage warnings."""
import csv
import hashlib
import io
import json
import re
from pathlib import Path

from . import paths, registry

ROLES = {
    "sft_train": "SFT training (question + approved answer) - teaches the model",
    "sft_val": "SFT validation (different questions + approved answer) - monitors SFT",
    "rl_train": "RL training (question + optional grading fields) - answers are generated and scored",
    "rl_val": "RL validation (different questions + grading fields) - checks reward-based performance",
    "test": "Manual test (question only) - you review the model's answers",
    "pref": "Human preferences (question + chosen answer + rejected answer) - trains the reward model for RLHF",
}
FIELDS = {
    "sft_train": ["id", "prompt", "completion"],
    "sft_val": ["id", "prompt", "completion"],
    "rl_train": ["id", "prompt", "required_checks", "avoid"],
    "rl_val": ["id", "prompt", "required_checks", "avoid"],
    "test": ["id", "prompt", "retired"],
    "pref": ["id", "prompt", "chosen", "rejected"],
}
LIST_FIELDS = {"required_checks", "avoid"}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(text).lower())).strip()


def clean_rows(role, rows):
    out = []
    for i, r in enumerate(rows):
        r = {k: v for k, v in dict(r).items() if k is not None}
        if not str(r.get("prompt", "")).strip():
            continue
        r["id"] = str(r.get("id") or f"{role}-{i + 1:03d}")
        for f in LIST_FIELDS & set(FIELDS[role]):
            v = r.get(f, [])
            if isinstance(v, str):
                v = [s.strip() for s in re.split(r"[;|]", v) if s.strip()] if v.strip() and not v.strip().startswith("[") else (json.loads(v) if v.strip() else [])
            r[f] = list(v or [])
        if role == "test":
            r["retired"] = str(r.get("retired", False)).lower() in ("true", "1", "yes")
        out.append(r)
    return out


def rows_hash(rows) -> str:
    blob = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_version(role, rows, parent_id=None, source="edit", note=""):
    """Save rows as a new immutable dataset version and register it. Returns the dataset id."""
    rows = clean_rows(role, rows)
    h = rows_hash(rows)
    latest = registry.latest_dataset(role)
    if latest and latest["hash"] == h and source != "sample":
        return latest["id"]  # identical content: no new version
    version = (latest["version"] if latest else 0) + 1
    path = paths.sub("datasets") / role / f"v{version}.jsonl"
    write_jsonl(path, rows)
    return registry.add_dataset(role, path, h, len(rows), parent_id=parent_id, source=source, note=note)


def load(did):
    d = registry.dataset(did)
    if not d:
        raise KeyError(f"Dataset {did} not found in registry")
    return read_jsonl(d["path"])


def parse_upload(name: str, content: bytes):
    text = content.decode("utf-8-sig")
    if name.lower().endswith(".csv"):
        return list(csv.DictReader(io.StringIO(text)))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def to_csv(rows) -> str:
    buf = io.StringIO()
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "id", k != "prompt", k))
    w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow({k: ("; ".join(v) if isinstance(v, list) else v) for k, v in r.items()})
    return buf.getvalue()


def quality_warnings(role, rows):
    warns = []
    seen = {}
    for r in rows:
        key = normalize(r.get("prompt", ""))
        if key in seen:
            warns.append(f"Exact duplicate question: '{r['prompt'][:60]}' (ids {seen[key]} and {r['id']})")
        seen[key] = r["id"]
    if role in ("sft_train", "sft_val"):
        for r in rows:
            if not str(r.get("completion", "")).strip():
                warns.append(f"Row {r['id']} has no approved answer")
    if role == "pref":
        for r in rows:
            if not str(r.get("chosen", "")).strip() or not str(r.get("rejected", "")).strip():
                warns.append(f"Row {r['id']} needs both a chosen and a rejected answer")
            elif normalize(r["chosen"]) == normalize(r["rejected"]):
                warns.append(f"Row {r['id']}: chosen and rejected are identical — no preference signal")
    if role in ("rl_train", "rl_val"):
        for r in rows:
            if not r.get("required_checks"):
                warns.append(f"Row {r['id']} has no required_checks (row-based reward rules give 0)")
    if len(rows) < 10:
        warns.append(f"Only {len(rows)} examples - fine for a learning smoke test, far too few for a useful model")
    return warns


TRAIN_ROLES = {"sft_train", "rl_train", "pref"}


def leakage(splits: dict):
    """splits: {dataset_id: rows}. Warns when a question is in more than one split and at least one of them is a
    validation or test split (overlap between training sets only, e.g. pref and rl_train, is not leakage)."""
    index = {}
    for did, rows in splits.items():
        for r in rows:
            if r.get("retired"):
                continue
            index.setdefault(normalize(r["prompt"]), set()).add(did)
    return [f"'{q[:70]}' appears in: {', '.join(sorted(d))}" for q, d in index.items()
            if len(d) > 1 and any(x.rsplit("-v", 1)[0] not in TRAIN_ROLES for x in d)]


def diff(old_rows, new_rows):
    """Row-level diff by id: added, removed and changed rows."""
    o = {r["id"]: r for r in old_rows}
    n = {r["id"]: r for r in new_rows}
    added = [n[k] for k in n if k not in o]
    removed = [o[k] for k in o if k not in n]
    changed = [(o[k], n[k]) for k in n if k in o and o[k] != n[k]]
    return added, removed, changed


def active_test_rows(rows):
    return [r for r in rows if not r.get("retired")]


def find_replace(rows, field, find, replace, case=False):
    """Replace text in one field of every row. Returns (new_rows, number_of_rows_changed)."""
    if not find:
        return rows, 0
    pat = re.compile(re.escape(find), 0 if case else re.IGNORECASE)
    out, n = [], 0
    for r in rows:
        v = r.get(field)
        if isinstance(v, str) and pat.search(v):
            r = dict(r, **{field: pat.sub(replace, v)})
            n += 1
        out.append(r)
    return out, n


# ---------- bundled SFT packs ----------
def list_packs():
    out = []
    for p in sorted(paths.PACK_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out.append({"slug": p.stem, "title": d["title"], "domain": d["domain"], "description": d["description"],
                    "n_train": len(d["train"]), "n_val": len(d["val"])})
    return out


def pack_rows(slug):
    d = json.loads((paths.PACK_DIR / f"{slug}.json").read_text(encoding="utf-8"))
    tag = slug.split("_", 1)[-1][:10]
    train = [dict(r, id=f"{tag}-t{i + 1:02d}") for i, r in enumerate(d["train"])]
    val = [dict(r, id=f"{tag}-v{i + 1:02d}") for i, r in enumerate(d["val"])]
    return d, train, val


def load_pack(slug, mode="replace"):
    """Save a pack as new sft_train + sft_val versions. mode='replace' uses only the pack,
    mode='append' adds the pack to the latest versions. Returns (train_id, val_id)."""
    d, train, val = pack_rows(slug)
    ids = []
    for role, rows in (("sft_train", train), ("sft_val", val)):
        latest = registry.latest_dataset(role)
        base = load(latest["id"]) if (mode == "append" and latest) else []
        have = {r["id"] for r in base}
        ids.append(save_version(role, base + [r for r in rows if r["id"] not in have],
                                parent_id=latest["id"] if latest else None, source="pack",
                                note=f"{'+ ' if mode == 'append' else ''}pack: {d['title']}"))
    return tuple(ids)


def bootstrap_samples():
    """Register the bundled synthetic sample datasets as v1 of each role (only once)."""
    for role in ROLES:
        if registry.latest_dataset(role):
            continue
        rows = read_jsonl(paths.SAMPLE_DIR / f"{role}.jsonl")
        save_version(role, rows, source="sample", note="Bundled synthetic tool-store sample (educational)")
