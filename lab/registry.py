"""Local SQLite registry: dataset versions, reward versions, model versions (lineage), runs, evals, reviews."""
import json
import sqlite3
import time
from contextlib import contextmanager

from . import paths

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY, role TEXT, version INTEGER, path TEXT, hash TEXT, n_rows INTEGER,
  parent_id TEXT, source TEXT, note TEXT, created REAL);
CREATE TABLE IF NOT EXISTS rewards (
  id TEXT PRIMARY KEY, version INTEGER, rules_json TEXT, approved INTEGER DEFAULT 0,
  parent_id TEXT, note TEXT, created REAL);
CREATE TABLE IF NOT EXISTS models (
  id TEXT PRIMARY KEY, kind TEXT, parent_id TEXT, base_model TEXT, base_revision TEXT,
  adapter_path TEXT, run_id TEXT, train_ds TEXT, val_ds TEXT, reward_id TEXT,
  preferred INTEGER DEFAULT 0, size_bytes INTEGER, reference_json TEXT, note TEXT,
  deleted INTEGER DEFAULT 0, created REAL);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, kind TEXT, status TEXT, label TEXT, parent_model_id TEXT, output_model_id TEXT,
  config_json TEXT, train_ds TEXT, val_ds TEXT, reward_id TEXT, hardware_json TEXT, libs_json TEXT,
  log_path TEXT, metrics_path TEXT, samples_path TEXT, error TEXT, pid INTEGER,
  started REAL, finished REAL, created REAL);
CREATE TABLE IF NOT EXISTS evals (
  id INTEGER PRIMARY KEY AUTOINCREMENT, model_id TEXT, dataset_id TEXT, reward_id TEXT,
  settings_json TEXT, results_json TEXT, mean_reward REAL, created REAL);
CREATE TABLE IF NOT EXISTS reviews (
  id INTEGER PRIMARY KEY AUTOINCREMENT, model_id TEXT, test_ds TEXT, question_id TEXT, question TEXT,
  answer TEXT, rating TEXT, correction TEXT, approved INTEGER DEFAULT 0, used_in_ds TEXT, created REAL);
CREATE TABLE IF NOT EXISTS traces (
  id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT, source TEXT, model_id TEXT, session TEXT,
  input TEXT, output TEXT, steps_json TEXT, latency REAL, meta_json TEXT, flag TEXT, created REAL);
CREATE TABLE IF NOT EXISTS agent_evals (
  id INTEGER PRIMARY KEY AUTOINCREMENT, model_id TEXT, kind TEXT, results_json TEXT, summary_json TEXT, created REAL);
CREATE TABLE IF NOT EXISTS progress (task_id TEXT PRIMARY KEY, done INTEGER, created REAL);
"""


@contextmanager
def conn():
    c = sqlite3.connect(paths.db_path(), timeout=30)
    c.row_factory = sqlite3.Row
    try:
        c.executescript(SCHEMA)
        yield c
        c.commit()
    finally:
        c.close()


def _rows(sql, args=()):
    with conn() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def _one(sql, args=()):
    r = _rows(sql, args)
    return r[0] if r else None


def _insert(table, row):
    row = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()}
    cols = ",".join(row)
    q = ",".join("?" * len(row))
    with conn() as c:
        cur = c.execute(f"INSERT INTO {table} ({cols}) VALUES ({q})", tuple(row.values()))
        return cur.lastrowid


def update(table, id_, **fields):
    fields = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in fields.items()}
    sets = ",".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE {table} SET {sets} WHERE id=?", (*fields.values(), id_))


# ---------- datasets ----------
def add_dataset(role, path, hash_, n_rows, parent_id=None, source="", note=""):
    version = (_one("SELECT MAX(version) v FROM datasets WHERE role=?", (role,))["v"] or 0) + 1
    did = f"{role}-v{version}"
    _insert("datasets", dict(id=did, role=role, version=version, path=str(path), hash=hash_, n_rows=n_rows,
                             parent_id=parent_id, source=source, note=note, created=time.time()))
    return did


def datasets(role=None):
    if role:
        return _rows("SELECT * FROM datasets WHERE role=? ORDER BY version DESC", (role,))
    return _rows("SELECT * FROM datasets ORDER BY role, version DESC")


def dataset(did):
    return _one("SELECT * FROM datasets WHERE id=?", (did,))


def latest_dataset(role):
    return _one("SELECT * FROM datasets WHERE role=? ORDER BY version DESC LIMIT 1", (role,))


# ---------- rewards ----------
def add_reward(rules, note="", parent_id=None):
    version = (_one("SELECT MAX(version) v FROM rewards")["v"] or 0) + 1
    rid = f"reward-v{version}"
    _insert("rewards", dict(id=rid, version=version, rules_json=json.dumps(rules), approved=0,
                            parent_id=parent_id, note=note, created=time.time()))
    return rid


def rewards():
    out = _rows("SELECT * FROM rewards ORDER BY version DESC")
    for r in out:
        r["rules"] = json.loads(r["rules_json"])
    return out


def reward(rid):
    r = _one("SELECT * FROM rewards WHERE id=?", (rid,))
    if r:
        r["rules"] = json.loads(r["rules_json"])
    return r


# ---------- models ----------
def _next_model_id(kind):
    n = len(_rows("SELECT id FROM models WHERE kind=?", (kind,))) + 1
    return f"{kind}-{n:03d}"


def ensure_base(base_model, revision=None):
    existing = _one("SELECT * FROM models WHERE kind='base' AND base_model=?", (base_model,))
    if existing:
        if revision and not existing["base_revision"]:
            update("models", existing["id"], base_revision=revision)
        return existing["id"]
    mid = _next_model_id("base")
    _insert("models", dict(id=mid, kind="base", parent_id=None, base_model=base_model, base_revision=revision,
                           adapter_path=None, created=time.time(), note="Original pretrained model"))
    return mid


def add_model(kind, parent_id, base_model, base_revision, adapter_path, run_id, train_ds, val_ds,
              reward_id=None, size_bytes=None, reference=None, mid=None):
    mid = mid or _next_model_id(kind)
    _insert("models", dict(id=mid, kind=kind, parent_id=parent_id, base_model=base_model,
                           base_revision=base_revision, adapter_path=str(adapter_path), run_id=run_id,
                           train_ds=train_ds, val_ds=val_ds, reward_id=reward_id, size_bytes=size_bytes,
                           reference_json=json.dumps(reference or {}), created=time.time()))
    return mid


def models(include_deleted=False):
    q = "SELECT * FROM models" + ("" if include_deleted else " WHERE deleted=0") + " ORDER BY created"
    return _rows(q)


def model(mid):
    return _one("SELECT * FROM models WHERE id=?", (mid,))


def lineage(mid):
    """Chain from the given model back to its base model: [mid, parent, ..., base]."""
    chain, seen = [], set()
    while mid and mid not in seen:
        seen.add(mid)
        m = model(mid)
        if not m:
            break
        chain.append(m)
        mid = m["parent_id"]
    return chain


def children(mid):
    return _rows("SELECT * FROM models WHERE parent_id=? AND deleted=0", (mid,))


def set_preferred(mid):
    with conn() as c:
        c.execute("UPDATE models SET preferred=0")
        c.execute("UPDATE models SET preferred=1 WHERE id=?", (mid,))


# ---------- runs ----------
def new_run(kind, **fields):
    n = len(_rows("SELECT id FROM runs")) + 1
    rid = f"run-{n:04d}"
    _insert("runs", dict(id=rid, kind=kind, status="queued", created=time.time(), **fields))
    return rid


def run(rid):
    r = _one("SELECT * FROM runs WHERE id=?", (rid,))
    if r:
        r["config"] = json.loads(r["config_json"] or "{}")
    return r


def runs(kind=None):
    if kind:
        return _rows("SELECT * FROM runs WHERE kind=? ORDER BY created DESC", (kind,))
    return _rows("SELECT * FROM runs ORDER BY created DESC")


# ---------- evals & reviews ----------
def add_eval(model_id, dataset_id, reward_id, settings, results, mean_reward):
    return _insert("evals", dict(model_id=model_id, dataset_id=dataset_id, reward_id=reward_id,
                                 settings_json=json.dumps(settings), results_json=json.dumps(results),
                                 mean_reward=mean_reward, created=time.time()))


def evals(model_id=None):
    q, a = "SELECT * FROM evals", ()
    if model_id:
        q, a = q + " WHERE model_id=?", (model_id,)
    out = _rows(q + " ORDER BY created DESC", a)
    for e in out:
        e["results"] = json.loads(e["results_json"])
        e["settings"] = json.loads(e["settings_json"] or "{}")
    return out


# ---------- traces (every RAG / agent / API call), agent evals, course progress ----------
def add_trace(kind, model_id, input, output, steps=None, latency=None, source="lab", session=None, meta=None):
    row = dict(kind=kind, source=source, model_id=model_id, session=session, input=input, output=output,
               latency=latency, flag="", created=time.time())
    tid = _insert("traces", dict(row, steps_json=json.dumps(steps or []), meta_json=json.dumps(meta or {})))
    try:  # mirror to the lab's Langfuse when it is running (no-op otherwise)
        from . import langfuse_stack
        langfuse_stack.send_async(dict(row, id=tid, steps=steps or [], meta=meta or {}))
    except Exception:  # noqa: BLE001
        pass
    return tid


def traces(kind=None, limit=500):
    q, a = "SELECT * FROM traces", ()
    if kind:
        q, a = q + " WHERE kind=?", (kind,)
    out = _rows(q + " ORDER BY id DESC LIMIT ?", (*a, limit))
    for t in out:
        t["steps"] = json.loads(t["steps_json"] or "[]")
        t["meta"] = json.loads(t["meta_json"] or "{}")
    return out


def add_agent_eval(model_id, kind, results, summary):
    return _insert("agent_evals", dict(model_id=model_id, kind=kind, results_json=json.dumps(results),
                                       summary_json=json.dumps(summary), created=time.time()))


def agent_evals():
    out = _rows("SELECT * FROM agent_evals ORDER BY id DESC")
    for e in out:
        e["results"], e["summary"] = json.loads(e["results_json"]), json.loads(e["summary_json"])
    return out


def set_progress(task_id, done=True):
    with conn() as c:
        c.execute("INSERT OR REPLACE INTO progress (task_id, done, created) VALUES (?,?,?)",
                  (task_id, int(done), time.time()))


def progress():
    return {r["task_id"]: bool(r["done"]) for r in _rows("SELECT * FROM progress")}


def add_review(**fields):
    return _insert("reviews", dict(created=time.time(), **fields))


def reviews(model_id=None):
    if model_id:
        return _rows("SELECT * FROM reviews WHERE model_id=? ORDER BY created DESC", (model_id,))
    return _rows("SELECT * FROM reviews ORDER BY created DESC")
