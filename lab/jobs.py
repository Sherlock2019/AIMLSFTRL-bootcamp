"""Launch training runs as background processes, cancel them, and read their metrics/logs."""
import json
import os
import subprocess
import sys

from . import paths, registry

SFT_PRESETS = {
    "Tiny (smoke test)": dict(lora_r=8, lora_alpha=16, lora_dropout=0.05, learning_rate=2e-4, batch_size=4,
                              grad_accum=1, epochs=1, max_steps=12, max_length=192, save_steps=0, eval_steps=4, seed=42),
    "Small": dict(lora_r=16, lora_alpha=32, lora_dropout=0.05, learning_rate=1e-4, batch_size=4,
                  grad_accum=2, epochs=3, max_steps=-1, max_length=256, save_steps=0, eval_steps=5, seed=42),
}
RL_PRESETS = {
    "Tiny (smoke test)": dict(lora_r=8, lora_alpha=16, lora_dropout=0.05, learning_rate=2e-5, batch_size=4,
                              grad_accum=1, num_generations=4, max_completion_length=48, max_steps=8,
                              temperature=1.0, beta=0.0, save_steps=0, seed=42),
    "Small": dict(lora_r=16, lora_alpha=32, lora_dropout=0.05, learning_rate=1e-5, batch_size=8,
                  grad_accum=1, num_generations=4, max_completion_length=64, max_steps=30,
                  temperature=1.0, beta=0.0, save_steps=0, seed=42),
}


RM_PRESETS = {
    "Tiny (smoke test)": dict(lora_r=8, lora_alpha=16, lora_dropout=0.05, learning_rate=1e-4, batch_size=4,
                              grad_accum=1, epochs=1, max_steps=20, max_length=256, val_frac=0.2, save_steps=0,
                              eval_steps=5, seed=42),
    "Small": dict(lora_r=16, lora_alpha=32, lora_dropout=0.05, learning_rate=5e-5, batch_size=4,
                  grad_accum=2, epochs=3, max_steps=-1, max_length=256, val_frac=0.2, save_steps=0,
                  eval_steps=5, seed=42),
}


def _next_output_id(kind):
    existing = {m["id"] for m in registry.models(include_deleted=True)}
    existing |= {r["output_model_id"] for r in registry.runs() if r["output_model_id"]}
    n = 1
    while f"{kind}-{n:03d}" in existing:
        n += 1
    return f"{kind}-{n:03d}"


def active_run():
    for r in registry.runs():
        if r["status"] in ("queued", "running"):
            if r["pid"] and not _alive(r["pid"]):
                registry.update("runs", r["id"], status="failed", error="Process exited unexpectedly (see log).")
                continue
            return r
    return None


def _alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # zombie check on Linux
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return True


def start(kind, parent_model_id, train_ds, val_ds, config, reward_id=None, wait=False):
    if active_run():
        raise RuntimeError("Another training run is active. Wait for it or cancel it first.")
    if not registry.model(parent_model_id):
        raise RuntimeError(f"Parent model {parent_model_id} not found.")
    out_id = _next_output_id(kind)
    rdir = paths.sub("runs")
    rid = registry.new_run(kind, label="Real training", parent_model_id=parent_model_id, output_model_id=out_id,
                           config_json=json.dumps(config), train_ds=train_ds, val_ds=val_ds, reward_id=reward_id)
    log_path, metrics_path, samples_path = (str(rdir / f"{rid}.{ext}") for ext in ("log", "metrics.jsonl", "samples.jsonl"))
    registry.update("runs", rid, log_path=log_path, metrics_path=metrics_path, samples_path=samples_path)
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    with open(log_path, "a") as out:
        p = subprocess.Popen([sys.executable, "-m", "lab.train", rid], cwd=str(paths.ROOT), stdout=out,
                             stderr=subprocess.STDOUT, env=env, start_new_session=True)
    registry.update("runs", rid, pid=p.pid)
    if wait:
        p.wait()
    return rid


def cancel(rid):
    (paths.sub("runs") / f"{rid}.cancel").write_text("cancel")


def read_jsonl(path):
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def tail(path, n=60):
    if not path or not os.path.exists(path):
        return ""
    with open(path, errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-n:])
