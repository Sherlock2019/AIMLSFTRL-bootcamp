"""The lab's own Langfuse: start/stop the Docker stack and send lab traces to it.

Secrets are generated on first start into docker/langfuse/.env (git-ignored). The lab project and its API keys are
created by Langfuse itself at boot (LANGFUSE_INIT_*), so traces can be sent immediately — no manual setup.
"""
import base64
import json
import os
import secrets
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone

from . import paths, services

PORT = int(os.environ.get("LAB_LANGFUSE_PORT", 3820))
DIR = paths.ROOT / "docker" / "langfuse"
ENV = DIR / ".env"
COMPOSE = ["docker", "compose", "-p", "aitrain-langfuse", "--env-file", str(ENV), "-f", str(DIR / "docker-compose.yml")]


def url():
    return f"http://localhost:{PORT}"


def ensure_env():
    if ENV.exists():
        return
    tok = lambda n=24: secrets.token_urlsafe(n)  # noqa: E731
    vals = {"LANGFUSE_PORT": PORT, "PG_PASSWORD": tok(), "CH_PASSWORD": tok(), "REDIS_PASSWORD": tok(),
            "S3_SECRET": tok(), "ENCRYPTION_KEY": secrets.token_hex(32), "SALT": tok(), "NEXTAUTH_SECRET": tok(),
            "LF_PUBLIC_KEY": f"pk-lf-lab-{secrets.token_hex(8)}", "LF_SECRET_KEY": f"sk-lf-lab-{secrets.token_hex(16)}",
            "LF_USER_EMAIL": "lab@ai-training.lab", "LF_USER_PASSWORD": tok(12)}
    ENV.write_text("".join(f"{k}={v}\n" for k, v in vals.items()))
    ENV.chmod(0o600)


def config():
    if not ENV.exists():
        return {}
    return dict(line.split("=", 1) for line in ENV.read_text().splitlines() if "=" in line)


def running():
    return services.port_open(PORT)


def start():
    """docker compose up -d (all images are standard; first start takes ~1 minute)."""
    ensure_env()
    r = subprocess.run(COMPOSE + ["up", "-d"], capture_output=True, text=True, timeout=900)
    return r.returncode == 0, (r.stderr or r.stdout)[-600:]


def stop():
    r = subprocess.run(COMPOSE + ["stop"], capture_output=True, text=True, timeout=300)
    return r.returncode == 0


def enabled():
    return running() and bool(config().get("LF_PUBLIC_KEY"))


# ---------- export lab traces ----------
def _iso(ts):
    return datetime.fromtimestamp(ts or time.time(), tz=timezone.utc).isoformat()


def to_events(t):
    """One lab trace → Langfuse ingestion events (trace + one span/generation per step)."""
    tid = f"lab-trace-{t['id']}"
    ev = [{"id": str(uuid.uuid4()), "type": "trace-create", "timestamp": _iso(t["created"]),
           "body": {"id": tid, "name": f"{t['kind']} · {t['source']}", "input": t["input"], "output": t["output"],
                    "sessionId": t.get("session") or None, "userId": "lab",
                    "metadata": {"model": t["model_id"], "latency_s": t["latency"], **(t.get("meta") or {})},
                    "tags": [t["kind"], t["model_id"] or ""], "timestamp": _iso(t["created"])}}]
    for i, s in enumerate(t.get("steps") or []):
        is_llm = s.get("step") in ("llm", "generate", "chat")
        ev.append({"id": str(uuid.uuid4()), "type": "generation-create" if is_llm else "span-create",
                   "timestamp": _iso(t["created"]),
                   "body": {"id": f"{tid}-s{i}", "traceId": tid, "name": s.get("name") or s.get("step"),
                            "startTime": _iso(t["created"]),
                            "input": s.get("arguments") or s.get("hits") or s.get("messages"),
                            "output": s.get("result") or s.get("output") or s.get("tool_calls"),
                            **({"model": t["model_id"]} if is_llm else {}), "metadata": s}})
    return ev


def send(traces):
    import httpx
    c = config()
    auth = base64.b64encode(f"{c['LF_PUBLIC_KEY']}:{c['LF_SECRET_KEY']}".encode()).decode()
    batch = [e for t in traces for e in to_events(t)]
    for i in range(0, len(batch), 200):
        r = httpx.post(f"{url()}/api/public/ingestion", json={"batch": batch[i:i + 200]}, timeout=30,
                       headers={"Authorization": f"Basic {auth}"})
        if r.status_code >= 300:
            raise RuntimeError(f"Langfuse ingestion → HTTP {r.status_code}: {r.text[:200]}")
    return len(traces)


def send_async(trace):
    """Called for every new lab trace; silently does nothing unless the lab's Langfuse is running."""
    if not enabled():
        return
    threading.Thread(target=lambda: _safe(send, [trace]), daemon=True).start()


def _safe(fn, *a):
    try:
        fn(*a)
    except Exception:  # noqa: BLE001
        pass


def trace_url(trace_id):
    return f"{url()}/project/ai-training-lab/traces/lab-trace-{trace_id}"


__all__ = ["start", "stop", "running", "enabled", "send", "send_async", "url", "trace_url", "config", "json"]
