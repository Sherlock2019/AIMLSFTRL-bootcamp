"""Start / stop background services from the dashboard: the FastAPI model server and Flowise."""
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from . import paths

API_PORT = int(os.environ.get("LAB_API_PORT", 8811))
FLOWISE_PORT = int(os.environ.get("LAB_FLOWISE_PORT", 3811))


def port_open(port, host="127.0.0.1"):
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def _log(name):
    return paths.sub("services") / f"{name}.log"


def _pidfile(name):
    return paths.sub("services") / f"{name}.pid"


def _spawn(name, cmd, env=None):
    log = open(_log(name), "ab")
    p = subprocess.Popen(cmd, cwd=str(paths.ROOT), stdout=log, stderr=subprocess.STDOUT,
                         env={**os.environ, **(env or {})}, start_new_session=True)
    _pidfile(name).write_text(str(p.pid))
    return p.pid


def stop(name):
    pf = _pidfile(name)
    if not pf.exists():
        return False
    try:
        os.killpg(int(pf.read_text()), 15)
    except (ProcessLookupError, ValueError, PermissionError):
        pass
    pf.unlink(missing_ok=True)
    return True


def tail(name, n=40):
    p = _log(name)
    return "".join(p.read_text(errors="replace").splitlines(True)[-n:]) if p.exists() else ""


# ---------- FastAPI ----------
def api_running():
    return port_open(API_PORT)


def start_api():
    if api_running():
        return None
    return _spawn("api", [sys.executable, "-m", "uvicorn", "lab.api:app", "--host", "0.0.0.0", "--port", str(API_PORT)],
                  env={"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})


# ---------- LangGraph Studio (local dev server; the Studio UI itself is hosted by LangSmith) ----------
STUDIO_PORT = int(os.environ.get("LAB_STUDIO_PORT", 2024))
STUDIO_URL = f"https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:{STUDIO_PORT}"


def studio_running():
    return port_open(STUDIO_PORT)


def start_studio():
    if studio_running():
        return None
    lg = Path(sys.executable).parent / "langgraph"
    return _spawn("studio", [str(lg), "dev", "--port", str(STUDIO_PORT), "--host", "127.0.0.1", "--no-browser",
                             "--no-reload"], env={"HF_HUB_OFFLINE": "1", "LANGGRAPH_NO_TELEMETRY": "1", "NO_COLOR": "1"})


# ---------- Flowise (Docker) ----------
# Runs the official image as its own container (npm install of Flowise fails to compile on Node 24).
# Inside the container, the lab API is reached at host.docker.internal:<API_PORT>.
FLOWISE_IMAGE = os.environ.get("LAB_FLOWISE_IMAGE", "flowiseai/flowise:3.1.2")
FLOWISE_CONTAINER = "aitrain-flowise"
FLOWISE_API_BASE = f"http://host.docker.internal:{API_PORT}/v1"


def _docker(*args, timeout=30):
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def docker_ok():
    return bool(shutil.which("docker")) and _docker("info", "--format", "{{.ServerVersion}}").returncode == 0


def flowise_image_present():
    return docker_ok() and _docker("image", "inspect", FLOWISE_IMAGE).returncode == 0


def flowise_running():
    return port_open(FLOWISE_PORT)


def start_flowise():
    """Start (or create) the lab's Flowise container. First run pulls the image if it is missing (~1 GB)."""
    if flowise_running():
        return "already running"
    if _docker("start", FLOWISE_CONTAINER).returncode == 0:
        return "started"
    r = _docker("run", "-d", "--name", FLOWISE_CONTAINER, "-p", f"127.0.0.1:{FLOWISE_PORT}:3000",
                "--add-host", "host.docker.internal:host-gateway", "-v", "aitrain-flowise-data:/root/.flowise",
                "--restart", "no", FLOWISE_IMAGE, timeout=900)
    return "created" if r.returncode == 0 else f"error: {r.stderr.strip()[:300]}"


def stop_flowise():
    return _docker("stop", FLOWISE_CONTAINER).returncode == 0


def flowise_log(n=40):
    r = _docker("logs", "--tail", str(n), FLOWISE_CONTAINER)
    return (r.stdout + r.stderr) if r.returncode == 0 else ""
