"""Where everything lives on disk. Override the artifacts folder with LAB_HOME (used by tests)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "datasets" / "sample_tool_store"
PACK_DIR = ROOT / "datasets" / "sft_packs"  # 10 bundled supervised (SFT) example packs


def home() -> Path:
    return Path(os.environ.get("LAB_HOME", ROOT / "artifacts"))


def sub(name: str) -> Path:
    p = home() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_path() -> Path:
    home().mkdir(parents=True, exist_ok=True)
    return home() / "lab.sqlite"


DEFAULT_BASE_MODEL = os.environ.get("LAB_BASE_MODEL", "Qwen/Qwen3-0.6B")
