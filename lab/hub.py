"""Import models from the Hugging Face Hub and register them as new base models."""
import os
from contextlib import contextmanager

from . import hardware, paths, registry

SUGGESTED = [
    ("Qwen/Qwen3-0.6B", "default lab model, fast, supports tool calls"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "tiny instruct model, good for quick comparisons"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "stronger; better as RAG answerer, agent or LLM judge (~6 GB in fp32)"),
    ("HuggingFaceTB/SmolLM2-135M-Instruct", "very small, shows how size limits quality"),
]
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "merges.txt", "tokenizer*"]


@contextmanager
def online():
    """Temporarily lift offline mode. The lab runs with HF_HUB_OFFLINE=1, and huggingface_hub
    reads that flag once at import time — so the env var alone is not enough; flip the constant too."""
    from huggingface_hub import constants
    old_env, old_const = os.environ.pop("HF_HUB_OFFLINE", None), constants.HF_HUB_OFFLINE
    constants.HF_HUB_OFFLINE = False
    try:
        yield
    finally:
        constants.HF_HUB_OFFLINE = old_const
        if old_env is not None:
            os.environ["HF_HUB_OFFLINE"] = old_env


def info(repo_id):
    """Size and metadata before download. Needs internet."""
    from huggingface_hub import HfApi
    with online():
        m = HfApi().model_info(repo_id, files_metadata=True)
    files = [s for s in (m.siblings or []) if s.rfilename.endswith(".safetensors")]
    size = sum((s.size or 0) for s in files)
    card = getattr(m, "card_data", None) or {}
    return {"repo_id": repo_id, "size_gb": size / 1e9, "has_safetensors": bool(files),
            "license": (card.get("license") if isinstance(card, dict) else getattr(card, "license", None)),
            "pipeline": getattr(m, "pipeline_tag", None), "downloads": getattr(m, "downloads", None),
            "gated": bool(getattr(m, "gated", False))}


def fit_advice(size_gb):
    """Rough GPU fit in fp32 (the lab trains in fp32 on GPUs without fast bf16)."""
    vram = hardware.check().get("gpu_mem_total_gb") or 0
    need_infer, need_train = size_gb * 2.2, size_gb * 3.5  # fp32 ~2x bf16 file size; + activations/optimizer for LoRA
    if not vram:
        return "unknown", f"No GPU detected — inference needs ~{need_infer:.1f} GB RAM and will be slow."
    if need_train <= vram * 0.9:
        return "ok", f"Fits: ~{need_infer:.1f} GB to run, ~{need_train:.1f} GB to LoRA-train (GPU {vram:.0f} GB)."
    if need_infer <= vram * 0.9:
        return "warn", f"Can run (~{need_infer:.1f} GB) but LoRA training may not fit in {vram:.0f} GB."
    return "no", f"Too big for {vram:.0f} GB GPU in fp32 (~{need_infer:.1f} GB just to run)."


def download(repo_id, token=None):
    from huggingface_hub import snapshot_download
    with online():  # the dashboard runs offline; allow this one download
        return snapshot_download(repo_id, allow_patterns=PATTERNS, token=token or None)


def register(repo_id):
    return registry.ensure_base(repo_id)


def is_cached(repo_id):
    """Weights present locally (single-file or sharded safetensors)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        return any(isinstance(try_to_load_from_cache(repo_id, f), str)
                   for f in ("model.safetensors", "model.safetensors.index.json"))
    except Exception:  # noqa: BLE001
        return False


def cached_models():
    return [r for r, _ in SUGGESTED if is_cached(r)]


def import_model(repo_id, token=None):
    """Download (if needed) and register. Returns the new/existing base model id."""
    if not is_cached(repo_id):
        download(repo_id, token)
    return register(repo_id)


__all__ = ["SUGGESTED", "info", "fit_advice", "download", "register", "import_model", "paths"]
