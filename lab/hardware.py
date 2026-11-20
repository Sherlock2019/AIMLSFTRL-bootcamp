"""Hardware check: OS, CPU, RAM, GPU, PyTorch device, disk, and a recommended mode."""
import os
import platform
import shutil

import psutil

from . import paths


def _os_name():
    if platform.system() == "Linux":
        try:
            if "microsoft" in open("/proc/version").read().lower():
                return "Windows (WSL Ubuntu)"
        except OSError:
            pass
        return "Linux"
    return {"Darwin": "macOS", "Windows": "Windows"}.get(platform.system(), platform.system())


def _cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def check():
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(paths.home() if paths.home().exists() else paths.ROOT)
    info = {
        "os": _os_name(), "cpu": f"{_cpu_name()} ({os.cpu_count()} threads)",
        "ram_total_gb": round(vm.total / 1e9, 1), "ram_available_gb": round(vm.available / 1e9, 1),
        "disk_free_gb": round(disk.free / 1e9, 1), "gpu": None, "gpu_mem_total_gb": None,
        "gpu_mem_free_gb": None, "device": "cpu", "torch": None, "gpu_problem": None,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            info["gpu"] = name
            free, total = torch.cuda.mem_get_info(0)
            info["gpu_mem_total_gb"], info["gpu_mem_free_gb"] = round(total / 1e9, 1), round(free / 1e9, 1)
            cap = torch.cuda.get_device_capability(0)
            info["gpu_capability"] = f"sm_{cap[0]}{cap[1]}"
            try:  # a real kernel launch: catches builds that do not support this GPU
                (torch.ones(2, device="cuda") * 2).sum().item()
                info["device"] = "cuda"
            except Exception as e:  # noqa: BLE001
                info["gpu_problem"] = (f"PyTorch {torch.__version__} cannot run on {name} ({info['gpu_capability']}): {e}. "
                                       "For GTX 10xx GPUs install the CUDA 12.6 build (scripts/setup_env.sh does this).")
            info["bf16"] = bool(torch.cuda.is_bf16_supported()) if info["device"] == "cuda" else False
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["device"] = "mps"
    except ImportError:
        info["gpu_problem"] = "PyTorch is not installed - run scripts/setup_env.sh"
    info["recommended"] = recommend(info)
    return info


def recommend(info):
    if info["device"] == "cuda" and (info["gpu_mem_free_gb"] or 0) >= 4:
        return ("GPU training", "Real SFT and GRPO with LoRA on a 0.6B model should fit. RL is still several times "
                "slower than SFT because it generates multiple answers per question.")
    if info["device"] == "cuda":
        return ("GPU (low memory)", "Little free GPU memory: use the tiny preset, short answers and 2 generations.")
    if info["device"] == "mps":
        return ("Apple GPU (MPS)", "Use the tiny preset. Some training kernels may fall back to CPU.")
    return ("CPU learning mode", "Real but very slow. Use the tiny preset (few examples, few steps, short sequences). "
            "GRPO on CPU can take a long time.")


def gpu_mem_used_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except Exception:  # noqa: BLE001
        pass
    return None


def model_cached(name):
    """True if the model's weights are already in the local Hugging Face cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(name, "config.json"), str) and \
            isinstance(try_to_load_from_cache(name, "model.safetensors"), str)
    except Exception:  # noqa: BLE001
        return False


def libs():
    out = {}
    for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "streamlit"):
        try:
            out[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            out[name] = None
    return out
