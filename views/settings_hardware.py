import streamlit as st

import ui_common as ui
from lab import hardware, paths, registry

st.title("🖥️ Settings & Hardware")
ui.explain("Checks what this computer can do before you train.",
           "Nothing — the lab detects your hardware.",
           "Nothing yet. This page only measures.",
           "The PyTorch device (cuda / mps / cpu) and free GPU memory decide which presets are realistic.",
           "Each training run stores the hardware it ran on in the registry.")
st.session_state["hw_seen"] = True
hw = hardware.check()
rows = [("Operating system", hw["os"]), ("CPU", hw["cpu"]),
        ("RAM", f"{hw['ram_available_gb']} GB available of {hw['ram_total_gb']} GB"),
        ("GPU", f"{hw['gpu']} ({hw.get('gpu_capability', '')}) — {hw['gpu_mem_free_gb']} GB free of "
                f"{hw['gpu_mem_total_gb']} GB" if hw["gpu"] else "none detected"),
        ("PyTorch device", f"{hw['device']} (torch {hw['torch']})"), ("Disk", f"{hw['disk_free_gb']} GB free"),
        ("Recommended mode", hw["recommended"][0])]
st.table({"Item": [r[0] for r in rows], "Display": [r[1] for r in rows]})
st.info(hw["recommended"][1])
if hw["gpu_problem"]:
    st.error(hw["gpu_problem"])
if hw["ram_available_gb"] < 3:
    st.warning("Less than 3 GB of free RAM. Close other programs before training.")
st.warning("RL (GRPO) generates several answers per question at every step, so it is much slower and uses more "
           "memory than SFT. Start with the Tiny preset. No configuration is promised to fit until it actually runs.")
st.caption("Precision: this lab trains in fp32 on GPUs without fast bf16 (e.g. GTX 10xx) and on CPU. "
           "Quantization is not used — it is hardware-dependent and not needed for a 0.6B model.")

st.markdown("### Base model")
base = next((m for m in registry.models() if m["kind"] == "base"), None)
name = base["base_model"] if base else paths.DEFAULT_BASE_MODEL
cached = hardware.model_cached(name)
st.write(f"**{name}** — {'✅ downloaded' if cached else '⬇️ not downloaded yet'} "
         f"(revision {base['base_revision'] or 'recorded at first training'})")
if not cached and st.button("Download model (needs internet, ~1.5 GB)"):
    with st.spinner("Downloading…"):
        from huggingface_hub import snapshot_download
        snapshot_download(name, allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt"])
    st.rerun()
st.caption("To use another compatible Hugging Face causal LM, start the lab with LAB_BASE_MODEL=<repo id>. "
           "Each base gets its own lineage; adapters trained on one base cannot be loaded on another.")

st.markdown("### Inference model in memory")
st.write("The dashboard keeps one base model loaded for generating answers. Unload it to free GPU memory "
         "before a large training run.")
if st.button("Unload inference model"):
    ui.unload_engine()
    st.success("Unloaded.")

st.markdown("### Library versions")
st.json(hardware.libs())
