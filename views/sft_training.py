import json

import streamlit as st

import run_panel
import ui_common as ui
from lab import jobs, registry

st.title("🎓 SFT Training — supervised fine-tuning")
ui.explain("The model studies question → approved-answer pairs and is nudged to produce those answers.",
           "A starting model (base or any earlier version), an SFT training dataset version and a validation version.",
           "TRL's SFTTrainer updates a small LoRA adapter so the approved answers become more likely.",
           "Training loss (fitting the examples) and validation loss (questions it did not train on). "
           "Then check real answers on the Evaluation page.",
           "A new LoRA adapter folder, its config, tokenizer files, training config, dataset ids, metrics, log and lineage.")
st.info("SFT uses your approved answers to adjust the model or its adapter. "
        "**A lower training loss alone does not prove that the model is better.**")

tab_new, tab_runs = st.tabs(["▶️ New run", "📈 Runs & live progress"])
with tab_new:
    default = "rl" if any(m["kind"] == "rl" for m in registry.models()) else "base"
    parent = ui.model_picker("Starting model", "sft_parent", default_kind=default,
                             help="Normally the latest RL model when starting a new cycle; the base model for cycle 1.")
    chain = registry.lineage(parent)
    st.caption("Lineage of the starting model: " + " ← ".join(m["id"] for m in chain))
    pm = registry.model(parent)
    c1, c2 = st.columns(2)
    with c1:
        train_ds = ui.dataset_picker("SFT training dataset", "sft_train", "sft_tr")
    with c2:
        val_ds = ui.dataset_picker("SFT validation dataset", "sft_val", "sft_va", allow_none=True)

    mode = st.radio("Mode", ["Beginner", "Advanced"], horizontal=True)
    preset = st.selectbox("Preset", list(jobs.SFT_PRESETS), help="Tiny ≈ 12 steps: a quick real smoke test.")
    cfg = dict(jobs.SFT_PRESETS[preset])
    inherit = pm["kind"] != "base"
    if inherit:
        ac = json.loads(open(f"{pm['adapter_path']}/adapter_config.json").read())
        cfg.update(lora_r=ac["r"], lora_alpha=ac["lora_alpha"])
        st.caption(f"LoRA shape is inherited from {parent} (r={ac['r']}, alpha={ac['lora_alpha']}): "
                   "training continues the parent's adapter weights rather than starting a new one.")
    if mode == "Advanced":
        a, b, c = st.columns(3)
        with a:
            cfg["learning_rate"] = st.number_input("Learning rate", value=cfg["learning_rate"], format="%.1e",
                                                   help="How big each update step is. Too high = unstable, too low = no change.")
            cfg["batch_size"] = st.number_input("Batch size", 1, 32, cfg["batch_size"],
                                                help="Examples processed together. Lower it if you run out of memory.")
            cfg["grad_accum"] = st.number_input("Gradient accumulation", 1, 32, cfg["grad_accum"],
                                                help="Add up several small batches before one update (bigger effective batch, same memory).")
        with b:
            cfg["epochs"] = st.number_input("Epochs", 1, 20, int(cfg["epochs"]), help="Passes over the training data (used when max steps = -1).")
            cfg["max_steps"] = st.number_input("Max steps (-1 = use epochs)", -1, 5000, cfg["max_steps"],
                                               help="Hard cap on update steps.")
            cfg["max_length"] = st.number_input("Max sequence length", 64, 2048, cfg["max_length"],
                                                help="Longer examples are cut. Lower = less memory.")
        with c:
            cfg["lora_r"] = st.number_input("LoRA rank r", 1, 128, cfg["lora_r"], disabled=inherit,
                                            help="Size of the adapter. Bigger = more capacity, more memory.")
            cfg["lora_alpha"] = st.number_input("LoRA alpha", 1, 256, cfg["lora_alpha"], disabled=inherit,
                                                help="Scales the adapter's effect (often 2 x r).")
            cfg["save_steps"] = st.number_input("Checkpoint every N steps (0 = off)", 0, 5000, cfg["save_steps"],
                                                help="Intermediate checkpoints use disk space.")
            cfg["eval_steps"] = st.number_input("Validate every N steps", 1, 1000, cfg["eval_steps"])
            cfg["seed"] = st.number_input("Random seed", 0, 10**6, cfg["seed"], help="Same seed + same data ≈ same result.")
    else:
        st.write(f"Beginner defaults: lr {cfg['learning_rate']}, batch {cfg['batch_size']}, max steps {cfg['max_steps']}, "
                 f"max length {cfg['max_length']}, LoRA r={cfg['lora_r']}, seed {cfg['seed']}.")
    active = jobs.active_run()
    if active:
        st.warning(f"{active['id']} is running — one training run at a time.")
    if st.button("🚀 Start real SFT training", type="primary", disabled=bool(active) or not train_ds):
        try:
            ui.unload_engine()  # free GPU memory used by the dashboard's inference model
            rid = jobs.start("sft", parent, train_ds, val_ds, cfg)
            st.success(f"Started {rid}. Open the 'Runs & live progress' tab.")
        except Exception as e:  # noqa: BLE001
            st.error(str(e))

with tab_runs:
    run_panel.show("sft")
