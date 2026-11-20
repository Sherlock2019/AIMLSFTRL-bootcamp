import io
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

import ui_common as ui
from lab import registry

st.title("🧬 Model Registry & Lineage")
ui.explain("Every model version, where it came from, and exactly what produced it.",
           "Nothing — versions are registered automatically when a training run completes.",
           "Load any version for inference, verify it reloads correctly, export it, or mark it as preferred.",
           "The parent chain: each new stage must start from the version you selected, never silently from the base.",
           "Per version: run id, parent, base model + revision, method, dataset versions + hashes, reward config, "
           "training config + seed, library versions, hardware, metrics, logs and artifact paths.")
st.caption("Every adapter here is a LoRA adapter on the base model. A child stage continues training its parent's "
           "adapter, so base + one adapter folder reproduces any version. An adapter is never a standalone model.")

ms = registry.models()
st.dataframe(pd.DataFrame([{
    "Version": m["id"] + (" ★" if m["preferred"] else ""),
    "Type": {"base": "Base model", "sft": "SFT", "rl": "RL", "rlhf": "RLHF", "rm": "Reward model"}.get(m["kind"], m["kind"]),
    "Parent": m["parent_id"] or "—", "Data": m["train_ds"] or "—", "Reward": m["reward_id"] or "—",
    "Result": "Base weights (Hugging Face)" if m["kind"] == "base" else "LoRA adapter",
    "Size (MB)": round(m["size_bytes"] / 1e6, 1) if m["size_bytes"] else "—"} for m in ms]),
    use_container_width=True, hide_index=True)

dot = ["digraph G { rankdir=LR; bgcolor=transparent; node [style=filled, shape=box, fontname=Helvetica, fontcolor=white];"]
colors = {"base": "#334155", "sft": "#0369a1", "rl": "#b45309", "rlhf": "#be185d", "rm": "#6d28d9"}
for m in ms:
    rm_used = (registry.run(m["run_id"]) or {}).get("config", {}).get("rm_id") if m["run_id"] else None
    lbl = f"{m['id']}\\n{m['train_ds'] or m['base_model']}" + (f"\\n{m['reward_id']}" if m["reward_id"] else "") \
        + (f"\\nreward: {rm_used}" if rm_used else "")
    dot.append(f'"{m["id"]}" [label="{lbl}", fillcolor="{colors.get(m["kind"], "#334155")}", '
               f'penwidth={3 if m["preferred"] else 1}];')
    if m["parent_id"]:
        dot.append(f'"{m["parent_id"]}" -> "{m["id"]}" [color="#94a3b8"];')
    if rm_used:
        dot.append(f'"{rm_used}" -> "{m["id"]}" [color="#a78bfa", style=dashed, label="reward"];')
dot.append("}")
st.graphviz_chart("\n".join(dot))

mid = st.selectbox("Inspect version", [m["id"] for m in ms], index=len(ms) - 1,
                   format_func=lambda i: ui.model_label(registry.model(i)))
m = registry.model(mid)
r = registry.run(m["run_id"]) if m["run_id"] else None
st.write("**Lineage:** " + " ← ".join(x["id"] for x in registry.lineage(mid)))

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Record")
    rec = {"model_id": mid, "type": m["kind"], "parent": m["parent_id"], "base_model": m["base_model"],
           "base_revision": m["base_revision"], "adapter_path": m["adapter_path"], "created": m["created"]}
    if r:
        rec.update({"run_id": r["id"], "label": r["label"], "status": r["status"],
                    "training_method": {"sft": "SFT (TRL SFTTrainer + LoRA)", "rm": "Reward model (TRL RewardTrainer + LoRA)",
                                        "rlhf": "RLHF: GRPO with a learned reward model (TRL GRPOTrainer + LoRA)"}
                    .get(r["kind"], "GRPO with reward rules (TRL GRPOTrainer + LoRA)"),
                    "datasets": {d: (registry.dataset(d) or {}).get("hash") for d in (r["train_ds"], r["val_ds"]) if d},
                    "reward": r["reward_id"], "config": r["config"],
                    "libraries": json.loads(r["libs_json"] or "{}"), "hardware": json.loads(r["hardware_json"] or "{}"),
                    "log": r["log_path"], "metrics": r["metrics_path"],
                    "training_seconds": round(r["finished"] - r["started"], 1) if r["finished"] else None})
        if r["reward_id"] and registry.reward(r["reward_id"]):
            rec["reward_rules"] = registry.reward(r["reward_id"])["rules"]
    st.json(rec, expanded=False)
    note = st.text_area("Notes", m["note"] or "", key=f"note_{mid}")
    if st.button("Save note"):
        registry.update("models", mid, note=note)
        st.success("Saved.")
with c2:
    st.markdown("#### Actions")
    q = st.text_input("Load & ask a question", "Can I promise a drill tomorrow?",
                      disabled=m["kind"] == "rm", help="Reward models score answers — try them on the Reward Model page.")
    if st.button("💬 Load model and answer", disabled=m["kind"] == "rm"):
        with st.spinner("Generating…"):
            try:
                st.write(ui.generate(mid, [q])[0])
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    ref = json.loads(m["reference_json"] or "{}")
    if ref.get("answers") and st.button("🔁 Verify reload reproduces saved answers"):
        with st.spinner("Reloading adapter and generating…"):
            got = ui.generate(mid, ref["questions"], max_new_tokens=ref.get("max_new_tokens", 64))
        same = sum(a == b for a, b in zip(got, ref["answers"]))
        (st.success if same == len(got) else st.warning)(
            f"{same}/{len(got)} answers identical to the ones recorded right after training "
            "(small differences can come from normal numerical variation).")
        st.dataframe(pd.DataFrame({"question": ref["questions"], "saved": ref["answers"], "reloaded": got}),
                     use_container_width=True, hide_index=True)
    if st.button("★ Mark as preferred"):
        registry.set_preferred(mid)
        st.rerun()
    if m["kind"] != "base" and m["adapter_path"] and Path(m["adapter_path"]).exists():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in Path(m["adapter_path"]).rglob("*"):
                if f.is_file() and "checkpoints" not in f.parts:
                    z.write(f, f.relative_to(m["adapter_path"]))
        st.download_button("⬇️ Export adapter (.zip)", buf.getvalue(), file_name=f"{mid}-adapter.zip")
        st.caption(f"Needs base model {m['base_model']} (revision {m['base_revision']}) to run.")
    if m["kind"] != "base":
        with st.expander("🗑️ Delete this version"):
            kids = registry.children(mid)
            if kids:
                st.warning(f"Children {[k['id'] for k in kids]} keep working (their adapters are complete), "
                           "but their lineage will point to a deleted version.")
            confirm = st.text_input(f"Type {mid} to confirm")
            if st.button("Delete adapter files", disabled=confirm != mid):
                shutil.rmtree(m["adapter_path"], ignore_errors=True)
                registry.update("models", mid, deleted=1, preferred=0)
                ui.unload_engine()
                st.success("Deleted. The registry keeps the record for history.")
                st.rerun()
