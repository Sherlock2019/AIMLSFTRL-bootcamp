import pandas as pd
import streamlit as st

import ui_common as ui
from lab import hub, registry

st.title("🤗 Model Hub")
ui.explain("Bring other open models from Hugging Face into the lab to train, evaluate, serve and compare.",
           "A Hugging Face repo id such as `Qwen/Qwen2.5-1.5B-Instruct` (and a token for gated models).",
           "Nothing yet — it is downloaded once into the local Hugging Face cache and registered as a new base model.",
           "Size, licence and whether it fits your GPU before you download.",
           "A new base model in the registry (base-00N). Every page's model picker can then use it.")

ui.step(1, "Choose a model", "Pick a suggestion or type any Hugging Face repo id (text-generation models only).")
sug = st.radio("Suggestions", [r for r, _ in hub.SUGGESTED] + ["Other…"], label_visibility="collapsed",
               format_func=lambda r: r if r == "Other…" else f"{r} — {dict(hub.SUGGESTED)[r]}"
               + ("  ✅ cached" if hub.is_cached(r) else ""))
repo = st.text_input("Repo id", placeholder="owner/model-name") if sug == "Other…" else sug
token = st.text_input("Hugging Face token (only for gated models, not saved)", type="password") if sug == "Other…" else ""

ui.step(2, "Check it fits", "Reads size and licence from the Hub (needs internet). Skipped if already cached.")
if repo and st.button("🔍 Check size & fit"):
    try:
        st.session_state["hub_info"] = hub.info(repo)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read {repo} from the Hub: {e}")
info = st.session_state.get("hub_info")
if info and info["repo_id"] == repo:
    level, msg = hub.fit_advice(info["size_gb"])
    c = st.columns(4)
    c[0].metric("Weights", f"{info['size_gb']:.2f} GB")
    c[1].metric("Licence", info["license"] or "?")
    c[2].metric("Task", info["pipeline"] or "?")
    c[3].metric("Downloads", f"{info['downloads']:,}" if info["downloads"] else "?")
    {"ok": st.success, "warn": st.warning, "no": st.error}.get(level, st.info)(msg)
    if not info["has_safetensors"]:
        st.error("No .safetensors weights in this repo — the lab only loads safetensors.")
    if info["gated"]:
        st.warning("Gated model: accept its licence on huggingface.co and paste a token above.")

ui.step(3, "Import", "Downloads once (if needed) and registers it as a base model.")
if repo and st.button("⬇️ Import into the lab", type="primary"):
    with st.spinner(f"Importing {repo}… (large models take a while)"):
        try:
            mid = hub.import_model(repo, token)
            st.success(f"Registered **{mid}** ({repo}). Train it on SFT Training, or evaluate it on Evaluation & Scoring.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Import failed: {e}")

st.markdown("### Base models in the lab")
st.dataframe(pd.DataFrame([{"id": m["id"], "Hugging Face repo": m["base_model"], "cached": hub.is_cached(m["base_model"]),
                            "trained versions": sum(1 for x in registry.models() if x["base_model"] == m["base_model"]
                                                    and x["kind"] != "base")}
                           for m in registry.models() if m["kind"] == "base"]), hide_index=True, width="stretch")
st.caption("Only one base model is kept in GPU memory at a time — switching models reloads (a few seconds).")
