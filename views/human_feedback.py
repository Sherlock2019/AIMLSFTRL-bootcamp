import pandas as pd
import streamlit as st

import ui_common as ui
from lab import data, registry

st.title("🗳️ Human Feedback")
ui.explain("Collect human preferences — the 'HF' in RLHF. You compare two answers and pick the better one.",
           "Questions, and your choice between two answers the model wrote (or your own better answer).",
           "It writes two different answers to the same question (sampling with different seeds).",
           "Pairs where the difference is clear; skip ties — they teach the reward model nothing.",
           "A new version of the PREF dataset (question, chosen, rejected). The Reward Model page trains on it.")
st.caption("Track C · RLHF — step 1 of 3:  **collect preferences** → train a reward model → RL with the reward model.")

ui.step(1, "Choose the model and the questions", "Normally your SFT model — you judge what it actually writes.")
c1, c2 = st.columns(2)
with c1:
    mid = ui.model_picker("Model that writes the answers", "hf_model", default_kind="sft")
src = c2.selectbox("Questions from", ["rl_train", "sft_train", "custom"],
                   help="Use training questions, never validation/test questions (that would leak).")
if src == "custom":
    qs = [q.strip() for q in st.text_area("One question per line").splitlines() if q.strip()]
else:
    did = registry.latest_dataset(src)
    qs = [r["prompt"] for r in data.load(did["id"])] if did else []
    st.caption(f"{len(qs)} questions from {did['id'] if did else '—'}.")

if "hf_pairs" not in st.session_state:
    st.session_state.update(hf_pairs=[], hf_i=0, hf_ans=None)
pairs = st.session_state["hf_pairs"]

ui.step(2, "Compare two answers", "Pick the answer a careful store expert would give. Skip if they are equally good or bad.")
if not (mid and qs):
    st.info("Pick a model and questions first.")
else:
    i = st.session_state["hf_i"] % len(qs)
    q = qs[i]
    temp = st.slider("Creativity (sampling temperature)", 0.3, 1.5, 0.9, 0.1,
                     help="Higher = the two answers differ more.")
    if st.session_state["hf_ans"] is None or st.session_state["hf_ans"][0] != q:
        if st.button("✍️ Generate two answers", type="primary"):
            with st.spinner("Writing two answers…"):
                a = ui.generate(mid, [q], temperature=temp, seed=11 * (i + 1), max_new_tokens=96)[0]
                b = ui.generate(mid, [q], temperature=temp, seed=11 * (i + 1) + 5, max_new_tokens=96)[0]
                if data.normalize(a) == data.normalize(b):
                    b = ui.generate(mid, [q], temperature=min(1.5, temp + 0.4), seed=11 * (i + 1) + 9, max_new_tokens=96)[0]
            st.session_state["hf_ans"] = (q, a, b)
            st.rerun()
        st.markdown(f"**Question {i + 1}/{len(qs)}:** {q}")
    else:
        _, a, b = st.session_state["hf_ans"]
        st.markdown(f"**Question {i + 1}/{len(qs)}:** {q}")
        ca, cb = st.columns(2)
        ca.markdown(f"<div class='card'><b>Answer A</b><br>{a}</div>", unsafe_allow_html=True)
        cb.markdown(f"<div class='card'><b>Answer B</b><br>{b}</div>", unsafe_allow_html=True)
        own = st.text_area("Both bad? Write a better answer (optional) — it becomes 'chosen', answer A 'rejected'", height=70)

        def record(chosen, rejected):
            pairs.append({"prompt": q, "chosen": chosen, "rejected": rejected, "model": mid})
            st.session_state.update(hf_i=st.session_state["hf_i"] + 1, hf_ans=None)
            st.rerun()
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("👈 A is better", type="primary"):
            record(a, b)
        if b2.button("👉 B is better", type="primary"):
            record(b, a)
        if b3.button("✍️ Use my answer", disabled=not own.strip()):
            record(own.strip(), a)
        if b4.button("⏭ Skip (tie)"):
            st.session_state.update(hf_i=st.session_state["hf_i"] + 1, hf_ans=None)
            st.rerun()

ui.step(3, "Save your preferences", "Adds your pairs to the newest PREF dataset as a new version.")
st.metric("Pairs collected this session", len(pairs))
if pairs:
    st.dataframe(pd.DataFrame(pairs), hide_index=True, width="stretch")
    if st.button("💾 Save as new PREF version", type="primary"):
        base = registry.latest_dataset("pref")
        old = data.load(base["id"]) if base else []
        n0 = len(old)
        rows = old + [{"id": f"pref-h{n0 + k + 1:03d}", "prompt": p["prompt"], "chosen": p["chosen"],
                       "rejected": p["rejected"]} for k, p in enumerate(pairs)]
        new = data.save_version("pref", rows, parent_id=base["id"] if base else None, source="human feedback",
                                note=f"+{len(pairs)} preferences on {pairs[0]['model']}")
        st.session_state["hf_pairs"] = []
        st.success(f"Saved **{new}** ({len(rows)} pairs). Next: train a reward model on it.")
cur = registry.latest_dataset("pref")
if cur:
    st.caption(f"Newest PREF dataset: {cur['id']} · {cur['n_rows']} pairs · {cur['source']} (view/edit it in Data Studio).")
