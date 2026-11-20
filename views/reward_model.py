import json

import pandas as pd
import streamlit as st

import run_panel
import ui_common as ui
from lab import data, jobs, registry

st.title("🏅 Reward Model")
ui.explain("Train a model that predicts which answer a human would prefer. RLHF then uses its score as the reward.",
           "A PREF dataset (question, chosen, rejected) and a base model to start from.",
           "A score head is added to the base model and trained (LoRA) so score(chosen) > score(rejected).",
           "Held-out accuracy: on pairs it never saw, how often does it prefer the human's choice? 50% = guessing.",
           "A reward model version (rm-00N) with its accuracy, and every held-out pair with both scores.")
st.caption("Track C · RLHF — step 2 of 3:  collect preferences → **train a reward model** → RL with the reward model.")

tab_new, tab_runs, tab_try = st.tabs(["▶️ Train", "📈 Runs & accuracy", "🧪 Try it"])
with tab_new:
    ui.step(1, "Choose the preference data", "More and clearer pairs = a better reward model. 20 pairs is a smoke test.")
    did = ui.dataset_picker("PREF dataset", "pref", "rm_ds")
    if did:
        rows = data.load(did)
        warns = data.quality_warnings("pref", rows)
        st.caption(f"{len(rows)} pairs · " + (f"{len(warns)} warning(s) — see Data Studio" if warns else "no quality warnings"))
        with st.expander("Preview pairs"):
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    ui.step(2, "Choose the starting model and settings", "Start from a base model; a fresh score head is added.")
    c1, c2 = st.columns(2)
    with c1:
        parent = ui.model_picker("Base model", "rm_parent", kinds=["base"])
    preset = c2.selectbox("Preset", list(jobs.RM_PRESETS))
    cfg = dict(jobs.RM_PRESETS[preset])
    with st.expander("Advanced"):
        a, b = st.columns(2)
        cfg["val_frac"] = a.slider("Held-out share of pairs", 0.1, 0.5, cfg["val_frac"], 0.05,
                                   help="These pairs are never trained on; accuracy is measured on them.")
        cfg["max_steps"] = int(b.number_input("Max steps (-1 = use epochs)", -1, 2000, cfg["max_steps"]))
        cfg["learning_rate"] = a.number_input("Learning rate", value=cfg["learning_rate"], format="%.1e")
        cfg["seed"] = int(b.number_input("Seed", 0, 10**6, cfg["seed"]))
    st.caption(f"{preset}: {cfg['max_steps'] if cfg['max_steps'] > 0 else str(cfg['epochs']) + ' epochs'} steps, "
               f"lr {cfg['learning_rate']}, LoRA r={cfg['lora_r']}, {int(cfg['val_frac'] * 100)}% held out.")
    ui.step(3, "Train", "Runs in the background; follow it in 'Runs & accuracy'.")
    active = jobs.active_run()
    if active:
        st.warning(f"{active['id']} is running — one training run at a time.")
    if st.button("🚀 Train reward model", type="primary", disabled=bool(active) or not (did and parent)):
        try:
            ui.unload_engine()
            st.success(f"Started {jobs.start('rm', parent, did, None, cfg)}. Open 'Runs & accuracy'.")
        except Exception as e:  # noqa: BLE001
            st.error(str(e))


def rm_details(r):
    m = registry.model(r["output_model_id"])
    if not m:
        return
    ref = json.loads(m["reference_json"] or "{}")
    c = st.columns(3)
    c[0].metric("Held-out accuracy", f"{ref.get('val_accuracy', 0):.0%}", help="50% = guessing, 100% = always agrees with humans")
    c[1].metric("Training accuracy", f"{ref.get('train_accuracy', 0):.0%}")
    c[2].metric("Mean score margin", f"{ref.get('val_margin', 0):+.2f}", help="score(chosen) − score(rejected) on held-out pairs")
    if (ref.get("train_accuracy") or 0) - (ref.get("val_accuracy") or 0) > 0.25:
        st.warning("Much better on training pairs than held-out pairs: the reward model memorised instead of "
                   "learning the preference. Collect more varied pairs.")
    st.caption(f"Measured on {ref.get('val_on', '?')} ({ref.get('n_val', 0)} pairs).")
    if ref.get("pairs"):
        st.dataframe(pd.DataFrame([dict(p, agrees="✅" if p["chosen_score"] > p["rejected_score"] else "❌")
                                   for p in ref["pairs"]]), hide_index=True, width="stretch")


with tab_runs:
    run_panel.show("rm", extra=rm_details)


@st.cache_resource(show_spinner="Loading reward model…", max_entries=1)
def scorer(rm_id):
    from lab.reward_model import Scorer
    return Scorer(rm_id)


with tab_try:
    st.caption("Score answers yourself: a good reward model gives the careful answer a higher score.")
    rm_id = ui.model_picker("Reward model", "rm_try", kinds=["rm"])
    q = st.text_input("Question", "Can I promise a customer their lawn mower arrives tomorrow?")
    a1 = st.text_area("Answer 1", "I will check warehouse stock and delivery slots before confirming a date.", height=70)
    a2 = st.text_area("Answer 2", "Yes, it will definitely arrive tomorrow.", height=70)
    if st.button("🏅 Score both", disabled=not rm_id):
        ui.unload_engine()
        s = scorer(rm_id).score([q, q], [a1, a2])
        c = st.columns(2)
        c[0].metric("Answer 1 score", f"{s[0]:+.2f}")
        c[1].metric("Answer 2 score", f"{s[1]:+.2f}")
        st.info(f"The reward model prefers **answer {1 if s[0] > s[1] else 2}**.")
