import pandas as pd
import streamlit as st

import ui_common as ui
from lab import data, registry

st.title("🧑‍⚖️ Human Review")
ui.explain("You judge answers to test questions the model has never trained on, and write better answers.",
           "Ratings (correct / partly correct / incorrect) and improved answers.",
           "It answers the test questions once, with greedy decoding.",
           "Where the model is still wrong — these corrections become the next SFT dataset.",
           "Every review (model, question, answer, rating, correction) is kept, even after it is used for training.")

c1, c2 = st.columns(2)
with c1:
    mid = ui.model_picker("Model to review", "hr_model", default_kind="rl")
with c2:
    tid = ui.dataset_picker("Test set version", "test", "hr_test")
test_rows = data.active_test_rows(data.load(tid)) if tid else []
st.caption(f"{len(test_rows)} active (unseen) test questions in {tid}.")

key = f"hr_ans_{mid}_{tid}"
if st.button("▶️ Generate answers", type="primary", disabled=not test_rows):
    with st.spinner("Generating…"):
        try:
            st.session_state[key] = ui.generate(mid, [r["prompt"] for r in test_rows], max_new_tokens=96)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))

answers = st.session_state.get(key)
if answers:
    with st.form("review_form"):
        entries = []
        for r, a in zip(test_rows, answers):
            st.markdown(f"**{r['id']} · {r['prompt']}**")
            st.markdown(f"> {a}")
            a1, a2 = st.columns([1, 3])
            rating = a1.radio("Your rating", ["correct", "partly correct", "incorrect"], index=None,
                              key=f"rt_{key}_{r['id']}", horizontal=False)
            corr = a2.text_area("Your correction (improved answer, optional)", key=f"co_{key}_{r['id']}", height=80)
            entries.append((r, a, rating, corr))
            st.divider()
        if st.form_submit_button("💾 Save reviews"):
            n = 0
            for r, a, rating, corr in entries:
                if rating:
                    registry.add_review(model_id=mid, test_ds=tid, question_id=r["id"], question=r["prompt"], answer=a,
                                        rating=rating, correction=corr.strip(), approved=0)
                    n += 1
            st.success(f"Saved {n} reviews.")

st.markdown("### Review history " + ui.tag("human"), unsafe_allow_html=True)
revs = registry.reviews()
if revs:
    st.dataframe(pd.DataFrame([{k: r[k] for k in ("id", "model_id", "question_id", "question", "answer", "rating",
                                                   "correction", "approved", "used_in_ds")} for r in revs]),
                 use_container_width=True, hide_index=True)

st.markdown("### ➕ Create a new SFT training dataset from approved corrections")
cands = [r for r in revs if r["correction"] and not r["used_in_ds"]]
if not cands:
    st.info("No unused corrections yet. Rate answers and write an improved answer above.")
else:
    pick = st.multiselect("Approve the corrections to add (only what you select is used)", [r["id"] for r in cands],
                          format_func=lambda i: next(f"#{r['id']} {r['question'][:60]} → {r['correction'][:60]}"
                                                     for r in cands if r["id"] == i))
    chosen = [r for r in cands if r["id"] in pick]
    base_sft = registry.latest_dataset("sft_train")
    st.warning(f"All {len(chosen)} selected questions come from the **manual test set**. Once used for training they "
               "are no longer unseen, so they will be **retired** from the test set (history is kept). "
               "Add replacement questions below afterwards.")
    ok = st.checkbox("I understand: retire these test questions and use them for training")
    if st.button("Create new SFT dataset version", type="primary", disabled=not (ok and chosen)):
        rows = data.load(base_sft["id"]) + [{"id": f"corr-{r['id']}", "prompt": r["question"],
                                             "completion": r["correction"]} for r in chosen]
        new_sft = data.save_version("sft_train", rows, parent_id=base_sft["id"], source="corrections",
                                    note=f"{base_sft['id']} + {len(chosen)} human corrections")
        latest_test = registry.latest_dataset("test")
        qids = {r["question_id"] for r in chosen}
        trows = [dict(t, retired=True) if t["id"] in qids else t for t in data.load(latest_test["id"])]
        new_test = data.save_version("test", trows, parent_id=latest_test["id"], source="retire",
                                     note=f"retired {sorted(qids)} (used in {new_sft})")
        for r in chosen:
            registry.update("reviews", r["id"], used_in_ds=new_sft, approved=1)
        st.success(f"Created {new_sft} and {new_test}. Next: add replacement test questions, then run SFT "
                   f"from your RL model with {new_sft}.")

st.markdown("### 🆕 Add replacement test questions")
newq = st.text_area("One new unseen question per line")
if st.button("Add to test set") and newq.strip():
    latest_test = registry.latest_dataset("test")
    trows = data.load(latest_test["id"])
    n0 = len(trows)
    trows += [{"id": f"test-{n0 + i + 1:03d}", "prompt": q.strip(), "retired": False}
              for i, q in enumerate(newq.splitlines()) if q.strip()]
    st.success(f"Created {data.save_version('test', trows, parent_id=latest_test['id'], source='edit', note='replacement questions')}")
