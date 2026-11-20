import json

import pandas as pd
import streamlit as st

import ui_common as ui
from lab import data, registry, rewards

st.title("🧪 Evaluation & Scoring")
ui.explain("Run a model on validation questions it was not trained on and inspect every answer.",
           "A model version, a validation dataset version and a reward version used as an automated check.",
           "It answers each question (greedy decoding, so results are repeatable).",
           "The actual answers next to the approved answers, the rule-based score and YOUR judgement.",
           "An evaluation record: model, dataset, reward version, settings, answers, scores and your ratings.")
st.warning("A keyword score is only a basic check. The model may mention the right words while giving an "
           "incorrect answer. Automated rule scores are not a definitive correctness score.")

c1, c2, c3 = st.columns(3)
with c1:
    mid = ui.model_picker("Model", "ev_model", default_kind="sft")
with c2:
    role = st.selectbox("Validation set", ["sft_val", "rl_val"],
                        help="rl_val has grading fields (required_checks / avoid) that row-based rules use.")
    did = ui.dataset_picker("Version", role, "ev_ds")
with c3:
    rws = registry.rewards()
    rid = st.selectbox("Reward version for the automated score", [r["id"] for r in rws],
                       format_func=lambda i: f"{i} ({'approved' if registry.reward(i)['approved'] else 'draft'})")
    max_new = st.number_input("Max answer tokens", 16, 512, 96)

if st.button("▶️ Run evaluation", type="primary", disabled=not (mid and did)):
    rows = data.load(did)
    rules = registry.reward(rid)["rules"]
    with st.spinner(f"Generating {len(rows)} answers with {mid}…"):
        try:
            answers = ui.generate(mid, [r["prompt"] for r in rows], max_new_tokens=int(max_new))
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not run {mid}: {e}")
            st.stop()
    results = []
    for r, a in zip(rows, answers):
        s = rewards.score(a, rules, r)
        results.append({"id": r["id"], "question": r["prompt"], "answer": a, "approved": r.get("completion", ""),
                        "reward": s["total"], "breakdown": s["breakdown"], "review": ""})
    mean = sum(x["reward"] for x in results) / max(1, len(results))
    eid = registry.add_eval(mid, did, rid, {"max_new_tokens": int(max_new), "temperature": 0, "source": "evaluation page"},
                            results, mean)
    st.session_state["ev_id"] = eid
    st.success(f"Evaluation #{eid} saved.")

evs = registry.evals()
if not evs:
    st.info("No evaluations yet.")
    st.stop()
ids = [e["id"] for e in evs]
cur = st.session_state.get("ev_id", ids[0])
eid = st.selectbox("Evaluation record", ids, index=ids.index(cur) if cur in ids else 0,
                   format_func=lambda i: next(f"#{e['id']} · {e['model_id']} on {e['dataset_id']} · {e['reward_id']} · "
                                              f"mean {e['mean_reward']:.2f}" for e in evs if e["id"] == i))
e = next(x for x in evs if x["id"] == eid)
reviewed = [x for x in e["results"] if x.get("review")]
c = st.columns(3)
c[0].markdown(f"{ui.tag('reward')}<br>mean rule score **{e['mean_reward']:.2f}** with {e['reward_id']}",
              unsafe_allow_html=True)
c[1].markdown(f"{ui.tag('human')}<br>**{sum(x['review'] == 'correct' for x in reviewed)} / {len(reviewed)}** "
              f"reviewed answers rated correct", unsafe_allow_html=True)
c[2].markdown(f"{ui.tag('val')}<br>dataset **{e['dataset_id']}** · {e['settings'].get('source', '')}",
              unsafe_allow_html=True)

df = pd.DataFrame([{"Question": x["question"], "Model answer": x["answer"], "Approved answer": x.get("approved", ""),
                    "Automated score": x["reward"], "Your review": x.get("review", "")} for x in e["results"]])
edited = st.data_editor(df, use_container_width=True, hide_index=True, key=f"ev_ed_{eid}",
                        disabled=["Question", "Model answer", "Approved answer", "Automated score"],
                        column_config={"Your review": st.column_config.SelectboxColumn(
                            options=["", "correct", "partly correct", "incorrect"])})
if st.button("💾 Save my review"):
    for x, rv in zip(e["results"], edited["Your review"].tolist()):
        x["review"] = rv or ""
    registry.update("evals", eid, results_json=json.dumps(e["results"]))
    st.success("Saved.")
    st.rerun()

disagree = [x for x, rv in zip(e["results"], edited["Your review"]) if rv == "incorrect" and x["reward"] > 0]
if disagree:
    st.error(f"{len(disagree)} answer(s) you rated incorrect still got a positive rule score — a sign the reward "
             "rules can be fooled. Use them as bad examples on the Reward Designer test bench.")

with st.expander("🔍 Score breakdown per answer"):
    for x in e["results"]:
        st.markdown(f"**{x['question']}**  \n_{x['answer']}_")
        st.dataframe(pd.DataFrame(x["breakdown"]), use_container_width=True, hide_index=True)
