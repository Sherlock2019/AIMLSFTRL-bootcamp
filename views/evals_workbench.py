import json

import pandas as pd
import streamlit as st

import ui_common as ui
from lab import data, evals, registry

st.title("🔬 Evals Workbench")
ui.explain("Find out *why* a model fails, whether your automated check can be trusted, and turn failures into fixes.",
           "An evaluation record from **Evaluation & Scoring**, plus your labels and notes.",
           "Nothing new — this page only analyses answers the model already gave.",
           "The top failure modes, reviewer agreement, how often the rule score agrees with humans, and a pass/fail gate.",
           "Your labels, failure tags, causes and corrections are saved inside the evaluation record.")

evs = registry.evals()
if not evs:
    st.info("No evaluations yet. Run one on **Evaluation & Scoring** first, then come back.")
    st.stop()

st.markdown("**Pick the evaluation to analyse**")
eid = st.selectbox("Evaluation", [e["id"] for e in evs], label_visibility="collapsed",
                   format_func=lambda i: next(f"#{e['id']} · {e['model_id']} on {e['dataset_id']} · "
                                              f"{len(e['results'])} answers · mean score {e['mean_reward']:.2f}"
                                              for e in evs if e["id"] == i))
e = next(x for x in evs if x["id"] == eid)
R = e["results"]
n_rev = sum(bool(x.get("review")) for x in R)
st.progress(n_rev / max(1, len(R)), text=f"{n_rev} of {len(R)} answers reviewed")


def persist(msg=None):
    registry.update("evals", eid, results_json=json.dumps(R))
    if msg:
        st.session_state["wb_msg"] = msg
    st.rerun()


if m := st.session_state.pop("wb_msg", None):
    st.success(m)

t1, t2, t3, t4, t5, t6 = st.tabs(["① Review", "② Error analysis", "③ Agreement", "④ Trust the judge?",
                                  "⑤ Search & charts", "⑥ Fix & gate"])

# ------------------------------------------------------------------ ① review card
with t1:
    st.caption("Look at answers **one at a time**. Label it, and for failures give it a short tag "
               "(e.g. *promised a date*) and a cause. Start with ~20–30 answers.")
    c1, c2 = st.columns(2)
    who = c1.text_input("Your name (reviewer)", st.session_state.get("wb_who", "me")).strip() or "me"
    st.session_state["wb_who"] = who
    strat = c2.selectbox("Which answers first?", list(evals.SAMPLING), help="\n".join(f"**{k}**: {v}" for k, v in evals.SAMPLING.items()))
    st.caption(f"↳ {evals.SAMPLING[strat]}")
    order = evals.order_for_review(R, strat, seed=eid, reviewer=who)
    pkey = f"wb_pos_{eid}_{strat}"
    pos = min(st.session_state.get(pkey, 0), max(0, len(order) - 1))
    if not order:
        st.success("Nothing left in this queue. 🎉 Pick another queue or go to ② Error analysis.")
    else:
        i = order[pos]
        x = R[i]
        st.markdown(f"**Answer {pos + 1} of {len(order)}** · automated score `{x['reward']:.2f}`")
        a, b = st.columns(2)
        a.markdown(f"<div class='card'><b>Question</b><br>{x['question']}<br><br><b>Model answer</b><br>{x['answer']}</div>",
                   unsafe_allow_html=True)
        b.markdown(f"<div class='card'><b>Approved answer</b><br>{x.get('approved') or '—'}</div>",
                   unsafe_allow_html=True)
        known = sorted({(y.get("failure_mode") or "").strip() for y in R} - {""})
        with st.form(f"card_{eid}_{i}"):
            cur = evals.human_label(x, who) or None
            lab = st.radio("Is the answer right?", evals.LABELS, horizontal=True,
                           index=evals.LABELS.index(cur) if cur in evals.LABELS else None)
            f1, f2 = st.columns(2)
            mode = f1.text_input("If not: failure tag (short)", x.get("failure_mode", ""),
                                 help="Reuse tags so they can be counted. Existing: " + (", ".join(known) or "none yet"))
            cause = f2.radio("If not: main cause", ["—"] + evals.CAUSES, horizontal=True,
                             index=(evals.CAUSES.index(x["cause"]) + 1) if x.get("cause") in evals.CAUSES else 0)
            corr = st.text_area("Better answer (optional — can become a training example)", x.get("correction", ""),
                                height=70)
            note = st.text_input("Note (optional)", x.get("note", ""))
            s1, s2, s3 = st.columns(3)
            go_save = s1.form_submit_button("💾 Save & next", type="primary")
            go_prev = s2.form_submit_button("◀ Previous")
            go_skip = s3.form_submit_button("Skip ▶")
        if known:
            st.caption("Existing tags: " + " · ".join(f"`{k}`" for k in known))
        with st.expander("What do the causes mean?"):
            for k, v in evals.CAUSE_HELP.items():
                st.markdown(f"- **{k}** — {v}")
        if go_save:
            if not lab:
                st.error("Pick a label first.")
            else:
                x.setdefault("reviews", {})[who] = lab
                x.update(review=lab, failure_mode=mode.strip(), cause=cause if cause != "—" else "",
                         correction=corr.strip(), note=note.strip())
                if strat != "Not reviewed yet":
                    st.session_state[pkey] = pos + 1
                persist()
        if go_prev:
            st.session_state[pkey] = max(0, pos - 1)
            st.rerun()
        if go_skip:
            st.session_state[pkey] = pos + 1
            st.rerun()

# ------------------------------------------------------------------ ② error analysis
with t2:
    st.caption("Count the tagged failures. **Fix the biggest bar first** — that is where an eval or a data fix pays off most.")
    fails = evals.failures(R)
    if not fails:
        st.info("No failures labelled yet. Review answers in ① and mark the wrong ones.")
    else:
        tbl = pd.DataFrame(evals.failure_table(R), columns=["Failure tag", "Count", "Share"])
        st.bar_chart(tbl.set_index("Failure tag")["Count"], horizontal=True)
        st.dataframe(tbl.assign(Share=lambda d: (d["Share"] * 100).round().astype(int).astype(str) + "%"),
                     hide_index=True, width="stretch")
        st.markdown("##### Where do the failures come from?")
        cc = evals.cause_counts(R)
        cols = st.columns(len(cc))
        for col, (k, v) in zip(cols, cc.items()):
            col.metric(k, v)
        top = max((k for k in cc if k in evals.CAUSES), key=cc.get, default=None)
        if top:
            st.info(f"Most failures are **{top}**: {evals.CAUSE_HELP[top]}.")

# ------------------------------------------------------------------ ③ agreement
with t3:
    st.caption("Do two people agree on what 'correct' means? Low agreement = the standard is unclear: "
               "write it down before trusting any score.")
    names = evals.reviewers(R)
    if len(names) < 2:
        st.info(f"Reviewers so far: {', '.join(names) or 'none'}. A second person should review the **same** "
                "answers in ① using their own name.")
    else:
        c1, c2 = st.columns(2)
        ra = c1.selectbox("Reviewer A", names, index=0)
        rb = c2.selectbox("Reviewer B", names, index=1)
        la = [evals.human_label(x, ra) for x in R]
        lb = [evals.human_label(x, rb) for x in R]
        k, n = evals.cohen_kappa(la, lb)
        same = sum(1 for p, q in zip(la, lb) if p and q and p == q)
        m = st.columns(3)
        m[0].metric("Answers both labelled", n)
        m[1].metric("Same label", f"{same}/{n}" if n else "—")
        m[2].metric("Cohen's kappa", f"{k:.2f}" if k is not None else "—", evals.kappa_meaning(k), delta_color="off")
        st.caption("Kappa corrects for agreement by chance: above 0.6 is usually good enough to automate.")
        dis = [x for x, p, q in zip(R, la, lb) if p and q and p != q]
        if dis:
            st.markdown("##### Disagreements — discuss these and write the rule down")
            st.dataframe(pd.DataFrame([{"Question": x["question"], "Answer": x["answer"], ra: evals.human_label(x, ra),
                                        rb: evals.human_label(x, rb)} for x in dis]), hide_index=True,
                         width="stretch")

# ------------------------------------------------------------------ ④ judge validation
with t4:
    st.caption("Any automated grader is a *judge*: the rule score, or an LLM judge. Before trusting one, check it "
               "against your labels, then estimate the **true** pass rate with an error bar.")
    src = st.radio("Which judge?", ["rule", "llm"], horizontal=True,
                   format_func=lambda s: "Rule score (Reward Designer)" if s == "rule" else "LLM judge (a model grades)")
    th = 0.5
    if src == "rule":
        th = st.slider("Score that counts as 'pass'", 0.0, 1.0, 0.5, 0.05)
    else:
        from lab import judge as judge_mod
        n_j = sum(bool(x.get("judge")) for x in R)
        with st.expander(f"Run the LLM judge ({n_j}/{len(R)} answers graded so far)", expanded=not n_j):
            jm = ui.model_picker("Judge model", "wb_judge_model", kinds=["base"],
                                 help="Pick a different (ideally stronger) model than the one being graded, "
                                      "e.g. import Qwen2.5-1.5B-Instruct on Model Hub.")
            crit = st.text_area("Criterion the judge applies", e["settings"].get("judge_criterion", judge_mod.DEFAULT_CRITERION),
                                height=90)
            if st.button("⚖️ Grade all answers", disabled=not jm):
                bar = st.progress(0.0)
                judge_mod.judge_eval(eid, jm, crit, progress=lambda i, n: bar.progress(i / n, text=f"{i}/{n}"))
                st.session_state["wb_msg"] = "LLM judge finished. Check its agreement with your labels below."
                st.rerun()
        if n_j:
            with st.expander("Judge verdicts and reasons"):
                st.dataframe(pd.DataFrame([{"Question": x["question"], "Answer": x["answer"], "Judge": x.get("judge", ""),
                                            "Reason": x.get("judge_reason", ""), "Human": x.get("review", "")} for x in R]),
                             hide_index=True, width="stretch")
    rep = evals.judge_report(R, th, source=src)
    if rep["n_labeled"] == 0:
        st.info("Label some answers in ① first (aim for 20+ with both passes and failures).")
    else:
        st.dataframe(pd.DataFrame({"Human: correct": [rep["tp"], rep["fn"]],
                                   "Human: not correct": [rep["fp"], rep["tn"]]},
                                  index=["Judge: pass", "Judge: fail"]), width="stretch")
        m = st.columns(4)
        fmt = lambda v: "—" if v is None else f"{v:.0%}"
        m[0].metric("Catches good answers (TPR)", fmt(rep["tpr"]))
        m[1].metric("Catches bad answers (TNR)", fmt(rep["tnr"]))
        m[2].metric("Judge pass rate (raw)", fmt(rep["judge_pass_rate"]))
        ci = rep["ci95"]
        m[3].metric("Estimated true pass rate", fmt(rep["corrected_pass_rate"]),
                    f"95% CI {ci[0]:.0%}–{ci[1]:.0%}" if ci else "need more labels for CI", delta_color="off")
        if rep["fp"]:
            st.warning(f"{rep['fp']} answer(s) passed the judge but a human said not correct — the rules can be "
                       "fooled. Tighten them on **Reward Designer**.")
        st.caption(f"Based on {rep['n_labeled']} human labels. Correction: (raw + TNR − 1) / (TPR + TNR − 1); "
                   "CI from 1,000 bootstrap resamples of the labelled answers.")

# ------------------------------------------------------------------ ⑤ search & charts
with t5:
    st.caption("Look across all answers at once to spot patterns.")
    q = st.text_input("🔎 Search questions, answers, notes and tags")
    hits = evals.search(R, q)
    c1, c2 = st.columns(2)
    c1.markdown("**Automated score distribution**")
    c1.bar_chart(pd.Series([round(x["reward"], 1) for x in R]).value_counts().sort_index())
    c2.markdown("**Human labels**")
    c2.bar_chart(pd.Series([x.get("review") or "not reviewed" for x in R]).value_counts())
    st.dataframe(pd.DataFrame([{"Question": R[i]["question"], "Answer": R[i]["answer"], "Score": R[i]["reward"],
                                "Label": R[i].get("review", ""), "Tag": R[i].get("failure_mode", ""),
                                "Cause": R[i].get("cause", "")} for i in hits]),
                 hide_index=True, width="stretch")
    st.caption(f"{len(hits)} of {len(R)} answers shown.")

# ------------------------------------------------------------------ ⑥ fix & gate
with t6:
    st.markdown("##### A · Turn failures into training examples")
    rows = evals.failures_to_sft_rows(R, eid)
    st.caption("Failed answers with a better answer (or an approved answer) become new SFT TRAIN examples.")
    if not rows:
        st.info("No failed answers with a better/approved answer yet.")
    else:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if e["dataset_id"].startswith(("sft_val", "rl_val")):
            st.warning("These questions come from a **validation** set. After training on them, that validation "
                       "set is no longer unseen — add fresh validation questions in Data Studio (leakage check will flag it).")
        if st.button(f"➕ Add {len(rows)} example(s) to SFT TRAIN", type="primary"):
            base = registry.latest_dataset("sft_train")
            have = {r["id"] for r in data.load(base["id"])}
            new = data.save_version("sft_train", data.load(base["id"]) + [r for r in rows if r["id"] not in have],
                                    parent_id=base["id"], source="eval-fixes", note=f"failures from evaluation #{eid}")
            persist(f"Created **{new}**. Train an SFT model from it, evaluate again, and compare.")

    st.markdown("##### B · Quality gate (for CI/CD)")
    st.caption("A release check: the model only ships if the evaluation meets these minimums.")
    g1, g2 = st.columns(2)
    ms = g1.number_input("Min mean rule score", 0.0, 1.0, 0.5, 0.05)
    mh = g2.number_input("Min human pass rate (reviewed answers)", 0.0, 1.0, 0.7, 0.05)
    ok, msgs = evals.gate(R, e["mean_reward"], ms, mh)
    (st.success if ok else st.error)(("✅ GATE PASSED" if ok else "❌ GATE FAILED") + " — " + " · ".join(msgs))
    st.code(f".venv/bin/python scripts/eval_gate.py --eval-id {eid} --min-score {ms} --min-human {mh}", language="bash")
    st.caption("Run this in your CI pipeline: exit code 0 = pass, 1 = fail. Add `--model … --dataset … --reward …` "
               "to run a fresh evaluation first.")

with st.expander("📚 How this page maps to an evals course"):
    st.markdown("""
| Topic | Where |
|---|---|
| Product evals vs model evals; is it data, instructions or model? | ① cause field, ② "where failures come from" |
| Error analysis (open + axial coding) | ① failure tags → ② ranked chart |
| Collaborative evaluation, agreement | ③ Cohen's kappa + disagreement list |
| Automated evaluators, judge validation, uncertainty | ④ TPR/TNR, corrected pass rate, bootstrap CI |
| CI/CD, turn failures into tests | ⑥ gate + `scripts/eval_gate.py`, failures → training rows |
| Interfaces for human review, sampling | ① one-at-a-time card with 4 sampling queues |
| Data analysis for traces | ⑤ search + charts |
| Improving the model with evals | ⑥ fixes → SFT → re-evaluate on **Compare Models** |
| LLM-as-judge | ④ "LLM judge" — run it, then validate it against your labels |
| RAG evaluation (IR metrics) | **RAG Lab** steps 3–4 |
| Multi-turn, tool use / agents, security tests | **Agent Lab** step 3, failed sessions show the first failing turn |
| Production monitoring | **Traces & Monitoring** |
""")
