import html
import json

import pandas as pd
import plotly.express as px
import streamlit as st

import ui_common as ui
from lab import data, jobs, registry, rewards

st.title("⚖️ Compare Models")
ui.explain("Put model versions side by side on the same questions with the same settings.",
           "Which versions to compare and which questions to ask.",
           "Each selected version answers every question (greedy decoding, same max length).",
           "Answers first. Then the separate charts: losses, rewards, rule scores, human ratings, cost.",
           "Side-by-side runs can be saved as evaluation records for each model.")

ms = [m for m in registry.models() if m["kind"] != "rm"]
pref = []
for kind in ("base", "sft", "rl", "rlhf"):
    pref += [m["id"] for m in ms if m["kind"] == kind][-1:]
st.caption("The finale of the three tracks: put the latest **SFT**, **RL** and **RLHF** models side by side.")
sel = st.multiselect("Models", [m["id"] for m in ms], default=pref,
                     format_func=lambda i: ui.model_label(registry.model(i)))

tab_ans, tab_metrics = st.tabs(["💬 Side-by-side answers", "📊 Metrics"])
with tab_ans:
    c1, c2, c3 = st.columns(3)
    role = c1.selectbox("Questions from", ["test", "rl_val", "sft_val", "custom"])
    did = None
    if role == "custom":
        qs = [q for q in c2.text_area("Questions (one per line)", "Can I promise a drill tomorrow?").splitlines() if q.strip()]
        rows = [{"prompt": q} for q in qs]
    else:
        with c2:
            did = ui.dataset_picker("Version", role, "cmp_ds")
        rows = data.active_test_rows(data.load(did)) if did else []
    with c3:
        rid = st.selectbox("Score with reward version", [r["id"] for r in registry.rewards()])
        max_new = st.number_input("Max answer tokens", 16, 512, 96, key="cmp_max")
    save = st.checkbox("Save as evaluation records", value=role != "custom" and role != "test")
    if st.button("▶️ Compare", type="primary", disabled=not (sel and rows)):
        rules = registry.reward(rid)["rules"]
        table = {"Question": [r["prompt"] for r in rows]}
        means = {}
        for mid in sel:
            with st.spinner(f"{mid}…"):
                try:
                    ans = ui.generate(mid, [r["prompt"] for r in rows], max_new_tokens=int(max_new))
                except Exception as e:  # noqa: BLE001
                    st.error(f"{mid}: {e}")
                    continue
            table[mid] = ans
            scored = [rewards.score(a, rules, r) for a, r in zip(ans, rows)]
            means[mid] = sum(s["total"] for s in scored) / len(scored)
            if save and did:
                registry.add_eval(mid, did, rid, {"max_new_tokens": int(max_new), "temperature": 0, "source": "compare"},
                                  [{"id": r.get("id"), "question": r["prompt"], "answer": a, "approved": r.get("completion", ""),
                                    "reward": s["total"], "breakdown": s["breakdown"], "review": ""}
                                   for r, a, s in zip(rows, ans, scored)], means[mid])
        st.session_state["cmp"] = (table, means, did or "custom", rid)
    if "cmp" in st.session_state:
        table, means, d_used, r_used = st.session_state["cmp"]
        mids = [k for k in table if k != "Question"]
        st.markdown(f"{ui.tag('reward')} mean rule score on **{d_used}** with **{r_used}** (same questions, same settings, "
                    "so these are comparable with each other):", unsafe_allow_html=True)
        cols = st.columns(len(mids) or 1)
        for c, mid in zip(cols, mids):
            c.metric(mid, f"{means[mid]:+.2f}")
        for i, q in enumerate(table["Question"]):
            st.markdown(f"**{i + 1}. {q}**")
            cols = st.columns(len(mids) or 1)
            for c, mid in zip(cols, mids):
                c.markdown(f'<div class="card"><small><b>{mid}</b></small><br>{html.escape(table[mid][i])}</div>',
                           unsafe_allow_html=True)
        with st.expander("Table view / export"):
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
        if d_used == "test" or (d_used or "").startswith("test"):
            st.caption("Test questions have no grading fields, so row-based rules contribute 0 here.")

with tab_metrics:
    runs = [r for r in registry.runs() if r["status"] == "completed" and r["output_model_id"] in sel]
    a, b = st.columns(2)
    with a:
        st.markdown(ui.tag("train") + ui.tag("val") + " **SFT loss**", unsafe_allow_html=True)
        rows_ = []
        for r in runs:
            if r["kind"] == "sft":
                for x in jobs.read_jsonl(r["metrics_path"]):
                    if "loss" in x:
                        rows_.append({"step": x["step"], "value": x["loss"], "series": f"{r['output_model_id']} train"})
                    if "eval_loss" in x and not x.get("final"):
                        rows_.append({"step": x["step"], "value": x["eval_loss"], "series": f"{r['output_model_id']} val"})
        if rows_:
            st.plotly_chart(px.line(pd.DataFrame(rows_), x="step", y="value", color="series", markers=True,
                                    template="plotly_dark", height=300), use_container_width=True)
            st.caption("Each SFT run starts from a different parent, so losses show fitting within a run, not overall quality.")
        else:
            st.info("Insufficient data: no completed SFT runs among the selected models.")
    with b:
        st.markdown(ui.tag("reward") + " **RL training reward**", unsafe_allow_html=True)
        rows_ = [{"step": x["step"], "reward": x["reward"],
                  "run": f"{r['output_model_id']} ({r['reward_id'] or r['config'].get('rm_id', '')})"}
                 for r in runs if r["kind"] in ("rl", "rlhf") for x in jobs.read_jsonl(r["metrics_path"]) if "reward" in x]
        if rows_:
            st.plotly_chart(px.line(pd.DataFrame(rows_), x="step", y="reward", color="run", markers=True,
                                    template="plotly_dark", height=300), use_container_width=True)
            if len({(r["reward_id"], r["config"].get("rm_id")) for r in runs if r["kind"] in ("rl", "rlhf")}) > 1:
                st.warning("These RL runs used different reward versions — their rewards are not directly comparable.")
        else:
            st.info("Insufficient data: no completed RL runs among the selected models.")

    st.markdown(ui.tag("val") + ui.tag("reward") + " **Validation rule scores**", unsafe_allow_html=True)
    ev = [e for e in registry.evals() if e["model_id"] in sel]
    if ev:
        edf = pd.DataFrame([{"model": e["model_id"], "setting": f"{e['dataset_id']} · {e['reward_id']}",
                             "mean reward": e["mean_reward"], "when": e["created"]} for e in ev])
        edf = edf.sort_values("when").groupby(["model", "setting"], as_index=False).last()
        st.plotly_chart(px.bar(edf, x="model", y="mean reward", color="setting", barmode="group",
                               template="plotly_dark", height=320), use_container_width=True)
        if edf["setting"].nunique() > 1:
            st.warning("Bars with different colours come from different datasets or reward versions: compare only "
                       "bars of the same colour.")
    else:
        st.info("Insufficient data: no evaluations for the selected models.")

    st.markdown(ui.tag("human") + " **Human review ratings**", unsafe_allow_html=True)
    rv = [r for r in registry.reviews() if r["model_id"] in sel]
    if rv:
        rdf = pd.DataFrame(rv).groupby(["model_id", "rating"], as_index=False).size()
        st.plotly_chart(px.bar(rdf, x="model_id", y="size", color="rating", template="plotly_dark", height=300,
                               color_discrete_map={"correct": "#22c55e", "partly correct": "#f59e0b",
                                                   "incorrect": "#ef4444"}), use_container_width=True)
    else:
        st.info("Insufficient data: no human reviews for the selected models.")

    st.markdown("**Cost and size**")
    info = []
    for mid in sel:
        m = registry.model(mid)
        r = registry.run(m["run_id"]) if m["run_id"] else None
        hw = json.loads(r["hardware_json"]) if r and r["hardware_json"] else {}
        info.append({"model": mid, "type": m["kind"], "parent": m["parent_id"] or "—",
                     "train data": m["train_ds"] or "—",
                     "examples": registry.dataset(m["train_ds"])["n_rows"] if m["train_ds"] else "—",
                     "training time (s)": round(r["finished"] - r["started"], 1) if r and r["finished"] else "—",
                     "adapter size (MB)": round(m["size_bytes"] / 1e6, 1) if m["size_bytes"] else "—",
                     "device": hw.get("gpu") or hw.get("device", "—")})
    st.dataframe(pd.DataFrame(info), use_container_width=True, hide_index=True)
