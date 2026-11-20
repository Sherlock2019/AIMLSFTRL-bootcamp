import json

import pandas as pd
import streamlit as st

import ui_common as ui
from lab import agent_evals, registry

st.title("📡 Traces & Monitoring")
ui.explain("Every RAG question, agent turn and API request is logged here — your production monitoring.",
           "Nothing: traces appear automatically. You flag bad ones and turn them into tests.",
           "Nothing new — this page reads logs.",
           "Volume, latency, failures, and the full step-by-step trace of any request.",
           "Flags on traces, and new agent test cases from flagged traces.")

import time  # noqa: E402

from lab import langfuse_stack, services  # noqa: E402

st.markdown("**🔭 Open the real monitoring tools** — the same traces in Langfuse, the agent graph in LangGraph Studio.")
lf, sd = st.columns(2)
with lf:
    with st.container(border=True):
        up = langfuse_stack.running()
        st.markdown(f"**Langfuse** (self-hosted, port {langfuse_stack.PORT}) · {'🟢 running' if up else '⚪ stopped'}")
        st.caption("LLM observability: traces, steps, latency, sessions. The lab's own Docker stack — separate from "
                   "any other project. New lab traces are sent automatically while it runs.")
        if not up:
            if st.button("▶️ Start Langfuse (Docker, ~1 min first time)"):
                with st.spinner("docker compose up …"):
                    ok, msg = langfuse_stack.start()
                    for _ in range(90):
                        if langfuse_stack.running():
                            break
                        time.sleep(2)
                st.rerun() if ok else st.error(msg)
        else:
            cfg = langfuse_stack.config()
            st.link_button("Open Langfuse ↗", f"{langfuse_stack.url()}/project/ai-training-lab/traces", type="primary")
            with st.popover("Login"):
                st.code(f"email:    {cfg.get('LF_USER_EMAIL')}\npassword: {cfg.get('LF_USER_PASSWORD')}")
                st.caption("Generated for this lab only; stored in docker/langfuse/.env (git-ignored).")
            a, b = st.columns(2)
            if a.button("⤴️ Send all existing traces"):
                try:
                    st.success(f"Sent {langfuse_stack.send(registry.traces(limit=2000))} traces.")
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))
            if b.button("⏹ Stop Langfuse"):
                langfuse_stack.stop()
                st.rerun()
with sd:
    with st.container(border=True):
        sup = services.studio_running()
        st.markdown(f"**LangGraph Studio** (agent graph, port {services.STUDIO_PORT}) · {'🟢 running' if sup else '⚪ stopped'}")
        st.caption("Runs `langgraph dev` on the Agent Lab graph; the Studio UI is LangSmith's web app (free login), "
                   "talking to your local server. Step through agent → tools → agent.")
        if not sup:
            if st.button("▶️ Start LangGraph dev server"):
                services.start_studio()
                for _ in range(60):
                    if services.studio_running():
                        break
                    time.sleep(1)
                st.rerun()
        else:
            st.link_button("Open in LangGraph Studio ↗", services.STUDIO_URL, type="primary")
            if st.button("⏹ Stop dev server"):
                services.stop("studio")
                st.rerun()
        with st.expander("Dev server log"):
            st.code(services.tail("studio", 25) or "(empty)")
st.divider()

tr = registry.traces(limit=2000)
if not tr:
    st.info("No traces yet. Ask something in RAG Lab, Agent Lab, or call the API.")
    st.stop()

ui.step(1, "Overview", "How much traffic, how fast, and how many flagged as bad.")
df = pd.DataFrame([{"id": t["id"], "kind": t["kind"], "source": t["source"], "model": t["model_id"], "session": t["session"],
                    "input": t["input"], "output": t["output"], "latency": t["latency"], "flag": t["flag"],
                    "tool calls": sum(1 for s in t["steps"] if s.get("step") == "tool"),
                    "time": pd.to_datetime(t["created"], unit="s")} for t in tr])
c = st.columns(4)
c[0].metric("Traces", len(df))
c[1].metric("Median latency", f"{df['latency'].median():.1f} s")
c[2].metric("p95 latency", f"{df['latency'].quantile(0.95):.1f} s")
c[3].metric("Flagged bad", int((df["flag"] == "bad").sum()))
a, b = st.columns(2)
a.markdown("**Requests by kind**")
a.bar_chart(df["kind"].value_counts())
b.markdown("**Latency per request (s)**")
b.line_chart(df.sort_values("time").set_index("time")["latency"])

ui.step(2, "Find traces", "Filter and search across inputs and outputs.")
f1, f2, f3 = st.columns([1, 1, 2])
kinds = f1.multiselect("Kind", sorted(df["kind"].unique()), default=sorted(df["kind"].unique()))
flag = f2.selectbox("Flag", ["all", "bad", "good", "unflagged"])
q = f3.text_input("🔎 Search")
v = df[df["kind"].isin(kinds)]
if flag != "all":
    v = v[v["flag"] == ("" if flag == "unflagged" else flag)]
if q:
    v = v[v["input"].str.contains(q, case=False, na=False) | v["output"].str.contains(q, case=False, na=False)]
st.dataframe(v.drop(columns=["session"]), hide_index=True, width="stretch")

ui.step(3, "Inspect, flag, turn into a test", "Open one trace. Flag it; if it is an agent failure, save it as a test so it never regresses.")
if v.empty:
    st.stop()
tid = st.selectbox("Trace", v["id"].tolist(), format_func=lambda i: f"#{i} · {df.loc[df.id == i, 'kind'].item()} · "
                                                                     f"{df.loc[df.id == i, 'input'].item()[:70]}")
t = next(x for x in tr if x["id"] == tid)
st.markdown(f"<div class='card'><b>Input</b><br>{t['input']}<br><br><b>Output</b><br>{t['output']}</div>", unsafe_allow_html=True)
st.caption(f"{t['kind']} · {t['source']} · model {t['model_id']} · {t['latency']} s · session {t['session'] or '–'}")
if langfuse_stack.running():
    st.link_button("Open this trace in Langfuse ↗", langfuse_stack.trace_url(tid))
with st.expander(f"Steps ({len(t['steps'])})", expanded=True):
    for i, s in enumerate(t["steps"]):
        st.markdown(f"`{i + 1}` **{s.get('step')}** " + (f"`{s.get('name')}`" if s.get("name") else ""))
        st.json(s, expanded=False)
c1, c2, c3 = st.columns(3)
if c1.button("👎 Flag bad"):
    registry.update("traces", tid, flag="bad")
    st.rerun()
if c2.button("👍 Mark good"):
    registry.update("traces", tid, flag="good")
    st.rerun()
if t["kind"] in ("agent", "api", "rag"):
    with c3.popover("🧪 Turn into agent test"):
        tools_used = sorted({s["name"] for s in t["steps"] if s.get("step") == "tool"})
        exp = st.multiselect("Tools it should call", ["check_stock", "delivery_slots", "policy_search", "calculator",
                                                      "create_ticket"], default=tools_used)
        inc = st.text_input("Answer must include (comma separated)")
        exc = st.text_input("Answer must NOT include (comma separated)")
        if st.button("Save test"):
            sid = agent_evals.add_scenario({"type": "tool", "turns": [t["input"]], "turn_tools": [exp],
                                            "must_include": [w.strip() for w in inc.split(",") if w.strip()],
                                            "must_not_include": [w.strip() for w in exc.split(",") if w.strip()],
                                            "from_trace": tid})
            st.success(f"Saved {sid} — it runs with the other scenarios in Agent Lab step 3.")
