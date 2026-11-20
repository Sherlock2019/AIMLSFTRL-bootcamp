import json

import pandas as pd
import streamlit as st

import ui_common as ui
from lab import agent, agent_evals, registry

ui.use_ui_engine()
st.title("🤖 Agent Lab")
ui.explain("A tool-using agent built with LangGraph: the model decides which tool to call, reads the result, and repeats.",
           "A model, which tools it may use, and your messages (multi-turn).",
           "Loops agent → tools → agent until it answers or hits the step budget. Tools are a safe sandbox with fake store data.",
           "Every step of the trace (model output, tool call, arguments, tool result) and eval scores per scenario.",
           "A trace per turn (Traces & Monitoring) and an agent-eval record per eval run.")

ui.step(1, "Set up the agent", "Choose the model, the tools it may call and how many steps it may take.")
c1, c2, c3 = st.columns([2, 3, 1])
with c1:
    mid = ui.model_picker("Model", "ag_model", default_kind="base",
                          help="Qwen models understand tool calls natively. Bigger = better tool choice.")
tools = c2.multiselect("Tools", list(agent.TOOLS), default=list(agent.TOOLS))
max_steps = c3.number_input("Max steps", 1, 8, 4)
with st.expander("How the graph works"):
    st.graphviz_chart("""digraph { rankdir=LR; node [shape=box, style=rounded, fontname=Helvetica];
        START [shape=circle]; END [shape=doublecircle];
        START -> agent; agent -> tools [label="tool call"]; tools -> agent [label="result"]; agent -> END [label="answer"]; }""")
    st.dataframe(pd.DataFrame([{"tool": n, "what it does": f.__doc__} for n, f in agent.TOOLS.items()]), hide_index=True,
                 width="stretch")


def show_steps(steps):
    for i, s in enumerate(steps):
        if s["step"] == "llm":
            calls = ", ".join(f"{c['name']}({json.dumps(c['arguments'])})" for c in s["tool_calls"])
            st.markdown(f"`{i + 1}` 🧠 **model** ({s['seconds']} s) → " + (f"calls {calls}" if calls else "answers"))
            if not calls:
                st.caption(s["output"][:400])
        else:
            st.markdown(f"`{i + 1}` 🛠️ **{s['name']}** `{json.dumps(s['arguments'])}` → `{json.dumps(s['result'])[:300]}`")


# ---------------------------------------------------------------- 2 chat
ui.step(2, "Talk to it (multi-turn)", "Each message is one turn; the agent remembers the conversation. Open 'trace' to see every step.")
if "ag_hist" not in st.session_state:
    import uuid
    st.session_state.update(ag_hist=[], ag_view=[], ag_sid=f"chat-{uuid.uuid4().hex[:6]}")
for turn in st.session_state["ag_view"]:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        with st.expander(f"trace · {len(turn['steps'])} steps · {turn['latency']} s"):
            show_steps(turn["steps"])
cA, cB = st.columns([5, 1])
if cB.button("🔄 New chat"):
    st.session_state.update(ag_hist=[], ag_view=[])
    st.rerun()
msg = st.chat_input("e.g. Do we have pressure washers at the warehouse, and is there a slot on Saturday?", disabled=not mid)
if msg:
    with st.spinner("Agent working…"):
        r = agent.run(mid, msg, history=st.session_state["ag_hist"], session=st.session_state["ag_sid"],
                      max_steps=int(max_steps), tools=tools)
    st.session_state["ag_hist"] = r["messages"]
    st.session_state["ag_view"].append({"user": msg, **r})
    st.rerun()

# ---------------------------------------------------------------- 3 evaluate
ui.step(3, "Evaluate the agent", "Run test scenarios: did it call the right tools with the right arguments, finish the task, and stay safe?")
scs = agent_evals.scenarios()
types = st.multiselect("Scenario types", list(agent_evals.TYPES), default=list(agent_evals.TYPES),
                       format_func=lambda t: f"{t} — {agent_evals.TYPES[t]} ({sum(s.get('type') == t for s in scs)})")
chosen = [s for s in scs if s.get("type", "tool") in types]
with st.expander(f"See the {len(chosen)} scenarios"):
    st.dataframe(pd.DataFrame([{"id": s["id"], "type": s.get("type"), "turns": " ⏵ ".join(s["turns"]),
                                "expected tools": " | ".join(",".join(t) or "–" for t in s.get("turn_tools", [])),
                                "must include": ", ".join(s.get("must_include", [])),
                                "must NOT include": ", ".join(s.get("must_not_include", []))} for s in chosen]),
                 hide_index=True, width="stretch")
if st.button(f"▶️ Run {len(chosen)} scenarios", type="primary", disabled=not (mid and chosen)):
    bar = st.progress(0.0)
    eid, _, _ = agent_evals.evaluate(mid, chosen, progress=lambda i, n: bar.progress(i / n, text=f"{i}/{n} scenarios"))
    st.session_state["ag_eval"] = eid
    st.rerun()

evs = registry.agent_evals()
if evs:
    eid = st.selectbox("Agent-eval record", [e["id"] for e in evs],
                       index=[e["id"] for e in evs].index(st.session_state.get("ag_eval", evs[0]["id"])),
                       format_func=lambda i: next(f"#{e['id']} · {e['model_id']} · {e['summary']['n']} scenarios · "
                                                  f"success {e['summary']['success_rate']:.0%}" for e in evs if e["id"] == i))
    e = next(x for x in evs if x["id"] == eid)
    s = e["summary"]
    c = st.columns(5)
    c[0].metric("Task success", f"{s['success_rate']:.0%}")
    c[1].metric("Right tools called", f"{s['tool_recall']:.0%}")
    for col, t in zip(c[2:], agent_evals.TYPES):
        col.metric(agent_evals.TYPES[t], f"{s[f'{t}_success']:.0%}" if f"{t}_success" in s else "—")
    df = pd.DataFrame([{"id": r["id"], "type": r["type"], "✓": "✅" if r["success"] else "❌", "first failure": r["first_failure"],
                        "tools called": ", ".join(r["called"]), "final answer": r["final"]} for r in e["results"]])
    st.dataframe(df, hide_index=True, width="stretch")
    bad = [r for r in e["results"] if not r["success"]]
    if bad:
        pick = st.selectbox("Open a failed session", [r["id"] for r in bad])
        r = next(x for x in bad if x["id"] == pick)
        st.error(f"Went wrong at: **{r['first_failure']}**")
        for i, t in enumerate(r["turns"]):
            st.markdown(f"**Turn {i + 1} · user:** {t['user']}")
            show_steps(t["steps"])
            st.markdown(f"**assistant:** {t['answer']}")
    if len({x["model_id"] for x in evs}) >= 2:
        st.markdown("##### Compare models")
        st.dataframe(pd.DataFrame([{"eval": f"#{x['id']}", "model": x["model_id"], "scenarios": x["summary"]["n"],
                                    "success": f"{x['summary']['success_rate']:.0%}",
                                    "tool recall": f"{x['summary']['tool_recall']:.0%}"} for x in evs]),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------- 4 add a test
ui.step(4, "Add your own test case", "Write a scenario the agent must pass. It is added to the eval set above.")
with st.form("ag_new"):
    turns = st.text_area("User turns (one per line)")
    f1, f2 = st.columns(2)
    exp = f1.multiselect("Tools that must be called (in the last turn)", list(agent.TOOLS))
    typ = f2.selectbox("Type", list(agent_evals.TYPES))
    inc = f1.text_input("Final answer must include (comma separated)")
    exc = f2.text_input("Answers must NOT include (comma separated)")
    if st.form_submit_button("➕ Add test case"):
        tl = [t.strip() for t in turns.splitlines() if t.strip()]
        if not tl:
            st.error("Write at least one user turn.")
        else:
            sid = agent_evals.add_scenario({"type": typ, "turns": tl, "turn_tools": [[] for _ in tl[:-1]] + [exp],
                                            "must_include": [w.strip() for w in inc.split(",") if w.strip()],
                                            "must_not_include": [w.strip() for w in exc.split(",") if w.strip()]})
            st.success(f"Added **{sid}**.")
