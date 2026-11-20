import json
import time

import streamlit as st

import ui_common as ui
from lab import services

API = f"http://localhost:{services.API_PORT}"
st.title("🔌 API & Integrations")
ui.explain("Serve your models to other software: a FastAPI server, LangChain clients, and Flowise visual flows.",
           "Which model to call (by registry id, e.g. sft-001) and what to send.",
           "The API process loads the model and answers requests. It speaks the OpenAI chat format, so most tools just work.",
           "The JSON response, the latency, and the trace logged for every request.",
           "Server logs in artifacts/services/ and a trace per request (Traces & Monitoring).")

# ---------------------------------------------------------------- 1 server
ui.step(1, "Start the API server", f"Runs `uvicorn lab.api:app` on port {services.API_PORT} in the background.")
up = services.api_running()
c1, c2, c3 = st.columns([2, 1, 1])
c1.markdown(f"Status: {'🟢 running' if up else '⚪ stopped'} · <{API}/docs> (interactive docs)")
if c2.button("▶️ Start", disabled=up, type="primary"):
    services.start_api()
    for _ in range(20):
        time.sleep(0.5)
        if services.api_running():
            break
    st.rerun()
if c3.button("⏹ Stop", disabled=not up):
    services.stop("api")
    time.sleep(1)
    st.rerun()
st.caption("The API loads its own copy of the model (separate process). On a small GPU, avoid training while it runs.")
c1, c2 = st.columns(2)
if up:
    c1.link_button("Open the FastAPI docs (Swagger UI) ↗", f"{API}/docs")
with st.expander("🧾 The FastAPI code — `lab/api.py` (this is the whole server)", expanded=False):
    import inspect

    from lab import api as api_module
    st.code(inspect.getsource(api_module), language="python", line_numbers=True)
with st.expander("Server log"):
    st.code(services.tail("api") or "(empty)")

ms = None
if up:
    import httpx
    try:
        ms = [m["id"] for m in httpx.get(f"{API}/v1/models", timeout=5).json()["data"]]
    except Exception as e:  # noqa: BLE001
        st.warning(f"API not answering yet: {e}")

# ---------------------------------------------------------------- 2 try it
ui.step(2, "Try it", "Send a real request. The first call loads the model (a few seconds).")
if not ms:
    st.info("Start the server first.")
else:
    import httpx
    c1, c2 = st.columns(2)
    model = c1.selectbox("Model id", ms, index=len(ms) - 1)
    ep = c2.radio("Endpoint", ["/v1/chat/completions", "/rag/ask", "/agent/run"], horizontal=True)
    text = st.text_input("Message", "Can I promise a customer delivery of a ladder tomorrow?")
    body = ({"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 128} if ep.startswith("/v1")
            else {"model": model, "question": text} if ep == "/rag/ask" else {"model": model, "message": text})
    if st.button("📨 Send", type="primary"):
        t0 = time.time()
        try:
            resp = httpx.post(API + ep, json=body, timeout=300)
            st.session_state["api_last"] = (resp.status_code, resp.json(), round(time.time() - t0, 2))
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
    if st.session_state.get("api_last"):
        code, js, sec = st.session_state["api_last"]
        st.caption(f"HTTP {code} · {sec} s")
        st.json(js, expanded=2)
    st.markdown("Same request from a terminal:")
    st.code(f"curl -s {API}{ep} -H 'Content-Type: application/json' -d '{json.dumps(body)}'", language="bash")

# ---------------------------------------------------------------- 3 LangChain
ui.step(3, "Call it from LangChain", "Your server is OpenAI-compatible, so LangChain's ChatOpenAI can use it with a different base_url.")
code = f'''from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(base_url="{API}/v1", api_key="not-needed", model="{(ms or ['sft-001'])[-1]}", max_tokens=128)
chain = ChatPromptTemplate.from_template("Answer for store staff: {{question}}") | llm
print(chain.invoke({{"question": "Can I give a 20% discount to an unhappy customer?"}}).content)'''
st.code(code, language="python")
if st.button("▶️ Run this snippet here", disabled=not ms):
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(base_url=f"{API}/v1", api_key="not-needed", model=ms[-1], max_tokens=128)
        out = (ChatPromptTemplate.from_template("Answer for store staff: {question}") | llm).invoke(
            {"question": "Can I give a 20% discount to an unhappy customer?"}).content
        st.success(out)
        from lab import registry
        registry.set_progress("m9-lc", True)
    except Exception as e:  # noqa: BLE001
        st.error(f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- 4 Flowise
ui.step(4, "Build a visual flow in Flowise", f"Flowise is a no-code builder for LLM apps. It runs locally on port {services.FLOWISE_PORT}.")
dock, run = services.docker_ok(), services.flowise_running()
c = st.columns(3)
c[0].metric("Docker", "ready" if dock else "missing")
c[1].metric("Flowise image", services.FLOWISE_IMAGE.split("/")[-1] if dock and services.flowise_image_present()
            else "will download (~1 GB)")
c[2].metric("Flowise", "🟢 running" if run else "⚪ stopped")
if not dock:
    st.error("Flowise runs as a Docker container here. Install/start Docker, then reload this page.")
else:
    b1, b2 = st.columns(2)
    if b1.button("▶️ Start Flowise", disabled=run, type="primary"):
        with st.spinner("Starting the Flowise container (first start takes ~30 s)…"):
            res = services.start_flowise()
            for _ in range(90):
                if services.flowise_running():
                    break
                time.sleep(1)
        if str(res).startswith("error"):
            st.error(res)
        else:
            st.rerun()
    if b2.button("⏹ Stop Flowise", disabled=not run):
        services.stop_flowise()
        st.rerun()
    if run:
        st.markdown(f"Open Flowise: **<http://localhost:{services.FLOWISE_PORT}>** (first visit: create a local admin account)")
    with st.expander("Flowise log"):
        st.code(services.flowise_log() or "(empty)")
st.markdown(f"""**Exercise — a chatflow powered by your own model**
1. Start the API server (step 1) and Flowise.
2. In Flowise: **Chatflows → Add New**.
3. Add **Chat Models → ChatOpenAI Custom**. Set *Model Name* = `{(ms or ['sft-001'])[-1]}` and *BasePath*
   (under Additional Parameters) = `{services.FLOWISE_API_BASE}` — Flowise runs in Docker, so it reaches your
   API through `host.docker.internal`. Create any credential as the API key (it is not checked).
4. Add **Chains → Conversation Chain** and connect the chat model to it.
5. Save, click the 💬 chat button and ask a store question. The request shows up in **Traces & Monitoring** (source `api`).
6. Bonus: the Conversation Chain asks for a **Buffer Memory** — add it and test a multi-turn conversation.""")
