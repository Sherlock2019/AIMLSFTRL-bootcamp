import pandas as pd
import streamlit as st

import ui_common as ui
from lab import rag, registry

ui.use_ui_engine()
st.title("📚 RAG Lab")
ui.explain("Retrieval-augmented generation: find the right document passages, then let the model answer from them.",
           "Documents (the bundled store policies, or your own .md/.txt/.pdf) and questions.",
           "It reads the retrieved passages and answers with citations like [1] (a LangChain chain: prompt | model | parser).",
           "Whether the right passages were retrieved (recall@k, MRR, nDCG) and whether answers cover the facts.",
           "A knowledge base in artifacts/kb/<name>/ and a trace for every question (Traces & Monitoring).")

# ---------------------------------------------------------------- 1 build
ui.step(1, "Build a knowledge base", "Split documents into chunks and index them. Smaller chunks = more precise, less context.")
kbs = rag.list_kbs()
with st.expander("➕ Build / rebuild", expanded=not kbs):
    src = st.radio("Documents", ["sample", "upload"], horizontal=True,
                   format_func=lambda s: "Bundled store policies (8 docs, 1 contains a prompt injection)" if s == "sample"
                   else "Upload my own files")
    files = st.file_uploader("Files", type=["md", "txt", "pdf"], accept_multiple_files=True) if src == "upload" else []
    c1, c2, c3, c4 = st.columns(4)
    name = c1.text_input("Name", "tool_store" if src == "sample" else "my_docs")
    size = c2.number_input("Chunk size (chars)", 100, 2000, 400, 50)
    overlap = c3.number_input("Overlap (chars)", 0, 500, 80, 20)
    emb = c4.checkbox("Embeddings (bge-small)", True, help="Enables meaning-based and hybrid search. Runs on CPU.")
    if st.button("🔨 Build", type="primary", disabled=(src == "upload" and not files)):
        with st.spinner("Chunking and indexing…"):
            if src == "sample":
                idx = rag.build_sample(name, size=size, overlap=overlap, embed=emb)
            else:
                idx = rag.build(name, {f.name: rag.read_upload(f.name, f.getvalue()) for f in files}, size, overlap, emb)
        st.session_state["rag_msg"] = f"Built **{name}**: {len(idx['chunks'])} chunks" + \
            ("" if idx.get("embed") or not emb else f" (embeddings failed: {idx.get('embed_error')})")
        st.rerun()
if m := st.session_state.pop("rag_msg", None):
    st.success(m)
kbs = rag.list_kbs()
if not kbs:
    st.stop()
kb = st.selectbox("Knowledge base", kbs)
idx = rag.load_index(kb)
st.caption(f"{len(idx['chunks'])} chunks from {len({c['doc'] for c in idx['chunks']})} documents · chunk {idx['size']} / "
           f"overlap {idx['overlap']} · embeddings {'on' if idx.get('embed') else 'off'}")
with st.expander("See the chunks"):
    st.dataframe(pd.DataFrame(idx["chunks"]), hide_index=True, width="stretch")
methods = [m for m in rag.METHODS if idx.get("embed") or m == "bm25"]

# ---------------------------------------------------------------- 2 ask
ui.step(2, "Ask a question", "Retrieve the top-k passages, then the model answers only from them.")
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    mid = ui.model_picker("Answering model", "rag_model", default_kind="base")
method = c2.selectbox("Retrieval", methods, format_func=lambda m: m.upper(), help="\n".join(f"**{k}**: {v}" for k, v in rag.METHODS.items()))
k = c3.slider("Top-k passages", 1, 6, 3)
q = st.text_input("Question", "How long does a customer have to return an opened box of screws?")
if st.button("💬 Ask", type="primary", disabled=not (mid and q.strip())):
    with st.spinner("Retrieving and generating…"):
        st.session_state["rag_last"] = rag.answer(mid, kb, q, k, method)
r = st.session_state.get("rag_last")
if r:
    st.markdown(f"<div class='card'><b>Answer</b> <small>({r['latency']} s)</small><br>{r['answer']}</div>", unsafe_allow_html=True)
    for i, h in enumerate(r["hits"]):
        st.markdown(f"**[{i + 1}]** `{h['doc']}` · score {h['score']}")
        st.caption(h["text"])

# ---------------------------------------------------------------- 3 measure retrieval
ui.step(3, "Measure retrieval", "Search-engine metrics on labelled questions: did the right document come back, and how high?")
items = rag.sample_eval()
st.caption(f"{len(items)} labelled questions (question → relevant documents) for the bundled policies"
           + ("" if kb == "tool_store" else " — they only make sense for the sample knowledge base."))
kk = st.slider("k for metrics", 1, 5, 3, key="rag_kk")
with st.expander("What do the metrics mean?"):
    st.markdown("- **Recall@k** — share of relevant documents found in the top k\n- **Precision@k** — share of the top k "
                "that are relevant\n- **MRR** — 1 / rank of the first relevant document (1.0 = always first)\n"
                "- **nDCG@k** — rewards relevant documents near the top")
if st.button("📏 Compare retrieval methods"):
    table, per = [], {}
    for m in methods:
        rows, mean = rag.eval_retrieval(kb, items, kk, m)
        per[m] = rows
        table.append({"Method": m.upper(), **{f"{x}@{kk}" if x in ('recall', 'precision', 'ndcg') else x.upper(): round(v, 3)
                                              for x, v in mean.items()}})
    st.session_state["rag_ir"] = (table, per)
    registry.set_progress("m10-ir", True)
if st.session_state.get("rag_ir"):
    table, per = st.session_state["rag_ir"]
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    pick = st.selectbox("Per-question detail for", list(per), format_func=str.upper)
    st.dataframe(pd.DataFrame(per[pick])[["id", "question", "retrieved", "relevant", "recall", "mrr"]], hide_index=True,
                 width="stretch")

# ---------------------------------------------------------------- 4 measure answers
ui.step(4, "Measure answers (end to end)", "Run every labelled question through RAG: are key facts covered, cited, and is the injection ignored?")
if st.button("🧪 Run end-to-end check", disabled=not mid):
    out, bar = [], st.progress(0.0)
    for i, it in enumerate(items):
        a = rag.answer(mid, kb, it["question"], k, method, source="rag eval")
        out.append({"id": it["id"], "question": it["question"], "answer": a["answer"], **rag.answer_checks(a["answer"], it)})
        bar.progress((i + 1) / len(items), text=f"{i + 1}/{len(items)}")
    st.session_state["rag_e2e"] = out
if st.session_state.get("rag_e2e"):
    out = st.session_state["rag_e2e"]
    c = st.columns(3)
    c[0].metric("Fact coverage", f"{sum(o['covers'] for o in out) / len(out):.0%}")
    c[1].metric("Answers with a citation", f"{sum(o['cited'] for o in out)}/{len(out)}")
    c[2].metric("Fell for the injection", sum(o["unsafe"] for o in out))
    st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch")

# ---------------------------------------------------------------- 5 show in Flowise
ui.step(5, "Show in Flowise", "See this RAG pipeline as a real Flowise chatflow (opens Flowise in a new window).")
from lab import flowise_flows, services  # noqa: E402

api_up, fl_up, key = services.api_running(), services.flowise_running(), flowise_flows.get_key()
c = st.columns(3)
c[0].markdown(f"{'🟢' if api_up else '⚪'} Lab API (:{services.API_PORT})")
c[1].markdown(f"{'🟢' if fl_up else '⚪'} Flowise (:{services.FLOWISE_PORT})")
c[2].markdown(f"{'🟢' if key else '⚪'} Flowise API key")
b1, b2 = st.columns(2)
if not api_up and b1.button("▶️ Start lab API"):
    services.start_api()
    import time
    for _ in range(30):
        if services.api_running():
            break
        time.sleep(0.5)
    st.rerun()
if not fl_up and b2.button("▶️ Start Flowise (Docker)"):
    with st.spinner("Starting Flowise… (first start ~30 s)"):
        res = services.start_flowise()
    st.error(res) if str(res).startswith("error") else st.rerun()
if fl_up and not key:
    st.markdown(f"One-time: open **<http://localhost:{services.FLOWISE_PORT}>**, create the local admin account, then "
                "**Settings → API Keys → Create** and paste the key here (stored only in your lab folder).")
    k = st.text_input("Flowise API key", type="password")
    if st.button("Save key", disabled=not k):
        ok, msg = flowise_flows.check_key(k)
        if ok:
            flowise_flows.set_key(k)
            st.rerun()
        st.error(f"Key rejected: {msg}")
if api_up and fl_up and key:
    st.caption(f"Creates **{flowise_flows.FLOW_NAME}**: ChatOpenAI Custom (model `rag:{kb}:{mid}` → your lab API) "
               "→ Conversation Chain ← Buffer Memory. Ask it a policy question in Flowise's chat.")
    if st.button("🔗 Create / update the chatflow", type="primary", disabled=not mid):
        try:
            fid, link = flowise_flows.upsert_rag_flow(mid, kb)
            st.session_state["fl_link"] = link
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not create the chatflow: {e}")
    if st.session_state.get("fl_link"):
        st.link_button("Open the RAG chatflow in Flowise ↗", st.session_state["fl_link"], type="primary")

with st.expander("🧩 The LangChain code behind step 2"):
    st.code('''from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from lab.lc import LabChatModel          # any lab model as a LangChain chat model
from lab import rag

prompt = ChatPromptTemplate.from_messages([("system", rag.RAG_SYSTEM),
                                           ("human", "Sources:\\n{context}\\n\\nQuestion: {question}")])
chain = prompt | LabChatModel(model_id="sft-001") | StrOutputParser()
hits = rag.retrieve("tool_store", question, k=3, method="hybrid")
print(chain.invoke({"context": rag.format_context(hits), "question": question}))''', language="python")
