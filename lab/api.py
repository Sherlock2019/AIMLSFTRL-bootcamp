"""FastAPI server: serve any lab model (base or trained LoRA version) to other apps.

Run:  .venv/bin/uvicorn lab.api:app --port 8811     (or the Start button on "API & Integrations")
- OpenAI-compatible: GET /v1/models, POST /v1/chat/completions  → works with LangChain ChatOpenAI, Flowise, curl, the openai SDK
- POST /rag/ask      retrieval-augmented answer with sources
- POST /agent/run    LangGraph tool-using agent (keeps a conversation per session_id)
Every request is logged as a trace (Traces & Monitoring page).
"""
import json
import time
import uuid
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import agent, engines, rag, registry

app = FastAPI(title="AI Training Lab API", version="1.0",
              description="Serve lab models. OpenAI-compatible chat + RAG + LangGraph agent.")
_sessions = {}


class ChatReq(BaseModel):
    model: str
    messages: List[dict]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.0
    stream: Optional[bool] = False
    tools: Optional[List[dict]] = None
    model_config = {"extra": "allow"}


class RagReq(BaseModel):
    model: str
    question: str
    kb: str = "tool_store"
    k: int = 3
    method: str = "bm25"


class AgentReq(BaseModel):
    model: str
    message: str
    session_id: Optional[str] = None


def parse_rag_model(name):
    """'rag:<kb>:<model_id>' → (kb, model_id) — a RAG pipeline exposed as an OpenAI 'model' (used by Flowise)."""
    if not str(name).startswith("rag:"):
        return None
    _, kb, mid = (name.split(":", 2) + ["", ""])[:3]
    return kb, mid


def _check_model(mid):
    rp = parse_rag_model(mid)
    if rp:
        if rp[0] not in rag.list_kbs() and rp[0] != "tool_store":
            raise HTTPException(404, f"Unknown knowledge base '{rp[0]}'. Available: {rag.list_kbs()}")
        mid = rp[1]
    m = registry.model(mid)
    if not m or m["deleted"] or m["kind"] == "rm":
        raise HTTPException(404, f"Unknown chat model '{mid}'. GET /v1/models lists the ids.")


def _text(content: Any):
    if isinstance(content, list):  # OpenAI content parts
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


@app.get("/health")
def health():
    return {"status": "ok", "models": len(registry.models())}


@app.get("/v1/models")
def list_models():
    chat = [m for m in registry.models() if m["kind"] != "rm"]
    data = [{"id": m["id"], "object": "model", "owned_by": "ai-training-lab", "base_model": m["base_model"],
             "kind": m["kind"]} for m in chat]
    data += [{"id": f"rag:{kb}:{m['id']}", "object": "model", "owned_by": "ai-training-lab", "kind": "rag pipeline"}
             for kb in rag.list_kbs() for m in chat]
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatReq):
    _check_model(req.model)
    t0 = time.time()
    msgs = []
    for m in req.messages:
        d = {"role": m.get("role", "user"), "content": _text(m.get("content"))}
        if m.get("tool_calls"):
            d["tool_calls"] = [{"type": "function", "function": {
                "name": c["function"]["name"],
                "arguments": json.loads(c["function"]["arguments"]) if isinstance(c["function"]["arguments"], str)
                else c["function"]["arguments"]}} for c in m["tool_calls"]]
        msgs.append(d)
    rp = parse_rag_model(req.model)
    if rp:  # RAG pipeline: retrieve from the knowledge base for the last user message, answer with sources
        kb, mid = rp
        if kb not in rag.list_kbs():
            rag.build_sample(embed=False)
        question = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
        r = rag.answer(mid, kb, question, k=3, method="hybrid" if rag.load_index(kb).get("embed") else "bm25",
                       source="api rag (e.g. Flowise)")
        text = r["answer"] + "\n\nSources: " + ", ".join(sorted({h["doc"] for h in r["hits"]}))
    else:
        text = engines.chat(req.model, msgs, tools=req.tools, max_new_tokens=req.max_tokens or 256,
                            temperature=req.temperature or 0.0)
    calls = agent.parse_tool_calls(text) if req.tools and not rp else []
    message = {"role": "assistant", "content": None if calls else text}
    if calls:
        message["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                                  "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]
    finish = "tool_calls" if calls else "stop"
    lat = round(time.time() - t0, 3)
    registry.add_trace("api", req.model, msgs[-1]["content"] if msgs else "", text,
                       [{"step": "chat", "messages": len(msgs), "tools": len(req.tools or []), "seconds": lat}], lat,
                       source="api /v1/chat/completions")
    cid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
    if req.stream:
        def gen():
            delta = {"role": "assistant", **({"tool_calls": [dict(c, index=i) for i, c in enumerate(message["tool_calls"])]}
                                             if calls else {"content": text})}
            for chunk in ({"delta": delta, "finish_reason": None}, {"delta": {}, "finish_reason": finish}):
                yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                             "model": req.model, "choices": [{"index": 0, **chunk}]}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return {"id": cid, "object": "chat.completion", "created": created, "model": req.model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


@app.post("/rag/ask")
def rag_ask(req: RagReq):
    _check_model(req.model)
    if req.kb not in rag.list_kbs():
        if req.kb == "tool_store":
            rag.build_sample(embed=False)
        else:
            raise HTTPException(404, f"Unknown knowledge base '{req.kb}'. Available: {rag.list_kbs()}")
    r = rag.answer(req.model, req.kb, req.question, req.k, req.method, source="api /rag/ask")
    return {"answer": r["answer"], "sources": [{"id": h["id"], "doc": h["doc"], "score": h["score"], "text": h["text"]}
                                               for h in r["hits"]], "latency": r["latency"]}


@app.post("/agent/run")
def agent_run(req: AgentReq):
    _check_model(req.model)
    sid = req.session_id or uuid.uuid4().hex[:8]
    r = agent.run(req.model, req.message, history=_sessions.get(sid, []), session=sid, source="api /agent/run")
    _sessions[sid] = r["messages"]
    return {"session_id": sid, "answer": r["answer"], "steps": r["steps"], "latency": r["latency"]}
