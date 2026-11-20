"""Flowise chatflow builder, RAG-as-a-model API, Langfuse events, notebooks, Studio graph (no model, no Docker)."""
import importlib
import json
import sys
from pathlib import Path

from lab import engines, flowise_flows, langfuse_stack, notebooks, rag, registry

ROOT = Path(__file__).resolve().parent.parent

FAKE_DEFS = {
    "chatOpenAICustom": {"name": "chatOpenAICustom", "label": "ChatOpenAI Custom", "version": 4, "type": "ChatOpenAI-Custom",
                         "baseClasses": ["ChatOpenAI-Custom", "BaseChatModel"],
                         "credential": {"label": "Connect Credential", "name": "credential", "type": "credential", "optional": True},
                         "inputs": [{"label": "Cache", "name": "cache", "type": "BaseCache", "optional": True},
                                    {"label": "Model Name", "name": "modelName", "type": "string"},
                                    {"label": "BasePath", "name": "basepath", "type": "string", "optional": True}]},
    "bufferMemory": {"name": "bufferMemory", "label": "Buffer Memory", "type": "BufferMemory",
                     "baseClasses": ["BufferMemory", "BaseChatMemory"],
                     "inputs": [{"label": "Memory Key", "name": "memoryKey", "type": "string", "default": "chat_history"}]},
    "conversationChain": {"name": "conversationChain", "label": "Conversation Chain", "type": "ConversationChain",
                          "baseClasses": ["ConversationChain", "BaseChain"],
                          "inputs": [{"label": "Chat Model", "name": "model", "type": "BaseChatModel"},
                                     {"label": "Memory", "name": "memory", "type": "BaseMemory"},
                                     {"label": "System Message", "name": "systemMessagePrompt", "type": "string"}]},
}


def test_flowise_flow_wiring():
    flow = flowise_flows.build_flow(FAKE_DEFS, "rag:tool_store:sft-001", "http://host.docker.internal:8811/v1")
    llm, mem, chain = flow["nodes"]
    assert llm["data"]["inputs"]["modelName"] == "rag:tool_store:sft-001"
    assert llm["data"]["inputs"]["basepath"].endswith(":8811/v1")
    assert [a["name"] for a in chain["data"]["inputAnchors"]] == ["model", "memory"]
    assert chain["data"]["inputs"]["model"] == "{{chatOpenAICustom_0.data.instance}}"
    assert {e["targetHandle"] for e in flow["edges"]} == {"conversationChain_0-input-model-BaseChatModel",
                                                         "conversationChain_0-input-memory-BaseMemory"}
    assert flow["edges"][0]["sourceHandle"] == "chatOpenAICustom_0-output-chatOpenAICustom-ChatOpenAI-Custom|BaseChatModel"
    json.dumps(flow)  # serialisable for the Flowise API


def test_api_serves_rag_pipeline_as_a_model(lab_home, monkeypatch):
    from fastapi.testclient import TestClient

    from lab import data
    from lab.api import app
    data.bootstrap_samples()
    registry.ensure_base("Qwen/Qwen3-0.6B")
    rag.build_sample(embed=False)
    monkeypatch.setattr(engines, "chat", lambda mid, msgs, **kw: "Customers have 30 days [1].")
    c = TestClient(app)
    ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
    assert "rag:tool_store:base-001" in ids
    r = c.post("/v1/chat/completions", json={"model": "rag:tool_store:base-001",
                                             "messages": [{"role": "user", "content": "Return window?"}]}).json()
    text = r["choices"][0]["message"]["content"]
    assert "30 days" in text and "Sources:" in text and "returns_policy.md" in text
    assert c.post("/v1/chat/completions", json={"model": "rag:nope:base-001", "messages": []}).status_code == 404


def test_langfuse_events_shape():
    t = {"id": 7, "kind": "agent", "source": "agent lab", "model_id": "sft-001", "session": "s1", "input": "q",
         "output": "a", "latency": 1.2, "created": 1_800_000_000, "meta": {"history_len": 0},
         "steps": [{"step": "llm", "output": "x", "tool_calls": []},
                   {"step": "tool", "name": "calculator", "arguments": {"expression": "1+1"}, "result": {"result": 2}}]}
    ev = langfuse_stack.to_events(t)
    assert [e["type"] for e in ev] == ["trace-create", "generation-create", "span-create"]
    assert ev[0]["body"]["id"] == "lab-trace-7" and all(e["body"].get("traceId", "lab-trace-7") == "lab-trace-7" for e in ev)
    assert langfuse_stack.enabled() in (True, False)  # never raises when the stack is down


def test_every_notebook_exists_and_matches_generator():
    sys.path.insert(0, str(ROOT / "scripts"))
    gen = importlib.import_module("make_notebooks")
    assert gen.PAGE_TO_NOTEBOOK == notebooks.BY_PAGE
    for page, f in notebooks.BY_PAGE.items():
        assert (ROOT / page).exists() and (ROOT / "notebooks" / f).exists(), f
        nb = json.loads((ROOT / "notebooks" / f).read_text())
        assert sum(c["cell_type"] == "code" for c in nb["cells"]) >= 2


def test_studio_graph_compiles(lab_home):
    mod = importlib.import_module("lab.studio_graph")
    assert set(mod.graph.get_graph().nodes) >= {"agent", "tools"}
    assert json.loads((ROOT / "langgraph.json").read_text())["graphs"]["store_agent"].endswith("studio_graph.py:graph")
