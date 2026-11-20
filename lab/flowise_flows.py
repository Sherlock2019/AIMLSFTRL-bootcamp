"""Create a real Flowise chatflow that runs the lab's RAG pipeline, so it can be opened in the Flowise UI.

Flow:  ChatOpenAI Custom (model = "rag:<kb>:<model_id>", BasePath = lab API)  →  Conversation Chain  ←  Buffer Memory
The lab API answers that model name with retrieval + generation + sources (see lab/api.py).

Node definitions are fetched from YOUR Flowise (GET /api/v1/nodes/<name>), so the chatflow matches its version.
Flowise 3 needs an API key for its REST API: create one in Flowise (Settings → API Keys) and paste it in the lab.
"""
import copy
import json

from . import paths, services

FLOW_NAME = "AI Training Lab · RAG"
NODES = ("chatOpenAICustom", "bufferMemory", "conversationChain")


def base_url():
    return f"http://localhost:{services.FLOWISE_PORT}"


def key_file():
    return paths.home() / "flowise_api_key"


def get_key():
    f = key_file()
    return f.read_text().strip() if f.exists() else ""


def set_key(k):
    key_file().write_text(k.strip())
    try:
        key_file().chmod(0o600)
    except OSError:
        pass


def _call(method, path, body=None, key=None):
    import httpx
    r = httpx.request(method, base_url() + path, json=body, timeout=30,
                      headers={"Authorization": f"Bearer {key or get_key()}", "Content-Type": "application/json"})
    if r.status_code >= 400:
        raise RuntimeError(f"Flowise {method} {path} → HTTP {r.status_code}: {r.text[:200]}")
    return r.json() if r.content else None


def check_key(key=None):
    try:
        _call("GET", "/api/v1/chatflows", key=key)
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def make_node(defn, node_id, inputs, x, y):
    """Build a canvas node from a Flowise node definition (same shape the Flowise UI saves)."""
    d = copy.deepcopy(defn)
    d["id"] = node_id
    params = d.get("inputs") if isinstance(d.get("inputs"), list) else d.get("inputParams", [])
    anchors, plain = [], []
    for p in params or []:
        p = dict(p, id=f"{node_id}-input-{p['name']}-{p['type']}")
        (plain if p["type"] in ("string", "number", "boolean", "password", "json", "code", "options",
                                "multiOptions", "asyncOptions", "credential", "file", "datagrid") else anchors).append(p)
    if d.get("credential"):
        plain.insert(0, dict(d["credential"], id=f"{node_id}-input-credential-credential"))
    d["inputParams"], d["inputAnchors"] = plain, anchors
    d["inputs"] = {p["name"]: inputs.get(p["name"], p.get("default", "")) for p in plain + anchors}
    d["outputAnchors"] = [{"id": f"{node_id}-output-{d['name']}-{'|'.join(d['baseClasses'])}", "name": d["name"],
                           "label": d.get("label", d["name"]), "type": " | ".join(d["baseClasses"])}]
    d["outputs"], d["selected"] = {}, False
    return {"id": node_id, "position": {"x": x, "y": y}, "type": "customNode", "data": d, "width": 300, "height": 600,
            "selected": False, "positionAbsolute": {"x": x, "y": y}, "dragging": False}


def edge(src, tgt, input_name):
    s = src["data"]["outputAnchors"][0]["id"]
    t = next(a["id"] for a in tgt["data"]["inputAnchors"] if a["name"] == input_name)
    return {"source": src["id"], "sourceHandle": s, "target": tgt["id"], "targetHandle": t, "type": "buttonedge",
            "id": f"{src['id']}-{s}-{tgt['id']}-{t}"}


def build_flow(defs, model_name, api_base=None):
    """defs = {node name: definition}. Returns the flowData dict."""
    api_base = api_base or services.FLOWISE_API_BASE
    llm = make_node(defs["chatOpenAICustom"], "chatOpenAICustom_0",
                    {"modelName": model_name, "basepath": api_base, "temperature": 0, "streaming": False,
                     "maxTokens": 256, "timeout": 300000}, 80, 80)
    mem = make_node(defs["bufferMemory"], "bufferMemory_0", {"memoryKey": "chat_history"}, 80, 760)
    chain = make_node(defs["conversationChain"], "conversationChain_0",
                      {"model": "{{chatOpenAICustom_0.data.instance}}", "memory": "{{bufferMemory_0.data.instance}}",
                       "systemMessagePrompt": "You are the AI Training Lab RAG assistant."}, 520, 300)
    return {"nodes": [llm, mem, chain], "edges": [edge(llm, chain, "model"), edge(mem, chain, "memory")],
            "viewport": {"x": 0, "y": 0, "zoom": 0.8}}


def upsert_rag_flow(model_id, kb="tool_store"):
    """Create or update the chatflow; returns (chatflow id, canvas URL)."""
    defs = {n: _call("GET", f"/api/v1/nodes/{n}") for n in NODES}
    flow = build_flow(defs, f"rag:{kb}:{model_id}")
    body = {"name": FLOW_NAME, "flowData": json.dumps(flow), "deployed": False, "isPublic": False, "type": "CHATFLOW"}
    rows = {r["name"]: r for r in (_call("GET", "/api/v1/chatflows") or [])}
    row = _call("PUT", f"/api/v1/chatflows/{rows[FLOW_NAME]['id']}", body) if FLOW_NAME in rows \
        else _call("POST", "/api/v1/chatflows", body)
    return row["id"], f"{base_url()}/canvas/{row['id']}"


def ask(flow_id, question):
    return _call("POST", f"/api/v1/prediction/{flow_id}", {"question": question})
