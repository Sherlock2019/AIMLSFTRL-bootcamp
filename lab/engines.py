"""One shared inference-engine cache for code outside Streamlit (FastAPI server, agents, RAG, judge, scripts).
Streamlit pages use ui_common.engine (st.cache_resource) and pass it in via set_provider()."""
from . import registry

_cache = {}
_provider = None


def set_provider(fn):
    """fn(base_model) -> Engine. Lets the Streamlit UI reuse its cached engine."""
    global _provider
    _provider = fn


def get(base_model):
    if _provider:
        return _provider(base_model)
    if base_model not in _cache:
        from .model_loader import Engine
        _cache.clear()  # keep at most one base model in memory
        _cache[base_model] = Engine(base_model)
    return _cache[base_model]


def chat(model_id, messages, **kw):
    return get(registry.model(model_id)["base_model"]).chat(model_id, messages, **kw)


def ask(model_id, question, **kw):
    return get(registry.model(model_id)["base_model"]).generate(model_id, [question], **kw)[0]
