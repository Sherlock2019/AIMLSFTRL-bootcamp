"""The Agent Lab's LangGraph graph, exported for LangGraph Studio (`langgraph dev`, see langgraph.json).

Studio shows the real graph (agent ⇄ tools), lets you send a message and step through every node.
The model is the newest trained chat model (or the base model); override with LAB_STUDIO_MODEL=<model id>.
Input in Studio:  {"messages": [{"role": "user", "content": "Do we have drills at the warehouse?"}], "steps": [], "turns": 0}
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from lab import agent, data, paths, registry  # noqa: E402


def _model():
    if os.environ.get("LAB_STUDIO_MODEL"):
        return os.environ["LAB_STUDIO_MODEL"]
    data.bootstrap_samples()
    base = registry.ensure_base(paths.DEFAULT_BASE_MODEL)
    for kind in ("rlhf", "rl", "sft"):
        ms = [m for m in registry.models() if m["kind"] == kind]
        if ms:
            return ms[-1]["id"]
    return base


graph = agent.build_graph(_model())
