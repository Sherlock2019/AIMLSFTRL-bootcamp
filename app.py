"""Local AI Training Lab - Streamlit entry point. Run: ./start.sh  (or streamlit run app.py)
Pages live in views/ (not pages/) so every URL is routed through this file's st.navigation.

Two sidebar layouts (switch at the bottom of the sidebar, remembered in the URL as ?ui=v1|v2):
  V2 (default): blocks follow the training path — Setup → Supervised → RL → RLHF → Build apps.
  V1: the original grouping (Train / Evaluate / Build apps) + the Training Path page."""
import streamlit as st

st.set_page_config(page_title="AI Training Lab", page_icon="🛰️", layout="wide")

import ui_common  # noqa: E402

ui_common.setup()

P = {  # one Page object per file (titles and icons as before)
    "overview": st.Page("views/overview.py", title="Overview", icon="🛰️", default=True),
    "path": st.Page("views/training_path.py", title="Training Path", icon="🎓"),
    "hw": st.Page("views/settings_hardware.py", title="Settings & Hardware", icon="🖥️"),
    "hub": st.Page("views/model_hub.py", title="Model Hub", icon="🤗"),
    "data": st.Page("views/data_studio.py", title="Data Studio", icon="🗂️"),
    "sft": st.Page("views/sft_training.py", title="SFT Training", icon="🧑‍🏫"),
    "eval": st.Page("views/evaluation.py", title="Evaluation & Scoring", icon="🧪"),
    "wb": st.Page("views/evals_workbench.py", title="Evals Workbench", icon="🔬"),
    "review": st.Page("views/human_review.py", title="Human Review", icon="🧑‍⚖️"),
    "rules": st.Page("views/reward_designer.py", title="Reward Designer", icon="🎯"),
    "rl": st.Page("views/rl_training.py", title="RL / RLHF Training", icon="🔁"),
    "hf": st.Page("views/human_feedback.py", title="Human Feedback", icon="🗳️"),
    "rm": st.Page("views/reward_model.py", title="Reward Model", icon="🏅"),
    "cmp": st.Page("views/compare_models.py", title="Compare Models", icon="⚖️"),
    "api": st.Page("views/api_integrations.py", title="API & Integrations", icon="🔌"),
    "rag": st.Page("views/rag_lab.py", title="RAG Lab", icon="📚"),
    "agent": st.Page("views/agent_lab.py", title="Agent Lab", icon="🤖"),
    "traces": st.Page("views/traces.py", title="Traces & Monitoring", icon="📡"),
    "registry": st.Page("views/model_registry.py", title="Model Registry", icon="🧬"),
}
V2 = {
    "Start": ["overview"],
    "1 · Setup": ["hw", "hub"],
    "2 · Supervised learning (SFT)": ["data", "sft", "eval", "wb", "review"],
    "3 · Reinforcement learning (RL)": ["rules", "rl"],
    "4 · RLHF (human feedback)": ["hf", "rm", "cmp"],
    "5 · Build apps": ["api", "rag", "agent", "traces"],
    "System": ["registry"],
}
V1 = {
    "Start": ["overview", "path"],
    "Train": ["data", "hub", "sft", "rules", "hf", "rm", "rl"],
    "Evaluate": ["eval", "wb", "cmp", "review"],
    "Build apps": ["rag", "agent", "api", "traces"],
    "System": ["registry", "hw"],
}
FILES = {k: p for k, p in {
    "overview": "views/overview.py", "path": "views/training_path.py", "hw": "views/settings_hardware.py",
    "hub": "views/model_hub.py", "data": "views/data_studio.py", "sft": "views/sft_training.py",
    "eval": "views/evaluation.py", "wb": "views/evals_workbench.py", "review": "views/human_review.py",
    "rules": "views/reward_designer.py", "rl": "views/rl_training.py", "hf": "views/human_feedback.py",
    "rm": "views/reward_model.py", "cmp": "views/compare_models.py", "api": "views/api_integrations.py",
    "rag": "views/rag_lab.py", "agent": "views/agent_lab.py", "traces": "views/traces.py",
    "registry": "views/model_registry.py"}.items()}

layout = "v1" if st.query_params.get("ui") == "v1" else "v2"
groups = V2 if layout == "v2" else V1
pg = st.navigation({sec: [P[k] for k in keys] for sec, keys in groups.items()})
ui_common.CURRENT_PAGE["path"] = next((FILES[k] for k, p in P.items() if p.title == pg.title), None)

with st.sidebar:
    st.divider()
    choice = st.radio("Sidebar layout", ["v2", "v1"], index=0 if layout == "v2" else 1, horizontal=True,
                      format_func={"v2": "V2 · course order", "v1": "V1 · classic"}.get)
    if choice != layout:
        st.query_params["ui"] = choice
        st.rerun()

pg.run()
