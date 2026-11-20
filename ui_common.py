"""Shared dashboard helpers: theme, explanations, model picker, cached inference engine."""
import streamlit as st

from lab import data, paths, registry, rewards

CSS = """
<style>
:root { --ok:#22c55e; --todo:#64748b; --accent:#38bdf8; --warn:#f59e0b; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }
.stage { display:inline-block; padding:10px 14px; margin:4px 2px; border-radius:12px; font-weight:600;
         border:1px solid rgba(148,163,184,.35); font-size:.9rem; }
.stage.ok { background:rgba(34,197,94,.15); border-color:var(--ok); color:var(--ok); }
.stage.todo { background:rgba(100,116,139,.12); color:#94a3b8; }
.stage.next { background:rgba(56,189,248,.15); border-color:var(--accent); color:var(--accent);
              box-shadow:0 0 12px rgba(56,189,248,.45); }
.arrow { color:#64748b; margin:0 2px; }
.tag { display:inline-block; padding:2px 8px; border-radius:999px; font-size:.75rem; font-weight:700; margin-right:6px; }
.tag.real { background:rgba(34,197,94,.18); color:#22c55e; }
.tag.train { background:rgba(56,189,248,.18); color:#38bdf8; }
.tag.val { background:rgba(168,85,247,.18); color:#c084fc; }
.tag.reward { background:rgba(245,158,11,.18); color:#f59e0b; }
.tag.human { background:rgba(236,72,153,.18); color:#f472b6; }
.card { border:1px solid rgba(148,163,184,.25); border-radius:14px; padding:14px 16px; background:rgba(15,23,42,.35); }
</style>
"""

TAGS = {"real": "REAL TRAINING", "train": "TRAINING METRIC", "val": "VALIDATION METRIC",
        "reward": "REWARD SCORE", "human": "HUMAN REVIEW"}


def tag(kind):
    return f'<span class="tag {kind}">{TAGS[kind]}</span>'


def setup():
    st.markdown(CSS, unsafe_allow_html=True)
    if "_boot" not in st.session_state:
        data.bootstrap_samples()
        registry.ensure_base(paths.DEFAULT_BASE_MODEL)
        if not registry.rewards():
            registry.add_reward(rewards.DEFAULT_RULES, note="Default draft rules (not yet approved)")
        st.session_state["_boot"] = True


CURRENT_PAGE = {"path": None}  # set by app.py before the page runs


def explain(what, provide, model_does, look_at, saved):
    with st.expander("ℹ️ About this step", expanded=False):
        st.markdown(f"**What is this step?** {what}\n\n**What do I provide?** {provide}\n\n"
                    f"**What does the model do?** {model_does}\n\n**What result should I look at?** {look_at}\n\n"
                    f"**What gets saved?** {saved}")
    course_box(CURRENT_PAGE["path"])


def course_box(page):
    """The training-path steps that belong to this page + its notebook, inside the page."""
    if not page:
        return
    from lab import course, notebooks
    mods, _, _ = course.status()
    mine = [(m, [r for r in rows if r[2] == page]) for m, rows in mods]
    mine = [(m, rows) for m, rows in mine if rows]
    nbp = notebooks.path_for(page)
    if not mine and not nbp:
        return
    n_ok = sum(r[4] for _, rows in mine for r in rows)
    n = sum(len(rows) for _, rows in mine)
    label = f"🎓 Course steps on this page · {n_ok}/{n} done" if n else "🎓 Course steps on this page"
    with st.expander(label, expanded=bool(n) and n_ok < n):
        for m, rows in mine:
            st.markdown(f"**{course.TRACKS[m['track']][0].split(' · ')[0]} · {m['title']}** — _{m['concept']}_")
            for tid, title, _, auto, ok in rows:
                if auto:
                    st.markdown(f"{'✅' if ok else '⬜'} {title}")
                else:
                    v = st.checkbox(f"{title} ✋", value=ok, key=f"cb_{tid}", help="Tick this yourself when done.")
                    if v != ok:
                        registry.set_progress(tid, v)
                        st.rerun()
        if nbp:
            st.markdown(f"📓 **Do it in code:** `notebooks/{nbp.name}` — the same step in Python, prefilled.")
            c1, c2 = st.columns(2)
            if notebooks.running():
                c1.link_button("Open in JupyterLab ↗", notebooks.url_for(page))
            elif c1.button("▶️ Start JupyterLab", key=f"jl_{page}"):
                import time
                notebooks.start()
                for _ in range(40):
                    if notebooks.running():
                        break
                    time.sleep(0.5)
                st.rerun()
            c2.download_button("⬇️ Download .ipynb", nbp.read_bytes(), file_name=nbp.name, key=f"dl_{page}")


def model_label(m):
    if m["kind"] == "base":
        return f"{m['id']} · base · {m['base_model']}"
    extra = f" · {m['reward_id']}" if m.get("reward_id") else ""
    star = " ★" if m.get("preferred") else ""
    return f"{m['id']} · {m['kind'].upper()} ← {m['parent_id']} · {m['train_ds']}{extra}{star}"


def model_picker(label, key, kinds=None, default_kind=None, help=None):
    # reward models ("rm") score answers but cannot write them: only listed when asked for explicitly
    ms = [m for m in registry.models() if (m["kind"] in kinds if kinds else m["kind"] != "rm")]
    if not ms:
        st.info("No models yet.")
        return None
    idx = len(ms) - 1
    if default_kind:
        cand = [i for i, m in enumerate(ms) if m["kind"] == default_kind]
        idx = cand[-1] if cand else idx
    choice = st.selectbox(label, ms, index=idx, format_func=model_label, key=key, help=help)
    return choice["id"]


def dataset_picker(label, role, key, allow_none=False):
    ds = registry.datasets(role)
    opts = ([None] if allow_none else []) + [d["id"] for d in ds]
    if not opts:
        return None
    fmt = lambda d: "— none —" if d is None else f"{d} ({registry.dataset(d)['n_rows']} rows, {registry.dataset(d)['source']})"
    # default = latest version (never "none", so validation is used unless the user opts out)
    return st.selectbox(label, opts, index=1 if allow_none and len(opts) > 1 else 0, key=key, format_func=fmt)


@st.cache_resource(show_spinner="Loading model into memory…", max_entries=1)
def engine(base_model):
    from lab.model_loader import Engine
    return Engine(base_model)


def use_ui_engine():
    """Make lab code (RAG, agent, judge) reuse the dashboard's cached engine instead of loading its own."""
    from lab import engines
    engines.set_provider(engine)


def step(n, title, help_text):
    """Numbered step header with a one-line explanation."""
    st.markdown(f"**Step {n} · {title}**")
    st.caption(help_text)


def unload_engine():
    engine.clear()
    try:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def generate(model_id, questions, **kw):
    m = registry.model(model_id)
    return engine(m["base_model"]).generate(model_id, questions, **kw)
