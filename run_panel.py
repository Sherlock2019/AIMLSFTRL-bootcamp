"""Live run panel shared by the SFT and RL pages (refreshes itself every few seconds while a run is active)."""
import pandas as pd
import plotly.express as px
import streamlit as st

import ui_common as ui
from lab import jobs, registry


def chart(df, ys, title, kind):
    ys = [y for y in ys if y in df.columns and df[y].notna().any()]
    if not ys:
        st.caption(f"{title}: no data yet")
        return
    long = df[["step"] + ys].melt("step").dropna()
    fig = px.line(long, x="step", y="value", color="variable", markers=True, title=title, template="plotly_dark")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), legend_title=None)
    st.markdown(ui.tag(kind), unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)


def show(kind, extra=None):
    """extra(run) renders more run details inside the auto-refreshing panel."""
    kinds = kind if isinstance(kind, tuple) else (kind,)
    runs = [r for r in registry.runs() if r["kind"] in kinds]
    if not runs:
        st.info("No runs yet.")
        return None
    rid = st.selectbox("Run", [r["id"] for r in runs], key=f"runsel_{'_'.join(kinds)}",
                       format_func=lambda i: f"{i} · {registry.run(i)['kind'].upper()} → {registry.run(i)['output_model_id']}")
    jobs.active_run()  # marks crashed processes as failed
    if registry.run(rid)["status"] in ("queued", "running"):
        _live(rid, kind, extra)
    else:
        _render(rid, kind, extra)  # finished runs render once, so charts stay zoomable
    return rid


@st.fragment(run_every=3)
def _live(rid, kind, extra=None):
    jobs.active_run()
    if registry.run(rid)["status"] not in ("queued", "running"):
        st.rerun()  # run just finished: switch to the static view (and refresh the run list)
    _render(rid, kind, extra)


def _render(rid, kind, extra=None):
    r = registry.run(rid)
    m = jobs.read_jsonl(r["metrics_path"])
    steps = [x for x in m if not x.get("final")]
    last = steps[-1] if steps else {}
    lr = next((x["learning_rate"] for x in reversed(steps) if x.get("learning_rate")), None)
    c = st.columns(6)
    c[0].markdown(f"{ui.tag('real')}<br>**{r['status'].upper()}**", unsafe_allow_html=True)
    c[1].metric("Step", f"{last.get('step', 0)} / {last.get('max_steps', '?')}")
    c[2].metric("Elapsed", f"{last.get('elapsed_s', 0):.0f} s")
    c[3].metric("Learning rate", f"{lr:.1e}" if lr else "—")
    c[4].metric("GPU peak", f"{last.get('gpu_mem_gb') or '—'} GB")
    c[5].metric("Process RAM", f"{last.get('ram_used_gb') or '—'} GB")
    st.caption(f"{r['parent_model_id']} → **{r['output_model_id']}** · train {r['train_ds']} · val {r['val_ds'] or '—'}"
               + (f" · reward {r['reward_id']}" if r["reward_id"] else "")
               + (f" · reward model {r['config']['rm_id']}" if r["config"].get("rm_id") else ""))
    if r["status"] in ("queued", "running"):
        st.progress(min(1.0, (last.get("step", 0) / last["max_steps"]) if last.get("max_steps") else 0.0))
        if st.button("⏹ Cancel run", key=f"cancel_{rid}"):
            jobs.cancel(rid)
            st.toast("Cancel requested — stops after the current step.")
    if r["error"]:
        st.error(r["error"])
    df = pd.DataFrame(m) if m else pd.DataFrame({"step": []})
    if r["kind"] in ("sft", "rm"):
        a, b = st.columns(2)
        with a:
            chart(df[df.get("loss").notna()] if "loss" in df else df, ["loss"], "Training loss", "train")
        with b:
            chart(df[df.get("eval_loss").notna()] if "eval_loss" in df else df, ["eval_loss"], "Validation loss", "val")
    else:
        a, b = st.columns(2)
        with a:
            chart(df, ["reward"], "Average reward per step (training prompts)", "reward")
        with b:
            chart(df, ["reward_std", "completions/mean_length"], "Reward spread & answer length", "train")
    with st.expander("📜 Log", expanded=r["status"] == "failed"):
        st.code(jobs.tail(r["log_path"], 40) or "(empty)", language="text")
    if extra:
        extra(r)
