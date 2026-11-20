import streamlit as st

import ui_common as ui
from lab import hardware, jobs, paths, registry, status

st.title("🛰️ AI Training Lab — Mission Control")
st.caption("Supervised training → evaluate → design rewards → reinforcement training → review → correct → again. "
           "Everything runs on this computer.")

p = status.pipeline()
nxt_stage = next((s for s in status.STAGES if not p["done"][s]), None)
chips = []
for s in status.STAGES:
    cls = "ok" if p["done"][s] else ("next" if s == nxt_stage else "todo")
    chips.append(f'<span class="stage {cls}">{"✓ " if p["done"][s] else ""}{s}</span>')
st.markdown('<span class="arrow">→</span>'.join(chips), unsafe_allow_html=True)

page, tip = p["next"]
st.info(f"**Next recommended action** ({page}): {tip}")

c1, c2, c3, c4 = st.columns(4)
base = next((m for m in registry.models() if m["kind"] == "base"), None)
c1.metric("Cycle", p["cycle"])
c2.metric("Base model", base["base_model"].split("/")[-1] if base else "—")
c3.metric("Current SFT model", p["latest_sft"]["id"] if p["latest_sft"] else "—")
c4.metric("Current RL model", p["latest_rl"]["id"] if p["latest_rl"] else "—")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("#### Dataset versions")
    for role in ("sft_train", "sft_val", "rl_train", "rl_val", "test"):
        d = registry.latest_dataset(role)
        st.write(f"`{role}` → **{d['id'] if d else '—'}** ({d['n_rows'] if d else 0} rows)")
with c2:
    st.markdown("#### Reward functions")
    for r in registry.rewards()[:4]:
        st.write(f"**{r['id']}** — {'✅ approved' if r['approved'] else '📝 draft'}")
    st.markdown("#### Latest training run")
    rs = registry.runs()
    if rs:
        r = rs[0]
        st.markdown(f"{ui.tag('real')} **{r['id']}** {r['kind'].upper()} → {r['output_model_id']} · "
                    f"**{r['status']}**", unsafe_allow_html=True)
    else:
        st.write("No runs yet.")
    act = jobs.active_run()
    if act:
        st.warning(f"Training in progress: {act['id']} ({act['kind'].upper()})")
with c3:
    st.markdown("#### Results (kept separate on purpose)")
    if p["latest_sft"]:
        runs = registry.runs("sft")
        last = next((r for r in runs if r["output_model_id"] == p["latest_sft"]["id"]), None)
        if last:
            m = [x for x in jobs.read_jsonl(last["metrics_path"]) if "loss" in x]
            v = [x for x in jobs.read_jsonl(last["metrics_path"]) if "eval_loss" in x]
            st.markdown(f"{ui.tag('train')} last SFT loss: **{m[-1]['loss']:.3f}**" if m else f"{ui.tag('train')} —",
                        unsafe_allow_html=True)
            st.markdown(f"{ui.tag('val')} last SFT val loss: **{v[-1]['eval_loss']:.3f}**" if v else f"{ui.tag('val')} —",
                        unsafe_allow_html=True)
    ev = registry.evals()
    if ev:
        e = ev[0]
        st.markdown(f"{ui.tag('reward')} {e['model_id']} on {e['dataset_id']} with {e['reward_id']}: "
                    f"mean **{e['mean_reward']:.2f}**", unsafe_allow_html=True)
    revs = registry.reviews()
    if revs:
        ok = sum(r["rating"] == "correct" for r in revs)
        st.markdown(f"{ui.tag('human')} {ok}/{len(revs)} answers rated correct (all models)", unsafe_allow_html=True)
    st.caption("Training loss, validation loss, reward scores and human ratings measure different things. "
               "There is no single 'quality' score here on purpose.")

with st.expander("🖥️ Hardware use", expanded=False):
    hw = hardware.check()
    st.write(f"Device **{hw['device']}** · GPU {hw['gpu'] or '—'} · free GPU mem {hw['gpu_mem_free_gb'] or '—'} GB · "
             f"free RAM {hw['ram_available_gb']} GB · free disk {hw['disk_free_gb']} GB")

st.markdown("### 🧭 Guided first run")
steps = [
    ("Check laptop hardware", "Settings & Hardware", bool(st.session_state.get("hw_seen"))),
    ("Download the small Qwen model", "Settings & Hardware", hardware.model_cached(base["base_model"]) if base else False),
    ("Inspect the sample training data", "Data Studio", p["done"]["Data"]),
    ("Run a tiny SFT job", "SFT Training", p["done"]["SFT"]),
    ("Review the model's answers", "Evaluation & Scoring", p["done"]["Evaluation"]),
    ("Test and approve reward rules", "Reward Designer", p["done"]["Reward Design"]),
    ("Run a tiny RL job", "RL Training", p["done"]["RL"]),
    ("Compare SFT and RL answers", "Compare Models", p["done"]["RL"]),
    ("Write a correction", "Human Review", p["done"]["Review"]),
    ("Create a new dataset version and start the next cycle", "Human Review", p["done"]["Next Cycle"]),
]
for i, (s, pg, d) in enumerate(steps, 1):
    st.write(f"{'✅' if d else '⬜'} **{i}. {s}** — *{pg}*")
st.caption(f"Artifacts folder: `{paths.home()}`")
