import streamlit as st

from lab import course, registry

st.title("🎓 Training Path")
st.caption("Build three complete workflows on the real lab: **A · supervised learning**, **B · reinforcement learning**, "
           "then **C · RLHF** (both combined with human feedback). ✅ ticks itself when the lab sees you did it; "
           "tasks marked ✋ you tick yourself.")

mods, done, total = course.status()
st.progress(done / max(1, total), text=f"{done} of {total} tasks done")

nxt = next(((m, r) for m, rows in mods for r in rows if not r[4]), None)
if nxt:
    m, (tid, title, page, auto, ok) = nxt
    c1, c2 = st.columns([4, 1])
    c1.info(f"**Next:** {m['title']} → {title}")
    c2.page_link(page, label="Go ▶", icon="🧭")
else:
    st.success("🎉 All modules complete.")

track = None
for num, (m, rows) in enumerate(mods, 1):
    if m.get("track") != track:
        track = m.get("track")
        title, idea = course.TRACKS[track]
        t_rows = [r for mm, rr in mods if mm.get("track") == track for r in rr]
        st.markdown(f"#### {title}  <small>· {sum(r[4] for r in t_rows)}/{len(t_rows)}</small>", unsafe_allow_html=True)
        st.caption(idea)
    n_ok = sum(r[4] for r in rows)
    icon = "✅" if n_ok == len(rows) else ("🟡" if n_ok else "⬜")
    with st.expander(f"{icon} {num}. {m['title']}  ·  {n_ok}/{len(rows)}", expanded=bool(nxt and nxt[0] is m)):
        st.markdown(f"_{m['concept']}_")
        for tid, title, page, auto, ok in rows:
            a, b, c = st.columns([0.6, 6, 1.4])
            if auto:
                a.markdown("✅" if ok else "⬜")
            else:
                v = a.checkbox("done", value=ok, key=f"cp_{tid}", label_visibility="collapsed")
                if v != ok:
                    registry.set_progress(tid, v)
                    st.rerun()
            b.markdown(f"{title}" + ("" if auto else "  ✋"))
            c.page_link(page, label="Open", icon="↗️")
