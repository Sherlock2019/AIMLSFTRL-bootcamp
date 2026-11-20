import json

import pandas as pd
import streamlit as st

import ui_common as ui
from lab import data, registry

st.title("🗂️ Data Studio")
ui.explain("Prepare and version the examples the model learns from and is tested on.",
           "Questions, approved answers (SFT) and grading fields (RL). Load a bundled pack, import CSV/JSONL or edit rows.",
           "Nothing — data is only read when you start a training or evaluation run.",
           "Row counts, duplicate warnings and questions that leak between splits.",
           "Every save is a new immutable version (a JSONL file + a content hash) recorded in the registry.")
st.caption("All bundled data is **synthetic and small** — good for learning the workflow, not proof a model is production-ready.")

st.markdown("**Step 1 · Choose the dataset**")
role = st.radio("Dataset", list(data.ROLES), horizontal=True, label_visibility="collapsed",
                format_func=lambda r: r.replace("_", " ").upper())
st.caption(data.ROLES[role])
versions = registry.datasets(role)
if not versions:
    st.info("No versions yet.")
    st.stop()

st.markdown("**Step 2 · Choose the version** (newest first)")
vid = st.selectbox("Version", [v["id"] for v in versions], label_visibility="collapsed",
                   format_func=lambda v: f"{v} · {registry.dataset(v)['n_rows']} rows · {registry.dataset(v)['source']}"
                                         f" · {registry.dataset(v)['note'][:50]}")
rows = data.load(vid)
d = registry.dataset(vid)
if d["parent_id"]:
    st.caption(f"Made from {d['parent_id']} · hash {d['hash']}")

is_sft = role.startswith("sft")
LIST = data.LIST_FIELDS & set(data.FIELDS[role])


def save(new_rows, source, note):
    new_id = data.save_version(role, new_rows, parent_id=vid, source=source, note=note)
    if new_id == vid:
        st.info("No changes — same content as this version.")
    else:
        st.session_state["ds_msg"] = f"Saved **{new_id}** ({len(data.load(new_id))} rows). It is now the newest version."
        st.rerun()


if msg := st.session_state.pop("ds_msg", None):
    st.success(msg)

st.markdown("**Step 3 · Work on it**")
tabs = (["📦 10 SFT packs", "📝 Edit one example"] if is_sft else []) + \
       ["✏️ Table edit", "🩺 Quality & leakage", "🔀 Changes", "⇅ Import / export"]
T = dict(zip(tabs, st.tabs(tabs)))

# ---------------------------------------------------------------- SFT packs
if is_sft:
    with T["📦 10 SFT packs"]:
        st.caption("Ready-made supervised datasets (question → approved answer). Loading one creates new "
                   "**SFT TRAIN** and **SFT VAL** versions together, so train and validation always match.")
        packs = data.list_packs()
        st.dataframe(pd.DataFrame([{"Pack": p["title"], "Domain": p["domain"], "Train": p["n_train"],
                                    "Val": p["n_val"], "What it teaches": p["description"]} for p in packs]),
                     hide_index=True, width="stretch")
        c1, c2 = st.columns([2, 1])
        slug = c1.selectbox("Pack", [p["slug"] for p in packs],
                            format_func=lambda s: next(p["title"] for p in packs if p["slug"] == s))
        mode = c2.radio("How", ["replace", "append"], horizontal=True,
                        format_func=lambda m: "Use only this pack" if m == "replace" else "Add to current data",
                        help="Add to current = latest SFT train/val + this pack (good for multi-domain training).")
        _, prev_tr, prev_va = data.pack_rows(slug)
        with st.expander(f"Preview ({len(prev_tr)} train + {len(prev_va)} val examples)"):
            st.dataframe(pd.DataFrame(prev_tr + prev_va), hide_index=True, width="stretch")
        if st.button("📦 Load pack", type="primary"):
            tr, va = data.load_pack(slug, mode)
            st.session_state["ds_msg"] = f"Loaded pack → **{tr}** and **{va}**. Pick them on the SFT Training page."
            st.rerun()

    # ------------------------------------------------------------ single-example editor
    with T["📝 Edit one example"]:
        st.caption("Easier than the table for long answers. Each save makes a new version; the old one is kept.")
        ids = [r["id"] for r in rows]
        q = st.text_input("🔎 Find an example", placeholder="type words from the question or answer")
        shown = [r["id"] for r in rows if q.lower() in (r["prompt"] + " " + r.get("completion", "")).lower()]
        pick = st.selectbox(f"Example ({len(shown)} of {len(rows)})", ["➕ New example"] + shown,
                            format_func=lambda i: i if i.startswith("➕") else
                            f"{i} · {next(r['prompt'] for r in rows if r['id'] == i)[:80]}")
        cur = next((r for r in rows if r["id"] == pick), {"id": "", "prompt": "", "completion": ""})
        with st.form(f"one_{vid}_{pick}"):
            p = st.text_area("Question (prompt)", cur["prompt"], height=90)
            c = st.text_area("Approved answer (completion) — what the model should say", cur.get("completion", ""),
                             height=120)
            note = st.text_input("Why this change? (version note)")
            b1, b2, b3 = st.columns(3)
            do_save = b1.form_submit_button("💾 Save", type="primary")
            do_dup = b2.form_submit_button("⧉ Save as copy", disabled=pick.startswith("➕"))
            do_del = b3.form_submit_button("🗑️ Delete", disabled=pick.startswith("➕"))
        if do_save or do_dup:
            if not p.strip() or not c.strip():
                st.error("Both the question and the approved answer are needed.")
            else:
                new = {"id": cur["id"], "prompt": p.strip(), "completion": c.strip()}
                if pick.startswith("➕") or do_dup:
                    n, base = len(rows) + 1, (cur["id"] or role)
                    while f"{base}-{n:03d}" in ids:
                        n += 1
                    new["id"] = f"{base}-{n:03d}"
                    out = rows + [new]
                else:
                    out = [new if r["id"] == pick else r for r in rows]
                save(out, "edit", note or f"{'added' if new['id'] not in ids else 'edited'} {new['id']}")
        if do_del:
            save([r for r in rows if r["id"] != pick], "edit", note or f"deleted {pick}")

        with st.expander("🔁 Find & replace across all examples"):
            f1, f2, f3 = st.columns([2, 2, 1])
            find = f1.text_input("Find")
            repl = f2.text_input("Replace with")
            field = f3.selectbox("In", ["completion", "prompt"])
            new_rows, n = data.find_replace(rows, field, find, repl)
            st.caption(f"{n} example(s) would change." if find else "Type text to find.")
            if st.button("Replace all", disabled=not n):
                save(new_rows, "edit", f"replace '{find}' → '{repl}' in {field} ({n} rows)")

# ---------------------------------------------------------------- table edit
df = pd.DataFrame([{k: ("; ".join(v) if k in LIST and isinstance(v, list) else v) for k, v in r.items()} for r in rows])
for f in data.FIELDS[role]:
    if f not in df.columns:
        df[f] = False if f == "retired" else ""
df = df[data.FIELDS[role] + [c for c in df.columns if c not in data.FIELDS[role]]]

with T["✏️ Table edit"]:
    st.caption("Edit cells directly, add rows at the bottom, select rows + Delete to remove."
               + (f" List fields ({', '.join(sorted(LIST))}): separate items with `;`." if LIST else ""))
    edited = st.data_editor(df, num_rows="dynamic", width="stretch", key=f"ed_{vid}", hide_index=True)
    note = st.text_input("Version note", placeholder="What did you change and why?", key="tbl_note")
    if st.button("💾 Save as new version", type="primary"):
        save([{k: v for k, v in r.items() if not (isinstance(v, float) and pd.isna(v))}
              for r in edited.to_dict("records")], "edit", note)
    used = [r for r in registry.runs() if vid in (r["train_ds"], r["val_ds"])]
    used_ev = [e for e in registry.evals() if e["dataset_id"] == vid]
    st.caption("Used by: " + (", ".join(f"{r['id']} ({r['kind'].upper()} → {r['output_model_id']})" for r in used)
                              or "no training runs yet") + (f" · {len(used_ev)} evaluations" if used_ev else ""))

with T["🩺 Quality & leakage"]:
    warns = data.quality_warnings(role, rows)
    st.metric("Examples", len(rows))
    for w in warns:
        st.warning(w)
    if not warns:
        st.success("No quality warnings.")
    st.markdown("##### Overlap between splits (latest version of each)")
    splits = {registry.latest_dataset(r)["id"]: data.load(registry.latest_dataset(r)["id"])
              for r in data.ROLES if registry.latest_dataset(r)}
    leaks = data.leakage(splits)
    for w in leaks:
        st.error("Possible leakage: " + w)
    if not leaks:
        st.success("No question appears in more than one split (after normalising case and punctuation).")

with T["🔀 Changes"]:
    if len(versions) < 2:
        st.info("Only one version so far — save an edit to see changes.")
    else:
        ids_v = [v["id"] for v in versions]
        a = st.selectbox("Older", ids_v, index=1, key="dfa")
        b = st.selectbox("Newer", ids_v, index=0, key="dfb")
        added, removed, changed = data.diff(data.load(a), data.load(b))
        st.write(f"**{len(added)}** added · **{len(removed)}** removed · **{len(changed)}** changed")
        for r in added:
            st.success(f"+ {r['id']}: {r['prompt']}")
        for r in removed:
            st.error(f"− {r['id']}: {r['prompt']}")
        for o, n in changed:
            st.warning(f"~ {n['id']}")
            st.json({"before": o, "after": n}, expanded=False)

with T["⇅ Import / export"]:
    st.download_button("⬇️ Export JSONL", "\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                       file_name=f"{vid}.jsonl")
    st.download_button("⬇️ Export CSV", data.to_csv(rows), file_name=f"{vid}.csv")
    up = st.file_uploader("Import CSV or JSONL as a new version of this dataset", type=["csv", "jsonl", "json"])
    if up and st.button("Import"):
        imp = None
        try:
            imp = data.parse_upload(up.name, up.getvalue())
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read file: {e}")
        need = ("prompt",) + (("completion",) if is_sft else ()) + (("chosen", "rejected") if role == "pref" else ())
        missing = [f for f in need if imp and f not in imp[0]]
        if missing:
            st.error(f"Missing required columns: {missing}. Expected: {data.FIELDS[role]}")
        elif imp:
            save(imp, "import", up.name)
