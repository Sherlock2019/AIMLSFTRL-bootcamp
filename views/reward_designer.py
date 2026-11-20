import pandas as pd
import streamlit as st

import ui_common as ui
from lab import data, registry, rewards

st.title("🎯 Reward Designer")
ui.explain("Turn what you learned from evaluation into explicit, testable scoring rules for RL.",
           "Rules (phrase checks, patterns, length limits, dataset-field checks) and their weights.",
           "Nothing yet — RL will use an approved version to score every generated answer.",
           "The test bench: do good answers score high, bad answers low, and can a rule be fooled?",
           "Each saved rule set is a numbered reward version. Only approved versions can be used for RL.")
st.caption("Rules are data interpreted by the lab — no Python code can be entered here. "
           "Evaluation results help you *draft* rules; they do not create a reliable reward function on their own.")

rws = registry.rewards()
rid = st.selectbox("Reward version", [r["id"] for r in rws],
                   format_func=lambda i: f"{i} · {'✅ approved' if registry.reward(i)['approved'] else '📝 draft'}"
                                         f" · {registry.reward(i)['note'] or ''}")
rw = registry.reward(rid)


def to_df(rules):
    return pd.DataFrame([{"enabled": r.get("enabled", True), "name": r["name"], "type": r["type"],
                          "keywords": "; ".join(r.get("params", {}).get("keywords", [])),
                          "pattern": r.get("params", {}).get("pattern", ""),
                          "max_words": r.get("params", {}).get("max_words", 60),
                          "condition": r.get("condition", "match"), "weight": float(r.get("weight", 0))} for r in rules])


def from_df(df):
    out = []
    for i, r in enumerate(df.to_dict("records"), 1):
        if not str(r.get("name") or "").strip():
            continue
        p = {}
        if r["type"] in ("keywords_any", "keywords_all"):
            p["keywords"] = [k.strip() for k in str(r.get("keywords") or "").split(";") if k.strip()]
        if r["type"] == "regex":
            p["pattern"] = str(r.get("pattern") or "")
        if r["type"] == "max_words":
            p["max_words"] = int(r.get("max_words") or 60)
        out.append({"id": f"r{i}", "name": r["name"], "type": r["type"], "params": p,
                    "condition": r.get("condition") or "match", "weight": float(r.get("weight") or 0),
                    "enabled": bool(r.get("enabled"))})
    return out


tab_edit, tab_bench, tab_suggest = st.tabs(["🧱 Rules", "🧪 Test bench", "💡 Suggested rules (unreviewed)"])
with tab_edit:
    if rw["approved"]:
        st.success(f"{rid} is approved and frozen. Editing below and saving creates a new draft version.")
    base_rules = st.session_state.pop("_rules_override", None) or rw["rules"]
    edited = st.data_editor(
        to_df(base_rules), num_rows="dynamic", use_container_width=True, hide_index=True, key=f"rules_{rid}_{len(base_rules)}",
        column_config={
            "type": st.column_config.SelectboxColumn("type", options=list(rewards.RULE_TYPES), required=True,
                                                     help="; ".join(f"{k}: {v}" for k, v in rewards.RULE_TYPES.items())),
            "condition": st.column_config.SelectboxColumn("condition", options=["match", "no_match"],
                                                          help="match = apply weight when the check is true; no_match = when false"),
            "weight": st.column_config.NumberColumn("weight", help="Positive = reward, negative = penalty", step=0.5),
            "keywords": st.column_config.TextColumn("keywords", help="For keyword rules: phrases separated by ;"),
        })
    st.session_state["_draft_rules"] = from_df(edited)
    with st.expander("Rule types explained"):
        for k, v in rewards.RULE_TYPES.items():
            st.write(f"**{k}** — {v}")
        st.write("Phrase matching: a phrase counts if it appears verbatim, or if every content word appears "
                 "(so 'check stock' also matches 'checking the stock'). It ignores negation — see the test bench.")
    note = st.text_input("Version note", key="rw_note", placeholder="What changed?")
    c1, c2 = st.columns(2)
    if c1.button("💾 Save as new draft version"):
        new = registry.add_reward(st.session_state["_draft_rules"], note=note, parent_id=rid)
        st.success(f"Saved {new}")
        st.rerun()
    if not rw["approved"] and c2.button(f"✅ Approve {rid} for RL", type="primary"):
        if st.session_state["_draft_rules"] != rw["rules"]:
            st.error("You have unsaved edits. Save them as a new version first, test it, then approve that version.")
        else:
            registry.update("rewards", rid, approved=1)
            st.success(f"{rid} approved.")
            st.rerun()

with tab_bench:
    rules = st.session_state.get("_draft_rules", rw["rules"])
    st.caption("Scores below use the rules currently in the editor (including unsaved edits).")
    summary = []
    for case in rewards.TEST_BENCH:
        s = rewards.score(case["answer"], rules, case["row"])
        summary.append({"Case": case["label"], "Answer": case["answer"] or "(empty)", "Total reward": s["total"]})
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
    st.info("Look for cases where a bad answer scores well. 'Keyword stuffing' and the negated promise show how "
            "simple phrase rules can be fooled — the RL model may learn to exploit exactly these gaps.")
    st.markdown("##### Try your own answer")
    ans = st.text_area("Answer", "Check warehouse stock and delivery availability first.")
    rl_rows = data.load(registry.latest_dataset("rl_train")["id"]) if registry.latest_dataset("rl_train") else []
    q = st.selectbox("Use grading fields from RL question", [None] + rl_rows,
                     format_func=lambda r: "— custom fields —" if r is None else r["prompt"])
    if q is None:
        req = st.text_input("required_checks (; separated)", "warehouse stock; delivery availability")
        avoid = st.text_input("avoid (; separated)", "guarantee")
        row = {"required_checks": [x.strip() for x in req.split(";") if x.strip()],
               "avoid": [x.strip() for x in avoid.split(";") if x.strip()]}
    else:
        row = q
    s = rewards.score(ans, rules, row)
    st.dataframe(pd.DataFrame([{"Rule": b["rule"], "Fired": "✓" if b["fired"] else "", "Contribution": b["contribution"],
                                "Detail": b["detail"]} for b in s["breakdown"]]),
                 use_container_width=True, hide_index=True)
    st.metric("Total reward", f"{s['total']:+g}")

with tab_suggest:
    st.warning("These are **unreviewed** drafts mined from frequent phrases in approved SFT validation answers. "
               "They are disabled by default. Edit, test and approve before relying on them.")
    val = registry.latest_dataset("sft_val")
    sugg = rewards.suggest_rules([r["completion"] for r in data.load(val["id"])]) if val else []
    if sugg:
        st.dataframe(to_df(sugg), use_container_width=True, hide_index=True)
        if st.button("Add suggestions to the editor (disabled)"):
            st.session_state["_rules_override"] = rw["rules"] + sugg
            st.rerun()
