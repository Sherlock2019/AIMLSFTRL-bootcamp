"""Controlled, no-code reward rules. No user Python is ever executed: rules are data interpreted here.

Rule = {id, name, type, params, condition ("match"|"no_match"), weight, enabled}
Types:
  keywords_any   - reward if ANY phrase in params.keywords is mentioned
  keywords_all   - reward if ALL phrases are mentioned
  regex          - reward if params.pattern matches (case-insensitive)
  empty          - reward if the answer is empty
  max_words      - reward if the answer is longer than params.max_words
  row_required   - weight x (fraction of the row's `required_checks` mentioned)
  row_avoid      - weight if any of the row's `avoid` phrases is mentioned
"""
import re
from collections import Counter

STOP = {"a", "an", "the", "and", "or", "to", "of", "for", "before", "is", "it", "be", "with", "on", "in", "at", "then"}
RULE_TYPES = {
    "keywords_any": "Mentions any of these phrases",
    "keywords_all": "Mentions all of these phrases",
    "regex": "Matches a text pattern (regular expression)",
    "empty": "Answer is empty",
    "max_words": "Answer is longer than N words",
    "row_required": "Covers the row's required_checks (dataset field)",
    "row_avoid": "Uses one of the row's avoid phrases (dataset field)",
}


def clean_answer(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    return text.replace("<think>", "").replace("</think>", "").strip()


def _norm_tokens(text):
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def mentions(answer: str, phrase: str) -> bool:
    """True if the phrase appears verbatim, or every content word of it appears (prefix match, so
    'check' matches 'checking'). Deliberately simple: it can be fooled, which is a lesson of the lab."""
    a = " ".join(_norm_tokens(answer))
    p = " ".join(_norm_tokens(phrase))
    if not p:
        return False
    if p in a:
        return True
    words = [w for w in p.split() if w not in STOP]
    toks = a.split()
    return bool(words) and all(any(t.startswith(w.rstrip("s")) for t in toks) for w in words)


def _eval_rule(rule, answer, row):
    t, p = rule["type"], rule.get("params", {}) or {}
    w = float(rule.get("weight", 0))
    if t == "row_required":
        req = list((row or {}).get("required_checks") or [])
        if not req:
            return False, 0.0, "row has no required_checks"
        found = [c for c in req if mentions(answer, c)]
        missing = [c for c in req if c not in found]
        return bool(found), w * len(found) / len(req), f"found {found or '-'}; missing {missing or '-'}"
    if t == "row_avoid":
        hits = [c for c in (row or {}).get("avoid") or [] if mentions(answer, c)]
        return bool(hits), (w if hits else 0.0), (f"used {hits}" if hits else "none used")
    if t == "keywords_any":
        hits = [k for k in p.get("keywords", []) if mentions(answer, k)]
        matched, detail = bool(hits), f"hit {hits}" if hits else "no hit"
    elif t == "keywords_all":
        kws = p.get("keywords", [])
        miss = [k for k in kws if not mentions(answer, k)]
        matched, detail = bool(kws) and not miss, f"missing {miss}" if miss else "all present"
    elif t == "regex":
        pat = str(p.get("pattern", ""))[:200]
        try:
            matched = bool(re.search(pat, answer, flags=re.I)) if pat else False
            detail = "pattern matched" if matched else "no match"
        except re.error as e:
            return False, 0.0, f"invalid pattern: {e}"
    elif t == "empty":
        matched, detail = not answer.strip(), "empty" if not answer.strip() else "not empty"
    elif t == "max_words":
        n = len(answer.split())
        matched, detail = n > int(p.get("max_words", 80)), f"{n} words"
    else:
        return False, 0.0, f"unknown rule type {t}"
    fire = matched if rule.get("condition", "match") == "match" else not matched
    return fire, (w if fire else 0.0), detail


def score(answer: str, rules, row=None):
    answer = clean_answer(answer)
    breakdown, total = [], 0.0
    for r in rules:
        if not r.get("enabled", True):
            continue
        fired, contrib, detail = _eval_rule(r, answer, row)
        total += contrib
        breakdown.append({"rule": r["name"], "type": r["type"], "fired": fired,
                          "contribution": round(contrib, 3), "detail": detail})
    return {"total": round(total, 3), "breakdown": breakdown}


def make_reward_func(rules, log_hook=None):
    """Build a TRL-compatible reward function: f(prompts, completions, **dataset_columns) -> [float]."""
    def tool_store_reward(prompts, completions, **kwargs):
        out = []
        for i, c in enumerate(completions):
            text = c if isinstance(c, str) else (c[-1]["content"] if c else "")
            row = {k: (v[i] if isinstance(v, list) and len(v) == len(completions) else None) for k, v in kwargs.items()}
            s = score(text, rules, row)
            out.append(s["total"])
            if log_hook:
                log_hook(prompts[i], clean_answer(text), s)
        return out
    return tool_store_reward


DEFAULT_RULES = [
    {"id": "r1", "name": "Mentions checking warehouse stock", "type": "keywords_any",
     "params": {"keywords": ["warehouse stock", "check stock", "stock level", "inventory"]},
     "condition": "match", "weight": 1.0, "enabled": True},
    {"id": "r2", "name": "Mentions checking delivery availability", "type": "keywords_any",
     "params": {"keywords": ["delivery availability", "delivery slot", "delivery schedule"]},
     "condition": "match", "weight": 1.0, "enabled": True},
    {"id": "r3", "name": "Promises a date or outcome without evidence", "type": "keywords_any",
     "params": {"keywords": ["guarantee", "i promise", "will definitely", "will arrive tomorrow"]},
     "condition": "match", "weight": -2.0, "enabled": True},
    {"id": "r4", "name": "Covers the row's required checks", "type": "row_required", "params": {},
     "condition": "match", "weight": 2.0, "enabled": True},
    {"id": "r5", "name": "Uses a phrase the row says to avoid", "type": "row_avoid", "params": {},
     "condition": "match", "weight": -2.0, "enabled": True},
    {"id": "r6", "name": "Answer is empty", "type": "empty", "params": {},
     "condition": "match", "weight": -2.0, "enabled": True},
    {"id": "r7", "name": "Answer is too long (over 60 words)", "type": "max_words", "params": {"max_words": 60},
     "condition": "match", "weight": -1.0, "enabled": True},
]

TEST_BENCH = [
    {"label": "Good", "answer": "Check warehouse stock and delivery availability first.",
     "row": {"required_checks": ["warehouse stock", "delivery availability"], "avoid": ["guarantee"]}},
    {"label": "Bad: unsupported promise", "answer": "Yes, it will arrive tomorrow.",
     "row": {"required_checks": ["warehouse stock", "delivery availability"], "avoid": ["guarantee"]}},
    {"label": "Tricky: right words + promise", "answer": "I will check stock, but I guarantee delivery tomorrow.",
     "row": {"required_checks": ["warehouse stock", "delivery availability"], "avoid": ["guarantee"]}},
    {"label": "Tricky: negated promise (rule fooled?)", "answer": "Don't guarantee anything; check warehouse stock and delivery availability.",
     "row": {"required_checks": ["warehouse stock", "delivery availability"], "avoid": ["guarantee"]}},
    {"label": "Keyword stuffing", "answer": "warehouse stock delivery availability warehouse stock",
     "row": {"required_checks": ["warehouse stock", "delivery availability"], "avoid": ["guarantee"]}},
    {"label": "Empty", "answer": "", "row": {"required_checks": ["warehouse stock"], "avoid": []}},
]


def suggest_rules(approved_answers, top_k=4):
    """Draft candidate keyword rules from frequent two-word phrases in approved answers.
    These are UNREVIEWED suggestions - the user must edit, test and approve them."""
    grams = Counter()
    for a in approved_answers:
        toks = [t for t in _norm_tokens(a) if t not in STOP]
        grams.update(" ".join(toks[i:i + 2]) for i in range(len(toks) - 1))
    out = []
    for i, (g, n) in enumerate(grams.most_common(top_k)):
        out.append({"id": f"s{i + 1}", "name": f"[UNREVIEWED suggestion] Mentions '{g}' (in {n} approved answers)",
                    "type": "keywords_any", "params": {"keywords": [g]}, "condition": "match",
                    "weight": 0.5, "enabled": False})
    return out
