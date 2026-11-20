"""Eval-practice helpers (pure functions, no model needed): error analysis, reviewer agreement,
judge validation with bias correction + bootstrap CI, review sampling, search, CI gate.

Each evaluation result row (see views/evaluation.py) may carry:
  review        primary human label: "correct" | "partly correct" | "incorrect" | ""
  reviews       {reviewer_name: label}  (several reviewers on the same answers)
  failure_mode  short free-text tag, e.g. "promised a date"   (open/axial coding)
  cause         one of CAUSES  (where the failure comes from)
  note          reviewer note
  correction    better answer written by a reviewer
"""
import random
from collections import Counter

LABELS = ["correct", "partly correct", "incorrect"]
CAUSES = ["data gap", "unclear instruction", "model limit"]
CAUSE_HELP = {
    "data gap": "the training data has no (or wrong) examples of this case → add/fix SFT rows",
    "unclear instruction": "the expected behaviour was never spelled out (approved answer / rule is vague) → clarify it",
    "model limit": "clear data and instructions, the model still fails → bigger model, more training, or accept it",
}


def human_label(x, reviewer=None):
    if reviewer:
        return (x.get("reviews") or {}).get(reviewer, "")
    return x.get("review", "")


def is_pass(label):
    return label == "correct"


# ---------- error analysis ----------
def failures(results):
    return [x for x in results if x.get("review") in ("incorrect", "partly correct")]


def failure_table(results):
    """[(failure_mode, count, share_of_failures)] sorted by count, untagged failures included."""
    fails = failures(results)
    c = Counter((x.get("failure_mode") or "").strip().lower() or "(not tagged yet)" for x in fails)
    n = max(1, len(fails))
    return [(m, k, k / n) for m, k in c.most_common()]


def cause_counts(results):
    c = Counter(x.get("cause") or "(not set)" for x in failures(results))
    return dict(c.most_common())


# ---------- reviewer agreement ----------
def reviewers(results):
    return sorted({n for x in results for n in (x.get("reviews") or {})})


def cohen_kappa(a, b):
    """Cohen's kappa for two equal-length label lists (empty labels skipped pairwise)."""
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return None, 0
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if pe == 1:
        return 1.0, n
    return (po - pe) / (1 - pe), n


def kappa_meaning(k):
    if k is None:
        return "not enough shared labels"
    return ("poor" if k < 0.2 else "fair" if k < 0.4 else "moderate" if k < 0.6
            else "substantial" if k < 0.8 else "almost perfect")


# ---------- judge validation (automated rule score vs human labels) ----------
def judge_pass(x, threshold, source="rule"):
    """source='rule': automated rule score >= threshold. source='llm': LLM judge verdict == pass."""
    return x.get("judge") == "pass" if source == "llm" else x["reward"] >= threshold


def confusion(results, threshold, reviewer=None, source="rule"):
    tp = fn = tn = fp = 0
    for x in results:
        h = human_label(x, reviewer)
        if not h or (source == "llm" and not x.get("judge")):
            continue
        j, hp = judge_pass(x, threshold, source), is_pass(h)
        tp += j and hp
        fn += (not j) and hp
        tn += (not j) and (not hp)
        fp += j and (not hp)
    return {"tp": tp, "fn": fn, "tn": tn, "fp": fp}


def _rates(c):
    tpr = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None
    tnr = c["tn"] / (c["tn"] + c["fp"]) if c["tn"] + c["fp"] else None
    return tpr, tnr


def corrected_pass_rate(p_obs, tpr, tnr):
    """Rogan-Gladen correction: true pass rate from the judge's observed pass rate and its error rates."""
    if tpr is None or tnr is None or tpr + tnr - 1 <= 0:
        return None
    return min(1.0, max(0.0, (p_obs + tnr - 1) / (tpr + tnr - 1)))


def judge_report(results, threshold, reviewer=None, n_boot=1000, seed=0, source="rule"):
    c = confusion(results, threshold, reviewer, source)
    tpr, tnr = _rates(c)
    judged = [x for x in results if source != "llm" or x.get("judge")]
    p_obs = sum(judge_pass(x, threshold, source) for x in judged) / max(1, len(judged))
    theta = corrected_pass_rate(p_obs, tpr, tnr)
    labeled = [x for x in judged if human_label(x, reviewer)]
    rng, boots = random.Random(seed), []
    if theta is not None and labeled:
        for _ in range(n_boot):
            s = [rng.choice(labeled) for _ in labeled]
            t2 = corrected_pass_rate(p_obs, *_rates(confusion(s, threshold, reviewer, source)))
            if t2 is not None:
                boots.append(t2)
    boots.sort()
    ci = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]) if len(boots) >= 20 else None
    return {**c, "tpr": tpr, "tnr": tnr, "n_labeled": len(labeled), "judge_pass_rate": p_obs,
            "corrected_pass_rate": theta, "ci95": ci}


# ---------- review sampling & search ----------
SAMPLING = {
    "Not reviewed yet": "answers nobody has labelled, lowest score first",
    "Lowest score first": "where the automated check is least happy",
    "Judge vs human disagree": "labelled answers where the rule score and the human disagree",
    "Random": "an unbiased sample, to estimate the true failure rate",
}


def order_for_review(results, strategy, threshold=0.5, seed=0, reviewer=None):
    idx = list(range(len(results)))
    if strategy == "Not reviewed yet":
        todo = [i for i in idx if not (human_label(results[i], reviewer) if reviewer else results[i].get("review"))]
        idx = sorted(todo, key=lambda i: results[i]["reward"])
    elif strategy == "Lowest score first":
        idx.sort(key=lambda i: results[i]["reward"])
    elif strategy == "Judge vs human disagree":
        idx = [i for i in idx if results[i].get("review")
               and (results[i]["reward"] >= threshold) != is_pass(results[i]["review"])]
    else:
        random.Random(seed).shuffle(idx)
    return idx


def search(results, query):
    q = query.lower().strip()
    if not q:
        return list(range(len(results)))
    return [i for i, x in enumerate(results)
            if q in " ".join(str(x.get(k, "")) for k in ("question", "answer", "approved", "note", "failure_mode")).lower()]


# ---------- failures → training rows ----------
def failures_to_sft_rows(results, eval_id):
    """Failed answers that have a reviewer correction (or an approved answer) become SFT training rows."""
    out = []
    for x in failures(results):
        ans = (x.get("correction") or x.get("approved") or "").strip()
        if ans:
            out.append({"id": f"fix-e{eval_id}-{x['id']}", "prompt": x["question"], "completion": ans})
    return out


# ---------- CI gate ----------
def gate(results, mean_reward, min_score=None, min_human=None):
    """Returns (passed, [messages]). Human threshold is applied only to reviewed rows."""
    msgs, ok = [], True
    if min_score is not None:
        good = mean_reward >= min_score
        ok &= good
        msgs.append(f"{'PASS' if good else 'FAIL'} mean rule score {mean_reward:.2f} (min {min_score})")
    if min_human is not None:
        rev = [x for x in results if x.get("review")]
        rate = sum(is_pass(x["review"]) for x in rev) / len(rev) if rev else 0.0
        good = bool(rev) and rate >= min_human
        ok &= good
        msgs.append(f"{'PASS' if good else 'FAIL'} human pass rate {rate:.2f} on {len(rev)} reviewed (min {min_human})")
    return ok, msgs
