"""LLM-as-judge: a model grades answers PASS/FAIL against a written criterion and the approved answer.
Validate it against human labels on the Evals Workbench (④) before trusting it."""
import json
import re

from . import engines, registry

DEFAULT_CRITERION = ("PASS if the answer gives the same guidance as the reference answer (same key action, e.g. "
                     "check stock first / ask a manager) and does not invent promises, dates or discounts. "
                     "Otherwise FAIL.")
JUDGE_TEMPLATE = """You are a strict grader.
Criterion: {criterion}

Question: {question}
Reference answer: {reference}
Answer to grade: {answer}

Reply in exactly this format:
VERDICT: PASS or FAIL
REASON: one short sentence"""


def parse(text):
    m = re.search(r"VERDICT\s*:\s*(PASS|FAIL)", text, flags=re.I)
    verdict = m.group(1).lower() if m else ("pass" if re.match(r"\s*pass\b", text, re.I) else
                                            "fail" if re.match(r"\s*fail\b", text, re.I) else "")
    r = re.search(r"REASON\s*:\s*(.+)", text, flags=re.I | re.S)
    return ("pass" if verdict == "pass" else "fail" if verdict == "fail" else ""), (r.group(1).strip()[:300] if r else text.strip()[:300])


def grade(judge_model, question, answer, reference, criterion=DEFAULT_CRITERION):
    msg = JUDGE_TEMPLATE.format(criterion=criterion, question=question, reference=reference or "(none)", answer=answer)
    raw = engines.chat(judge_model, [{"role": "user", "content": msg}], max_new_tokens=80)
    return parse(raw)


def judge_eval(eval_id, judge_model, criterion=DEFAULT_CRITERION, progress=None):
    """Grade every answer of an evaluation record; stores judge / judge_reason on each result row."""
    e = next(x for x in registry.evals() if x["id"] == eval_id)
    R = e["results"]
    for i, x in enumerate(R):
        x["judge"], x["judge_reason"] = grade(judge_model, x["question"], x["answer"], x.get("approved", ""), criterion)
        if progress:
            progress(i + 1, len(R))
    settings = dict(e["settings"], judge_model=judge_model, judge_criterion=criterion)
    registry.update("evals", eval_id, results_json=json.dumps(R), settings_json=json.dumps(settings))
    return R
