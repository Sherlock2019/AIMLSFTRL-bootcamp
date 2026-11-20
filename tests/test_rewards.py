from lab import rewards

ROW = {"required_checks": ["warehouse stock", "delivery availability"], "avoid": ["guarantee"]}
SPEC_RULES = [
    {"name": "stock", "type": "keywords_any", "params": {"keywords": ["warehouse stock"]}, "weight": 1},
    {"name": "delivery", "type": "keywords_any", "params": {"keywords": ["delivery availability"]}, "weight": 1},
    {"name": "promise", "type": "keywords_any", "params": {"keywords": ["guarantee", "will arrive tomorrow"]}, "weight": -2},
    {"name": "empty", "type": "empty", "params": {}, "weight": -2},
]


def test_spec_example_scores_plus_two():
    s = rewards.score("Check warehouse stock and delivery availability first.", SPEC_RULES)
    assert s["total"] == 2
    assert [b["contribution"] for b in s["breakdown"]] == [1, 1, 0, 0]


def test_unsupported_promise_detected_with_positive_phrase():
    s = rewards.score("I will check stock, but I guarantee delivery tomorrow.", rewards.DEFAULT_RULES, ROW)
    by = {b["rule"]: b["contribution"] for b in s["breakdown"]}
    assert by["Promises a date or outcome without evidence"] == -2
    assert by["Uses a phrase the row says to avoid"] == -2
    assert s["total"] < 0


def test_empty_and_long_and_no_match():
    assert rewards.score("", SPEC_RULES)["total"] == -2
    long = {"name": "long", "type": "max_words", "params": {"max_words": 3}, "weight": -1}
    assert rewards.score("one two three four", [long])["total"] == -1
    neg = {"name": "no stock", "type": "keywords_any", "params": {"keywords": ["stock"]}, "condition": "no_match", "weight": -1}
    assert rewards.score("hello", [neg])["total"] == -1 and rewards.score("stock", [neg])["total"] == 0


def test_row_required_is_fractional_and_disabled_rules_skipped():
    rule = {"name": "req", "type": "row_required", "params": {}, "weight": 2}
    assert rewards.score("check warehouse stock", [rule], ROW)["total"] == 1
    assert rewards.score("check warehouse stock", [dict(rule, enabled=False)], ROW)["total"] == 0


def test_bad_regex_is_safe_and_think_tags_stripped():
    r = {"name": "bad", "type": "regex", "params": {"pattern": "("}, "weight": 1}
    assert rewards.score("x", [r])["total"] == 0
    assert rewards.clean_answer("<think>\nhmm\n</think>\n\nCheck stock.") == "Check stock."


def test_trl_reward_func_receives_row_columns():
    f = rewards.make_reward_func(rewards.DEFAULT_RULES)
    out = f(prompts=["p", "p"], completions=["Check warehouse stock and delivery availability.", "I guarantee it."],
            required_checks=[ROW["required_checks"]] * 2, avoid=[ROW["avoid"]] * 2, trainer_state=None)
    assert out[0] > 0 > out[1]
