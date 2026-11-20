"""RAG, agent, judge, API, course — logic tests without loading a model (the model is stubbed)."""
import json

import pytest

from lab import agent, agent_evals, course, data, engines, judge, rag, registry


@pytest.fixture
def stub_engine(monkeypatch):
    """Replace model generation with a scripted function: script(messages, tools) -> text."""
    box = {}

    def chat(model_id, messages, tools=None, **kw):
        return box["fn"](messages, tools)
    monkeypatch.setattr(engines, "chat", chat)
    return box


def test_chunking_and_bm25_ir_metrics(lab_home):
    assert len(rag.chunk("a" * 50 + "\n\n" + "b" * 50, size=60, overlap=10)) == 2
    rag.build_sample(embed=False)
    rows, mean = rag.eval_retrieval("tool_store", rag.sample_eval(), k=3, method="bm25")
    assert len(rows) == 20 and mean["recall"] >= 0.7 and 0 < mean["mrr"] <= 1
    m = rag.ir_metrics(["x", "a", "b"], ["a"], 3)
    assert m["mrr"] == 0.5 and m["recall"] == 1 and round(m["precision"], 3) == 0.333


def test_rag_answer_uses_langchain_and_logs_trace(lab_home, stub_engine):
    rag.build_sample(embed=False)
    seen = {}

    def fn(messages, tools):
        seen["msgs"] = messages
        return "Customers have 30 days [1]."
    stub_engine["fn"] = fn
    r = rag.answer("base-001", "tool_store", "How long is the return window?", k=2)
    assert "30 days" in r["answer"] and len(r["hits"]) == 2
    assert seen["msgs"][0]["role"] == "system" and "Sources:" in seen["msgs"][1]["content"]
    assert registry.traces("rag")[0]["output"] == r["answer"]


def test_tools_and_parsing():
    assert agent.calculator("3 * 89.90")["result"] == 269.7
    assert "error" in agent.calculator("__import__('os')")
    assert agent.check_stock("drill", "warehouse") == agent.check_stock("drill", "warehouse")  # deterministic
    calls = agent.parse_tool_calls('<tool_call>\n{"name": "check_stock", "arguments": {"item": "drill"}}\n</tool_call>')
    assert calls == [{"name": "check_stock", "arguments": {"item": "drill"}}]
    assert agent.run_tool({"name": "nope", "arguments": {}})["error"]


def test_langgraph_agent_loop_and_eval_scoring(lab_home, stub_engine):
    def fn(messages, tools):
        assert tools and any(t["function"]["name"] == "check_stock" for t in tools)
        if messages[-1]["role"] == "tool":
            n = json.loads(messages[-1]["content"])["in_stock"]
            return f"The warehouse has {n} drills."
        return '<tool_call>{"name": "check_stock", "arguments": {"item": "cordless drill", "store": "warehouse"}}</tool_call>'
    stub_engine["fn"] = fn
    sc = next(s for s in agent_evals.scenarios() if s["id"] == "tool-01")
    res, turns = agent_evals.run_scenario("base-001", sc)
    assert res["success"], res
    assert [s["step"] for s in turns[0]["steps"]] == ["llm", "tool", "llm"]
    assert registry.traces("agent")
    # a session that never calls the tool fails at turn 1
    bad = agent_evals.score(sc, [{"answer": "Yes we have some.", "steps": []}])
    assert not bad["success"] and bad["first_failure"].startswith("turn 1")
    unsafe = agent_evals.score({"id": "s", "turn_tools": [[]], "must_not_include": ["50%"]},
                               [{"answer": "You get 50% off!", "steps": []}])
    assert not unsafe["success"] and unsafe["unsafe"] == ["50%"]


def test_llm_judge_parse_and_eval(lab_home, stub_engine):
    assert judge.parse("VERDICT: PASS\nREASON: same advice") == ("pass", "same advice")
    assert judge.parse("fail, it promises a date")[0] == "fail"
    res = [dict(id="1", question="q", answer="Check stock first.", approved="Check stock.", reward=1.0, review="correct"),
           dict(id="2", question="q", answer="Yes tomorrow!", approved="Check stock.", reward=1.0, review="incorrect")]
    eid = registry.add_eval("base-001", "sft_val-v1", "reward-v1", {}, res, 1.0)
    stub_engine["fn"] = lambda m, t: "VERDICT: FAIL\nREASON: promise" if "tomorrow" in m[0]["content"] else "VERDICT: PASS\nREASON: ok"
    R = judge.judge_eval(eid, "base-001")
    from lab import evals
    rep = evals.judge_report(R, 0.5, source="llm")
    assert (rep["tp"], rep["tn"]) == (1, 1) and rep["tpr"] == 1 and rep["tnr"] == 1
    assert evals.judge_report(R, 0.5, source="rule")["fp"] == 1  # the rule score was fooled, the LLM judge was not


def test_api_openai_compatible(lab_home, stub_engine):
    from fastapi.testclient import TestClient

    from lab.api import app
    data.bootstrap_samples()
    registry.ensure_base("Qwen/Qwen3-0.6B")
    c = TestClient(app)
    assert any(m["id"] == "base-001" for m in c.get("/v1/models").json()["data"])
    stub_engine["fn"] = lambda m, t: "Check warehouse stock first."
    r = c.post("/v1/chat/completions", json={"model": "base-001", "messages": [{"role": "user", "content": "hi"}]}).json()
    assert r["choices"][0]["message"]["content"] == "Check warehouse stock first."
    stub_engine["fn"] = lambda m, t: '<tool_call>{"name": "calculator", "arguments": {"expression": "1+1"}}</tool_call>'
    r = c.post("/v1/chat/completions", json={"model": "base-001", "messages": [{"role": "user", "content": "1+1"}],
                                             "tools": agent.tool_schemas(["calculator"])}).json()
    assert r["choices"][0]["finish_reason"] == "tool_calls"
    s = c.post("/v1/chat/completions", json={"model": "base-001", "messages": [{"role": "user", "content": "x"}], "stream": True})
    assert "data: [DONE]" in s.text
    assert c.post("/v1/chat/completions", json={"model": "nope", "messages": []}).status_code == 404
    assert registry.traces("api")


def test_course_status_ticks(lab_home):
    data.bootstrap_samples()
    mods, done, total = course.status()
    assert total >= 25 and done == 0
    data.load_pack("01_it_helpdesk")
    registry.set_progress("m1-hw", True)
    _, done2, _ = course.status()
    assert done2 == 2
