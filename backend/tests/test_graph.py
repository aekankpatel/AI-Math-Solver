"""LangGraph workflow tests, including a fake LLM planner (no network)."""

import pytest

from app.agents import graph as graph_mod
from app.agents.llm_planner import LLMPlanUnsupported, PlannerOutput, plan_output_to_tool_call
from app.schemas import ProblemCategory, ToolCall, VerificationStatus


def test_offline_pipeline_end_to_end():
    state = graph_mod.run_solver("Integrate x^2 from 0 to 3")
    assert state["status"] == "solved"
    assert state["planner"] == "rules"
    assert state["tool_output"].answer_text == "9"
    assert state["verification"].status == VerificationStatus.VERIFIED
    assert state["explanation"]
    assert state["visualization"]["data"]


def test_unsupported_short_circuits():
    state = graph_mod.run_solver("tell me a joke")
    assert state["status"] == "unsupported"
    assert "tool_output" not in state or state["tool_output"] is None


def test_tool_error_is_reported():
    state = graph_mod.run_solver("inverse of [[1,2],[2,4]]")
    assert state["status"] == "error"
    assert "singular" in state["error"]


def test_llm_plan_is_used(monkeypatch):
    call = ToolCall(tool="differentiate", category=ProblemCategory.CALCULUS, args={"expression": "x^2"})
    monkeypatch.setattr(graph_mod, "plan_with_llm", lambda p, c=None: call)
    state = graph_mod.run_solver("what's the slope function of x squared?")
    assert state["planner"] == "llm"
    assert state["tool_output"].answer_text == "2*x"


def test_bad_llm_plan_falls_back_to_rules(monkeypatch):
    bad = ToolCall(tool="differentiate", category=ProblemCategory.CALCULUS, args={"expression": "import os"})
    monkeypatch.setattr(graph_mod, "plan_with_llm", lambda p, c=None: bad)
    state = graph_mod.run_solver("derivative of x^3")
    assert state["status"] == "solved"
    assert state["planner"] == "rules"
    assert state["tool_output"].answer_text == "3*x**2"


def test_llm_failure_falls_back_to_rules(monkeypatch):
    def boom(p, c=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(graph_mod, "plan_with_llm", boom)
    assert graph_mod.run_solver("solve x^2 = 9")["status"] == "solved"


def test_llm_unsupported_is_respected(monkeypatch):
    def unsupported(p, c=None):
        raise LLMPlanUnsupported("Differential equations are not supported yet.")

    monkeypatch.setattr(graph_mod, "plan_with_llm", unsupported)
    state = graph_mod.run_solver("solve y'' + y = 0")
    assert state["status"] == "unsupported"
    assert "Differential" in state["error"]


def test_planner_output_mapping_drops_irrelevant_fields():
    out = PlannerOutput(
        supported=True,
        category="integrals",
        tool="integrate",
        expression="x",
        lower="0",
        upper="1",
        matrix=[[1]],
        data=[1.0],
    )
    call = plan_output_to_tool_call(out)
    assert call.args == {"expression": "x", "lower": "0", "upper": "1"}
    with pytest.raises(LLMPlanUnsupported):
        plan_output_to_tool_call(PlannerOutput(supported=False, category="unsupported", tool="none", clarification="?"))
