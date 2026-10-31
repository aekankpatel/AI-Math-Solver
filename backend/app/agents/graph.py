"""LangGraph Solver–Verifier–Tutor workflow.

    classify/plan ──► solve (tool) ──► verify ──► explain ──► visualize ──► END
          │                │
          └── unsupported ─┴── error ──────────────────────────────────────► END

Planning uses the Groq LLM (structured tool selection) when configured and
falls back to the deterministic rule planner otherwise or on failure.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.llm_planner import LLMPlanUnsupported, plan_with_llm
from app.agents.rule_planner import PlanningError, plan_with_rules
from app.agents.tutor import explain
from app.math_engine.parser import ExpressionError
from app.math_engine.tools import ToolError, ToolOutput, run_tool
from app.math_engine.verifier import verify
from app.math_engine.visualize import visualize
from app.schemas import ProblemCategory, Step, ToolCall, Verification

log = logging.getLogger(__name__)


class SolverState(TypedDict, total=False):
    problem: str
    context: str | None
    tool_call: ToolCall | None
    planner: str
    fallback_call: ToolCall | None
    tool_output: ToolOutput | None
    verification: Verification | None
    explanation: str | None
    visualization: dict[str, Any] | None
    status: str
    error: str | None
    llm_used: bool


def plan_node(state: SolverState) -> SolverState:
    problem, context = state["problem"], state.get("context")
    rules_call: ToolCall | None = None
    rules_error: str | None = None
    try:
        rules_call = plan_with_rules(problem)
    except (PlanningError, ExpressionError) as exc:
        rules_error = str(exc)

    try:
        llm_call = plan_with_llm(problem, context)
    except LLMPlanUnsupported as exc:
        # The LLM judged the problem unsupported/ambiguous: ask rather than guess.
        return {"status": "unsupported", "error": str(exc), "planner": "llm", "llm_used": True}
    except Exception as exc:  # provider/network/validation errors -> deterministic fallback
        log.warning("LLM planning failed, using rules: %s", exc)
        llm_call = None

    if llm_call is not None:
        return {"tool_call": llm_call, "planner": "llm", "fallback_call": rules_call, "llm_used": True}
    if rules_call is not None:
        return {"tool_call": rules_call, "planner": "rules"}
    return {"status": "unsupported", "error": rules_error or "Unsupported problem.", "planner": "rules"}


def solve_node(state: SolverState) -> SolverState:
    call = state["tool_call"]
    assert call is not None
    try:
        return {"tool_output": run_tool(call.tool, call.args)}
    except (ToolError, ExpressionError) as exc:
        fallback = state.get("fallback_call")
        if fallback is not None and fallback != call:
            try:
                return {
                    "tool_output": run_tool(fallback.tool, fallback.args),
                    "tool_call": fallback,
                    "planner": "rules",
                }
            except (ToolError, ExpressionError):
                pass
        return {"status": "error", "error": str(exc)}


def verify_node(state: SolverState) -> SolverState:
    call, out = state["tool_call"], state["tool_output"]
    assert call is not None and out is not None
    return {"verification": verify(call.tool, out)}


def explain_node(state: SolverState) -> SolverState:
    call, out = state["tool_call"], state["tool_output"]
    assert call is not None and out is not None
    text, used = explain(state["problem"], call.tool, out.answer_text, out.steps, state.get("verification"))
    return {"explanation": text, "llm_used": state.get("llm_used", False) or used}


def visualize_node(state: SolverState) -> SolverState:
    call, out = state["tool_call"], state["tool_output"]
    assert call is not None and out is not None
    return {"visualization": visualize(call.tool, out), "status": "solved"}


def _after_plan(state: SolverState) -> str:
    return "end" if state.get("status") == "unsupported" else "solve"


def _after_solve(state: SolverState) -> str:
    return "end" if state.get("status") == "error" else "verify"


@lru_cache(maxsize=1)
def build_graph() -> Any:
    g = StateGraph(SolverState)
    g.add_node("plan", plan_node)
    g.add_node("solve", solve_node)
    g.add_node("verify", verify_node)
    g.add_node("explain", explain_node)
    g.add_node("visualize", visualize_node)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", _after_plan, {"solve": "solve", "end": END})
    g.add_conditional_edges("solve", _after_solve, {"verify": "verify", "end": END})
    g.add_edge("verify", "explain")
    g.add_edge("explain", "visualize")
    g.add_edge("visualize", END)
    return g.compile()


def run_solver(problem: str, context: str | None = None) -> SolverState:
    return build_graph().invoke({"problem": problem, "context": context, "llm_used": False})


def state_steps(state: SolverState) -> list[Step]:
    out = state.get("tool_output")
    return out.steps if out else []


def category_of(state: SolverState) -> ProblemCategory:
    call = state.get("tool_call")
    return call.category if call else ProblemCategory.UNSUPPORTED
