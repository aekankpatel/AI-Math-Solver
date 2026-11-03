"""LangGraph Solver–Verifier–Tutor workflow with intent verification.

    plan ──► intent ──► solve ──► verify ──► explain ──► visualize ──► END
      │        │          │
      │        │          └── tool error ─────────────────────────────► END
      │        └── ambiguous / mismatch (after one repair attempt) ──► END (needs clarification)
      └── unsupported ───────────────────────────────────────────────► END

* **plan**: Groq LLM planner (structured tool call) when configured, with the
  deterministic rule planner as fallback.
* **intent**: checks that the planned tool and arguments answer the question
  that was asked (``app/agents/intent.py``). A mismatch gets one repair attempt
  (safe slot fixes stated in the question, or the alternative plan). Otherwise
  the system asks for clarification instead of answering the wrong question.
* **verify**: independent *mathematical* verification of the tool result.
* The final ``overall`` status is "verified" only if the math is verified AND
  the intent is confirmed.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agents.intent import check_intent
from app.agents.llm_planner import LLMPlanUnsupported, plan_with_llm
from app.agents.nl_planner import ClarificationNeeded, plan_from_intent
from app.agents.rule_planner import PlanningError, plan_with_rules
from app.agents.tutor import explain, template_explanation
from app.math_engine.parser import ExpressionError
from app.math_engine.tools import ToolError, ToolOutput, run_tool
from app.math_engine.verifier import verify
from app.math_engine.visualize import visualize
from app.schemas import (
    IntentStatus,
    IntentVerification,
    OverallStatus,
    ProblemCategory,
    Step,
    ToolCall,
    Verification,
    VerificationStatus,
)

log = logging.getLogger(__name__)

_ACCEPTABLE = (IntentStatus.CONFIRMED, IntentStatus.ASSUMED)


class SolverState(TypedDict, total=False):
    problem: str
    context: str | None
    tool_call: ToolCall | None
    planner: str
    fallback_call: ToolCall | None
    alternatives: list[ToolCall]
    intent: IntentVerification | None
    tool_output: ToolOutput | None
    verification: Verification | None
    overall: OverallStatus | None
    explanation: str | None
    visualization: dict[str, Any] | None
    status: str
    error: str | None
    llm_used: bool
    llm_plan: ToolCall | None
    llm_error: str | None
    # Set by the API service: the Groq planner already ran in the main process (network stays
    # outside the computation sandbox). Keys: call (ToolCall | None), error, unsupported.
    llm_result: dict[str, Any] | None
    explain_with_llm: bool


def plan_node(state: SolverState) -> SolverState:
    """Produce a primary plan and alternatives: LLM (if configured), then the rule
    planner, then the intent-based NL planner. The intent node picks the first
    one that faithfully matches the question."""
    problem, context = state["problem"], state.get("context")
    rules_call: ToolCall | None = None
    nl_call: ToolCall | None = None
    errors: list[str] = []
    clarification: str | None = None
    try:
        rules_call = plan_with_rules(problem).model_copy(update={"source": "rules"})
    except (PlanningError, ExpressionError) as exc:
        errors.append(str(exc))
    try:
        nl_call = plan_from_intent(problem)
    except ClarificationNeeded as exc:
        clarification = str(exc)
    except (PlanningError, ExpressionError) as exc:
        errors.append(str(exc))

    llm_error: str | None = None
    pre = state.get("llm_result")
    try:
        if pre is not None:  # planned outside the sandbox
            if pre.get("unsupported"):
                raise LLMPlanUnsupported(pre["unsupported"])
            if pre.get("error"):
                raise RuntimeError(pre["error"])
            llm_call = pre.get("call")
        else:
            llm_call = plan_with_llm(problem, context)
    except LLMPlanUnsupported as exc:
        # The LLM judged the problem unsupported/ambiguous: ask rather than guess.
        return {"status": "unsupported", "error": str(exc), "planner": "llm", "llm_used": True}
    except Exception as exc:  # provider/network/validation errors -> offline planners, recorded (never silent)
        log.warning("LLM planning failed, using offline planners: %s", exc)
        llm_call, llm_error = None, f"{type(exc).__name__}: {exc}"[:700]

    candidates = [
        c for c in (llm_call.model_copy(update={"source": "llm"}) if llm_call else None, rules_call, nl_call) if c
    ]
    if candidates:
        primary, rest = candidates[0], candidates[1:]
        return {
            "tool_call": primary,
            "planner": primary.source or "rules",
            "alternatives": rest,
            "fallback_call": rest[0] if rest else None,
            "llm_used": llm_call is not None,
            "llm_plan": candidates[0] if llm_call else None,
            "llm_error": llm_error,
        }
    if clarification:
        return {"status": "needs_clarification", "error": clarification, "planner": "nl", "llm_error": llm_error}
    return {
        "status": "unsupported",
        "error": errors[0] if errors else "Unsupported problem.",
        "planner": "rules",
        "llm_error": llm_error,
    }


def _check(problem: str, call: ToolCall) -> tuple[IntentVerification, dict[str, Any]]:
    iv, fixes = check_intent(problem, call, call.ambiguities, call.assumptions)
    if call.source == "nl" and iv.status == IntentStatus.CONFIRMED:
        # the NL planner and the checker share the same extraction: not independent
        iv.status = IntentStatus.ASSUMED
        iv.assumptions.append(
            "This interpretation was built from the same language analysis that checks it; please confirm it."
        )
    return iv, fixes


def intent_node(state: SolverState) -> SolverState:
    problem, call = state["problem"], state["tool_call"]
    assert call is not None
    iv, fixes = _check(problem, call)
    if iv.status in _ACCEPTABLE:
        return {"intent": iv}

    declared = any(c.name == "planner-declared ambiguity" and not c.passed for c in iv.checks)
    if iv.status == IntentStatus.MISMATCH and fixes:
        repaired = call.model_copy(update={"args": {**call.args, **fixes}})
        iv2, _ = _check(problem, repaired)
        if iv2.status in _ACCEPTABLE:
            iv2.repaired = True
            iv2.assumptions.append(
                "Adjusted the plan to match the question: " + ", ".join(f"{k} = {v}" for k, v in fixes.items()) + "."
            )
            return {"intent": iv2, "tool_call": repaired}

    if not declared:
        for alt in state.get("alternatives") or []:
            if alt == call:
                continue
            iva, _ = _check(problem, alt)
            if iva.status in _ACCEPTABLE:
                iva.assumptions.append(f"The first plan did not match the question; used the {alt.source} plan.")
                return {"intent": iva, "tool_call": alt, "planner": alt.source or "rules"}

    return {"intent": iv, "status": "needs_clarification", "error": iv.clarification}


def solve_node(state: SolverState) -> SolverState:
    call = state["tool_call"]
    assert call is not None
    try:
        return {"tool_output": run_tool(call.tool, call.args)}
    except (ToolError, ExpressionError) as exc:
        for alt in state.get("alternatives") or []:
            if alt == call:
                continue
            iva, _ = _check(state["problem"], alt)
            if iva.status not in _ACCEPTABLE:  # never run an alternative that misreads the question
                continue
            try:
                return {
                    "tool_output": run_tool(alt.tool, alt.args),
                    "tool_call": alt,
                    "planner": alt.source or "rules",
                    "intent": iva,
                }
            except (ToolError, ExpressionError):
                continue
        return {"status": "error", "error": str(exc)}


def overall_status(verification: Verification | None, intent: IntentVerification | None) -> OverallStatus:
    if intent is None or intent.status in (IntentStatus.AMBIGUOUS, IntentStatus.MISMATCH):
        return OverallStatus.NEEDS_CLARIFICATION
    if verification is None or verification.status == VerificationStatus.UNVERIFIED:
        return OverallStatus.UNVERIFIED
    if verification.status == VerificationStatus.FAILED:
        return OverallStatus.FAILED
    if verification.status == VerificationStatus.APPROXIMATE:
        return OverallStatus.APPROXIMATE
    if intent.status == IntentStatus.CONFIRMED:
        return OverallStatus.VERIFIED
    return OverallStatus.VERIFIED_WITH_ASSUMPTIONS


def verify_node(state: SolverState) -> SolverState:
    call, out = state["tool_call"], state["tool_output"]
    assert call is not None and out is not None
    verification = verify(call.tool, out)
    return {"verification": verification, "overall": overall_status(verification, state.get("intent"))}


def explain_node(state: SolverState) -> SolverState:
    call, out = state["tool_call"], state["tool_output"]
    assert call is not None and out is not None
    if state.get("explain_with_llm", True):
        text, used = explain(
            state["problem"], call.tool, out.answer_text, out.steps, state.get("verification"), state.get("intent")
        )
    else:  # the API service may add an LLM explanation outside the sandbox
        text, used = (
            template_explanation(call.tool, out.answer_text, state.get("verification"), state.get("intent")),
            False,
        )
    return {"explanation": text, "llm_used": state.get("llm_used", False) or used}


def visualize_node(state: SolverState) -> SolverState:
    call, out = state["tool_call"], state["tool_output"]
    assert call is not None and out is not None
    return {"visualization": visualize(call.tool, out), "status": "solved"}


def _after_plan(state: SolverState) -> str:
    return "end" if state.get("status") in ("unsupported", "needs_clarification") else "intent"


def _after_intent(state: SolverState) -> str:
    return "end" if state.get("status") == "needs_clarification" else "solve"


def _after_solve(state: SolverState) -> str:
    return "end" if state.get("status") == "error" else "verify"


@lru_cache(maxsize=1)
def build_graph() -> Any:
    g = StateGraph(SolverState)
    g.add_node("plan", plan_node)
    g.add_node("intent", intent_node)
    g.add_node("solve", solve_node)
    g.add_node("verify", verify_node)
    g.add_node("explain", explain_node)
    g.add_node("visualize", visualize_node)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", _after_plan, {"intent": "intent", "end": END})
    g.add_conditional_edges("intent", _after_intent, {"solve": "solve", "end": END})
    g.add_conditional_edges("solve", _after_solve, {"verify": "verify", "end": END})
    g.add_edge("verify", "explain")
    g.add_edge("explain", "visualize")
    g.add_edge("visualize", END)
    return g.compile()


def run_solver(
    problem: str,
    context: str | None = None,
    *,
    llm_result: dict[str, Any] | None = None,
    explain_with_llm: bool = True,
) -> SolverState:
    state: SolverState = {
        "problem": problem,
        "context": context,
        "llm_used": False,
        "explain_with_llm": explain_with_llm,
    }
    if llm_result is not None:
        state["llm_result"] = llm_result
    return build_graph().invoke(state)


def state_steps(state: SolverState) -> list[Step]:
    out = state.get("tool_output")
    return out.steps if out else []


def category_of(state: SolverState) -> ProblemCategory:
    call = state.get("tool_call")
    return call.category if call else ProblemCategory.UNSUPPORTED
