"""Computation jobs executed inside the sandbox workers (see ``app/sandbox.py``).

A job receives plain data, runs the Solver–Verifier graph without any network
access (the LLM plan, if any, was made beforehand in the API process) and returns
plain data: a ``SolveResponse`` plus the session focus expressions as strings.
No SymPy object ever leaves the worker.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

from app.agents.graph import SolverState, category_of, run_solver, state_steps
from app.schemas import OverallStatus, ProblemCategory, SolveResponse, ToolCall

MAX_ANSWER_CHARS = 20_000


def to_response(session_id: str, problem: str, state: SolverState) -> SolveResponse:
    out = state.get("tool_output")
    status = state.get("status", "error")
    return SolveResponse(
        session_id=session_id,
        problem=problem,
        status=status if status in ("solved", "unsupported", "error", "needs_clarification") else "error",
        category=category_of(state) if status != "unsupported" else ProblemCategory.UNSUPPORTED,
        tool_call=state.get("tool_call"),
        answer_text=out.answer_text if out else None,
        answer_latex=out.answer_latex if out else None,
        steps=state_steps(state),
        explanation=state.get("explanation"),
        verification=state.get("verification"),
        intent=state.get("intent"),
        overall_status=state.get("overall")
        or (OverallStatus.NEEDS_CLARIFICATION if status == "needs_clarification" else None),
        visualization=state.get("visualization"),
        error=state.get("error"),
        llm_used=state.get("planner") == "llm",
        planner=state.get("planner") if state.get("planner") in ("llm", "rules", "nl") else "rules",
    )


def focus_expressions(state: SolverState) -> tuple[str | None, str | None]:
    """(focus, original): what "it" / "the function" refer to in a follow-up question."""
    out, call = state.get("tool_output"), state.get("tool_call")
    if not out or not call:
        return None, None
    d = out.data
    focus: Any = None
    original: Any = d.get("expr") if isinstance(d.get("expr"), sp.Basic) else d.get("input")
    if call.tool == "differentiate":
        focus = d.get("result")
    elif call.tool == "integrate":
        focus = d.get("antiderivative") if "value" not in d else d.get("expr")
    elif call.tool in ("simplify_expression", "factor_expression", "expand_expression"):
        focus = d.get("result")
    elif call.tool in ("solve_equation", "limit"):
        focus = d.get("expr")
    f = sp.sstr(focus) if isinstance(focus, sp.Basic) and not focus.has(sp.Integral) else None
    o = sp.sstr(original) if isinstance(original, sp.Basic) else None
    return f, o


def solve_job(problem: str, context: str | None, llm_result: dict[str, Any]) -> dict[str, Any]:
    call = llm_result.get("call")
    if isinstance(call, dict):
        llm_result = {**llm_result, "call": ToolCall.model_validate(call)}
    state = run_solver(problem, context, llm_result=llm_result, explain_with_llm=False)
    response = to_response("", problem, state)
    if response.answer_text and len(response.answer_text) > MAX_ANSWER_CHARS:
        return {
            "response": SolveResponse(
                session_id="",
                problem=problem,
                status="error",
                category=response.category,
                tool_call=response.tool_call,
                error="The result is too large to display.",
                planner=response.planner,
                llm_used=response.llm_used,
            ),
            "focus": None,
            "original": None,
            "llm_error": state.get("llm_error"),
        }
    focus, original = focus_expressions(state)
    return {"response": response, "focus": focus, "original": original, "llm_error": state.get("llm_error")}


def _sleep_job(seconds: float) -> str:  # used by tests of the sandbox itself
    import time

    time.sleep(seconds)
    return "slept"


def _allocate_job(megabytes: int) -> int:  # used by tests of the memory limit
    block = bytearray(megabytes * 1024 * 1024)
    return len(block)


def _env_job(name: str) -> str | None:  # used by tests: workers must not see secrets
    import os

    return os.environ.get(name)


_JOBS = {"solve": solve_job, "sleep": _sleep_job, "allocate": _allocate_job, "env": _env_job}


def run_job(name: str, kwargs: dict[str, Any]) -> Any:
    return _JOBS[name](**kwargs)
