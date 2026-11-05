"""Application service: Groq planning under the usage budget, sandboxed computation,
API responses, sessions and follow-up questions.

Request flow for a problem:

1. Groq planner (API process, network): only if configured, within the deployment's
   rolling daily token budget, not paused after a Groq rate limit, and within this
   client's daily share. Its plan is data (a ToolCall); it never computes anything.
2. Sandbox worker (no network, no secrets, time and memory limits): offline planners,
   intent verification, the math tools, verification and plots.
3. Optional LLM tutor explanation (API process), again within the budget.

When the Groq planner is not used, the response says why (``llm_status``) and the
UI shows ``planner_notice``. Nothing about keys or raw provider errors is returned.
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from app.agents import llm as llm_mod
from app.agents.llm_planner import LLMPlanParts, LLMPlanUnsupported, plan_with_llm
from app.agents.multipart import MAX_PARTS, numbers_grounded, split_parts
from app.agents.spectral_planner import is_spectral_question
from app.agents.tutor import answer_followup, explain
from app.budget import get_budget
from app.config import get_settings
from app.math_engine.latex import normalize_latex
from app.ratelimit import ClientLimits
from app.sandbox import SandboxBusy, SandboxCrashed, SandboxTimeout, get_runner
from app.schemas import OverallStatus, ProblemCategory, SolveResponse

log = logging.getLogger(__name__)


@dataclass
class Session:
    turns: list[SolveResponse] = field(default_factory=list)
    focus_expression: str | None = None  # what "it" refers to in a follow-up
    original_expression: str | None = None

    @property
    def last_solved(self) -> SolveResponse | None:
        return next((t for t in reversed(self.turns) if t.status == "solved"), None)


MAX_TURNS_PER_SESSION = 50


class SessionStore:
    """Small thread-safe LRU store; sessions are in memory only."""

    def __init__(self, max_sessions: int) -> None:
        self._data: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_sessions

    def get_or_create(self, session_id: str | None) -> tuple[str, Session]:
        with self._lock:
            session = self._data.pop(session_id, None) if session_id else None
            # unknown ids get a fresh server-generated id (clients cannot choose session ids)
            sid = session_id if session is not None and session_id else uuid.uuid4().hex
            session = session or Session()
            self._data[sid] = session
            while len(self._data) > self._max:
                self._data.popitem(last=False)
            return sid, session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._data.get(session_id)
            if session is not None:
                self._data.move_to_end(session_id)
            return session


_store = SessionStore(get_settings().max_sessions)


def get_store() -> SessionStore:
    return _store


@lru_cache(maxsize=1)
def get_client_limits() -> ClientLimits:
    s = get_settings()
    return ClientLimits(s.rate_limit_per_minute, s.rate_limit_per_day, s.llm_calls_per_client_per_day)


def _charge(tokens: int | None) -> None:
    get_budget().record(tokens)


def _on_failure(message: str) -> None:
    _llm_failed(message)


llm_mod.set_usage_listener(_charge, _on_failure)


_NOTICES = {
    "disabled": None,  # offline mode by configuration: the badge says "offline planner", no warning
    "budget": "The AI planner has reached today's free usage budget, so the offline planner answered. "
    "It handles standard notation and common phrasings.",
    "cooldown": "The AI planner is temporarily rate-limited, so the offline planner answered.",
    "client_quota": "You have used today's share of AI planner requests, so the offline planner answered.",
    "error": "The AI planner did not return a usable plan, so the offline planner answered.",
    "rejected": "The AI planner's reading did not match the question, so the offline planner's reading was used.",
}


def llm_state() -> str:
    """available | disabled | budget | cooldown (for /api/health and the UI)."""
    if not get_settings().llm_configured:
        return "disabled"
    d = get_budget().check()
    return "available" if d.allowed else d.reason


def _gate_llm(client: str | None, calls: int = 1) -> str:
    """'ok' or the reason the LLM must not be called now."""
    s = get_settings()
    if not s.llm_configured:
        return "disabled"
    d = get_budget().check(s.llm_tokens_per_call_estimate * calls)
    if not d.allowed:
        return d.reason
    if not get_client_limits().take_llm_call(client):
        return "client_quota"
    return "ok"


def _after_llm_call() -> None:
    get_budget().observe_headers(llm_mod.rate_limits())


def _llm_failed(message: str) -> str:
    """Record a failed Groq call; returns the llm_status to report."""
    budget = get_budget()
    budget.record(None, success=False)  # unknown usage: charge the estimate (conservative)
    paused = budget.failure(message)
    return "cooldown" if paused in ("rate_limit", "quota", "errors") else "error"


def _plan_remote(problem: str, context: str | None, client: str | None) -> tuple[dict[str, Any], str]:
    """Run the Groq planner in this process. Returns (llm_result for the sandbox, llm_status)."""
    gate = _gate_llm(client)
    if gate != "ok":
        return {"call": None}, gate
    try:
        call = plan_with_llm(problem, context)
        _after_llm_call()
        return {"call": call}, "used"
    except LLMPlanUnsupported as exc:
        _after_llm_call()
        return {"call": None, "unsupported": str(exc)}, "unsupported"
    except LLMPlanParts as exc:  # a multi-part question: the parts are planned separately
        _after_llm_call()
        return {"call": None, "parts": exc.parts}, "used"
    except Exception as exc:  # provider/network/validation: offline planners take over
        _after_llm_call()
        message = f"{type(exc).__name__}: {exc}"
        status = _llm_failed(message)
        log.warning("Groq planner unavailable (%s): %s", status, message[:300])
        return {"call": None, "error": message[:700]}, status


def _error_response(problem: str, sid: str, code: str, message: str, llm_status: str) -> SolveResponse:
    return SolveResponse(
        session_id=sid,
        problem=problem,
        status="error",
        category=ProblemCategory.UNSUPPORTED,
        error=message,
        error_code=code,  # type: ignore[arg-type]
        llm_status=llm_status,  # type: ignore[arg-type]
    )


_TIMEOUT_MESSAGE = (
    "This computation was stopped because it exceeded the {limit:g}-second time limit. "
    "Very large powers, factorials or expansions can take too long; try a smaller or simpler input."
)


def _compute(problem: str, context: str | None, client: str | None) -> tuple[SolveResponse, str | None, str | None]:
    """Solve a question; a question with several parts is split and every part is solved and
    verified on its own. Returns (response, focus, original)."""
    text = normalize_latex(problem)
    if not is_spectral_question(text):  # spectral multi-part questions are answered by one tool
        parts = split_parts(text)
        if parts:
            return _compute_parts(problem, parts, "rules", context, client)
    llm_result, llm_status = _plan_remote(text, context, client)
    if llm_result.get("parts"):
        parts = [normalize_latex(p) for p in llm_result["parts"]]
        if 2 <= len(parts) <= MAX_PARTS and numbers_grounded(parts, text):
            return _compute_parts(problem, parts, "llm", context, client)
        log.info("Ignoring an LLM split whose parts are not grounded in the question")
        llm_result, llm_status = {"call": None, "error": "the proposed split changed the question"}, "rejected"
    return _compute_single(problem, context, client, llm_result, llm_status)


_TRUST = ["failed", "needs_clarification", "unverified", "approximate", "verified_with_assumptions", "verified"]


def _compute_parts(
    problem: str, parts: list[str], split_by: str, context: str | None, client: str | None
) -> tuple[SolveResponse, str | None, str | None]:
    """Solve every part through the full pipeline and combine the results."""
    results: list[tuple[SolveResponse, str | None, str | None]] = []
    for part in parts:
        llm_result, llm_status = _plan_remote(part, context, client)
        if llm_result.get("parts"):  # no recursive splitting: plan the part offline
            llm_result, llm_status = {"call": None, "error": "nested split"}, "rejected"
        results.append(_compute_single(part, context, client, llm_result, llm_status))
    responses = [r for r, _, _ in results]
    solved = [r for r in responses if r.status == "solved"]

    def trust(r: SolveResponse) -> int:
        if r.status != "solved":
            return _TRUST.index("needs_clarification")
        return _TRUST.index(r.overall_status.value) if r.overall_status else _TRUST.index("unverified")

    worst = min(responses, key=trust)
    overall = OverallStatus(_TRUST[trust(worst)])
    if len(solved) == len(responses):
        status = "solved"
    elif solved:
        status = "solved"  # partly answered: the overall badge (weakest part) says so
    elif any(r.status == "needs_clarification" for r in responses):
        status = "needs_clarification"
    else:
        status = "error"
    lines = []
    for i, r in enumerate(responses, 1):
        if r.status == "solved":
            lines.append(f"{i}. {r.answer_text}")
        else:
            lines.append(f"{i}. not answered ({(r.error or r.status)[:120]})")
    unanswered = [i for i, r in enumerate(responses, 1) if r.status != "solved"]
    planners = {r.planner for r in responses}
    notices = [r.planner_notice for r in responses if r.planner_notice]
    combined = SolveResponse(
        session_id="",
        problem=problem,
        status=status,  # type: ignore[arg-type]
        category=(solved[0] if solved else responses[0]).category,
        answer_text="; ".join(lines),
        overall_status=overall,
        error=(f"Part {', '.join(map(str, unanswered))} could not be answered; see below.") if unanswered else None,
        llm_used=any(r.llm_used for r in responses),
        planner="llm" if "llm" in planners else responses[0].planner,
        llm_status="used" if any(r.llm_status == "used" for r in responses) else responses[0].llm_status,
        planner_notice=notices[0] if notices else None,
        parts=responses,
        split_by=split_by,  # type: ignore[arg-type]
    )
    last = next(((f, o) for r, f, o in reversed(results) if r.status == "solved"), (None, None))
    return combined, last[0], last[1]


def _compute_single(
    problem: str, context: str | None, client: str | None, llm_result: dict[str, Any], llm_status: str
) -> tuple[SolveResponse, str | None, str | None]:
    """Compute one planned question in the sandbox (the Groq plan, if any, was made already)."""
    s = get_settings()
    try:
        result = get_runner().run("solve", {"problem": problem, "context": context, "llm_result": llm_result})
    except SandboxTimeout:
        msg = _TIMEOUT_MESSAGE.format(limit=s.sandbox_timeout_seconds)
        return _error_response(problem, "", "timeout", msg, llm_status), None, None
    except SandboxCrashed as exc:
        return _error_response(problem, "", "memory", str(exc), llm_status), None, None
    response: SolveResponse = result["response"]
    if response.error == "The result is too large to display.":
        response.error_code = "too_large"
    if llm_status == "used" and response.planner != "llm":
        llm_status = "rejected"
    response.llm_status = llm_status  # type: ignore[assignment]
    if response.planner != "llm":
        response.planner_notice = _NOTICES.get(llm_status)

    if s.llm_explanations and response.status == "solved" and response.answer_text:
        if _gate_llm(client) == "ok":
            text, used = explain(
                problem,
                response.tool_call.tool if response.tool_call else "",
                response.answer_text,
                response.steps,
                response.verification,
                response.intent,
            )
            _after_llm_call()
            if used:
                response.explanation = text
    response.llm_used = response.planner == "llm"
    return response, result.get("focus"), result.get("original")


def _remember(session: Session, response: SolveResponse, focus: str | None, original: str | None) -> None:
    session.turns.append(response)
    del session.turns[:-MAX_TURNS_PER_SESSION]
    if focus:
        session.focus_expression = focus
    if original:
        session.original_expression = original


def solve_problem(
    problem: str, session_id: str | None = None, context: str | None = None, client: str | None = None
) -> SolveResponse:
    """Solve one problem. Raises SandboxBusy when every computation worker is busy."""
    sid, session = get_store().get_or_create(session_id)
    response, focus, original = _compute(problem, context, client)
    response.session_id = sid
    _remember(session, response, focus, original)
    return response


_COMPUTE_WORDS = re.compile(
    r"\b(integrate|integral|antiderivative|derivative|differentiate|solve|roots?|zeros?|limit|factor|expand|"
    r"simplify|evaluate|determinant|inverse|eigen\w*|mean|median|probability|choose)\b",
    re.I,
)
_PRONOUNS = re.compile(r"\b(it|this|that|the result|the answer|the derivative|the antiderivative)\b", re.I)
_ORIGINAL = re.compile(r"\b(the original( function| expression)?|the function|f\(x\))\b", re.I)


def resolve_followup(question: str, session: Session) -> str:
    """Replace references like 'it' / 'the function' with concrete expressions."""
    text = question.strip()
    if session.original_expression:
        text = _ORIGINAL.sub(f"({session.original_expression})", text)
    if session.focus_expression:
        text = _PRONOUNS.sub(f"({session.focus_expression})", text)
    text = re.sub(r"^\s*(now|then|and|ok(ay)?|great)[,\s]+", "", text, flags=re.I)
    return text


def handle_followup(session_id: str, question: str, client: str | None = None) -> SolveResponse:
    session = get_store().get(session_id)
    if session is None:
        raise KeyError(session_id)
    last = session.last_solved
    if _COMPUTE_WORDS.search(question) or "=" in question:
        resolved = resolve_followup(question, session)
        context = None
        if last is not None:
            context = f"Previous problem: {last.problem}\nPrevious answer: {(last.answer_text or '')[:500]}"
        response, focus, original = _compute(resolved, context, client)
        if response.status == "solved" or response.error_code is not None:
            response.session_id = session_id
            _remember(session, response, focus, original)
            return response
    if last is None:
        return SolveResponse(
            session_id=session_id,
            problem=question,
            status="error",
            category=ProblemCategory.UNSUPPORTED,
            error="There is no solved problem in this session to ask about yet.",
        )
    gate = _gate_llm(client)
    if gate == "ok":
        text, used = answer_followup(question, last.problem, last.answer_text or "", last.steps, last.verification)
        _after_llm_call()
    else:
        text, used = answer_followup(
            question, last.problem, last.answer_text or "", last.steps, last.verification, use_llm=False
        )
    response = SolveResponse(
        session_id=session_id,
        problem=question,
        status="answered",
        category=last.category,
        explanation=text,
        llm_used=used,
        planner="llm" if used else "rules",
        llm_status="used" if used else (gate if gate != "ok" else "error"),  # type: ignore[arg-type]
    )
    _remember(session, response, None, None)
    return response


__all__ = ["SandboxBusy", "get_store", "handle_followup", "llm_state", "resolve_followup", "solve_problem"]
