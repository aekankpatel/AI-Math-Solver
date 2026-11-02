"""Application service: runs the graph, builds API responses, manages sessions
and follow-up questions."""

from __future__ import annotations

import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

import sympy as sp

from app.agents.graph import SolverState, category_of, run_solver, state_steps
from app.agents.tutor import answer_followup
from app.config import get_settings
from app.schemas import OverallStatus, ProblemCategory, SolveResponse


@dataclass
class Session:
    turns: list[SolveResponse] = field(default_factory=list)
    focus_expression: str | None = None  # what "it" refers to in a follow-up
    original_expression: str | None = None

    @property
    def last_solved(self) -> SolveResponse | None:
        return next((t for t in reversed(self.turns) if t.status == "solved"), None)


class SessionStore:
    """Small thread-safe LRU store; sessions are in memory only (MVP)."""

    def __init__(self, max_sessions: int) -> None:
        self._data: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_sessions

    def get_or_create(self, session_id: str | None) -> tuple[str, Session]:
        with self._lock:
            sid = session_id or uuid.uuid4().hex
            session = self._data.pop(sid, None) or Session()
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


def _to_response(session_id: str, problem: str, state: SolverState) -> SolveResponse:
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
        llm_used=bool(state.get("llm_used")),
        planner=state.get("planner") if state.get("planner") in ("llm", "rules", "nl") else "rules",
    )


def _update_focus(session: Session, state: SolverState) -> None:
    out, call = state.get("tool_output"), state.get("tool_call")
    if not out or not call:
        return
    d = out.data
    focus: sp.Basic | None = None
    original: sp.Basic | None = d.get("expr") if isinstance(d.get("expr"), sp.Basic) else d.get("input")
    if call.tool == "differentiate":
        focus = d.get("result")
    elif call.tool == "integrate":
        focus = d.get("antiderivative") if "value" not in d else d.get("expr")
    elif call.tool in ("simplify_expression", "factor_expression", "expand_expression"):
        focus = d.get("result")
    elif call.tool in ("solve_equation", "limit"):
        focus = d.get("expr")
    if isinstance(focus, sp.Basic) and not focus.has(sp.Integral):
        session.focus_expression = sp.sstr(focus)
    if isinstance(original, sp.Basic):
        session.original_expression = sp.sstr(original)


def solve_problem(problem: str, session_id: str | None = None, context: str | None = None) -> SolveResponse:
    sid, session = get_store().get_or_create(session_id)
    state = run_solver(problem, context)
    response = _to_response(sid, problem, state)
    session.turns.append(response)
    _update_focus(session, state)
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


def handle_followup(session_id: str, question: str) -> SolveResponse:
    session = get_store().get(session_id)
    if session is None:
        raise KeyError(session_id)
    last = session.last_solved
    if _COMPUTE_WORDS.search(question) or "=" in question:
        resolved = resolve_followup(question, session)
        context = None
        if last is not None:
            context = f"Previous problem: {last.problem}\nPrevious answer: {last.answer_text}"
        state = run_solver(resolved, context)
        if state.get("status") == "solved":
            response = _to_response(session_id, resolved, state)
            session.turns.append(response)
            _update_focus(session, state)
            return response
    if last is None:
        return SolveResponse(
            session_id=session_id,
            problem=question,
            status="error",
            category=ProblemCategory.UNSUPPORTED,
            error="There is no solved problem in this session to ask about yet.",
        )
    text, used = answer_followup(question, last.problem, last.answer_text or "", last.steps, last.verification)
    response = SolveResponse(
        session_id=session_id,
        problem=question,
        status="answered",
        category=last.category,
        explanation=text,
        llm_used=used,
        planner="llm" if used else "rules",
    )
    session.turns.append(response)
    return response
