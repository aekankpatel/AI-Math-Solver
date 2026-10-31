"""Tutor: turns verified results into explanations and answers follow-ups.

The LLM (when configured) is given the *already computed and verified* answer
and the deterministic steps, and is told to explain them without introducing
new numbers. Without an LLM, template explanations are used.
"""

from __future__ import annotations

import logging

from app.agents.llm import get_chat_model
from app.schemas import Step, Verification, VerificationStatus

log = logging.getLogger(__name__)

TUTOR_SYSTEM_PROMPT = """You are the tutor in MathForge AI. You receive a math problem, the
answer computed by a computer algebra system, the solution steps, and an independent
verification report. Explain the solution clearly for a student in 3-6 short sentences or a
compact list. Rules:
- Do NOT recompute or change the answer and do NOT introduce numbers that are not in the
  provided material. The computed answer is authoritative.
- If verification status is not "verified", say plainly how confident the result is.
- Use plain text with inline LaTeX between $...$ when helpful. No headings."""

FOLLOWUP_SYSTEM_PROMPT = """You are the tutor in MathForge AI answering a follow-up question about
a solved problem. Base your answer only on the provided problem, computed answer, steps and
verification. Do not perform new arithmetic: if the question requires a new computation, say
that it should be submitted as a new problem and suggest the exact phrasing. Be concise.
Use inline LaTeX between $...$ when helpful."""

_STATUS_SENTENCE = {
    VerificationStatus.VERIFIED: "The result passed independent verification (an exact check plus independent numeric checks).",
    VerificationStatus.APPROXIMATE: "The result was confirmed numerically, but not proven exactly, so treat it as a high-precision approximation.",
    VerificationStatus.UNVERIFIED: "No independent check was possible, so the result is unverified.",
    VerificationStatus.FAILED: "Warning: the independent check disagreed with the solver, so this result should not be trusted.",
}


def _material(problem: str, answer: str, steps: list[Step], verification: Verification | None) -> str:
    lines = [f"Problem: {problem}", f"Computed answer: {answer}", "Steps:"]
    for i, s in enumerate(steps, 1):
        lines.append(f"{i}. {s.title}. {s.detail} {s.latex or ''}".strip())
    if verification:
        lines.append(f"Verification status: {verification.status.value}. {verification.summary}")
        for c in verification.checks:
            lines.append(f"- {c.method}: {'pass' if c.passed else 'FAIL'} {c.detail}")
    return "\n".join(lines)


def template_explanation(tool: str, answer: str, verification: Verification | None) -> str:
    intro = {
        "solve_equation": "We rewrote the equation as f = 0 and found every value of the variable that makes it true.",
        "solve_system": "We solved the equations simultaneously by eliminating variables.",
        "simplify_expression": "We rewrote the expression in a simpler equivalent form.",
        "factor_expression": "We wrote the expression as a product of irreducible factors.",
        "expand_expression": "We multiplied out all products and collected like terms.",
        "differentiate": "We applied the standard differentiation rules term by term.",
        "integrate": "We found an antiderivative and, for definite integrals, applied the Fundamental Theorem of Calculus.",
        "limit": "We examined the behaviour of the function as the variable approaches the point.",
        "matrix_operation": "We carried out the requested matrix computation with exact rational arithmetic.",
        "descriptive_statistics": "We computed the standard summary statistics of the data set.",
        "probability_distribution": "We used the distribution's probability function to compute the requested probability.",
        "combinatorics": "We applied the standard counting formula.",
    }.get(tool, "We computed the result with a computer algebra system.")
    status = _STATUS_SENTENCE[verification.status] if verification else ""
    return f"{intro} Result: {answer}. {status}".strip()


def explain(
    problem: str, tool: str, answer: str, steps: list[Step], verification: Verification | None
) -> tuple[str, bool]:
    """Return (explanation, llm_used)."""
    model = get_chat_model()
    if model is not None:
        try:
            msg = model.invoke(
                [("system", TUTOR_SYSTEM_PROMPT), ("human", _material(problem, answer, steps, verification))]
            )
            text = str(getattr(msg, "content", "")).strip()
            if text:
                return text, True
        except Exception as exc:  # network/provider errors fall back to templates
            log.warning("LLM explanation failed: %s", exc)
    return template_explanation(tool, answer, verification), False


def answer_followup(
    question: str, problem: str, answer: str, steps: list[Step], verification: Verification | None
) -> tuple[str, bool]:
    model = get_chat_model()
    if model is not None:
        try:
            msg = model.invoke(
                [
                    ("system", FOLLOWUP_SYSTEM_PROMPT),
                    ("human", _material(problem, answer, steps, verification) + f"\n\nFollow-up question: {question}"),
                ]
            )
            text = str(getattr(msg, "content", "")).strip()
            if text:
                return text, True
        except Exception as exc:
            log.warning("LLM follow-up failed: %s", exc)
    q = question.lower()
    if any(w in q for w in ("verif", "sure", "correct", "trust", "check")):
        detail = "; ".join(
            f"{c.method}: {'passed' if c.passed else 'failed'}" for c in (verification.checks if verification else [])
        )
        status = _STATUS_SENTENCE[verification.status] if verification else "No verification is available."
        return f"{status} Checks performed: {detail or 'none'}.", False
    if "step" in q or "why" in q or "how" in q or "explain" in q:
        lines = [f"{i}. {s.title}" + (f": {s.detail}" if s.detail else "") for i, s in enumerate(steps, 1)]
        return "Here is how the solution proceeds:\n" + "\n".join(lines), False
    return (
        "Without an LLM configured I can answer questions about the steps or the verification, or solve a "
        "new related problem. Try e.g. 'now integrate it', 'explain the steps' or 'how was this verified?'."
    ), False
