"""Tutor: turns verified results into explanations and answers follow-ups.

The LLM (when configured) is given the *already computed and verified* answer
and the deterministic steps, and is told to explain them without introducing
new numbers. Without an LLM, template explanations are used.
"""

from __future__ import annotations

import logging
import re

from app.agents.llm import charge_usage, get_chat_model, report_failure
from app.schemas import IntentStatus, IntentVerification, Step, Verification, VerificationStatus

log = logging.getLogger(__name__)


def normalize_math_delimiters(text: str) -> str:
    """LLMs often write \\[...\\] / \\(...\\); the UI renders $$...$$ / $...$, so convert them."""
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: "$$" + m.group(1).strip() + "$$", text, flags=re.S)
    text = re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m.group(1).strip() + "$", text, flags=re.S)
    return text


TUTOR_SYSTEM_PROMPT = """You are the tutor in Mathenza. You receive a math problem, the
answer computed by a computer algebra system, the solution steps, and an independent
verification report. Explain the solution clearly for a student in 3-6 short sentences or a
compact list. Rules:
- Do NOT recompute or change the answer and do NOT introduce numbers that are not in the
  provided material. The computed answer is authoritative.
- Start by stating how the question was interpreted (one sentence) and mention any listed
  assumptions, so the student can spot a misreading.
- Explain WHY the method works, not only what was done; point out the key idea or common
  mistake for this kind of problem in one sentence.
- If verification status is not "verified", say plainly how confident the result is.
- Use plain text with inline LaTeX between $...$ (display math between $$...$$) when helpful.
  Never use \\( \\) or \\[ \\] delimiters. No headings."""

FOLLOWUP_SYSTEM_PROMPT = """You are the tutor in Mathenza answering a follow-up question about
a solved problem. Base your answer only on the provided problem, computed answer, steps and
verification. Do not perform new arithmetic: if the question requires a new computation, say
that it should be submitted as a new problem and suggest the exact phrasing. Be concise.
Use inline LaTeX between $...$ (display math between $$...$$) when helpful; never use \\( \\) or \\[ \\]
delimiters."""

_STATUS_SENTENCE = {
    VerificationStatus.VERIFIED: "The result passed independent verification (an exact check plus independent numeric checks).",
    VerificationStatus.APPROXIMATE: "The result was confirmed numerically, but not proven exactly, so treat it as a high-precision approximation.",
    VerificationStatus.UNVERIFIED: "No independent check was possible, so the result is unverified.",
    VerificationStatus.FAILED: "Warning: the independent check disagreed with the solver, so this result should not be trusted.",
}


_INTENT_SENTENCE = {
    IntentStatus.CONFIRMED: "Every number and condition in the question was matched to the computation.",
    IntentStatus.ASSUMED: "The computation is consistent with the question, but relies on the assumptions above.",
}

_KEY_IDEAS = {
    "solve_equation": "Key idea: a solution must make both sides equal, so every root is checked by substituting it back.",
    "solve_system": "Key idea: a solution of a system must satisfy every equation at the same time.",
    "simplify_expression": "Key idea: simplifying never changes the value, so the result must agree with the original "
    "wherever both are defined.",
    "factor_expression": "Key idea: factoring rewrites a sum as a product; multiplying the factors back must give the original.",
    "expand_expression": "Key idea: expanding distributes every product; collecting like terms gives the polynomial form.",
    "differentiate": "Key idea: the derivative measures the instantaneous rate of change; for products and compositions "
    "remember the product and chain rules.",
    "integrate": "Key idea: integration reverses differentiation, so differentiating the antiderivative must return "
    "the integrand. A definite integral is F(b) − F(a).",
    "limit": "Key idea: a limit describes the value the function approaches, even when it is undefined at the point.",
    "matrix_operation": "Key idea: matrix results can be checked directly, e.g. A·A⁻¹ = I or A·x = b.",
    "descriptive_statistics": "Key idea: the sample standard deviation divides by n − 1, the population one by n.",
    "probability_distribution": "Key idea: for discrete distributions, 'fewer than k' means P(X ≤ k − 1); "
    "for continuous ones P(X < k) = P(X ≤ k).",
    "combinatorics": "Key idea: use permutations when order matters and combinations when it does not.",
}


def _material(
    problem: str,
    answer: str,
    steps: list[Step],
    verification: Verification | None,
    intent: IntentVerification | None = None,
) -> str:
    lines = [f"Problem: {problem}"]
    if intent:
        lines.append(f"Interpretation: {intent.interpretation} (intent check: {intent.status.value})")
        lines += [f"Assumption: {a}" for a in intent.assumptions]
    lines += [f"Computed answer: {answer}", "Steps:"]
    for i, s in enumerate(steps, 1):
        lines.append(f"{i}. {s.title}. {s.detail} {s.latex or ''}".strip())
    if verification:
        lines.append(f"Verification status: {verification.status.value}. {verification.summary}")
        for c in verification.checks:
            lines.append(f"- {c.method}: {'pass' if c.passed else 'FAIL'} {c.detail}")
    return "\n".join(lines)


def template_explanation(
    tool: str, answer: str, verification: Verification | None, intent: IntentVerification | None = None
) -> str:
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
        "summation": "We found a closed form for the sum (or added the terms) and checked it independently.",
        "sequence": "We identified the common difference or ratio and applied the formula for the n-th term or the sum.",
        "critical_points": "We set the derivative to zero, solved, and classified each point with the second "
        "derivative (or the sign of the first derivative).",
        "percentage": "A percentage is a fraction of 100, so we converted it and multiplied.",
        "evaluate_expression": "We substituted the value into the expression exactly.",
        "matrix_spectrum": "We used the eigenvalues: for a function of the matrix, they are the function of the "
        "eigenvalues, so the trace is their sum and the determinant their product.",
    }.get(tool, "We computed the result with a computer algebra system.")
    parts = []
    if intent and intent.interpretation:
        parts.append(f"I read the question as: {intent.interpretation}")
        if intent.assumptions:
            parts.append("Assumptions: " + " ".join(a.rstrip(".") + "." for a in intent.assumptions))
    parts += [intro, f"Result: {answer}."]
    if tool in _KEY_IDEAS:
        parts.append(_KEY_IDEAS[tool])
    if verification:
        parts.append(_STATUS_SENTENCE[verification.status])
    if intent and intent.status in _INTENT_SENTENCE:
        parts.append(_INTENT_SENTENCE[intent.status])
    return " ".join(parts).strip()


def explain(
    problem: str,
    tool: str,
    answer: str,
    steps: list[Step],
    verification: Verification | None,
    intent: IntentVerification | None = None,
) -> tuple[str, bool]:
    """Return (explanation, llm_used)."""
    model = get_chat_model()
    if model is not None:
        try:
            msg = model.invoke(
                [("system", TUTOR_SYSTEM_PROMPT), ("human", _material(problem, answer, steps, verification, intent))]
            )
            charge_usage(msg)
            text = normalize_math_delimiters(str(getattr(msg, "content", "")).strip())
            if text:
                return text, True
        except Exception as exc:  # network/provider errors fall back to templates
            report_failure(exc)
            log.warning("LLM explanation failed: %s", type(exc).__name__)
    return template_explanation(tool, answer, verification, intent), False


def answer_followup(
    question: str,
    problem: str,
    answer: str,
    steps: list[Step],
    verification: Verification | None,
    use_llm: bool = True,
) -> tuple[str, bool]:
    model = get_chat_model() if use_llm else None
    if model is not None:
        try:
            msg = model.invoke(
                [
                    ("system", FOLLOWUP_SYSTEM_PROMPT),
                    ("human", _material(problem, answer, steps, verification) + f"\n\nFollow-up question: {question}"),
                ]
            )
            charge_usage(msg)
            text = normalize_math_delimiters(str(getattr(msg, "content", "")).strip())
            if text:
                return text, True
        except Exception as exc:
            report_failure(exc)
            log.warning("LLM follow-up failed: %s", type(exc).__name__)
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
