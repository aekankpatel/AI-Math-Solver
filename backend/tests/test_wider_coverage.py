"""Questions found by a stress test of unseen phrasings: wrong-question answers must be refused,
and the newly supported forms must be answered and verified."""

import pytest

from app.agents.graph import run_solver
from app.math_engine.tools import run_tool
from app.math_engine.verifier import verify
from app.schemas import ToolCall
from app.service import solve_problem


def _solve(q):
    s = run_solver(q)
    return s, (s["tool_output"].answer_text if s.get("tool_output") else None)


# --------------------------------------------------------------------------- never answer a different question


def test_given_mean_with_a_missing_value_is_not_answered_as_summary_statistics():
    r = solve_problem("The mean of 4 numbers is 10. Three of them are 8, 9 and 12. Find the fourth.")
    assert r.status != "solved"


@pytest.mark.parametrize("q", ["Evaluate x^3 - 3x at x = 2", "What is x^2 + 1 when x = 3?"])
def test_evaluation_is_not_answered_as_root_finding(q):
    wrong = ToolCall(
        tool="solve_equation",
        category="algebra",
        args={"equation": q.split(" at ")[0].split("is ")[-1].split(" when")[0]},
    )
    s = run_solver(q, llm_result={"call": wrong}, explain_with_llm=False)
    assert s["planner"] != "llm"  # the root-finding plan is rejected
    assert s["tool_call"].tool == "evaluate_expression"


def test_area_between_curves_uses_both_curves():
    q = "Find the area between y = x^2 and y = x from 0 to 1"
    s, answer = _solve(q)
    assert answer == "1/6" and s["overall"].value == "verified"
    one_curve = ToolCall(tool="integrate", category="integrals", args={"expression": "x^2", "lower": "0", "upper": "1"})
    s = run_solver(q, llm_result={"call": one_curve}, explain_with_llm=False)
    assert s["planner"] != "llm" and s["tool_output"].answer_text == "1/6"


# --------------------------------------------------------------------------- newly supported


@pytest.mark.parametrize(
    "q, answer",
    [
        ("Evaluate x^3 - 3x at x = 2", "2"),
        ("What is x^2 + 1 when x = 3?", "10"),
        ("Solve for x: |2x - 3| = 5", "x = -1, x = 4"),
        ("Find the area between y = sin(x) and y = cos(x) from 0 to pi", "2*sqrt(2)"),
        ("If a matrix A is 2x2 with det(A) = 3, what is det(2A)?", "det(2A) = 12"),
    ],
)
def test_newly_supported_questions(q, answer):
    r = solve_problem(q)
    assert r.status == "solved" and r.answer_text == answer
    assert r.overall_status.value in ("verified", "verified_with_assumptions")


def test_trace_and_determinant_identities():
    r = solve_problem("A 3x3 matrix A has determinant 5 and trace 4. Find det(A^-1), det(3A) and tr(2A).")
    assert r.answer_text == "1. det(A^-1) = 1/5; 2. det(3A) = 135; 3. tr(2A) = 8"
    assert r.overall_status.value == "verified"
    r = solve_problem("A 3x3 matrix A has determinant 5. Find tr(A^2).")
    assert r.status != "solved"  # not determined by the facts given


@pytest.mark.parametrize(
    "expr, lower, upper, value",
    [("abs(x^2 - x)", "0", "1", "1/6"), ("abs(x^3 - x)", "-1", "1", "1/2"), ("abs(x)", "-2", "3", "13/2")],
)
def test_absolute_value_integrals_are_exact(expr, lower, upper, value):
    out = run_tool("integrate", {"expression": expr, "lower": lower, "upper": upper})
    assert out.answer_text == value and verify("integrate", out).status.value == "verified"


def test_absolute_value_integral_verifier_catches_a_wrong_value():
    import copy

    import sympy as sp

    out = run_tool("integrate", {"expression": "abs(x^2 - x)", "lower": "0", "upper": "1"})
    bad = copy.copy(out)
    bad.data = dict(out.data, value=sp.Rational(1, 5))
    assert verify("integrate", bad).status.value == "failed"


@pytest.mark.parametrize(
    "q, answers",
    [
        ("Find the inverse of the matrix [[1,2],[3,4]] and its determinant", ["[[-2, 1], [3/2, -1/2]]", "-2"]),
        ("Let g(t) = t^2 e^t. Find g'(t) and g''(t).", ["t*(t + 2)*exp(t)", "(t**2 + 4*t + 2)*exp(t)"]),
        ("Let f(x) = x^3 - 3x. Find f'(x) and f(2).", ["3*x**2 - 3", "2"]),
    ],
)
def test_more_multi_part_forms(q, answers):
    r = solve_problem(q)
    assert [p.answer_text for p in r.parts] == answers


def test_differentiate_and_simplify_is_one_task():
    r = solve_problem("Differentiate sin(x)cos(x) and simplify")
    assert r.parts is None and r.answer_text == "cos(2*x)"


# --------------------------------------------------------------------------- word problems may name their unknowns


@pytest.mark.parametrize(
    "q, tool, args, answer",
    [
        (
            "The mean of 4 numbers is 10. Three of them are 8, 9 and 12. Find the fourth.",
            "solve_equation",
            {"equation": "(8+9+12+x)/4 = 10"},
            "x = 11",
        ),
        (
            "A rectangle has perimeter 20 and area 24. Find its dimensions.",
            "solve_system",
            {"equations": ["2x + 2y = 20", "x*y = 24"]},
            "(x = 4, y = 6) or (x = 6, y = 4)",
        ),
    ],
)
def test_word_problem_equations_with_named_unknowns_are_accepted_as_assumed(q, tool, args, answer):
    s = run_solver(q, llm_result={"call": ToolCall(tool=tool, category="algebra", args=args)}, explain_with_llm=False)
    assert s["planner"] == "llm" and s["tool_output"].answer_text == answer
    assert s["overall"].value == "verified_with_assumptions"  # never "fully verified"


@pytest.mark.parametrize(
    "q, tool, args",
    [
        # precomputed: 10 is half the perimeter, worked out by the planner
        (
            "A rectangle has perimeter 20 and area 24. Find its dimensions.",
            "solve_system",
            {"equations": ["x + y = 10", "x*y = 24"]},
        ),
        # a renamed variable in a written equation is not an "unknown to name"
        ("Solve 2x + 3 = 7", "solve_equation", {"equation": "2y + 3 = 7"}),
        # helper symbols in an expression (not an equation) stay rejected
        (
            "find the area of a triangle with sides 3, 4 and 5",
            "simplify_expression",
            {"expression": "sqrt(s*(s-3)*(s-4)*(s-5))"},
        ),
    ],
)
def test_unfaithful_word_problem_plans_stay_rejected(q, tool, args):
    s = run_solver(q, llm_result={"call": ToolCall(tool=tool, category="algebra", args=args)}, explain_with_llm=False)
    # rejected: either a faithful offline plan answers instead, or the user is asked to clarify
    assert s["planner"] != "llm" or (s["status"] == "needs_clarification" and s["intent"].status.value == "mismatch")
