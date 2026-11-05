"""Sums, sequences, critical points, percentages and probability word problems; readable explanations."""

import copy

import pytest
import sympy as sp

from app.agents.graph import run_solver
from app.math_engine.tools import run_tool
from app.math_engine.verifier import verify
from app.schemas import ToolCall
from app.service import solve_problem


@pytest.mark.parametrize(
    "q, answer, trust",
    [
        ("Compute the sum of the first 100 positive integers", "5050", "verified"),
        ("Find the sum of the first 10 odd numbers", "100", "verified"),
        ("What is the sum of the first 50 even numbers?", "2550", "verified"),
        ("The sum of the first 12 cubes", "6084", "verified"),
        ("sum of k^2 for k = 1 to 10", "385", "verified"),
        ("Evaluate the sum of 2^k for k = 0 to 10", "2047", "verified"),
        ("Find the sum of (1/2)^k from k = 1 to infinity", "1", "approximate"),
        ("Find the 10th term of the arithmetic sequence 3, 7, 11, ...", "a_10 = 39 (arithmetic)", "verified"),
        ("What is the 6th term of the geometric sequence 3, -6, 12, ...?", "a_6 = -96 (geometric)", "verified"),
        ("Sum of the first 10 terms of 4, 9, 14, 19, ...", "S_10 = 265 (arithmetic)", "verified"),
        (
            "Find the critical points of f(x) = x^3 - 3x",
            "x = -1: local maximum, f = 2; x = 1: local minimum, f = -2",
            "verified",
        ),
        ("Find the turning points of y = x^2 - 6x + 5", "x = 3: local minimum, f = -4", "verified"),
        ("What is 15% of 240?", "36", "verified"),
        ("12 is what percent of 48?", "25%", "verified"),
        ("Decrease 400 by 15%", "340", "verified"),
        ("Find the percentage change from 80 to 120", "50%", "verified"),
        (
            "What is the probability of getting at least one six in 4 rolls of a fair die?",
            "P(X ≥ 1) = 671/1296 ≈ 0.517747",
            None,
        ),
        ("A coin is flipped 8 times. Probability of at most 2 heads?", "P(X ≤ 2) = 37/256 ≈ 0.144531", None),
        (
            "A bag has 5 red and 3 blue balls. Two are drawn without replacement. Probability both are red?",
            "P(X = 2) = 5/14 ≈ 0.357143",
            None,
        ),
        (
            "An urn has 6 green and 4 yellow balls. 3 balls are drawn without replacement. Probability that exactly 2 are green?",
            "P(X = 2) = 1/2 ≈ 0.5",
            None,
        ),
        ("A die is rolled 5 times. Probability of no sixes?", "P(X = 0) = 3125/7776 ≈ 0.401878", None),
    ],
)
def test_new_question_types(q, answer, trust):
    r = solve_problem(q)
    assert r.status == "solved" and r.answer_text == answer
    if trust:
        assert r.overall_status.value == trust
    else:
        assert r.overall_status.value in ("verified", "verified_with_assumptions")


def test_70_percent_of_a_number_is_still_an_equation():
    # an equation with an unknown is not a percentage calculation; a faithful equation plan solves it
    r = solve_problem("70% of a number x is 14. Find x.")
    assert r.tool_call is None or r.tool_call.tool != "percentage"
    plan = ToolCall(tool="solve_equation", category="algebra", args={"equation": "70/100 * x = 14"})
    s = run_solver("70% of a number x is 14. Find x.", llm_result={"call": plan}, explain_with_llm=False)
    assert s["tool_output"].answer_text == "x = 20"


@pytest.mark.parametrize(
    "tool, args, mutate",
    [
        ("summation", {"expression": "k", "lower": "1", "upper": "100", "variable": "k"}, {"value": sp.Integer(5051)}),
        (
            "summation",
            {"expression": "k^2", "lower": "1", "upper": "n", "variable": "k"},
            {"value": sp.Symbol("n") ** 3 / 3},
        ),
        ("sequence", {"terms": [3, 7, 11], "n": 10}, {"value": sp.Integer(40)}),
        ("percentage", {"operation": "of", "value": 15, "whole": 240}, {"result": sp.Integer(35)}),
        (
            "critical_points",
            {"expression": "x^3 - 3x"},
            {
                "points": [
                    (sp.Integer(-1), "local minimum", sp.Integer(2)),
                    (sp.Integer(1), "local minimum", sp.Integer(-2)),
                ]
            },
        ),
        ("critical_points", {"expression": "x^3 - 3x"}, {"points": [(sp.Integer(1), "local minimum", sp.Integer(-2))]}),
    ],
)
def test_verifiers_catch_wrong_results(tool, args, mutate):
    out = run_tool(tool, args)
    assert verify(tool, out).status.value == "verified"
    bad = copy.copy(out)
    bad.data = {**out.data, **mutate}
    assert verify(tool, bad).status.value == "failed"


def test_hypergeometric_matches_scipy_and_rejects_impossible_draws():
    out = run_tool(
        "probability_distribution",
        {"distribution": "hypergeometric", "params": {"success": 5, "failure": 3, "draws": 2}, "query": "pmf", "x": 2},
    )
    assert (
        out.data["value"] == sp.Rational(5, 14) and verify("probability_distribution", out).status.value == "verified"
    )
    with pytest.raises(Exception, match="no more draws"):
        run_tool(
            "probability_distribution",
            {
                "distribution": "hypergeometric",
                "params": {"success": 1, "failure": 1, "draws": 5},
                "query": "pmf",
                "x": 1,
            },
        )


def test_percentage_plan_of_the_wrong_kind_is_corrected():
    wrong = ToolCall(tool="percentage", category="algebra", args={"operation": "of", "value": "20", "whole": "80"})
    s = run_solver("What percent of 80 is 20?", llm_result={"call": wrong}, explain_with_llm=False)
    assert s["tool_output"].answer_text == "25%"


def test_sum_plan_with_the_wrong_terms_is_corrected():
    wrong = ToolCall(
        tool="summation", category="algebra", args={"expression": "k^2", "lower": "1", "upper": "100", "variable": "k"}
    )
    s = run_solver(
        "Compute the sum of the first 100 positive integers", llm_result={"call": wrong}, explain_with_llm=False
    )
    assert s["tool_output"].answer_text == "5050"


# --------------------------------------------------------------------------- readable explanations


def test_written_explanation_uses_typeset_maths():
    s = run_solver("Find the derivative of x^3 sin(x) at x = 1")
    assert "$" in s["explanation"] and "x**2" not in s["explanation"]
