"""Provenance of plan arguments: the planner must not do the math itself.

Found by the real Groq benchmark: for "find the area of a triangle with sides 3,
4 and 5" the LLM applied Heron's formula itself and passed `sqrt(36)` to the
simplify tool; the result was reported as "verified with assumptions" although
the math engine only checked sqrt(36) = 6.

Rule (app/agents/intent.py):
1. math equivalent to math written in the question -> accepted (any valid transformation,
   including ones that introduce new constants, is proven by numeric equivalence)
2. math reconstructed from words -> accepted as an *assumption* only if every number and symbol
   is traceable to the question (digits, number words, structural constants) and it is not a constant
3. anything else -> rejected (fallback to a faithful offline plan, otherwise clarification)
"""

import pytest

from app.agents import graph as graph_mod
from app.agents import llm_planner
from app.agents.intent import check_intent
from app.schemas import IntentStatus as IS
from app.schemas import OverallStatus, ProblemCategory, ToolCall, VerificationStatus

TRIANGLE = "find the area of a triangle with sides 3, 4 and 5"


def status(problem, tool, **args):
    category = ProblemCategory(llm_planner._TOOL_CATEGORY[tool])  # a consistent category, as the planner derives it
    iv, fixes = check_intent(problem, ToolCall(tool=tool, category=category, args=args, source="llm"))
    return iv.status, iv.clarification or "", fixes


# --- the triangle-area case and other precomputed answers ------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "sqrt(36)",  # what gpt-oss-120b actually sent
        "sqrt(6*(6-3)*(6-4)*(6-5))",  # Heron with the semi-perimeter already computed
        "6",
        "3*4/2",  # right-triangle shortcut, computed by the planner
        "sqrt(s*(s-3)*(s-4)*(s-5))",  # undefined symbol s
    ],
)
def test_triangle_area_plans_are_rejected(expression):
    st, why, fixes = status(TRIANGLE, "simplify_expression", expression=expression)
    assert st == IS.MISMATCH and fixes == {}
    assert "math engine" in why or "symbol" in why


@pytest.mark.parametrize(
    "problem, tool, args",
    [
        ("Integrate x^2 from 0 to 3", "simplify_expression", {"expression": "9"}),
        ("derivative of x^3 at x = 2", "simplify_expression", {"expression": "3*2^2"}),
        ("Binomial n=10 p=0.5, P(X=3)", "simplify_expression", {"expression": "120/1024"}),
        ("10 choose 3", "simplify_expression", {"expression": "10*9*8/6"}),
        ("Find the determinant of [[1,2],[3,4]]", "simplify_expression", {"expression": "1*4 - 2*3"}),
        ("What is the mean of 2, 4 and 9?", "simplify_expression", {"expression": "15/3"}),
    ],
)
def test_planner_precomputed_answers_are_rejected(problem, tool, args):
    st, _, _ = status(problem, tool, **args)
    assert st == IS.MISMATCH


# --- incorrect (or merely unverifiable) intermediate calculations ------------------------


WORDS = "Three times a number x plus 2 is 11. Find x."


@pytest.mark.parametrize("equation", ["3*x = 8", "3*x = 9", "x = 3", "3*x = 13"])
def test_intermediate_calculations_in_word_problems_are_rejected(equation):
    # 3x = 8 and 3x = 13 are wrong; 3x = 9 is right, but the subtraction was done by the planner
    # and cannot be verified by the math engine.
    st, why, _ = status(WORDS, "solve_equation", equation=equation)
    assert st == IS.MISMATCH, why


def test_faithful_word_translation_is_accepted_as_an_assumption():
    st, _, _ = status(WORDS, "solve_equation", equation="3*x + 2 = 11")
    assert st == IS.ASSUMED


# --- valid transformations (may introduce constants) ------------------------------------


@pytest.mark.parametrize(
    "problem, tool, args",
    [
        ("Solve for y: 2y/3 + 4 = 10", "solve_equation", {"equation": "2*y + 12 = 30", "variable": "y"}),
        ("simplify 2(x + 3) - 4x", "simplify_expression", {"expression": "6 - 2*x"}),
        ("simplify sqrt(50)", "simplify_expression", {"expression": "5*sqrt(2)"}),
        ("Solve x^2 - 5x + 6 = 0", "solve_equation", {"equation": "2*x^2 - 10*x + 12 = 0"}),
        ("d/dx e^(2x)", "differentiate", {"expression": "exp(2*x)"}),
        ("differentiate sqrt(x)", "differentiate", {"expression": "x**(1/2)"}),
        ("integrate 1/(1 + x^2)", "integrate", {"expression": "(x**2 + 1)**(-1)"}),
    ],
)
def test_equivalent_rewrites_are_confirmed(problem, tool, args):
    st, why, _ = status(problem, tool, **args)
    assert st == IS.CONFIRMED, why


@pytest.mark.parametrize(
    "problem, args",
    [
        ("What values of x satisfy x squared minus 9 equals 0?", {"equation": "x^2 - 9 = 0"}),
        ("70% of a number x is 14. Find x.", {"equation": "0.7*x = 14"}),
        ("70% of a number x is 14. Find x.", {"equation": "70/100*x = 14"}),
        ("Half of a number x is 7. Find x.", {"equation": "x/2 = 7"}),
    ],
)
def test_translations_with_structural_constants_are_assumed(problem, args):
    st, why, _ = status(problem, "solve_equation", **args)
    assert st == IS.ASSUMED, why


# --- adversarial plans -------------------------------------------------------------------


@pytest.mark.parametrize(
    "problem, tool, args",
    [
        # right numbers, wrong structure
        ("Solve x^2 - 5x + 6 = 0", "solve_equation", {"equation": "x^2 - 6x + 5 = 0"}),
        # a correct constant smuggled in place of the variable expression
        ("Find the derivative of x^3 sin(x)", "differentiate", {"expression": "x^3*sin(1)"}),
        # pre-solved equation passed as the problem
        ("solve 2x + 3 = 11", "solve_equation", {"equation": "x = 4"}),
        # invented symbol to dodge the constant check
        (TRIANGLE, "solve_equation", {"equation": "A = sqrt(36)", "variable": "A"}),
        # number laundering through a word problem
        ("A number x doubled is 10. Find x.", "solve_equation", {"equation": "x = 10/2"}),
    ],
)
def test_adversarial_plans_are_rejected(problem, tool, args):
    st, why, _ = status(problem, tool, **args)
    assert st == IS.MISMATCH, why


def test_persuasive_self_report_cannot_rescue_a_precomputed_plan():
    call = ToolCall(
        tool="simplify_expression",
        category=ProblemCategory.ALGEBRA,
        args={"expression": "sqrt(36)"},
        source="llm",
        interpretation="Area of the triangle by Heron's formula, sqrt(s(s-a)(s-b)(s-c)).",
        rationale="Heron's formula with s = 6.",
    )
    iv, _ = check_intent(TRIANGLE, call)
    assert iv.status == IS.MISMATCH


# --- end to end: unsupported operations are never shown as verified ------------------------


class _Model:
    model_name = "fake"

    def __init__(self, out):
        self.out = out

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        return {"raw": None, "parsed": llm_planner.PlannerOutput.model_validate(self.out), "parsing_error": None}


def test_triangle_area_through_the_pipeline_is_not_verified(monkeypatch):
    out = {
        "tool": "simplify_expression",
        "category": "algebra",
        "expression": "sqrt(36)",
        "interpretation": "Area via Heron's formula.",
    }
    monkeypatch.setattr(llm_planner, "get_chat_model", lambda: _Model(out))
    state = graph_mod.run_solver(TRIANGLE)
    assert state.get("tool_output") is None  # nothing computed, nothing claimed
    assert state["status"] in ("needs_clarification", "unsupported")
    assert state.get("overall") in (None, OverallStatus.NEEDS_CLARIFICATION)


def test_valid_llm_rewrite_still_runs_and_verifies(monkeypatch):
    out = {"tool": "solve_equation", "equation": "2*y + 12 = 30", "variable": "y"}
    monkeypatch.setattr(llm_planner, "get_chat_model", lambda: _Model(out))
    state = graph_mod.run_solver("Solve for y: 2y/3 + 4 = 10")
    assert state["status"] == "solved" and state["planner"] == "llm"
    assert state["tool_output"].answer_text == "y = 9"
    assert state["verification"].status == VerificationStatus.VERIFIED
    assert state["overall"] == OverallStatus.VERIFIED


def test_precomputed_llm_plan_falls_back_to_a_faithful_offline_plan(monkeypatch):
    out = {"tool": "simplify_expression", "expression": "9"}  # the LLM "integrated" by itself
    monkeypatch.setattr(llm_planner, "get_chat_model", lambda: _Model(out))
    state = graph_mod.run_solver("Integrate x^2 from 0 to 3")
    assert state["status"] == "solved" and state["planner"] != "llm"
    assert state["tool_call"].tool == "integrate" and state["tool_output"].answer_text == "9"
