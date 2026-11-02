"""Intent verification: the plan must answer the question that was asked.

Covers natural-language ambiguity, incorrect tool selection, missing
constraints and misinterpreted parameters, plus the end-to-end rule that a
mathematically verified result is NOT "fully verified" when the question was
misread or relied on assumptions.
"""

import pytest

from app.agents import graph as graph_mod
from app.agents.intent import check_intent, extract_intent
from app.agents.nl_planner import ClarificationNeeded, plan_from_intent
from app.schemas import IntentStatus as IS
from app.schemas import OverallStatus, ProblemCategory, ToolCall, Verification, VerificationStatus


def call(tool, **args):
    return ToolCall(tool=tool, category=ProblemCategory.ALGEBRA, args=args)


def status(problem, c):
    iv, _ = check_intent(problem, c)
    return iv.status


# --- structured intent extraction ---------------------------------------------


def test_extracts_slots_from_natural_language():
    it = extract_intent("What's the second derivative of e^(3x) at x = 0?")
    assert (it.task, it.order, it.point) == ("derivative", 2, "0")
    it = extract_intent("Find the area under y = x^2 between x = 1 and x = 4")
    assert (it.task, it.lower, it.upper) == ("integral", "1", "4")
    it = extract_intent("Integrate 4x^3 over the interval [0, 2]")
    assert (it.lower, it.upper) == ("0", "2")
    it = extract_intent("What does sin(2x)/x approach as x tends to 0 from the right?")
    assert (it.task, it.limit_point, it.direction) == ("limit", "0", "+")
    it = extract_intent("Find the positive solution of x^2 = 9")
    assert (it.task, it.domain) == ("solve", "positive")


def test_probability_relations_and_named_distribution():
    it = extract_intent(
        "Waiting times are exponential with rate 0.5 per minute. What is the probability of waiting more than 4 minutes?"
    )
    assert it.distribution == "exponential"  # the named distribution beats the "per minute" (Poisson) cue
    assert extract_intent("probability of fewer than 2 emails, Poisson rate 4 per minute").relation == (
        "cdf<",
        "2",
        None,
    )
    assert extract_intent("chance of at least 3 successes in 5 binomial trials").relation == ("greater>=", "3", None)


# --- incorrect tool selection ----------------------------------------------------


@pytest.mark.parametrize(
    "problem, bad",
    [
        (
            "Determine the rank of the matrix [[1, 2], [2, 4]]",
            call("matrix_operation", operation="determinant", matrix=[[1, 2], [2, 4]]),
        ),
        ("integrate cos(x)", call("differentiate", expression="cos(x)")),
        ("factor x^2 - 9", call("expand_expression", expression="x^2 - 9")),
        ("x + y = 3, x - y = 1", call("solve_equation", equation="x + y = 3")),
        (
            "In how many ways can a committee of 3 be chosen from 10 people?",
            call("combinatorics", operation="permutations", n=10, k=3),
        ),
    ],
)
def test_wrong_tool_or_operation_is_a_mismatch(problem, bad):
    assert status(problem, bad) == IS.MISMATCH


# --- misinterpreted parameters ---------------------------------------------------------


@pytest.mark.parametrize(
    "problem, bad",
    [
        ("Integrate x^2 from 0 to 3", call("integrate", expression="x^2", lower="3", upper="0")),
        ("second derivative of x^4", call("differentiate", expression="x^4", order=1)),
        ("derivative of x^2 at x = 3", call("differentiate", expression="x^2", point="2")),
        ("limit of 1/x as x -> 0 from the right", call("limit", expression="1/x", point="0", direction="-")),
        (
            "Binomial n=10 p=0.5, P(X < 3)",
            call("probability_distribution", distribution="binomial", params={"n": 10, "p": 0.5}, query="cdf", x=3),
        ),
        (
            "normal mean 100 std 15 P(X > 130)",
            call(
                "probability_distribution", distribution="normal", params={"mean": 100, "std": 15}, query="cdf", x=130
            ),
        ),
        ("10 choose 3", call("combinatorics", operation="combinations", n=3, k=10)),
        ("mean of 2, 4, 4, 5", call("descriptive_statistics", data=[2, 4, 5])),
        ("Solve x^2 - 5x + 6 = 0", call("solve_equation", equation="x^2 - 5x - 6 = 0")),
        ("Find f''(x) for f(x) = x^3 + 2x^2", call("differentiate", expression="x^3 + 2x^2", variable="f", order=2)),
    ],
)
def test_misread_parameters_are_mismatches(problem, bad):
    assert status(problem, bad) == IS.MISMATCH


def test_already_solved_equation_is_not_the_problem():
    # "How quickly does 4t^2 change at t = 2?" once became "solve t = 2"
    assert status("How quickly does 4t^2 change at t = 2?", call("solve_equation", equation="t = 2")) == IS.MISMATCH


# --- missing constraints and ambiguity ------------------------------------------------------


def test_ignored_domain_constraint_is_repaired_by_the_workflow():
    iv, fixes = check_intent(
        "Find the positive solution of x^2 - x - 6 = 0", call("solve_equation", equation="x^2 - x - 6 = 0")
    )
    assert iv.status == IS.MISMATCH and fixes == {"domain": "positive"}
    state = graph_mod.run_solver("Find the positive solution of x^2 - x - 6 = 0")
    assert state["status"] == "solved"
    assert [str(s) for s in state["tool_output"].data["solutions"]] == ["3"]


@pytest.mark.parametrize(
    "problem, plan",
    [
        (
            "Normal distribution with mean 10: what is P(X < 12)?",
            call("probability_distribution", distribution="normal", params={"mean": 10, "std": 1}, query="cdf", x=12),
        ),
        ("Solve x + y = 10", call("solve_equation", equation="x + y = 10")),
        ("Integrate x^2 from 1", call("integrate", expression="x^2")),
        ("Find the limit of (x^2 - 1)/(x - 1)", call("limit", expression="(x^2 - 1)/(x - 1)", point="1")),
        ("Differentiate a*b + c", call("differentiate", expression="a*b + c", variable="a")),
    ],
)
def test_missing_information_requires_clarification(problem, plan):
    iv, _ = check_intent(problem, plan)
    assert iv.status in (IS.AMBIGUOUS, IS.MISMATCH)
    assert iv.clarification


@pytest.mark.parametrize(
    "problem",
    [
        "Solve x + y = 10",
        "Normal distribution with mean 10: what is P(X < 12)?",
        "Integrate x^2 from 1",
        "How many ways can 3 people be chosen from a group?",
        "What is the probability of getting 3 heads?",
        "solve x^2 > 4",
        "solve y'' + y = 0",
    ],
)
def test_pipeline_never_guesses_missing_information(problem):
    state = graph_mod.run_solver(problem)
    assert state.get("status") in ("needs_clarification", "unsupported", "error")
    assert state.get("tool_output") is None


def test_unused_numbers_make_intent_assumed_not_confirmed():
    iv, _ = check_intent("For a 2023 project, solve 4x - 3 = 13", call("solve_equation", equation="4x - 3 = 13"))
    assert iv.status == IS.ASSUMED
    assert any("2023" in a for a in iv.assumptions)


# --- end-to-end trust -------------------------------------------------------------------


def test_overall_status_combines_math_and_intent():
    v = Verification(status=VerificationStatus.VERIFIED)
    from app.schemas import IntentVerification

    confirmed = IntentVerification(status=IS.CONFIRMED)
    assumed = IntentVerification(status=IS.ASSUMED)
    mismatch = IntentVerification(status=IS.MISMATCH)
    assert graph_mod.overall_status(v, confirmed) == OverallStatus.VERIFIED
    assert graph_mod.overall_status(v, assumed) == OverallStatus.VERIFIED_WITH_ASSUMPTIONS
    assert graph_mod.overall_status(v, mismatch) == OverallStatus.NEEDS_CLARIFICATION
    assert graph_mod.overall_status(Verification(status=VerificationStatus.FAILED), confirmed) == OverallStatus.FAILED


def test_math_verified_misreading_is_never_fully_verified(monkeypatch):
    # The planner misreads "rank" as determinant; the determinant is mathematically correct,
    # but the result must not be presented as verified.
    bad = ToolCall(
        tool="matrix_operation",
        category=ProblemCategory.LINEAR_ALGEBRA,
        args={"operation": "determinant", "matrix": [[1, 2], [2, 4]]},
        source="llm",
    )
    monkeypatch.setattr(graph_mod, "plan_with_llm", lambda p, c=None: bad)
    monkeypatch.setattr(graph_mod, "plan_with_rules", lambda p: (_ for _ in ()).throw(graph_mod.PlanningError("x")))
    monkeypatch.setattr(graph_mod, "plan_from_intent", lambda p: (_ for _ in ()).throw(graph_mod.PlanningError("x")))
    state = graph_mod.run_solver("Determine the rank of the matrix [[1, 2], [2, 4]]")
    assert state["status"] == "needs_clarification"
    assert state.get("tool_output") is None


def test_misreading_falls_back_to_a_faithful_alternative(monkeypatch):
    bad = ToolCall(
        tool="matrix_operation",
        category=ProblemCategory.LINEAR_ALGEBRA,
        args={"operation": "determinant", "matrix": [[1, 2], [2, 4]]},
        source="llm",
    )
    monkeypatch.setattr(graph_mod, "plan_with_llm", lambda p, c=None: bad)
    state = graph_mod.run_solver("Determine the rank of the matrix [[1, 2], [2, 4]]")
    assert state["status"] == "solved"
    assert state["tool_call"].args["operation"] == "rank" and state["tool_output"].answer_text == "1"


def test_llm_declared_ambiguity_is_respected(monkeypatch):
    plan = ToolCall(
        tool="solve_equation",
        category=ProblemCategory.ALGEBRA,
        args={"equation": "x + y = 10"},
        ambiguities=["Two unknowns but one equation."],
        source="llm",
    )
    monkeypatch.setattr(graph_mod, "plan_with_llm", lambda p, c=None: plan)
    state = graph_mod.run_solver("Solve x + y = 10 for x")
    assert state["status"] == "needs_clarification"
    assert "Two unknowns" in state["error"]


def test_nl_planner_plans_are_capped_at_assumed():
    state = graph_mod.run_solver(
        "Weights are normal with mean 60 and standard deviation 5. What fraction weigh more than 70?"
    )
    assert state["status"] == "solved"
    assert state["intent"].status == IS.ASSUMED
    assert state["overall"] != OverallStatus.VERIFIED


def test_nl_planner_asks_instead_of_defaulting():
    with pytest.raises(ClarificationNeeded):
        plan_from_intent("Normal distribution with mean 10: what is P(X < 12)?")
    with pytest.raises(ClarificationNeeded):
        plan_from_intent("How many ways can 3 people be chosen from a group?")


def test_explanation_states_interpretation_and_assumptions():
    state = graph_mod.run_solver("For a 2023 project, solve 4x - 3 = 13")
    text = state["explanation"]
    assert "I read the question as" in text and "2023" in text and "Key idea" in text
