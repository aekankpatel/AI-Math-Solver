"""Multi-part questions whose parts need different tools: every part is solved and verified."""

import pytest

from app import service
from app.agents.llm_planner import LLMPlanParts, PlannerOutput, plan_output_to_tool_call
from app.agents.multipart import numbers_grounded, split_parts
from app.config import get_settings
from eval.dataset import CASES


@pytest.mark.parametrize(
    "problem, parts",
    [
        (
            "Let f(x) = x^2 sin(x). (a) Find f'(x). (b) Integrate f(x) from 0 to pi.",
            ["Find the derivative of x^2 sin(x).", "Integrate (x^2 sin(x)) from 0 to pi."],
        ),
        (
            "Solve x^2 - 5x + 6 = 0 and find the determinant of [[1,2],[3,4]]",
            ["Solve x^2 - 5x + 6 = 0", "Find the determinant of [[1,2],[3,4]]"],
        ),
        (
            "1. Solve 2x + 3 = 7. 2. Compute the mean of 2, 4, 6. 3. Find the limit of sin(x)/x as x -> 0.",
            ["Solve 2x + 3 = 7.", "Compute the mean of 2, 4, 6.", "Find the limit of sin(x)/x as x -> 0."],
        ),
        (
            "Find the derivative and the integral of x^2 sin(x)",
            ["Find the derivative of x^2 sin(x)", "Find the integral of x^2 sin(x)"],
        ),
        ("Solve 2x + 3 = 7. Then compute the mean of 2, 4, 6.", ["Solve 2x + 3 = 7", "Compute the mean of 2, 4, 6"]),
        (
            "Differentiate x^3 sin(x) and then integrate it from 0 to 1",
            ["Differentiate x^3 sin(x)", "Integrate (x^3 sin(x)) from 0 to 1"],
        ),
        (
            "A fair coin is tossed 6 times. (a) Find the probability of exactly 2 heads. (b) Find the probability of at most 2 heads.",
            [
                "A fair coin is tossed 6 times. Find the probability of exactly 2 heads.",
                "A fair coin is tossed 6 times. Find the probability of at most 2 heads.",
            ],
        ),
    ],
)
def test_split_into_self_contained_parts(problem, parts):
    assert split_parts(problem) == parts


@pytest.mark.parametrize(
    "problem",
    [
        "mean and standard deviation of 2, 4, 4, 4, 5, 5, 7, 9",
        "Solve the system: 2x + y = 5; x - y = 1",
        "x + 2y = 7, 3x - y = 7",
        "Find the derivative of x^3 sin(x)",
        "normal mean 100 std 15, P(X > 130)",
        "If x = 2.5 and y = 3, solve 2x + y = z for z",
    ],
)
def test_single_task_questions_are_not_split(problem):
    assert split_parts(problem) is None


def test_no_benchmark_question_is_split():
    assert [c["id"] for c in CASES if split_parts(c["problem"])] == []


def test_every_part_is_answered_and_verified():
    r = service.solve_problem(
        "1. Solve 2x + 3 = 7. 2. Compute the mean of 2, 4, 6. 3. Find the determinant of [[4,7],[2,6]]."
    )
    assert r.status == "solved" and len(r.parts) == 3 and r.split_by == "rules"
    assert [p.status for p in r.parts] == ["solved"] * 3
    assert r.parts[0].answer_text == "x = 2" and r.parts[2].answer_text == "10"
    assert r.parts[1].tool_call.tool == "descriptive_statistics"
    assert r.overall_status.value == "verified"
    assert r.answer_text.startswith("1. x = 2; 2. mean = 4")


def test_overall_status_is_the_weakest_part():
    r = service.solve_problem("Solve x^2 - 5x + 6 = 0 and find the limit of sin(x)/x as x -> 0")
    assert [p.overall_status.value for p in r.parts] == ["verified", "approximate"]
    assert r.overall_status.value == "approximate"


def test_an_unanswerable_part_is_reported_not_hidden():
    r = service.solve_problem("(a) Solve 2x + 3 = 7. (b) Write a poem about matrices.")
    assert r.parts[0].status == "solved" and r.parts[1].status != "solved"
    assert r.overall_status.value == "needs_clarification"  # never shown as verified
    assert "Part 2 could not be answered" in r.error
    assert "2. not answered" in r.answer_text


def test_follow_up_refers_to_the_last_answered_part():
    r = service.solve_problem("Solve x^2 - 4 = 0 and find the derivative of x^3")
    follow = service.handle_followup(r.session_id, "now integrate it")
    # as for single questions, "it" after a derivative is the derivative's result (3x^2)
    assert follow.status == "solved" and follow.answer_text == "x**3 + C"


def test_spectral_questions_stay_one_tool():
    q = "Let A be a 3x3 matrix with eigenvalues 1, -2 and 3. 1. Find det(A^3 - 2A). 2. Find the trace of A^2."
    r = service.solve_problem(q)
    assert r.parts is None and r.tool_call.tool == "matrix_spectrum"
    assert r.answer_text == "1. det(A^3 - 2A) = 84; 2. tr(A^2) = 14"


# --------------------------------------------------------------------------- splits proposed by Groq


@pytest.fixture
def llm_on(monkeypatch):
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_unitTestNotARealKey000")
    get_settings.cache_clear()
    from app.budget import get_budget

    get_budget.cache_clear()
    service.get_client_limits.cache_clear()


Q_WORDS = "Work out how fast x^3 grows, and the area under x^2 between 0 and 1"


def test_grounded_llm_split_is_used(llm_on, monkeypatch):
    def planner(problem, context=None):
        if problem == Q_WORDS:
            raise LLMPlanParts(["Find the derivative of x^3", "Integrate x^2 from 0 to 1"])
        raise RuntimeError("plan the parts offline")

    monkeypatch.setattr(service, "plan_with_llm", planner)
    r = service.solve_problem(Q_WORDS)
    assert r.split_by == "llm" and [p.answer_text for p in r.parts] == ["3*x**2", "1/3"]


def test_llm_split_that_changes_numbers_is_rejected(llm_on, monkeypatch):
    def planner(problem, context=None):
        if problem == Q_WORDS:
            raise LLMPlanParts(["Find the derivative of x^4", "Integrate x^2 from 0 to 5"])
        raise RuntimeError("offline")

    monkeypatch.setattr(service, "plan_with_llm", planner)
    r = service.solve_problem(Q_WORDS)
    assert r.parts is None and r.split_by is None and r.status != "solved"


def test_numbers_grounded():
    assert numbers_grounded(["Integrate x^2 from 0 to 1"], "area under x^2 between 0 and 1")
    assert not numbers_grounded(["Integrate x^2 from 0 to 5"], "area under x^2 between 0 and 1")


def test_planner_schema_parts():
    out = PlannerOutput(tool="none", parts=["Find the derivative of x^3", "Integrate x^2 from 0 to 1"])
    with pytest.raises(LLMPlanParts) as exc:
        plan_output_to_tool_call(out)
    assert exc.value.parts == ["Find the derivative of x^3", "Integrate x^2 from 0 to 1"]
