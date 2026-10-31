import pytest

from app.agents.rule_planner import PlanningError, plan_with_rules


@pytest.mark.parametrize(
    "problem, tool, expected_args",
    [
        ("Solve x^2 - 5x + 6 = 0", "solve_equation", {"equation": "x^2 - 5x + 6 = 0"}),
        ("solve 2t + 3 = 11 for t", "solve_equation", {"variable": "t"}),
        ("x + y = 3, x - y = 1", "solve_system", {"equations": ["x + y = 3", "x - y = 1"]}),
        ("factor x^2 - 9", "factor_expression", {"expression": "x^2 - 9"}),
        ("Find the derivative of x^3 sin(x)", "differentiate", {"expression": "x^3 sin(x)", "order": 1}),
        ("second derivative of x^4", "differentiate", {"order": 2}),
        ("derivative of x^2 at x = 3", "differentiate", {"point": "3"}),
        ("d/dt t^2", "differentiate", {"variable": "t"}),
        ("Integrate x^2 from 0 to 3", "integrate", {"expression": "x^2", "lower": "0", "upper": "3"}),
        ("integral of x e^x dx", "integrate", {"expression": "x e^x", "lower": None}),
        ("integrate exp(-x^2) from -infinity to infinity", "integrate", {"lower": "-oo", "upper": "oo"}),
        ("limit of sin(x)/x as x -> 0", "limit", {"expression": "sin(x)/x", "point": "0"}),
        ("lim x->oo (1+1/x)^x", "limit", {"point": "oo"}),
        ("determinant of [[1,2],[3,4]]", "matrix_operation", {"operation": "determinant"}),
        ("solve [[2,1],[1,3]] x = [3,5]", "matrix_operation", {"operation": "solve", "vector": [3, 5]}),
        ("mean of 1, 2, 3, 4", "descriptive_statistics", {"data": [1, 2, 3, 4]}),
        ("Binomial n=10 p=0.5, P(X=3)", "probability_distribution", {"distribution": "binomial", "query": "pmf"}),
        ("Binomial n=10 p=0.5, P(X<3)", "probability_distribution", {"query": "cdf", "x": "2"}),
        (
            "normal mean 100 std 15 P(X > 130)",
            "probability_distribution",
            {"distribution": "normal", "query": "greater"},
        ),
        (
            "flip a fair coin 10 times, probability of exactly 3 heads",
            "probability_distribution",
            {"distribution": "binomial"},
        ),
        ("10 choose 3", "combinatorics", {"operation": "combinations"}),
        ("5!", "combinatorics", {"operation": "factorial"}),
    ],
)
def test_rule_planner(problem, tool, expected_args):
    call = plan_with_rules(problem)
    assert call.tool == tool
    for k, v in expected_args.items():
        assert call.args.get(k) == v, (k, call.args)


@pytest.mark.parametrize(
    "problem",
    ["write me a poem", "what is the capital of France?", "limit of 1/x", "determinant", "Binomial P(X=3)"],
)
def test_rule_planner_refuses_unsupported_or_ambiguous(problem):
    with pytest.raises((PlanningError, ValueError)):
        plan_with_rules(problem)
