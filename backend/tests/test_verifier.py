"""The verifier must confirm correct results AND catch wrong ones.

Wrong results are produced by tampering with a correct ToolOutput, simulating
a buggy solver or a hallucinated answer.
"""

import pytest
import sympy as sp

from app.math_engine.tools import run_tool
from app.math_engine.verifier import verify
from app.schemas import VerificationStatus as VS

x = sp.Symbol("x")


@pytest.mark.parametrize(
    "tool, args",
    [
        ("solve_equation", {"equation": "x^2 - 5x + 6 = 0"}),
        ("solve_equation", {"equation": "x^3 = 2"}),
        ("solve_system", {"equations": ["x + y = 3", "x - y = 1"]}),
        ("factor_expression", {"expression": "x^3 - 8"}),
        ("differentiate", {"expression": "x^3*sin(x)"}),
        ("integrate", {"expression": "x^2", "lower": "0", "upper": "3"}),
        ("integrate", {"expression": "x*cos(x)"}),
        ("integrate", {"expression": "exp(-x^2)", "lower": "0", "upper": "oo"}),
        ("matrix_operation", {"operation": "inverse", "matrix": [[4, 7], [2, 6]]}),
        ("matrix_operation", {"operation": "eigenvalues", "matrix": [[2, 1], [1, 2]]}),
        ("matrix_operation", {"operation": "determinant", "matrix": [[2, 1, 0], [1, 3, 1], [0, 1, 4]]}),
        ("matrix_operation", {"operation": "solve", "matrix": [[2, 1], [1, 3]], "vector": [3, 5]}),
        ("descriptive_statistics", {"data": [1, 2, 3, 4, 4]}),
        (
            "probability_distribution",
            {"distribution": "binomial", "params": {"n": 10, "p": 0.3}, "query": "greater", "x": 4},
        ),
        (
            "probability_distribution",
            {"distribution": "normal", "params": {"mean": 100, "std": 15}, "query": "greater", "x": 130},
        ),
        ("combinatorics", {"operation": "combinations", "n": 52, "k": 5}),
    ],
)
def test_correct_results_are_verified(tool, args):
    v = verify(tool, run_tool(tool, args))
    assert v.status == VS.VERIFIED, v
    assert v.checks and all(c.passed for c in v.checks)


@pytest.mark.parametrize(
    "tool, args",
    [
        ("solve_equation", {"equation": "cos(x) = x"}),
        ("limit", {"expression": "sin(x)/x", "point": "0"}),
        ("limit", {"expression": "(1 + 1/x)^x", "point": "oo"}),
        ("differentiate", {"expression": "exp(2x)", "order": 3}),
    ],
)
def test_numeric_only_results_are_marked_approximate(tool, args):
    assert verify(tool, run_tool(tool, args)).status == VS.APPROXIMATE


def test_detects_wrong_equation_solution():
    out = run_tool("solve_equation", {"equation": "x^2 - 5x + 6 = 0"})
    out.data["solutions"] = [sp.Integer(2), sp.Integer(4)]  # 4 is wrong, 3 missing
    v = verify("solve_equation", out)
    assert v.status == VS.FAILED
    assert sum(not c.passed for c in v.checks) >= 2  # bad substitution + missed root


def test_detects_missing_root():
    out = run_tool("solve_equation", {"equation": "x^2 - 5x + 6 = 0"})
    out.data["solutions"] = [sp.Integer(2)]
    v = verify("solve_equation", out)
    assert v.status == VS.FAILED
    assert any("missed" in c.detail for c in v.checks)


def test_detects_wrong_derivative():
    out = run_tool("differentiate", {"expression": "x^3*sin(x)"})
    out.data["result"] = 3 * x**2 * sp.cos(x)
    assert verify("differentiate", out).status == VS.FAILED


def test_detects_wrong_definite_integral():
    out = run_tool("integrate", {"expression": "x^2", "lower": "0", "upper": "3"})
    out.data["value"] = sp.Integer(10)
    assert verify("integrate", out).status == VS.FAILED


def test_detects_wrong_antiderivative():
    out = run_tool("integrate", {"expression": "x*cos(x)"})
    out.data["antiderivative"] = x * sp.sin(x)
    assert verify("integrate", out).status == VS.FAILED


def test_detects_wrong_limit():
    out = run_tool("limit", {"expression": "sin(x)/x", "point": "0"})
    out.data["value"] = sp.Integer(2)
    assert verify("limit", out).status == VS.FAILED


def test_detects_wrong_simplification():
    out = run_tool("factor_expression", {"expression": "x^2 - 9"})
    out.data["result"] = (x - 3) ** 2
    assert verify("factor_expression", out).status == VS.FAILED


def test_detects_wrong_matrix_results():
    det = run_tool("matrix_operation", {"operation": "determinant", "matrix": [[1, 2], [3, 4]]})
    det.data["result"] = sp.Integer(2)
    assert verify("matrix_operation", det).status == VS.FAILED

    inv = run_tool("matrix_operation", {"operation": "inverse", "matrix": [[4, 7], [2, 6]]})
    inv.data["result"] = inv.data["result"].T
    assert verify("matrix_operation", inv).status == VS.FAILED

    ev = run_tool("matrix_operation", {"operation": "eigenvalues", "matrix": [[2, 1], [1, 2]]})
    ev.data["result"] = {sp.Integer(3): 1, sp.Integer(2): 1}
    assert verify("matrix_operation", ev).status == VS.FAILED


def test_detects_wrong_statistics_and_probability():
    st = run_tool("descriptive_statistics", {"data": [1, 2, 3]})
    st.data["stats"]["mean"] = sp.Integer(3)
    assert verify("descriptive_statistics", st).status == VS.FAILED

    pr = run_tool(
        "probability_distribution", {"distribution": "binomial", "params": {"n": 10, "p": 0.5}, "query": "pmf", "x": 3}
    )
    pr.data["numeric"] = 0.2
    assert verify("probability_distribution", pr).status == VS.FAILED

    cb = run_tool("combinatorics", {"operation": "combinations", "n": 10, "k": 3})
    cb.data["result"] = sp.Integer(121)
    assert verify("combinatorics", cb).status == VS.FAILED


def test_verifier_never_raises():
    out = run_tool("solve_equation", {"equation": "x^2 = 4"})
    out.data.clear()
    v = verify("solve_equation", out)
    assert v.status == VS.UNVERIFIED


def test_unknown_tool_is_unverified():
    out = run_tool("solve_equation", {"equation": "x = 1"})
    assert verify("nonexistent", out).status == VS.UNVERIFIED
