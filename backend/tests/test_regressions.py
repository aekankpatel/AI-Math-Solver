"""Regression tests for correctness bugs found during the Phase 2 audit.

Each test names the original faulty behaviour so it cannot silently return.
"""

import pytest
import sympy as sp

from app.math_engine.tools import ToolError, ToolOutput, run_tool
from app.math_engine.verifier import _close, verify
from app.schemas import VerificationStatus as VS

x = sp.Symbol("x")


# --- equations ---------------------------------------------------------------


def test_periodic_equation_returns_general_solution():
    # Was: "x = 0, x = pi" (incomplete) and the verifier reported FAILED.
    out = run_tool("solve_equation", {"equation": "sin(x) = 0"})
    assert out.data["period"] == 2 * sp.pi
    assert set(out.data["solutions"]) == {0, sp.pi}
    assert "k any integer" in out.answer_text
    assert verify("solve_equation", out).status == VS.VERIFIED


def test_tan_general_solution_verified():
    out = run_tool("solve_equation", {"equation": "tan(x) = 1"})
    assert out.data["solutions"] == [sp.pi / 4] and out.data["period"] == sp.pi
    assert verify("solve_equation", out).status == VS.VERIFIED


def test_periodic_tampered_base_detected():
    out = run_tool("solve_equation", {"equation": "sin(x) = 0"})
    out.data["solutions"] = [sp.Integer(0)]  # drops x = pi + 2k*pi
    assert verify("solve_equation", out).status == VS.FAILED


def test_no_solution_equation_does_not_return_garbage_roots():
    # Was: Newton iterations drifting to x ≈ -2e7 were reported as "approximate" roots.
    with pytest.raises(ToolError, match="no solutions"):
        run_tool("solve_equation", {"equation": "1/(x - 1) = 0"})


def test_verifier_rejects_decay_pseudo_root():
    out = ToolOutput(
        answer_text="x = -23624830.75",
        answer_latex="",
        approximate=True,
        data={"expr": 1 / (x - 1), "var": x, "solutions": [sp.Float(-23624830.75)], "real_solutions": []},
    )
    assert verify("solve_equation", out).status == VS.FAILED


def test_no_real_solutions_reported():
    with pytest.raises(ToolError, match="no real solutions"):
        run_tool("solve_equation", {"equation": "sin(x) = 2"})


def test_decimals_are_exact():
    # Was: x = 4.00000000000000 (binary floating point).
    out = run_tool("solve_equation", {"equation": "0.5x = 2"})
    assert out.data["solutions"] == [4] and out.answer_text == "x = 4"
    assert run_tool("simplify_expression", {"expression": "0.1 + 0.2"}).data["result"] == sp.Rational(3, 10)


def test_absolute_value_equation_exact():
    # Was: approximate floats ±3.00000000000.
    out = run_tool("solve_equation", {"equation": "abs(x) = 3"})
    assert set(out.data["solutions"]) == {-3, 3} and not out.approximate


def test_missing_complex_roots_detected():
    out = run_tool("solve_equation", {"equation": "x^3 = 1"})
    out.data["solutions"] = [sp.Integer(1)]  # real root only
    v = verify("solve_equation", out)
    assert v.status == VS.FAILED
    assert any("numpy.roots" in c.method and not c.passed for c in v.checks)


def test_extraneous_polynomial_root_detected():
    out = run_tool("solve_equation", {"equation": "x^2 + 1 = 0"})
    out.data["solutions"] = [sp.I, -sp.I, sp.Integer(5)]
    assert verify("solve_equation", out).status == VS.FAILED


# --- integrals -----------------------------------------------------------------


def test_integral_across_pole_reports_divergence():
    # Was: "Invalid arguments for integrate: cannot unpack non-iterable ComplexInfinity".
    with pytest.raises(ToolError, match="diverges.*x = 0"):
        run_tool("integrate", {"expression": "1/x", "lower": "-1", "upper": "1"})


def test_divergent_integral_not_trivially_verified():
    # Was: "verified" because |inf - inf| <= rel*inf evaluated True.
    out = run_tool("integrate", {"expression": "1/x", "lower": "0", "upper": "1"})
    assert out.data["value"] == sp.oo and out.answer_text.startswith("diverges")
    v = verify("integrate", out)
    assert v.status == VS.APPROXIMATE
    assert any("divergence" in c.method and c.passed for c in v.checks)


def test_convergent_integral_claimed_divergent_is_caught():
    out = run_tool("integrate", {"expression": "1/sqrt(x)", "lower": "0", "upper": "1"})
    assert out.data["value"] == 2
    out.data["value"] = sp.oo
    assert verify("integrate", out).status == VS.FAILED


def test_divergent_integral_claimed_finite_is_not_verified():
    out = run_tool("integrate", {"expression": "1/x", "lower": "1", "upper": "oo"})
    out.data["value"] = sp.Integer(20)
    assert verify("integrate", out).status in (VS.FAILED, VS.UNVERIFIED)


def test_reversed_limits_and_improper_integrals():
    assert run_tool("integrate", {"expression": "x^2", "lower": "3", "upper": "0"}).data["value"] == -9
    out = run_tool("integrate", {"expression": "1/sqrt(x)", "lower": "0", "upper": "1"})
    assert verify("integrate", out).status == VS.VERIFIED


def test_infinity_comparison():
    assert not _close(complex(float("inf")), 5.0)
    assert not _close(float("inf"), float("-inf"))
    assert _close(float("inf"), float("inf"))


# --- limits ----------------------------------------------------------------------


def test_oscillating_limit_does_not_exist():
    # Was: answer "AccumBounds(-1, 1)".
    with pytest.raises(ToolError, match="does not exist.*oscillates"):
        run_tool("limit", {"expression": "sin(1/x)", "point": "0"})


def test_two_sided_infinite_limit_does_not_exist():
    # Was: answer "zoo" marked approximate.
    with pytest.raises(ToolError, match="left-hand limit is -oo"):
        run_tool("limit", {"expression": "1/x", "point": "0"})


def test_infinite_limit_sign_is_checked():
    out = run_tool("limit", {"expression": "-1/x^2", "point": "0"})
    assert out.data["value"] == -sp.oo
    assert verify("limit", out).status == VS.APPROXIMATE
    out.data["value"] = sp.oo  # wrong sign used to pass
    assert verify("limit", out).status == VS.FAILED


# --- calculus over the reals -----------------------------------------------------


def test_derivative_of_abs_is_real():
    # Was: Piecewise expression in re(x), im(x) and Derivative(re(x), x).
    out = run_tool("differentiate", {"expression": "abs(x)"})
    assert out.data["result"] == sp.sign(out.data["var"])


# --- probability -----------------------------------------------------------------


def test_reversed_interval_is_rejected():
    # Was: P(1 <= X <= -1) = -0.6827 (a negative probability).
    with pytest.raises(ToolError, match="Invalid interval"):
        run_tool(
            "probability_distribution",
            {"distribution": "normal", "params": {"mean": 0, "std": 1}, "query": "between", "x": 1, "x2": -1},
        )


# --- robustness ------------------------------------------------------------------


def test_internal_library_errors_become_tool_errors(monkeypatch):
    import app.math_engine.tools as tools

    def boom(expression):
        raise TypeError("internal failure")

    monkeypatch.setitem(tools.TOOLS, "differentiate", boom)
    with pytest.raises(ToolError, match="could not be completed"):
        run_tool("differentiate", {"expression": "x"})
