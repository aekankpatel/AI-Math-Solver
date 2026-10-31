"""Mathematical correctness of each tool against known answers."""

import math

import pytest
import sympy as sp

from app.math_engine.tools import ToolError, run_tool

x = sp.Symbol("x")


def solutions(out):
    return {sp.nsimplify(s) for s in out.data["solutions"]}


class TestAlgebra:
    def test_quadratic(self):
        out = run_tool("solve_equation", {"equation": "x^2 - 5x + 6 = 0"})
        assert solutions(out) == {2, 3}
        assert not out.approximate

    def test_linear_with_variable(self):
        out = run_tool("solve_equation", {"equation": "3t + 2 = 11", "variable": "t"})
        assert out.data["solutions"] == [3]

    def test_complex_roots(self):
        out = run_tool("solve_equation", {"equation": "x^2 + 1 = 0"})
        assert set(out.data["solutions"]) == {sp.I, -sp.I}
        assert out.data["real_solutions"] == []

    def test_transcendental_falls_back_to_numeric(self):
        out = run_tool("solve_equation", {"equation": "cos(x) = x"})
        assert out.approximate
        assert abs(float(out.data["solutions"][0]) - 0.7390851332151607) < 1e-9

    def test_no_solution(self):
        with pytest.raises(ToolError, match="no solutions"):
            run_tool("solve_equation", {"equation": "x = x + 1"})

    def test_identity(self):
        with pytest.raises(ToolError, match="identity"):
            run_tool("solve_equation", {"equation": "2(x + 1) = 2x + 2"})

    def test_ambiguous_variables(self):
        with pytest.raises(ToolError, match="Ambiguous"):
            run_tool("solve_equation", {"equation": "a + b = 1"})

    def test_system(self):
        out = run_tool("solve_system", {"equations": ["x + y = 3", "x - y = 1"]})
        sol = out.data["solutions"][0]
        assert sol[sp.Symbol("x")] == 2 and sol[sp.Symbol("y")] == 1

    def test_inconsistent_system(self):
        with pytest.raises(ToolError, match="inconsistent"):
            run_tool("solve_system", {"equations": ["x + y = 1", "x + y = 2"]})

    def test_factor_expand_simplify(self):
        assert run_tool("factor_expression", {"expression": "x^2 - 9"}).data["result"] == (x - 3) * (x + 3)
        assert run_tool("expand_expression", {"expression": "(x+1)^2"}).data["result"] == x**2 + 2 * x + 1
        assert run_tool("simplify_expression", {"expression": "sin(x)^2 + cos(x)^2"}).data["result"] == 1


class TestCalculus:
    def test_product_rule(self):
        out = run_tool("differentiate", {"expression": "x^3*sin(x)"})
        assert sp.simplify(out.data["result"] - (3 * x**2 * sp.sin(x) + x**3 * sp.cos(x))) == 0

    def test_higher_order_and_point(self):
        out = run_tool("differentiate", {"expression": "x^4", "order": 2, "point": "2"})
        assert out.data["result"] == 12 * x**2
        assert out.data["value"] == 48

    def test_invalid_order(self):
        with pytest.raises(ToolError):
            run_tool("differentiate", {"expression": "x", "order": 0})

    def test_limit_sinx_over_x(self):
        assert run_tool("limit", {"expression": "sin(x)/x", "point": "0"}).data["value"] == 1

    def test_limit_at_infinity(self):
        assert run_tool("limit", {"expression": "(1 + 1/x)^x", "point": "oo"}).data["value"] == sp.E

    def test_one_sided_limit(self):
        assert run_tool("limit", {"expression": "1/x", "point": "0", "direction": "+"}).data["value"] == sp.oo
        assert run_tool("limit", {"expression": "1/x", "point": "0", "direction": "-"}).data["value"] == -sp.oo


class TestIntegrals:
    def test_definite(self):
        assert run_tool("integrate", {"expression": "x^2", "lower": "0", "upper": "3"}).data["value"] == 9

    def test_indefinite(self):
        out = run_tool("integrate", {"expression": "x*exp(x)"})
        assert sp.simplify(sp.diff(out.data["antiderivative"], x) - x * sp.exp(x)) == 0
        assert out.answer_text.endswith("+ C")

    def test_gaussian(self):
        out = run_tool("integrate", {"expression": "exp(-x^2)", "lower": "-oo", "upper": "oo"})
        assert out.data["value"] == sp.sqrt(sp.pi)

    def test_missing_limit(self):
        with pytest.raises(ToolError):
            run_tool("integrate", {"expression": "x", "lower": "0"})

    def test_non_elementary_indefinite(self):
        with pytest.raises(ToolError, match="elementary"):
            run_tool("integrate", {"expression": "exp(x)/log(x)"})


class TestLinearAlgebra:
    def test_determinant(self):
        assert (
            run_tool("matrix_operation", {"operation": "determinant", "matrix": [[1, 2], [3, 4]]}).data["result"] == -2
        )

    def test_inverse(self):
        out = run_tool("matrix_operation", {"operation": "inverse", "matrix": [[4, 7], [2, 6]]})
        assert out.data["result"] == sp.Matrix(
            [[sp.Rational(3, 5), sp.Rational(-7, 10)], [sp.Rational(-1, 5), sp.Rational(2, 5)]]
        )

    def test_singular_inverse(self):
        with pytest.raises(ToolError, match="singular"):
            run_tool("matrix_operation", {"operation": "inverse", "matrix": [[1, 2], [2, 4]]})

    def test_eigenvalues(self):
        out = run_tool("matrix_operation", {"operation": "eigenvalues", "matrix": [[2, 1], [1, 2]]})
        assert out.data["result"] == {3: 1, 1: 1}

    def test_solve_linear_system(self):
        out = run_tool("matrix_operation", {"operation": "solve", "matrix": [[2, 1], [1, 3]], "vector": [3, 5]})
        assert list(out.data["result"]) == [sp.Rational(4, 5), sp.Rational(7, 5)]

    def test_rank_and_dimension_errors(self):
        assert run_tool("matrix_operation", {"operation": "rank", "matrix": [[1, 2], [2, 4]]}).data["result"] == 1
        with pytest.raises(ToolError):
            run_tool("matrix_operation", {"operation": "determinant", "matrix": [[1, 2, 3], [4, 5, 6]]})
        with pytest.raises(ToolError):
            run_tool("matrix_operation", {"operation": "multiply", "matrix": [[1, 2]], "other": [[1, 2]]})


class TestProbabilityStatistics:
    def test_descriptive(self):
        out = run_tool("descriptive_statistics", {"data": [2, 4, 4, 4, 5, 5, 7, 9]})
        s = out.data["stats"]
        assert s["mean"] == 5 and s["median"] == sp.Rational(9, 2)
        assert s["population_std"] == 2
        assert s["mode"] == [4]

    def test_binomial(self):
        out = run_tool(
            "probability_distribution",
            {"distribution": "binomial", "params": {"n": 10, "p": 0.5}, "query": "pmf", "x": 3},
        )
        assert out.data["value"] == sp.Rational(15, 128)

    def test_binomial_cdf(self):
        out = run_tool(
            "probability_distribution",
            {"distribution": "binomial", "params": {"n": 10, "p": 0.5}, "query": "cdf", "x": 3},
        )
        assert out.data["value"] == sp.Rational(176, 1024)

    def test_normal_68_rule(self):
        out = run_tool(
            "probability_distribution",
            {"distribution": "normal", "params": {"mean": 0, "std": 1}, "query": "between", "x": -1, "x2": 1},
        )
        assert abs(out.data["numeric"] - 0.6826894921370859) < 1e-12

    def test_poisson(self):
        out = run_tool(
            "probability_distribution", {"distribution": "poisson", "params": {"lambda": 3}, "query": "cdf", "x": 2}
        )
        assert abs(out.data["numeric"] - 8.5 * math.exp(-3)) < 1e-12

    def test_exponential_and_uniform(self):
        e = run_tool(
            "probability_distribution",
            {"distribution": "exponential", "params": {"rate": 2}, "query": "greater", "x": 1},
        )
        assert abs(e.data["numeric"] - math.exp(-2)) < 1e-12
        u = run_tool(
            "probability_distribution",
            {"distribution": "uniform", "params": {"a": 0, "b": 10}, "query": "between", "x": 2, "x2": 5},
        )
        assert abs(u.data["numeric"] - 0.3) < 1e-12

    def test_invalid_parameters(self):
        with pytest.raises(ToolError):
            run_tool("probability_distribution", {"distribution": "binomial", "params": {"n": 10, "p": 1.5}, "x": 1})
        with pytest.raises(ToolError):
            run_tool("probability_distribution", {"distribution": "normal", "params": {"mean": 0, "std": -1}, "x": 1})
        with pytest.raises(ToolError):
            run_tool("probability_distribution", {"distribution": "normal", "params": {}, "query": "pmf", "x": 1})

    def test_combinatorics(self):
        assert run_tool("combinatorics", {"operation": "combinations", "n": 10, "k": 3}).data["result"] == 120
        assert run_tool("combinatorics", {"operation": "permutations", "n": 5, "k": 2}).data["result"] == 20
        assert run_tool("combinatorics", {"operation": "factorial", "n": 6}).data["result"] == 720


def test_unknown_tool_and_arguments_rejected():
    with pytest.raises(ToolError):
        run_tool("run_python", {"code": "print(1)"})
    with pytest.raises(ToolError, match="Unexpected arguments"):
        run_tool("differentiate", {"expression": "x", "code": "import os"})
