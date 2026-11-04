"""Mathenza benchmark dataset.

Every case states the problem in natural language, the tool that should be
selected, canonical tool arguments (used by the planner-independent ``oracle``
mode) and the expected answer ("truth").

Ground truth is written by hand from textbook definitions/closed forms in SymPy
syntax. It is NOT produced by the application or by an LLM at runtime, and it
is independently cross-checked by ``eval/ground_truth.py`` with mpmath, NumPy
and SciPy (no application code).

Truth types
-----------
roots           complete solution set of one equation (complex values allowed)
roots_periodic  real solutions b + k*period
system          list of solution dicts
expr            expression equivalence (optionally ``form``: factored/expanded)
antiderivative  equal up to an additive constant
number          scalar (oo / -oo allowed for divergence and infinite limits)
matrix / vector exact matrices/vectors
multiset        eigenvalues with multiplicity
stats           mean, median, population_std, sample_std
reject          the system should refuse (error / unsupported) rather than answer

``tamper`` cases replace part of a correct tool output with a wrong value to
measure whether the verifier catches incorrect solutions.
"""

from __future__ import annotations

from typing import Any

Case = dict[str, Any]
CASES: list[Case] = []


def _add(cid: str, category: str, problem: str, tool: str | None, args: dict | None, truth: dict, **extra: Any) -> None:
    CASES.append(
        {"id": cid, "category": category, "problem": problem, "tool": tool, "args": args, "truth": truth, **extra}
    )


def roots(*values: str) -> dict:
    return {"type": "roots", "values": list(values)}


def num(value: str) -> dict:
    return {"type": "number", "value": value}


def expr(value: str, form: str | None = None) -> dict:
    return {"type": "expr", "value": value, "form": form}


REJECT = {"type": "reject"}

# ------------------------------------------------------------------ algebra: equations (25)
EQ = "solve_equation"
for i, (problem, equation, truth, var) in enumerate(
    [
        ("Solve x^2 - 5x + 6 = 0", "x^2 - 5x + 6 = 0", roots("2", "3"), None),
        ("solve 2x + 3 = 11", "2x + 3 = 11", roots("4"), None),
        ("Solve 3t - 7 = 2 for t", "3t - 7 = 2", roots("3"), "t"),
        ("find the roots of x^3 - 6x^2 + 11x - 6", "x^3 - 6x^2 + 11x - 6", roots("1", "2", "3"), None),
        ("solve x^2 + 1 = 0", "x^2 + 1 = 0", roots("I", "-I"), None),
        ("solve x^3 = 8", "x^3 = 8", roots("2", "-1 + sqrt(3)*I", "-1 - sqrt(3)*I"), None),
        ("solve x^4 - 5x^2 + 4 = 0", "x^4 - 5x^2 + 4 = 0", roots("-2", "-1", "1", "2"), None),
        ("solve x^2 - 2x + 1 = 0", "x^2 - 2x + 1 = 0", roots("1"), None),
        ("solve x^2 - 2 = 0", "x^2 - 2 = 0", roots("sqrt(2)", "-sqrt(2)"), None),
        ("solve 2x^2 + 3x - 2 = 0", "2x^2 + 3x - 2 = 0", roots("1/2", "-2"), None),
        ("solve sqrt(x) = x - 2", "sqrt(x) = x - 2", roots("4"), None),
        ("solve exp(x) = 5", "exp(x) = 5", roots("log(5)"), None),
        ("solve log(x) = 2", "log(x) = 2", roots("exp(2)"), None),
        ("solve cos(x) = x", "cos(x) = x", roots("0.73908513321516064166"), None),
        ("solve sin(x) = 0", "sin(x) = 0", {"type": "roots_periodic", "bases": ["0", "pi"], "period": "2*pi"}, None),
        (
            "solve 2cos(x) = 1",
            "2cos(x) = 1",
            {"type": "roots_periodic", "bases": ["pi/3", "5*pi/3"], "period": "2*pi"},
            None,
        ),
        ("solve tan(x) = 1", "tan(x) = 1", {"type": "roots_periodic", "bases": ["pi/4"], "period": "pi"}, None),
        ("solve x/(x + 1) = 2", "x/(x + 1) = 2", roots("-2"), None),
        ("solve (x - 1)(x + 2)(x - 3) = 0", "(x - 1)(x + 2)(x - 3) = 0", roots("1", "-2", "3"), None),
        ("solve 0.5x + 1.5 = 4", "0.5x + 1.5 = 4", roots("5"), None),
        ("solve abs(x - 1) = 2", "abs(x - 1) = 2", roots("-1", "3"), None),
        ("solve x^3 - x = 0", "x^3 - x = 0", roots("-1", "0", "1"), None),
        ("solve 2^x = 8", "2^x = 8", roots("3"), None),
        ("solve x^2 + 4x + 13 = 0", "x^2 + 4x + 13 = 0", roots("-2 + 3*I", "-2 - 3*I"), None),
        ("solve exp(x) - 3x = 0", "exp(x) - 3x = 0", roots("0.61906128673594511836", "1.5121345516578424739"), None),
    ],
    1,
):
    _add(f"eq-{i:02d}", "algebra", problem, EQ, {"equation": equation, "variable": var}, truth)

# ------------------------------------------------------------------ algebra: systems (8)
for i, (problem, eqs, sols) in enumerate(
    [
        ("x + y = 3, x - y = 1", ["x + y = 3", "x - y = 1"], [{"x": "2", "y": "1"}]),
        ("solve 2x + 3y = 13 and x - y = -1", ["2x + 3y = 13", "x - y = -1"], [{"x": "2", "y": "3"}]),
        (
            "x + y + z = 6, x - y = 0, x + z = 4",
            ["x + y + z = 6", "x - y = 0", "x + z = 4"],
            [{"x": "2", "y": "2", "z": "2"}],
        ),
        (
            "solve x^2 + y^2 = 25, x - y = 1",
            ["x^2 + y^2 = 25", "x - y = 1"],
            [{"x": "4", "y": "3"}, {"x": "-3", "y": "-4"}],
        ),
        ("y = x^2, y = 2x", ["y = x^2", "y = 2x"], [{"x": "0", "y": "0"}, {"x": "2", "y": "4"}]),
        ("3a + b = 5, a - b = 3", ["3a + b = 5", "a - b = 3"], [{"a": "2", "b": "-1"}]),
        ("x + 2y = 7, 3x - y = 7", ["x + 2y = 7", "3x - y = 7"], [{"x": "3", "y": "2"}]),
        ("solve xy = 6 and x + y = 5", ["xy = 6", "x + y = 5"], [{"x": "2", "y": "3"}, {"x": "3", "y": "2"}]),
    ],
    1,
):
    _add(f"sys-{i:02d}", "algebra", problem, "solve_system", {"equations": eqs}, {"type": "system", "solutions": sols})

# ------------------------------------------------------------------ algebra: simplify / factor / expand (18)
for i, (problem, e, t) in enumerate(
    [
        ("simplify sin(x)^2 + cos(x)^2", "sin(x)^2 + cos(x)^2", "1"),
        ("simplify (x^2 - 1)/(x - 1)", "(x^2 - 1)/(x - 1)", "x + 1"),
        ("simplify (x^2 + 2x + 1)/(x + 1)", "(x^2 + 2x + 1)/(x + 1)", "x + 1"),
        ("simplify 2(x + 3) - 4x", "2(x + 3) - 4x", "6 - 2*x"),
        ("simplify exp(log(x))", "exp(log(x))", "x"),
        ("simplify 1/2 + 1/3", "1/2 + 1/3", "5/6"),
        ("simplify sqrt(50)", "sqrt(50)", "5*sqrt(2)"),
    ],
    1,
):
    _add(f"simp-{i:02d}", "algebra", problem, "simplify_expression", {"expression": e}, expr(t))
for i, (problem, e, t) in enumerate(
    [
        ("factor x^2 - 9", "x^2 - 9", "(x - 3)*(x + 3)"),
        ("factor x^3 - 8", "x^3 - 8", "(x - 2)*(x**2 + 2*x + 4)"),
        ("factor x^2 + 5x + 6", "x^2 + 5x + 6", "(x + 2)*(x + 3)"),
        ("factor 2x^2 - 8", "2x^2 - 8", "2*(x - 2)*(x + 2)"),
        ("factor x^4 - 1", "x^4 - 1", "(x - 1)*(x + 1)*(x**2 + 1)"),
        ("factor x^3 + 3x^2 + 3x + 1", "x^3 + 3x^2 + 3x + 1", "(x + 1)**3"),
    ],
    1,
):
    _add(f"fact-{i:02d}", "algebra", problem, "factor_expression", {"expression": e}, expr(t, "factored"))
for i, (problem, e, t) in enumerate(
    [
        ("expand (x + 1)^3", "(x + 1)^3", "x**3 + 3*x**2 + 3*x + 1"),
        ("expand (x - 2)(x + 5)", "(x - 2)(x + 5)", "x**2 + 3*x - 10"),
        ("expand (a + b)^2", "(a + b)^2", "a**2 + 2*a*b + b**2"),
        ("expand (2x - 1)^2", "(2x - 1)^2", "4*x**2 - 4*x + 1"),
        ("expand x(x + 1)(x + 2)", "x(x + 1)(x + 2)", "x**3 + 3*x**2 + 2*x"),
    ],
    1,
):
    _add(f"exp-{i:02d}", "algebra", problem, "expand_expression", {"expression": e}, expr(t, "expanded"))

# ------------------------------------------------------------------ calculus: derivatives (16)
D = "differentiate"
for i, (problem, args, truth) in enumerate(
    [
        ("Find the derivative of x^3 sin(x)", {"expression": "x^3 sin(x)"}, expr("3*x**2*sin(x) + x**3*cos(x)")),
        ("d/dx e^(2x)", {"expression": "e^(2x)"}, expr("2*exp(2*x)")),
        ("derivative of ln(x^2 + 1)", {"expression": "ln(x^2 + 1)"}, expr("2*x/(x**2 + 1)")),
        ("differentiate sqrt(x)", {"expression": "sqrt(x)"}, expr("1/(2*sqrt(x))")),
        ("derivative of tan(x)", {"expression": "tan(x)"}, expr("1/cos(x)**2")),
        ("derivative of x*exp(x)", {"expression": "x*exp(x)"}, expr("(x + 1)*exp(x)")),
        ("derivative of sin(x^2)", {"expression": "sin(x^2)"}, expr("2*x*cos(x**2)")),
        (
            "derivative of (x^2 + 1)/(x - 1)",
            {"expression": "(x^2 + 1)/(x - 1)"},
            expr("(2*x*(x - 1) - (x**2 + 1))/(x - 1)**2"),
        ),
        ("derivative of x^x", {"expression": "x^x"}, expr("x**x*(log(x) + 1)")),
        ("second derivative of x^4", {"expression": "x^4", "order": 2}, expr("12*x**2")),
        ("second derivative of sin(x)", {"expression": "sin(x)", "order": 2}, expr("-sin(x)")),
        ("third derivative of x^5", {"expression": "x^5", "order": 3}, expr("60*x**2")),
        ("derivative of x^2 at x = 3", {"expression": "x^2", "point": "3"}, num("6")),
        ("derivative of sin(x) at x = pi/3", {"expression": "sin(x)", "point": "pi/3"}, num("1/2")),
        ("d/dt t^3 + 2t", {"expression": "t^3 + 2t", "variable": "t"}, expr("3*t**2 + 2")),
        ("derivative of atan(x)", {"expression": "atan(x)"}, expr("1/(1 + x**2)")),
    ],
    1,
):
    _add(f"diff-{i:02d}", "calculus", problem, D, args, truth)

# ------------------------------------------------------------------ calculus: limits (12)
L = "limit"
for i, (problem, args, value) in enumerate(
    [
        ("limit of sin(x)/x as x -> 0", {"expression": "sin(x)/x", "point": "0"}, "1"),
        ("lim x->oo (1+1/x)^x", {"expression": "(1+1/x)^x", "point": "oo"}, "E"),
        ("limit of (x^2 - 1)/(x - 1) as x -> 1", {"expression": "(x^2 - 1)/(x - 1)", "point": "1"}, "2"),
        ("limit of (1 - cos(x))/x^2 as x -> 0", {"expression": "(1 - cos(x))/x^2", "point": "0"}, "1/2"),
        ("limit of (exp(x) - 1)/x as x -> 0", {"expression": "(exp(x) - 1)/x", "point": "0"}, "1"),
        ("limit of x*sin(1/x) as x -> 0", {"expression": "x*sin(1/x)", "point": "0"}, "0"),
        ("limit of (3x^2 + 1)/(x^2 - 4) as x -> oo", {"expression": "(3x^2 + 1)/(x^2 - 4)", "point": "oo"}, "3"),
        ("limit of 1/x^2 as x -> 0", {"expression": "1/x^2", "point": "0"}, "oo"),
        ("limit of log(x) as x -> 0+", {"expression": "log(x)", "point": "0", "direction": "+"}, "-oo"),
        ("limit of sqrt(x^2 + x) - x as x -> oo", {"expression": "sqrt(x^2 + x) - x", "point": "oo"}, "1/2"),
        ("limit of x^x as x -> 0+", {"expression": "x^x", "point": "0", "direction": "+"}, "1"),
        ("limit of 1/x as x -> 0 from the right", {"expression": "1/x", "point": "0", "direction": "+"}, "oo"),
    ],
    1,
):
    _add(f"lim-{i:02d}", "calculus", problem, L, args, num(value))

# ------------------------------------------------------------------ integrals (21)
I_ = "integrate"
for i, (problem, e, t) in enumerate(
    [
        ("integral of x^2 dx", "x^2", "x**3/3"),
        ("integral of x e^x dx", "x e^x", "(x - 1)*exp(x)"),
        ("integrate cos(x)", "cos(x)", "sin(x)"),
        ("integrate 1/x", "1/x", "log(x)"),
        ("integrate x*cos(x)", "x*cos(x)", "x*sin(x) + cos(x)"),
        ("integrate 1/(1 + x^2)", "1/(1 + x^2)", "atan(x)"),
        ("integrate sin(x)^2", "sin(x)^2", "x/2 - sin(2*x)/4"),
        ("integrate 3x^2 + 2x + 1", "3x^2 + 2x + 1", "x**3 + x**2 + x"),
        ("integrate exp(3x)", "exp(3x)", "exp(3*x)/3"),
    ],
    1,
):
    _add(f"int-{i:02d}", "integrals", problem, I_, {"expression": e}, {"type": "antiderivative", "value": t})
for i, (problem, e, lo, hi, value) in enumerate(
    [
        ("Integrate x^2 from 0 to 3", "x^2", "0", "3", "9"),
        ("integrate sin(x) from 0 to pi", "sin(x)", "0", "pi", "2"),
        ("integrate exp(-x^2) from -infinity to infinity", "exp(-x^2)", "-oo", "oo", "sqrt(pi)"),
        ("integrate 1/x^2 from 1 to infinity", "1/x^2", "1", "oo", "1"),
        ("integrate 1/sqrt(x) from 0 to 1", "1/sqrt(x)", "0", "1", "2"),
        ("integrate x*sin(x) from 0 to pi", "x*sin(x)", "0", "pi", "pi"),
        ("integrate exp(-x) from 0 to infinity", "exp(-x)", "0", "oo", "1"),
        ("integrate x^3 from -1 to 1", "x^3", "-1", "1", "0"),
        ("integrate abs(x) from -2 to 1", "abs(x)", "-2", "1", "5/2"),
        ("integrate sin(x^2) from 0 to 1", "sin(x^2)", "0", "1", "0.31026830172338110180"),
        ("integrate 1/x from 0 to 1", "1/x", "0", "1", "oo"),
        ("integrate log(x) from 1 to e", "log(x)", "1", "e", "1"),
    ],
    1,
):
    _add(f"dint-{i:02d}", "integrals", problem, I_, {"expression": e, "lower": lo, "upper": hi}, num(value))

# ------------------------------------------------------------------ linear algebra (17)
M = "matrix_operation"
for i, (problem, args, truth) in enumerate(
    [
        ("Find the determinant of [[1,2],[3,4]]", {"operation": "determinant", "matrix": [[1, 2], [3, 4]]}, num("-2")),
        (
            "determinant of [[2,1,0],[1,3,1],[0,1,4]]",
            {"operation": "determinant", "matrix": [[2, 1, 0], [1, 3, 1], [0, 1, 4]]},
            num("18"),
        ),
        (
            "det of [[1,2,3],[4,5,6],[7,8,9]]",
            {"operation": "determinant", "matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]},
            num("0"),
        ),
        (
            "inverse of [[4,7],[2,6]]",
            {"operation": "inverse", "matrix": [[4, 7], [2, 6]]},
            {"type": "matrix", "value": [["3/5", "-7/10"], ["-1/5", "2/5"]]},
        ),
        (
            "invert [[1,2,0],[0,1,0],[0,0,2]]",
            {"operation": "inverse", "matrix": [[1, 2, 0], [0, 1, 0], [0, 0, 2]]},
            {"type": "matrix", "value": [["1", "-2", "0"], ["0", "1", "0"], ["0", "0", "1/2"]]},
        ),
        (
            "eigenvalues of [[2,1],[1,2]]",
            {"operation": "eigenvalues", "matrix": [[2, 1], [1, 2]]},
            {"type": "multiset", "values": ["1", "3"]},
        ),
        (
            "eigenvalues of [[4,1],[2,3]]",
            {"operation": "eigenvalues", "matrix": [[4, 1], [2, 3]]},
            {"type": "multiset", "values": ["2", "5"]},
        ),
        (
            "eigenvalues of [[1,1],[0,1]]",
            {"operation": "eigenvalues", "matrix": [[1, 1], [0, 1]]},
            {"type": "multiset", "values": ["1", "1"]},
        ),
        (
            "eigenvalues of [[0,-1],[1,0]]",
            {"operation": "eigenvalues", "matrix": [[0, -1], [1, 0]]},
            {"type": "multiset", "values": ["I", "-I"]},
        ),
        (
            "eigenvalues of [[2,0,0],[0,3,4],[0,4,9]]",
            {"operation": "eigenvalues", "matrix": [[2, 0, 0], [0, 3, 4], [0, 4, 9]]},
            {"type": "multiset", "values": ["1", "2", "11"]},
        ),
        ("rank of [[1,2],[2,4]]", {"operation": "rank", "matrix": [[1, 2], [2, 4]]}, num("1")),
        (
            "rank of [[1,0,0],[0,1,0],[0,0,1]]",
            {"operation": "rank", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            num("3"),
        ),
        (
            "transpose of [[1,2,3],[4,5,6]]",
            {"operation": "transpose", "matrix": [[1, 2, 3], [4, 5, 6]]},
            {"type": "matrix", "value": [["1", "4"], ["2", "5"], ["3", "6"]]},
        ),
        (
            "multiply [[1,2],[3,4]] and [[5,6],[7,8]]",
            {"operation": "multiply", "matrix": [[1, 2], [3, 4]], "other": [[5, 6], [7, 8]]},
            {"type": "matrix", "value": [["19", "22"], ["43", "50"]]},
        ),
        (
            "solve [[2,1],[1,3]] x = [3,5]",
            {"operation": "solve", "matrix": [[2, 1], [1, 3]], "vector": [3, 5]},
            {"type": "vector", "value": ["4/5", "7/5"]},
        ),
        (
            "solve [[1,1,1],[0,2,5],[2,5,-1]] x = [6,-4,27]",
            {"operation": "solve", "matrix": [[1, 1, 1], [0, 2, 5], [2, 5, -1]], "vector": [6, -4, 27]},
            {"type": "vector", "value": ["5", "3", "-2"]},
        ),
        (
            "rref of [[1,2,3],[4,5,6]]",
            {"operation": "rref", "matrix": [[1, 2, 3], [4, 5, 6]]},
            {"type": "matrix", "value": [["1", "0", "-1"], ["0", "1", "2"]]},
        ),
    ],
    1,
):
    _add(f"lin-{i:02d}", "linear_algebra", problem, M, args, truth)

# ------------------------------------------------------------------ statistics (9)
for i, (problem, data, mean, median, pstd, sstd) in enumerate(
    [
        (
            "mean and standard deviation of 2, 4, 4, 4, 5, 5, 7, 9",
            [2, 4, 4, 4, 5, 5, 7, 9],
            "5",
            "9/2",
            "2",
            "sqrt(32/7)",
        ),
        ("statistics of [1, 2, 3, 4, 5]", [1, 2, 3, 4, 5], "3", "3", "sqrt(2)", "sqrt(5/2)"),
        ("mean of 10, 20, 30", [10, 20, 30], "20", "20", "sqrt(200/3)", "10"),
        ("mean and median of 1.5, 2.5, 3.5", [1.5, 2.5, 3.5], "5/2", "5/2", "sqrt(2/3)", "1"),
        ("variance of -3, 0, 3, 6", [-3, 0, 3, 6], "3/2", "3/2", "sqrt(45/4)", "sqrt(15)"),
        ("standard deviation of 7, 7, 7, 7", [7, 7, 7, 7], "7", "7", "0", "0"),
        (
            "find the median of 3, 1, 4, 1, 5, 9, 2, 6",
            [3, 1, 4, 1, 5, 9, 2, 6],
            "31/8",
            "7/2",
            "sqrt(423)/8",
            "sqrt(423/56)",
        ),
        ("average of 100, 200", [100, 200], "150", "150", "50", "50*sqrt(2)"),
        ("summary statistics of [12, 15, 11, 14, 13]", [12, 15, 11, 14, 13], "13", "13", "sqrt(2)", "sqrt(5/2)"),
    ],
    1,
):
    _add(
        f"stat-{i:02d}",
        "statistics",
        problem,
        "descriptive_statistics",
        {"data": data},
        {"type": "stats", "mean": mean, "median": median, "population_std": pstd, "sample_std": sstd},
    )

# ------------------------------------------------------------------ probability (16)
P = "probability_distribution"
for i, (problem, args, value) in enumerate(
    [
        (
            "Binomial n=10 p=0.5, P(X=3)",
            {"distribution": "binomial", "params": {"n": 10, "p": 0.5}, "query": "pmf", "x": 3},
            "15/128",
        ),
        (
            "Binomial n=10 p=0.5, P(X <= 3)",
            {"distribution": "binomial", "params": {"n": 10, "p": 0.5}, "query": "cdf", "x": 3},
            "11/64",
        ),
        (
            "binomial n = 20, p = 0.3, P(X >= 5)",
            {"distribution": "binomial", "params": {"n": 20, "p": 0.3}, "query": "greater", "x": 5},
            "1 - Sum(binomial(20, k)*(3/10)**k*(7/10)**(20 - k), (k, 0, 4))",
        ),
        (
            "flip a fair coin 10 times, probability of exactly 5 heads",
            {"distribution": "binomial", "params": {"n": 10, "p": 0.5}, "query": "pmf", "x": 5},
            "63/256",
        ),
        (
            "binomial n=5 p=0.2, P(X > 1)",
            {"distribution": "binomial", "params": {"n": 5, "p": 0.2}, "query": "greater", "x": 2},
            "0.26272",
        ),
        (
            "poisson lambda = 3, P(X <= 2)",
            {"distribution": "poisson", "params": {"lambda": 3}, "query": "cdf", "x": 2},
            "17*exp(-3)/2",
        ),
        (
            "poisson lambda=2, P(X=0)",
            {"distribution": "poisson", "params": {"lambda": 2}, "query": "pmf", "x": 0},
            "exp(-2)",
        ),
        (
            "poisson with rate 4, P(X >= 2)",
            {"distribution": "poisson", "params": {"lambda": 4}, "query": "greater", "x": 2},
            "1 - 5*exp(-4)",
        ),
        (
            "normal mean 0 std 1, P(X <= 1.96)",
            {"distribution": "normal", "params": {"mean": 0, "std": 1}, "query": "cdf", "x": 1.96},
            "(1 + erf(49*sqrt(2)/50))/2",
        ),
        (
            "normal mean 100 std 15 P(X > 130)",
            {"distribution": "normal", "params": {"mean": 100, "std": 15}, "query": "greater", "x": 130},
            "(1 - erf(sqrt(2)))/2",
        ),
        (
            "standard normal P(-1 < X < 1)",
            {"distribution": "normal", "params": {"mean": 0, "std": 1}, "query": "between", "x": -1, "x2": 1},
            "erf(sqrt(2)/2)",
        ),
        (
            "normal mean 50 std 10, P(40 < X < 60)",
            {"distribution": "normal", "params": {"mean": 50, "std": 10}, "query": "between", "x": 40, "x2": 60},
            "erf(sqrt(2)/2)",
        ),
        (
            "uniform from 0 to 10, P(X < 3)",
            {"distribution": "uniform", "params": {"a": 0, "b": 10}, "query": "cdf", "x": 3},
            "3/10",
        ),
        (
            "uniform between 2 and 6, P(3 < X < 5)",
            {"distribution": "uniform", "params": {"a": 2, "b": 6}, "query": "between", "x": 3, "x2": 5},
            "1/2",
        ),
        (
            "exponential rate 2, P(X > 1)",
            {"distribution": "exponential", "params": {"rate": 2}, "query": "greater", "x": 1},
            "exp(-2)",
        ),
        (
            "exponential with rate 0.5, P(X <= 2)",
            {"distribution": "exponential", "params": {"rate": 0.5}, "query": "cdf", "x": 2},
            "1 - exp(-1)",
        ),
    ],
    1,
):
    _add(f"prob-{i:02d}", "probability", problem, P, args, num(value))

# ------------------------------------------------------------------ combinatorics (8)
C = "combinatorics"
for i, (problem, args, value) in enumerate(
    [
        ("10 choose 3", {"operation": "combinations", "n": 10, "k": 3}, "120"),
        ("C(52, 5)", {"operation": "combinations", "n": 52, "k": 5}, "2598960"),
        ("5!", {"operation": "factorial", "n": 5}, "120"),
        ("factorial of 10", {"operation": "factorial", "n": 10}, "3628800"),
        ("P(5, 2)", {"operation": "permutations", "n": 5, "k": 2}, "20"),
        ("how many ways to choose 2 people from 8", {"operation": "combinations", "n": 8, "k": 2}, "28"),
        ("0!", {"operation": "factorial", "n": 0}, "1"),
        ("permutations(10, 3)", {"operation": "permutations", "n": 10, "k": 3}, "720"),
    ],
    1,
):
    _add(f"comb-{i:02d}", "probability", problem, C, args, num(value))

# ------------------------------------------------------------------ edge cases: unsolvable / invalid (13)
for i, (problem, tool, args) in enumerate(
    [
        ("solve x = x + 1", EQ, {"equation": "x = x + 1"}),
        ("solve 2(x + 1) = 2x + 2", EQ, {"equation": "2(x + 1) = 2x + 2"}),
        ("solve 1/(x - 1) = 0", EQ, {"equation": "1/(x - 1) = 0"}),
        ("solve sin(x) = 2", EQ, {"equation": "sin(x) = 2"}),
        ("inverse of [[1,2],[2,4]]", M, {"operation": "inverse", "matrix": [[1, 2], [2, 4]]}),
        ("solve [[1,1],[1,1]] x = [1,2]", M, {"operation": "solve", "matrix": [[1, 1], [1, 1]], "vector": [1, 2]}),
        ("determinant of [[1,2,3],[4,5,6]]", M, {"operation": "determinant", "matrix": [[1, 2, 3], [4, 5, 6]]}),
        ("x + y = 1, x + y = 2", "solve_system", {"equations": ["x + y = 1", "x + y = 2"]}),
        ("limit of sin(1/x) as x -> 0", L, {"expression": "sin(1/x)", "point": "0"}),
        ("limit of 1/x as x -> 0", L, {"expression": "1/x", "point": "0"}),
        ("integrate 1/x from -1 to 1", I_, {"expression": "1/x", "lower": "-1", "upper": "1"}),
        ("10 choose 11", C, {"operation": "combinations", "n": 10, "k": 11}),
        (
            "Binomial n=10 p=1.5, P(X=3)",
            P,
            {"distribution": "binomial", "params": {"n": 10, "p": 1.5}, "query": "pmf", "x": 3},
        ),
    ],
    1,
):
    _add(f"edge-{i:02d}", "edge", problem, tool, args, REJECT)

# ------------------------------------------------------------------ ambiguous inputs (8)
for i, (problem, tool, args) in enumerate(
    [
        ("solve a + b = 1", EQ, {"equation": "a + b = 1"}),
        ("determinant", None, None),
        ("Binomial P(X=3)", None, None),
        ("limit of 1/x", None, None),
        ("integrate", None, None),
        ("solve", None, None),
        ("normal distribution P(X < 1)", None, None),
        ("integrate x from 0", I_, {"expression": "x", "lower": "0"}),
    ],
    1,
):
    _add(f"amb-{i:02d}", "ambiguous", problem, tool, args, REJECT)

# ------------------------------------------------------------------ unsupported / unsafe (8)
for i, problem in enumerate(
    [
        "write me a poem about triangles",
        "solve y'' + y = 0",
        "prove that sqrt(2) is irrational",
        "solve x^2 > 4",
        "__import__('os').system('ls')",
        "what is the capital of France?",
        "integrate x*y dy dx over the unit square",
        "find the area of a triangle with sides 3, 4 and 5",
    ],
    1,
):
    _add(f"unsup-{i:02d}", "unsupported", problem, None, None, REJECT)

# ------------------------------------------------------------------ deliberately incorrect solutions (34)
TAMPER: list[Case] = []


def _tamper(cid: str, tool: str, args: dict, mutate: dict, note: str) -> None:
    TAMPER.append({"id": cid, "category": "tamper", "tool": tool, "args": args, "mutate": mutate, "note": note})


_tamper("tam-01", EQ, {"equation": "x^2 - 5x + 6 = 0"}, {"solutions": ["2", "4"]}, "wrong root")
_tamper("tam-02", EQ, {"equation": "x^2 - 5x + 6 = 0"}, {"solutions": ["2"]}, "missing real root")
_tamper("tam-03", EQ, {"equation": "x^3 = 1"}, {"solutions": ["1"]}, "missing complex roots")
_tamper("tam-04", EQ, {"equation": "sin(x) = 0"}, {"solutions": ["0"]}, "incomplete periodic family")
_tamper("tam-05", EQ, {"equation": "cos(x) = x"}, {"solutions": ["0.75"]}, "inaccurate numeric root")
_tamper("tam-06", EQ, {"equation": "sqrt(x) = x - 2"}, {"solutions": ["1", "4"]}, "extraneous root")
_tamper("tam-07", EQ, {"equation": "x^2 + 4x + 13 = 0"}, {"solutions": ["-2 + 3*I"]}, "missing conjugate root")
_tamper(
    "tam-08",
    "solve_system",
    {"equations": ["x + y = 3", "x - y = 1"]},
    {"system": [{"x": "2", "y": "2"}]},
    "wrong system solution",
)
_tamper("tam-09", "factor_expression", {"expression": "x^2 - 9"}, {"result": "(x - 3)**2"}, "wrong factorisation")
_tamper("tam-10", "expand_expression", {"expression": "(x + 1)^3"}, {"result": "x**3 + 3*x**2 + 1"}, "dropped term")
_tamper("tam-11", "simplify_expression", {"expression": "(x^2 - 1)/(x - 1)"}, {"result": "x - 1"}, "sign error")
_tamper("tam-12", D, {"expression": "x^3 sin(x)"}, {"result": "3*x**2*cos(x)"}, "product rule misapplied")
_tamper("tam-13", D, {"expression": "e^(2x)"}, {"result": "exp(2*x)"}, "chain rule forgotten")
_tamper("tam-14", D, {"expression": "x^4", "order": 2}, {"result": "4*x**3"}, "first instead of second derivative")
_tamper("tam-15", D, {"expression": "x^2", "point": "3"}, {"value": "9"}, "evaluated f instead of f'")
_tamper("tam-16", D, {"expression": "ln(x^2 + 1)"}, {"result": "1/(x**2 + 1)"}, "missing inner derivative")
_tamper("tam-17", I_, {"expression": "x*cos(x)"}, {"antiderivative": "x*sin(x)"}, "incomplete integration by parts")
_tamper("tam-18", I_, {"expression": "x^2", "lower": "0", "upper": "3"}, {"value": "10"}, "arithmetic slip")
_tamper(
    "tam-19", I_, {"expression": "exp(-x^2)", "lower": "-oo", "upper": "oo"}, {"value": "pi"}, "missing square root"
)
_tamper(
    "tam-20",
    I_,
    {"expression": "1/sqrt(x)", "lower": "0", "upper": "1"},
    {"value": "oo"},
    "convergent integral claimed divergent",
)
_tamper("tam-21", I_, {"expression": "sin(x)", "lower": "0", "upper": "pi"}, {"value": "0"}, "sign of cosine")
_tamper("tam-22", L, {"expression": "sin(x)/x", "point": "0"}, {"value": "2"}, "wrong limit")
_tamper(
    "tam-23", L, {"expression": "log(x)", "point": "0", "direction": "+"}, {"value": "oo"}, "wrong sign of infinity"
)
_tamper("tam-24", L, {"expression": "(1+1/x)^x", "point": "oo"}, {"value": "1"}, "1^oo fallacy")
_tamper("tam-25", M, {"operation": "determinant", "matrix": [[1, 2], [3, 4]]}, {"result": "2"}, "ad + bc")
_tamper(
    "tam-26",
    M,
    {"operation": "inverse", "matrix": [[4, 7], [2, 6]]},
    {"result_matrix": [["3/5", "-1/5"], ["-7/10", "2/5"]]},
    "transposed inverse",
)
_tamper(
    "tam-27",
    M,
    {"operation": "eigenvalues", "matrix": [[2, 1], [1, 2]]},
    {"eigenvalues": {"3": 1, "2": 1}},
    "wrong eigenvalue",
)
_tamper("tam-28", M, {"operation": "rank", "matrix": [[1, 2], [2, 4]]}, {"result": "2"}, "rank of singular matrix")
_tamper(
    "tam-29",
    M,
    {"operation": "solve", "matrix": [[2, 1], [1, 3]], "vector": [3, 5]},
    {"result_matrix": [["1"], ["1"]]},
    "wrong solution vector",
)
_tamper(
    "tam-30",
    M,
    {"operation": "multiply", "matrix": [[1, 2], [3, 4]], "other": [[5, 6], [7, 8]]},
    {"result_matrix": [["19", "22"], ["43", "51"]]},
    "one wrong entry",
)
_tamper("tam-31", "descriptive_statistics", {"data": [1, 2, 3]}, {"stats.mean": "3"}, "wrong mean")
_tamper(
    "tam-32",
    "descriptive_statistics",
    {"data": [2, 4, 4, 4, 5, 5, 7, 9]},
    {"stats.sample_std": "2"},
    "population std reported as sample std",
)
_tamper(
    "tam-33",
    P,
    {"distribution": "binomial", "params": {"n": 10, "p": 0.5}, "query": "pmf", "x": 3},
    {"value": "1/5", "numeric": 0.2},
    "wrong probability",
)
_tamper(
    "tam-34",
    C,
    {"operation": "combinations", "n": 10, "k": 3},
    {"result": "720"},
    "permutations instead of combinations",
)

ALL_TOOLS = (
    "solve_equation",
    "solve_system",
    "simplify_expression",
    "factor_expression",
    "expand_expression",
    "differentiate",
    "integrate",
    "limit",
    "matrix_operation",
    "descriptive_statistics",
    "probability_distribution",
    "combinatorics",
)

# ------------------------------------------------------------------ held-out paraphrases (31)
# Written after the rule planner was frozen and never used to tune it: these
# measure robustness to natural phrasing (expected to favour the LLM planner).
for i, (problem, tool, args, truth) in enumerate(
    [
        ("What values of x satisfy x squared minus 9 equals 0?", EQ, {"equation": "x^2 - 9 = 0"}, roots("3", "-3")),
        ("Find x if 4x - 7 = 9", EQ, {"equation": "4x - 7 = 9"}, roots("4")),
        ("Find all real x such that x^2 = 16", EQ, {"equation": "x^2 = 16", "domain": "real"}, roots("4", "-4")),
        (
            "Solve the system: 2x + y = 5; x - y = 1",
            "solve_system",
            {"equations": ["2x + y = 5", "x - y = 1"]},
            {"type": "system", "solutions": [{"x": "2", "y": "1"}]},
        ),
        (
            "Simplify the expression (x^2 - 4)/(x + 2)",
            "simplify_expression",
            {"expression": "(x^2 - 4)/(x + 2)"},
            expr("x - 2"),
        ),
        (
            "Factorise x^2 - 7x + 12",
            "factor_expression",
            {"expression": "x^2 - 7x + 12"},
            expr("(x - 3)*(x - 4)", "factored"),
        ),
        (
            "Multiply out (x + 3)(x - 3)",
            "expand_expression",
            {"expression": "(x + 3)(x - 3)"},
            expr("x**2 - 9", "expanded"),
        ),
        ("What is the slope of y = x^3 at x = 2?", D, {"expression": "x^3", "point": "2"}, num("12")),
        ("How fast is sin(x) changing at x = 0?", D, {"expression": "sin(x)", "point": "0"}, num("1")),
        ("Differentiate f(x) = x^2 e^x with respect to x", D, {"expression": "x^2 e^x"}, expr("(x**2 + 2*x)*exp(x)")),
        ("What's the derivative of cos(3x)?", D, {"expression": "cos(3x)"}, expr("-3*sin(3*x)")),
        (
            "Compute the area under y = x^2 between x = 0 and x = 2",
            I_,
            {"expression": "x^2", "lower": "0", "upper": "2"},
            num("8/3"),
        ),
        ("Evaluate the integral of x^3 from 0 to 1", I_, {"expression": "x^3", "lower": "0", "upper": "1"}, num("1/4")),
        (
            "What's the antiderivative of 1/(x+1)?",
            I_,
            {"expression": "1/(x+1)"},
            {"type": "antiderivative", "value": "log(x + 1)"},
        ),
        (
            "integrate e^(-2x) dx from 0 to infinity",
            I_,
            {"expression": "e^(-2x)", "lower": "0", "upper": "oo"},
            num("1/2"),
        ),
        (
            "What does (x^3 - 8)/(x - 2) approach as x approaches 2?",
            L,
            {"expression": "(x^3 - 8)/(x - 2)", "point": "2"},
            num("12"),
        ),
        (
            "Find the limit as x goes to infinity of (2x + 1)/(x - 3)",
            L,
            {"expression": "(2x + 1)/(x - 3)", "point": "oo"},
            num("2"),
        ),
        (
            "Calculate the inverse of the matrix [[2, 0], [0, 4]]",
            M,
            {"operation": "inverse", "matrix": [[2, 0], [0, 4]]},
            {"type": "matrix", "value": [["1/2", "0"], ["0", "1/4"]]},
        ),
        (
            "What are the eigenvalues of the matrix [[3, 0], [0, -1]]?",
            M,
            {"operation": "eigenvalues", "matrix": [[3, 0], [0, -1]]},
            {"type": "multiset", "values": ["3", "-1"]},
        ),
        (
            "Multiply the matrices [[1,0],[0,1]] and [[2,3],[4,5]]",
            M,
            {"operation": "multiply", "matrix": [[1, 0], [0, 1]], "other": [[2, 3], [4, 5]]},
            {"type": "matrix", "value": [["2", "3"], ["4", "5"]]},
        ),
        (
            "Determine the rank of the matrix [[1, 2, 3], [2, 4, 6], [1, 0, 1]]",
            M,
            {"operation": "rank", "matrix": [[1, 2, 3], [2, 4, 6], [1, 0, 1]]},
            num("2"),
        ),
        (
            "What is the determinant of the identity matrix [[1,0,0],[0,1,0],[0,0,1]]?",
            M,
            {"operation": "determinant", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            num("1"),
        ),
        (
            "Give me the average and spread of these test scores: 70, 80, 90",
            "descriptive_statistics",
            {"data": [70, 80, 90]},
            {"type": "stats", "mean": "80", "median": "80", "population_std": "sqrt(200/3)", "sample_std": "10"},
        ),
        (
            "Work out the median of 4, 8, 15, 16, 23, 42",
            "descriptive_statistics",
            {"data": [4, 8, 15, 16, 23, 42]},
            {
                "type": "stats",
                "mean": "18",
                "median": "31/2",
                "population_std": "sqrt(455/3)",
                "sample_std": "sqrt(182)",
            },
        ),
        (
            "A coin is tossed 8 times. What is the probability of getting at most 2 heads?",
            P,
            {"distribution": "binomial", "params": {"n": 8, "p": 0.5}, "query": "cdf", "x": 2},
            num("37/256"),
        ),
        (
            "Heights are normally distributed with mean 170 and standard deviation 10. What fraction are taller than 190?",
            P,
            {"distribution": "normal", "params": {"mean": 170, "std": 10}, "query": "greater", "x": 190},
            num("(1 - erf(sqrt(2)))/2"),
        ),
        (
            "Calls arrive as a Poisson process with mean 5 per hour. Probability of exactly 3 calls in an hour?",
            P,
            {"distribution": "poisson", "params": {"lambda": 5}, "query": "pmf", "x": 3},
            num("125*exp(-5)/6"),
        ),
        (
            "Find P(Z < 1.5) for a standard normal variable Z",
            P,
            {"distribution": "normal", "params": {"mean": 0, "std": 1}, "query": "cdf", "x": 1.5},
            num("(1 + erf(3*sqrt(2)/4))/2"),
        ),
        ("How many ways can 4 books be arranged on a shelf?", C, {"operation": "factorial", "n": 4}, num("24")),
        (
            "In how many ways can a committee of 3 be chosen from 10 people?",
            C,
            {"operation": "combinations", "n": 10, "k": 3},
            num("120"),
        ),
        ("What is 15 choose 4?", C, {"operation": "combinations", "n": 15, "k": 4}, num("1365")),
    ],
    1,
):
    _add(f"para-{i:02d}", "paraphrase", problem, tool, args, truth, heldout=True)

# ------------------------------------------------------------------ UNSEEN real-world set (Phase 3)
# Frozen in its own commit BEFORE any Phase 3 change to the planner or intent
# layer. Never used for tuning. Includes word problems, contractions, explicit
# domain constraints ("positive", "real"), distractor numbers and problems with
# missing or ambiguous information (expected outcome: ask for clarification).
CLARIFY = {"type": "reject", "reason": "clarify"}
for i, (problem, tool, args, truth) in enumerate(
    [
        (
            "A rectangle satisfies x^2 + 3x = 40. What positive value of x works?",
            EQ,
            {"equation": "x^2 + 3x = 40", "domain": "positive"},
            {"type": "roots", "values": ["5"], "domain": "positive"},
        ),
        (
            "What are the real roots of x^3 - x^2 + x - 1 = 0?",
            EQ,
            {"equation": "x^3 - x^2 + x - 1 = 0", "domain": "real"},
            {"type": "roots", "values": ["1"], "domain": "real"},
        ),
        (
            "Find the positive solution of t^2 - 2t - 3 = 0",
            EQ,
            {"equation": "t^2 - 2t - 3 = 0", "variable": "t", "domain": "positive"},
            {"type": "roots", "values": ["3"], "domain": "positive"},
        ),
        ("If 3(x - 2) = 2x + 5, what is x?", EQ, {"equation": "3(x - 2) = 2x + 5"}, roots("11")),
        ("Solve for y: 2y/3 + 4 = 10", EQ, {"equation": "2y/3 + 4 = 10", "variable": "y"}, roots("9")),
        (
            "Two numbers x and y satisfy x + y = 12 and x - y = 4. Find them.",
            "solve_system",
            {"equations": ["x + y = 12", "x - y = 4"]},
            {"type": "system", "solutions": [{"x": "8", "y": "4"}]},
        ),
        (
            "For a 2023 physics project, find all t with -5t^2 + 20t = 0",
            EQ,
            {"equation": "-5t^2 + 20t = 0", "variable": "t"},
            roots("0", "4"),
        ),
        ("Solve 2^(x+1) = 32", EQ, {"equation": "2^(x+1) = 32"}, roots("4")),
        (
            "What's the rate of change of f(t) = t^2 + 3t when t = 1?",
            D,
            {"expression": "t^2 + 3t", "variable": "t", "point": "1"},
            num("5"),
        ),
        ("Find the instantaneous rate of change of x^3 at x = -2", D, {"expression": "x^3", "point": "-2"}, num("12")),
        ("Find f''(x) for f(x) = x^3 + 2x^2", D, {"expression": "x^3 + 2x^2", "order": 2}, expr("6*x + 4")),
        ("What is d/dx of ln(3x)?", D, {"expression": "ln(3x)"}, expr("1/x")),
        (
            "Find the second derivative of e^(3x) at x = 0",
            D,
            {"expression": "e^(3x)", "order": 2, "point": "0"},
            num("9"),
        ),
        (
            "Compute the definite integral of 2x + 1 over [0, 4]",
            I_,
            {"expression": "2x + 1", "lower": "0", "upper": "4"},
            num("20"),
        ),
        (
            "Find the area under the curve y = sin(x) from x = 0 to x = pi/2",
            I_,
            {"expression": "sin(x)", "lower": "0", "upper": "pi/2"},
            num("1"),
        ),
        ("Integrate 6x^2 with respect to x", I_, {"expression": "6x^2"}, {"type": "antiderivative", "value": "2*x**3"}),
        (
            "Evaluate the integral of x e^(-x) from 0 to infinity",
            I_,
            {"expression": "x e^(-x)", "lower": "0", "upper": "oo"},
            num("1"),
        ),
        ("What value does sin(3x)/x approach as x tends to 0?", L, {"expression": "sin(3x)/x", "point": "0"}, num("3")),
        (
            "Evaluate lim_{x->oo} (5x^3 - 2)/(2x^3 + x)",
            L,
            {"expression": "(5x^3 - 2)/(2x^3 + x)", "point": "oo"},
            num("5/2"),
        ),
        (
            "limit of (sqrt(x) - 2)/(x - 4) when x approaches 4",
            L,
            {"expression": "(sqrt(x) - 2)/(x - 4)", "point": "4"},
            num("1/4"),
        ),
        (
            "Find the inverse of the 2x2 matrix [[1, 2], [3, 4]]",
            M,
            {"operation": "inverse", "matrix": [[1, 2], [3, 4]]},
            {"type": "matrix", "value": [["-2", "1"], ["3/2", "-1/2"]]},
        ),
        ("Compute det([[3, 8], [4, 6]])", M, {"operation": "determinant", "matrix": [[3, 8], [4, 6]]}, num("-14")),
        (
            "What's the transpose of [[1, 2], [3, 4], [5, 6]]?",
            M,
            {"operation": "transpose", "matrix": [[1, 2], [3, 4], [5, 6]]},
            {"type": "matrix", "value": [["1", "3", "5"], ["2", "4", "6"]]},
        ),
        (
            "Find the eigenvalues for A = [[5, 4], [1, 2]]",
            M,
            {"operation": "eigenvalues", "matrix": [[5, 4], [1, 2]]},
            {"type": "multiset", "values": ["1", "6"]},
        ),
        (
            "Solve the linear system Ax = b with A = [[1, 2], [3, 5]] and b = [5, 13]",
            M,
            {"operation": "solve", "matrix": [[1, 2], [3, 5]], "vector": [5, 13]},
            {"type": "vector", "value": ["1", "2"]},
        ),
        ("What is the rank of [[2, 4], [1, 2]]?", M, {"operation": "rank", "matrix": [[2, 4], [1, 2]]}, num("1")),
        (
            "The daily temperatures were 21, 23, 19, 25 and 22 degrees. What were the mean and standard deviation?",
            "descriptive_statistics",
            {"data": [21, 23, 19, 25, 22]},
            {"type": "stats", "mean": "22", "median": "22", "population_std": "2", "sample_std": "sqrt(5)"},
        ),
        (
            "Find the variance of the sample 2, 4, 6, 8",
            "descriptive_statistics",
            {"data": [2, 4, 6, 8]},
            {"type": "stats", "mean": "5", "median": "5", "population_std": "sqrt(5)", "sample_std": "sqrt(20/3)"},
        ),
        (
            "Find the median and mean of 5, 3, 9, 1",
            "descriptive_statistics",
            {"data": [5, 3, 9, 1]},
            {"type": "stats", "mean": "9/2", "median": "4", "population_std": "sqrt(35)/2", "sample_std": "sqrt(35/3)"},
        ),
        (
            "A basketball player makes 70% of free throws. If she takes 10 shots, what is the probability she makes exactly 7?",
            P,
            {"distribution": "binomial", "params": {"n": 10, "p": 0.7}, "query": "pmf", "x": 7},
            num("binomial(10, 7)*(7/10)**7*(3/10)**3"),
        ),
        (
            "Each of 12 light bulbs fails independently with probability 0.1. What is the probability that at most 1 fails?",
            P,
            {"distribution": "binomial", "params": {"n": 12, "p": 0.1}, "query": "cdf", "x": 1},
            num("(9/10)**12 + 12*(1/10)*(9/10)**11"),
        ),
        (
            "Find P(X >= 3) for a binomial with 6 trials and success probability 0.5",
            P,
            {"distribution": "binomial", "params": {"n": 6, "p": 0.5}, "query": "greater", "x": 3},
            num("21/32"),
        ),
        (
            "Emails arrive at an average rate of 4 per minute (Poisson). What is the probability of fewer than 2 emails in a minute?",
            P,
            {"distribution": "poisson", "params": {"lambda": 4}, "query": "cdf", "x": 1},
            num("5*exp(-4)"),
        ),
        (
            "Find P(X > 3) for a Poisson distribution with mean 2",
            P,
            {"distribution": "poisson", "params": {"lambda": 2}, "query": "greater", "x": 4},
            num("1 - 19*exp(-2)/3"),
        ),
        (
            "Test scores are normally distributed with mean 75 and standard deviation 8. What proportion scored below 91?",
            P,
            {"distribution": "normal", "params": {"mean": 75, "std": 8}, "query": "cdf", "x": 91},
            num("(1 + erf(sqrt(2)))/2"),
        ),
        (
            "IQ is normal with mean 100 and sd 15. What fraction lies between 85 and 115?",
            P,
            {"distribution": "normal", "params": {"mean": 100, "std": 15}, "query": "between", "x": 85, "x2": 115},
            num("erf(sqrt(2)/2)"),
        ),
        (
            "For a standard normal Z, find P(Z > -1)",
            P,
            {"distribution": "normal", "params": {"mean": 0, "std": 1}, "query": "greater", "x": -1},
            num("(1 + erf(sqrt(2)/2))/2"),
        ),
        (
            "A bus arrives uniformly at random between 0 and 20 minutes. What's the probability you wait more than 15 minutes?",
            P,
            {"distribution": "uniform", "params": {"a": 0, "b": 20}, "query": "greater", "x": 15},
            num("1/4"),
        ),
        (
            "A component's lifetime is exponential with rate 0.2 per year. What is the probability it lasts more than 5 years?",
            P,
            {"distribution": "exponential", "params": {"rate": 0.2}, "query": "greater", "x": 5},
            num("exp(-1)"),
        ),
        (
            "How many different 5-card hands can be dealt from a 52-card deck?",
            C,
            {"operation": "combinations", "n": 52, "k": 5},
            num("2598960"),
        ),
        (
            "In how many ways can 6 runners finish first, second and third?",
            C,
            {"operation": "permutations", "n": 6, "k": 3},
            num("120"),
        ),
        ("How many ways can 7 people line up?", C, {"operation": "factorial", "n": 7}, num("5040")),
        (
            "Expand (x + 2)^4",
            "expand_expression",
            {"expression": "(x + 2)^4"},
            expr("x**4 + 8*x**3 + 24*x**2 + 32*x + 16", "expanded"),
        ),
        (
            "Factor completely: 3x^2 - 12",
            "factor_expression",
            {"expression": "3x^2 - 12"},
            expr("3*(x - 2)*(x + 2)", "factored"),
        ),
        (
            "Simplify (x^2 - 9)/(x^2 + 6x + 9)",
            "simplify_expression",
            {"expression": "(x^2 - 9)/(x^2 + 6x + 9)"},
            expr("(x - 3)/(x + 3)"),
        ),
        # --- missing or ambiguous information: the system should ask, not guess
        ("Solve x + y = 10", None, None, CLARIFY),
        ("Find the derivative", None, None, CLARIFY),
        ("What is the probability of getting 3 heads?", None, None, CLARIFY),
        ("Integrate x^2 from 1", None, None, CLARIFY),
        ("Find the limit of (x^2 - 1)/(x - 1)", None, None, CLARIFY),
        ("Normal distribution with mean 10: what is P(X < 12)?", None, None, CLARIFY),
        ("Find the eigenvalues", None, None, CLARIFY),
        ("Differentiate a*b + c", None, None, CLARIFY),
        ("Solve a^2 + b = 4", None, None, CLARIFY),
        ("Compute the mean of the data", None, None, CLARIFY),
        ("How many ways can 3 people be chosen from a group?", None, None, CLARIFY),
    ],
    1,
):
    _add(f"unseen-{i:02d}", "unseen", problem, tool, args, truth, split="unseen")

# ------------------------------------------------------------------ tampered interpretations (Phase 3)
# A correct plan ("args") and a deliberately wrong reading of the same question
# ("bad": overrides; "bad_tool": a different tool). The intent checker should
# flag every tampered plan (mismatch/ambiguous) and accept every control.
INTENT_TAMPER: list[Case] = []


def _itamper(
    cid: str,
    problem: str,
    tool: str,
    args: dict,
    bad: dict,
    note: str,
    bad_tool: str | None = None,
    control_expect: str = "accept",
) -> None:
    INTENT_TAMPER.append(
        {
            "id": cid,
            "problem": problem,
            "tool": tool,
            "args": args,
            "bad": bad,
            "bad_tool": bad_tool,
            "note": note,
            "control_expect": control_expect,
        }
    )


_itamper(
    "it-01",
    "Determine the rank of the matrix [[1, 2], [2, 4]]",
    M,
    {"operation": "rank", "matrix": [[1, 2], [2, 4]]},
    {"operation": "determinant"},
    "wrong matrix operation",
)
_itamper(
    "it-02",
    "inverse of [[4,7],[2,6]]",
    M,
    {"operation": "inverse", "matrix": [[4, 7], [2, 6]]},
    {"operation": "transpose"},
    "transpose instead of inverse",
)
_itamper(
    "it-03",
    "eigenvalues of [[2,1],[1,2]]",
    M,
    {"operation": "eigenvalues", "matrix": [[2, 1], [1, 2]]},
    {"matrix": [[2, 1], [1, 3]]},
    "matrix entry misread",
)
_itamper(
    "it-04",
    "solve [[2,1],[1,3]] x = [3,5]",
    M,
    {"operation": "solve", "matrix": [[2, 1], [1, 3]], "vector": [3, 5]},
    {"vector": [5, 3]},
    "right-hand side swapped",
)
_itamper(
    "it-05",
    "Integrate x^2 from 0 to 3",
    I_,
    {"expression": "x^2", "lower": "0", "upper": "3"},
    {"lower": "3", "upper": "0"},
    "integration limits swapped",
)
_itamper(
    "it-06",
    "Integrate x^2 from 0 to 3",
    I_,
    {"expression": "x^2", "lower": "0", "upper": "3"},
    {"lower": None, "upper": None},
    "definite integral computed as indefinite",
)
_itamper(
    "it-07",
    "integral of x e^x dx",
    I_,
    {"expression": "x e^x"},
    {"lower": "0", "upper": "1"},
    "invented integration limits",
)
_itamper(
    "it-08", "integrate 3x^2 + 2x + 1", I_, {"expression": "3x^2 + 2x + 1"}, {"expression": "3x^2 + 2x"}, "dropped term"
)
_itamper(
    "it-09", "integrate cos(x)", I_, {"expression": "cos(x)"}, {}, "differentiated instead of integrated", bad_tool=D
)
_itamper(
    "it-10", "second derivative of x^4", D, {"expression": "x^4", "order": 2}, {"order": 1}, "wrong derivative order"
)
_itamper(
    "it-11",
    "derivative of x^2 at x = 3",
    D,
    {"expression": "x^2", "point": "3"},
    {"point": None},
    "evaluation point ignored",
)
_itamper(
    "it-12",
    "derivative of x^2 at x = 3",
    D,
    {"expression": "x^2", "point": "3"},
    {"point": "2"},
    "wrong evaluation point",
)
_itamper(
    "it-13",
    "Find the derivative of x^3 sin(x)",
    D,
    {"expression": "x^3 sin(x)"},
    {"expression": "x^3 cos(x)"},
    "misread function",
)
_itamper(
    "it-14", "d/dt t^3 + 2t", D, {"expression": "t^3 + 2t", "variable": "t"}, {"expression": "t^3 + 2"}, "misread term"
)
_itamper(
    "it-15",
    "limit of sin(x)/x as x -> 0",
    L,
    {"expression": "sin(x)/x", "point": "0"},
    {"point": "oo"},
    "wrong limit point",
)
_itamper(
    "it-16",
    "limit of 1/x as x -> 0 from the right",
    L,
    {"expression": "1/x", "point": "0", "direction": "+"},
    {"direction": "-"},
    "wrong side",
)
_itamper(
    "it-17",
    "Solve x^2 - 5x + 6 = 0",
    EQ,
    {"equation": "x^2 - 5x + 6 = 0"},
    {"equation": "x^2 - 5x - 6 = 0"},
    "sign misread",
)
_itamper(
    "it-18",
    "Find the positive solution of x^2 - x - 6 = 0",
    EQ,
    {"equation": "x^2 - x - 6 = 0", "domain": "positive"},
    {"domain": "complex"},
    "ignored 'positive' constraint",
)
_itamper(
    "it-19",
    "What are the real roots of x^4 - 1 = 0?",
    EQ,
    {"equation": "x^4 - 1 = 0", "domain": "real"},
    {"domain": "complex"},
    "ignored 'real' constraint",
)
_itamper(
    "it-20",
    "Solve 3t - 7 = 2 for t",
    EQ,
    {"equation": "3t - 7 = 2", "variable": "t"},
    {"equation": "3t - 7 = 3"},
    "wrong constant",
)
_itamper(
    "it-21",
    "x + y = 3, x - y = 1",
    "solve_system",
    {"equations": ["x + y = 3", "x - y = 1"]},
    {"equation": "x + y = 3"},
    "second equation ignored",
    bad_tool=EQ,
)
_itamper(
    "it-22",
    "factor x^2 - 9",
    "factor_expression",
    {"expression": "x^2 - 9"},
    {},
    "expanded instead of factored",
    bad_tool="expand_expression",
)
_itamper(
    "it-23",
    "Binomial n=10 p=0.5, P(X < 3)",
    P,
    {"distribution": "binomial", "params": {"n": 10, "p": 0.5}, "query": "cdf", "x": 2},
    {"x": 3},
    "strict inequality treated as <=",
)
_itamper(
    "it-24",
    "Binomial n=10 p=0.5, P(X > 3)",
    P,
    {"distribution": "binomial", "params": {"n": 10, "p": 0.5}, "query": "greater", "x": 4},
    {"x": 3},
    "P(X > 3) computed as P(X >= 3)",
)
_itamper(
    "it-25",
    "binomial n = 20, p = 0.3, P(X >= 5)",
    P,
    {"distribution": "binomial", "params": {"n": 20, "p": 0.3}, "query": "greater", "x": 5},
    {"params": {"n": 20, "p": 0.5}},
    "success probability misread",
)
_itamper(
    "it-26",
    "normal mean 100 std 15 P(X > 130)",
    P,
    {"distribution": "normal", "params": {"mean": 100, "std": 15}, "query": "greater", "x": 130},
    {"query": "cdf"},
    "complement event",
)
_itamper(
    "it-27",
    "normal mean 100 std 15 P(X > 130)",
    P,
    {"distribution": "normal", "params": {"mean": 100, "std": 15}, "query": "greater", "x": 130},
    {"params": {"mean": 100, "std": 1}},
    "wrong standard deviation",
)
_itamper(
    "it-28",
    "poisson lambda = 3, P(X <= 2)",
    P,
    {"distribution": "poisson", "params": {"lambda": 3}, "query": "cdf", "x": 2},
    {"distribution": "binomial", "params": {"n": 3, "p": 0.5}},
    "wrong distribution",
)
_itamper(
    "it-29",
    "Normal distribution with mean 50: what is P(X < 60)?",
    P,
    {"distribution": "normal", "params": {"mean": 50, "std": 10}, "query": "cdf", "x": 60},
    {"params": {"mean": 50, "std": 1}},
    "missing standard deviation silently defaulted",
    control_expect="clarify",
)  # no std in the text: asking is correct
_itamper(
    "it-30",
    "uniform between 2 and 6, P(3 < X < 5)",
    P,
    {"distribution": "uniform", "params": {"a": 2, "b": 6}, "query": "between", "x": 3, "x2": 5},
    {"x2": 6},
    "wrong bound",
)
_itamper(
    "it-31",
    "In how many ways can a committee of 3 be chosen from 10 people?",
    C,
    {"operation": "combinations", "n": 10, "k": 3},
    {"operation": "permutations"},
    "order treated as mattering",
)
_itamper(
    "it-32",
    "how many ways to choose 2 people from 8",
    C,
    {"operation": "combinations", "n": 8, "k": 2},
    {"n": 2, "k": 8},
    "n and k swapped",
)
_itamper(
    "it-33",
    "mean and standard deviation of 2, 4, 4, 4, 5, 5, 7, 9",
    "descriptive_statistics",
    {"data": [2, 4, 4, 4, 5, 5, 7, 9]},
    {"data": [2, 4, 4, 5, 5, 7, 9]},
    "dropped a data value",
)
_itamper(
    "it-34",
    "statistics of [1, 2, 3, 4, 5]",
    "descriptive_statistics",
    {"data": [1, 2, 3, 4, 5]},
    {"data": [1, 2, 3, 4, 6]},
    "misread data value",
)
_itamper(
    "it-35",
    "Solve x + y = 10 for y",
    EQ,
    {"equation": "x + y = 10", "variable": "y"},
    {"variable": "x"},
    "solved for the wrong variable",
)
_itamper(
    "it-36",
    "simplify (x^2 - 1)/(x - 1)",
    "simplify_expression",
    {"expression": "(x^2 - 1)/(x - 1)"},
    {"expression": "(x^2 - 1)/(x + 1)"},
    "misread denominator",
)

# ------------------------------------------------------------------ NL tuning set (Phase 3)
# Natural-language phrasings used to DEVELOP the intent-based planner. Kept
# separate from the held-out paraphrases and the frozen unseen set.
for i, (problem, tool, args, truth) in enumerate(
    [
        ("What's the value of x when 5x + 2 = 17?", EQ, {"equation": "5x + 2 = 17"}, roots("3")),
        ("For which x does x^2 - 4x + 3 = 0 hold?", EQ, {"equation": "x^2 - 4x + 3 = 0"}, roots("1", "3")),
        (
            "Give the positive root of x^2 - 2x - 8 = 0",
            EQ,
            {"equation": "x^2 - 2x - 8 = 0", "domain": "positive"},
            {"type": "roots", "values": ["4"], "domain": "positive"},
        ),
        (
            "List the real solutions of x^3 + x = 0",
            EQ,
            {"equation": "x^3 + x = 0", "domain": "real"},
            {"type": "roots", "values": ["0"], "domain": "real"},
        ),
        ("Find y such that 7 - 2y = 1", EQ, {"equation": "7 - 2y = 1", "variable": "y"}, roots("3")),
        (
            "Both 2a + b = 7 and a - b = 2 hold. Find a and b.",
            "solve_system",
            {"equations": ["2a + b = 7", "a - b = 2"]},
            {"type": "system", "solutions": [{"a": "3", "b": "1"}]},
        ),
        (
            "Factorize x^2 + 7x + 10",
            "factor_expression",
            {"expression": "x^2 + 7x + 10"},
            expr("(x + 2)*(x + 5)", "factored"),
        ),
        (
            "Please multiply out (2x + 1)(x - 3)",
            "expand_expression",
            {"expression": "(2x + 1)(x - 3)"},
            expr("2*x**2 - 5*x - 3", "expanded"),
        ),
        ("Can you simplify (x^3 - x)/x ?", "simplify_expression", {"expression": "(x^3 - x)/x"}, expr("x**2 - 1")),
        (
            "What's the slope of the tangent to y = x^2 + 1 at x = 3?",
            D,
            {"expression": "x^2 + 1", "point": "3"},
            num("6"),
        ),
        ("How quickly does 4t^2 change at t = 2?", D, {"expression": "4t^2", "variable": "t", "point": "2"}, num("16")),
        ("Give me the derivative of x^4 - 3x", D, {"expression": "x^4 - 3x"}, expr("4*x**3 - 3")),
        ("What's the second derivative of cos(x)?", D, {"expression": "cos(x)", "order": 2}, expr("-cos(x)")),
        (
            "Differentiate g(x) = sqrt(x) + x with respect to x",
            D,
            {"expression": "sqrt(x) + x"},
            expr("1/(2*sqrt(x)) + 1"),
        ),
        (
            "What's the area under y = 3x^2 between x = 1 and x = 2?",
            I_,
            {"expression": "3x^2", "lower": "1", "upper": "2"},
            num("7"),
        ),
        ("Integrate 4x^3 over the interval [0, 2]", I_, {"expression": "4x^3", "lower": "0", "upper": "2"}, num("16")),
        (
            "What's the antiderivative of sin(x)?",
            I_,
            {"expression": "sin(x)"},
            {"type": "antiderivative", "value": "-cos(x)"},
        ),
        (
            "Find the integral of e^(2x) from 0 to 1",
            I_,
            {"expression": "e^(2x)", "lower": "0", "upper": "1"},
            num("(exp(2) - 1)/2"),
        ),
        (
            "What does (x^2 - 4)/(x - 2) tend to as x tends to 2?",
            L,
            {"expression": "(x^2 - 4)/(x - 2)", "point": "2"},
            num("4"),
        ),
        (
            "As x goes to infinity, what does (3x + 1)/(x + 5) approach?",
            L,
            {"expression": "(3x + 1)/(x + 5)", "point": "oo"},
            num("3"),
        ),
        ("What is the limit of tan(x)/x when x approaches 0?", L, {"expression": "tan(x)/x", "point": "0"}, num("1")),
        (
            "Let A = [[2, 0], [0, 5]]. What's det(A)?",
            M,
            {"operation": "determinant", "matrix": [[2, 0], [0, 5]]},
            num("10"),
        ),
        (
            "Please invert the matrix [[2, 1], [1, 1]]",
            M,
            {"operation": "inverse", "matrix": [[2, 1], [1, 1]]},
            {"type": "matrix", "value": [["1", "-1"], ["-1", "2"]]},
        ),
        (
            "Compute the eigenvalues of A = [[6, 2], [2, 3]]",
            M,
            {"operation": "eigenvalues", "matrix": [[6, 2], [2, 3]]},
            {"type": "multiset", "values": ["2", "7"]},
        ),
        (
            "Here are some numbers: 4, 7, 10. What is their average?",
            "descriptive_statistics",
            {"data": [4, 7, 10]},
            {"type": "stats", "mean": "7", "median": "7", "population_std": "sqrt(6)", "sample_std": "3"},
        ),
        (
            "Find the standard deviation of the sample 1, 3, 5",
            "descriptive_statistics",
            {"data": [1, 3, 5]},
            {"type": "stats", "mean": "3", "median": "3", "population_std": "sqrt(8/3)", "sample_std": "2"},
        ),
        (
            "A coin is fair. Tossing it 6 times, what is the probability of exactly 2 heads?",
            P,
            {"distribution": "binomial", "params": {"n": 6, "p": 0.5}, "query": "pmf", "x": 2},
            num("15/64"),
        ),
        (
            "A machine produces defects with probability 0.05. In 20 independent items, what is the chance of no more than 1 defect?",
            P,
            {"distribution": "binomial", "params": {"n": 20, "p": 0.05}, "query": "cdf", "x": 1},
            num("(19/20)**20 + 20*(1/20)*(19/20)**19"),
        ),
        (
            "A shooter hits the target 80% of the time. Out of 5 shots, what is the probability of at least 4 hits?",
            P,
            {"distribution": "binomial", "params": {"n": 5, "p": 0.8}, "query": "greater", "x": 4},
            num("5*(4/5)**4*(1/5) + (4/5)**5"),
        ),
        (
            "Customers arrive at a rate of 3 per hour (Poisson). What is the probability that exactly 2 arrive in an hour?",
            P,
            {"distribution": "poisson", "params": {"lambda": 3}, "query": "pmf", "x": 2},
            num("9*exp(-3)/2"),
        ),
        (
            "Typos occur on average 2 per page (Poisson). Probability of more than 1 typo on a page?",
            P,
            {"distribution": "poisson", "params": {"lambda": 2}, "query": "greater", "x": 2},
            num("1 - 3*exp(-2)"),
        ),
        (
            "Weights are normal with mean 60 and standard deviation 5. What fraction weigh more than 70?",
            P,
            {"distribution": "normal", "params": {"mean": 60, "std": 5}, "query": "greater", "x": 70},
            num("(1 - erf(sqrt(2)))/2"),
        ),
        (
            "Scores follow a normal distribution with mean 500 and sd 100. Probability a score is between 400 and 600?",
            P,
            {"distribution": "normal", "params": {"mean": 500, "std": 100}, "query": "between", "x": 400, "x2": 600},
            num("erf(sqrt(2)/2)"),
        ),
        (
            "A train arrives uniformly between 0 and 30 minutes. Probability it arrives within 10 minutes?",
            P,
            {"distribution": "uniform", "params": {"a": 0, "b": 30}, "query": "cdf", "x": 10},
            num("1/3"),
        ),
        (
            "Waiting times are exponential with rate 0.5 per minute. What is the probability of waiting more than 4 minutes?",
            P,
            {"distribution": "exponential", "params": {"rate": 0.5}, "query": "greater", "x": 4},
            num("exp(-2)"),
        ),
        ("How many ways can 5 books be arranged on a shelf?", C, {"operation": "factorial", "n": 5}, num("120")),
        (
            "How many ways are there to choose 4 cards from 10?",
            C,
            {"operation": "combinations", "n": 10, "k": 4},
            num("210"),
        ),
        (
            "A club of 12 people elects a president and a treasurer. How many possible outcomes?",
            C,
            {"operation": "permutations", "n": 12, "k": 2},
            num("132"),
        ),
        # missing information in the tuning set too
        ("Integrate sin(x) from 0", None, None, CLARIFY),
        ("A coin is tossed several times. What is the probability of 2 heads?", None, None, CLARIFY),
        ("Solve 2x + 3y = 6", None, None, CLARIFY),
        ("Find the inverse", None, None, CLARIFY),
    ],
    1,
):
    _add(f"nl-{i:02d}", "dev_nl", problem, tool, args, truth, split="dev_nl")
