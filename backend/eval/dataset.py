"""MathForge benchmark dataset.

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
        ("Find all real x such that x^2 = 16", EQ, {"equation": "x^2 = 16"}, roots("4", "-4")),
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
