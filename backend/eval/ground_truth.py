"""Independent validation of the benchmark's ground truth.

This module deliberately does NOT import any application code. SymPy is used
only to *parse* the dataset strings into functions; every check is numerical
and uses mpmath (high-precision root finding, differentiation, quadrature,
linear algebra), NumPy or SciPy:

    python -m eval.ground_truth        # exits non-zero if any truth is wrong
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from typing import Any

import mpmath as mp
import numpy as np
import sympy as sp
from scipy import stats
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from eval.dataset import CASES

mp.mp.dps = 40
_T = standard_transformations + (convert_xor, implicit_multiplication_application)
_LOCALS = {"e": sp.E, "E": sp.E, "I": sp.I, "pi": sp.pi, "oo": sp.oo, "ln": sp.log}


def P(text: Any) -> sp.Expr:
    """Parse a dataset string (problem notation or SymPy syntax)."""
    if isinstance(text, (int, float)):
        return sp.nsimplify(text, rational=True)
    return parse_expr(str(text), local_dict=dict(_LOCALS), transformations=_T)


def V(text: Any) -> mp.mpc:
    value = P(text)
    if value in (sp.oo, -sp.oo):
        return mp.inf if value == sp.oo else -mp.inf
    re_part, im_part = sp.N(value, 45).as_real_imag()
    if im_part == 0:
        return mp.mpf(str(re_part))
    return mp.mpc(mp.mpf(str(re_part)), mp.mpf(str(im_part)))


def F(text: Any, var: str = "x") -> Callable[[Any], Any]:
    expr = P(text)
    x = sp.Symbol(var)
    expr = expr.subs({s: x for s in expr.free_symbols if s.name == var})
    return sp.lambdify(x, expr, modules="mpmath")


def close(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if mp.isinf(a) or mp.isinf(b):
        return a == b
    return abs(a - b) <= tol * max(1, abs(a), abs(b))


def _sides(eq: str) -> Callable[[Any], Any]:
    lhs, _, rhs = eq.partition("=")
    expr = P(f"({lhs}) - ({rhs or 0})")
    names = sorted(s.name for s in expr.free_symbols)
    var = names[0] if len(names) == 1 else "x"
    return F(f"({lhs}) - ({rhs or 0})", var)


def _real_scan(f: Callable, lo: float, hi: float, n: int = 6001) -> list[mp.mpf]:
    xs = np.linspace(lo, hi, n)
    roots = []
    prev = None
    for a, b in zip(xs[:-1], xs[1:], strict=True):
        try:
            fa, fb = f(mp.mpf(a)), f(mp.mpf(b))
        except (ZeroDivisionError, ValueError):
            continue
        if abs(mp.im(fa)) > 1e-20 or abs(mp.im(fb)) > 1e-20:
            continue
        if mp.re(fa) * mp.re(fb) <= 0:
            try:
                r = mp.findroot(lambda t: mp.re(f(t)), (mp.mpf(a), mp.mpf(b)), solver="anderson")
            except Exception:
                continue
            if abs(f(r)) < 1e-20 and (prev is None or abs(r - prev) > 1e-9):
                roots.append(r)
                prev = r
    return roots


# ----------------------------------------------------------------------- validators


def check_roots(case: dict) -> str | None:
    eq = case["args"]["equation"]
    f = _sides(eq)
    truth = [V(v) for v in case["truth"]["values"]]
    for r in truth:
        if abs(f(r)) > 1e-15:  # decimal truths carry 20 significant digits
            return f"f({r}) = {f(r)} is not zero"
    domain = case["truth"].get("domain", "complex")

    def in_domain(z: Any) -> bool:
        if domain == "complex":
            return True
        real = abs(mp.im(z)) < 1e-20
        return real and (domain == "real" or mp.re(z) > 0)

    for r in truth:
        if not in_domain(r):
            return f"truth value {r} violates domain {domain}"
    expr = P(eq.split("=")[0]) - P(eq.split("=")[1] if "=" in eq else 0)
    sym = sorted(expr.free_symbols, key=lambda s: s.name)[0]
    num, _ = sp.fraction(sp.together(expr))
    if num.is_polynomial(sym):
        poly = sp.Poly(num, sym)
        coeffs = [complex(c) for c in poly.all_coeffs()]
        found = mp.polyroots(coeffs, maxsteps=200, extraprec=200)
        found = [z for z in found if mp.isfinite(f(z)) if not _is_pole(f, z) and in_domain(z)]
        missing = [z for z in found if not any(close(z, t, 1e-10) for t in truth)]
        if missing:
            return f"truth misses polynomial roots {missing}"
    for r in _real_scan(f, -50, 50):
        if in_domain(r) and not any(close(r, t, 1e-10) for t in truth):
            return f"truth misses real root {r}"
    return None


def _is_pole(f: Callable, z: Any) -> bool:
    try:
        return not mp.isfinite(f(z + mp.mpf("1e-30")))
    except ZeroDivisionError:
        return True


def check_periodic(case: dict) -> str | None:
    f = _sides(case["args"]["equation"])
    bases = [V(b) for b in case["truth"]["bases"]]
    period = V(case["truth"]["period"])
    for b in bases:
        if abs(f(b)) > 1e-25:
            return f"f({b}) != 0"
    for r in _real_scan(f, -20, 20, 8001):
        if not any(abs((r - b) / period - mp.nint((r - b) / period)) < 1e-12 for b in bases):
            return f"root {r} not in the periodic family"
    return None


def check_system(case: dict) -> str | None:
    eqs = case["args"]["equations"]
    for sol in case["truth"]["solutions"]:
        subs = {sp.Symbol(k): P(v) for k, v in sol.items()}
        for e in eqs:
            lhs, rhs = e.split("=")
            val = complex(sp.N((P(lhs) - P(rhs)).subs(subs), 30))
            if abs(val) > 1e-20:
                return f"{sol} does not satisfy {e}"
    exprs = [P(e.split("=")[0]) - P(e.split("=")[1]) for e in eqs]
    symbols = sorted(set().union(*(x.free_symbols for x in exprs)), key=lambda s: s.name)
    # count check: Groebner-free numeric search from many starts must find no other solution
    rng = np.random.default_rng(0)
    fs = [sp.lambdify(symbols, x, "mpmath") for x in exprs]
    truth_pts = [[V(sol[s.name]) for s in symbols] for sol in case["truth"]["solutions"]]
    if len(exprs) == len(symbols):
        for _ in range(30):
            start = [mp.mpf(float(v)) for v in rng.uniform(-6, 6, len(symbols))]
            try:
                root = mp.findroot(fs, start)
            except Exception:
                continue
            pt = [root[i] for i in range(len(symbols))] if len(symbols) > 1 else [root]
            if max(abs(f(*pt)) for f in fs) > 1e-20:
                continue
            if not any(all(close(a, b, 1e-10) for a, b in zip(pt, t, strict=True)) for t in truth_pts):
                return f"truth misses solution {pt}"
    return None


def _points(n: int = 5) -> list[mp.mpf]:
    return [mp.mpf(v) for v in (0.37, 0.81, 1.23, 1.67, 2.11)[:n]]


def check_expr_equiv(case: dict) -> str | None:
    src, truth = P(case["args"]["expression"]), P(case["truth"]["value"])
    symbols = sorted(src.free_symbols | truth.free_symbols, key=lambda s: s.name)
    fa, fb = sp.lambdify(symbols, src, "mpmath"), sp.lambdify(symbols, truth, "mpmath")
    for _i, x0 in enumerate(_points()):
        vals = [x0 + mp.mpf(j) / 7 for j in range(len(symbols))] if symbols else []
        a, b = fa(*vals), fb(*vals)
        if not close(a, b, 1e-25):
            return f"input and truth differ at {vals}: {a} vs {b}"
    form = case["truth"].get("form")
    if form == "factored" and not (truth.is_Mul or truth.is_Pow):
        return "truth is not written as a product"
    if form == "expanded" and truth.is_Mul:
        return "truth is not expanded"
    return None


def check_derivative(case: dict) -> str | None:
    args = case["args"]
    var = args.get("variable") or "x"
    f = F(args["expression"], var)
    order = int(args.get("order", 1))
    if case["truth"]["type"] == "number":
        x0 = V(args["point"])
        num_value = mp.diff(f, x0, order)
        return None if close(num_value, V(case["truth"]["value"]), 1e-20) else f"f'({x0}) = {num_value}"
    g = F(case["truth"]["value"], var)
    for x0 in _points():
        if not close(mp.diff(f, x0, order), g(x0), 1e-20):
            return f"mismatch at {x0}: {mp.diff(f, x0, order)} vs {g(x0)}"
    return None


def check_antiderivative(case: dict) -> str | None:
    f, big_f = F(case["args"]["expression"]), F(case["truth"]["value"])
    for x0 in _points():
        if not close(mp.diff(big_f, x0), f(x0), 1e-20):
            return f"F'({x0}) != f({x0})"
    return None


def check_definite(case: dict) -> str | None:
    args = case["args"]
    f = F(args["expression"])
    lo, hi = V(args["lower"]), V(args["upper"])
    truth = V(case["truth"]["value"])
    nodes = [lo, hi]
    if mp.isfinite(lo) and mp.isfinite(hi) and lo < 0 < hi:
        nodes = [lo, mp.mpf(0), hi]
    if mp.isinf(truth):
        # divergence: truncated integrals keep growing (log-type growth for 1/x)
        vals = [mp.quad(f, [lo + mp.mpf(10) ** -k, hi]) for k in (4, 8, 12)]
        ok = vals[0] < vals[1] < vals[2] and (vals[2] - vals[1]) > 0.5 * (vals[1] - vals[0])
        return None if ok and truth == mp.inf else f"not divergent to {truth}: {vals}"
    value = mp.quad(f, nodes)
    return None if close(value, truth, 1e-15) else f"quad = {value}, truth = {truth}"


def check_limit(case: dict) -> str | None:
    args = case["args"]
    f = F(args["expression"])
    pt, truth = V(args["point"]), V(case["truth"]["value"])
    direction = args.get("direction", "+-")
    mp.mp.dps = 200  # avoid cancellation, e.g. 1 - cos(1e-30)
    try:
        exps = (10, 30, 60)
        if mp.isinf(pt):
            paths = [[mp.mpf(10) ** k * (1 if pt > 0 else -1) for k in exps]]
        else:
            sides = {"+": [1], "-": [-1], "+-": [1, -1]}[direction]
            paths = [[pt + s * mp.mpf(10) ** -k for k in exps] for s in sides]
        values = [[f(x) for x in path] for path in paths]
    finally:
        mp.mp.dps = 40
    for vals in values:
        if mp.isinf(truth):
            sign = 1 if truth > 0 else -1
            real = all(abs(mp.im(v)) < 1e-20 for v in vals)
            grows = all(sign * mp.re(b) > sign * mp.re(a) for a, b in zip(vals, vals[1:], strict=False))
            if not (real and grows and sign * mp.re(vals[-1]) > 2 * abs(vals[0])):
                return f"values {vals} do not diverge to {truth}"
        elif not close(vals[1], truth, 1e-12):
            return f"value near point {vals[1]} != {truth}"
    return None


def _mat(rows: list[list[Any]]) -> mp.matrix:
    return mp.matrix([[V(v) for v in r] for r in rows])


def check_matrix(case: dict) -> str | None:
    args, truth = case["args"], case["truth"]
    op = args["operation"]
    A = _mat(args["matrix"])
    if op == "determinant":
        return None if close(mp.det(A), V(truth["value"])) else f"det = {mp.det(A)}"
    if op == "rank":
        r = int(np.linalg.matrix_rank(np.array(args["matrix"], dtype=float)))
        return None if r == int(truth["value"]) else f"rank = {r}"
    if op == "eigenvalues":
        ev = sorted(mp.eig(A)[0], key=lambda z: (float(mp.re(z)), float(mp.im(z))))
        want = sorted((V(v) for v in truth["values"]), key=lambda z: (float(mp.re(z)), float(mp.im(z))))
        ok = len(ev) == len(want) and all(close(a, b, 1e-15) for a, b in zip(ev, want, strict=True))
        return None if ok else f"eig = {ev}"
    if op == "solve":
        x = mp.lu_solve(A, mp.matrix([V(v) for v in args["vector"]]))
        want = [V(v) for v in truth["value"]]
        return None if all(close(x[i], want[i]) for i in range(len(want))) else f"x = {x}"
    T = _mat(truth["value"])
    if op == "inverse":
        expected = mp.inverse(A)
    elif op == "transpose":
        expected = A.T
    elif op == "multiply":
        expected = A * _mat(args["other"])
    elif op == "rref":
        # T must be in RREF and span the same row space as A
        Tn, An = np.array(T.tolist(), dtype=float), np.array(A.tolist(), dtype=float)
        r = np.linalg.matrix_rank
        same_space = r(An) == r(Tn) == r(np.vstack([An, Tn]))
        pivots = [next((j for j, v in enumerate(row) if abs(v) > 1e-12), None) for row in Tn]
        nz = [p for p in pivots if p is not None]
        rref = nz == sorted(nz) and all(
            abs(Tn[i, p] - 1) < 1e-12 and sum(abs(Tn[:, p])) - 1 < 1e-12 for i, p in enumerate(nz)
        )
        return None if same_space and rref else "not the RREF of A"
    else:
        return f"unknown op {op}"
    ok = (
        expected.rows == T.rows
        and expected.cols == T.cols
        and all(close(expected[i, j], T[i, j]) for i in range(T.rows) for j in range(T.cols))
    )
    return None if ok else f"expected {expected}"


def check_stats(case: dict) -> str | None:
    data = np.array(case["args"]["data"], dtype=float)
    t = case["truth"]
    ref = {
        "mean": np.mean(data),
        "median": np.median(data),
        "population_std": np.std(data),
        "sample_std": np.std(data, ddof=1),
    }
    for k, v in ref.items():
        if not math.isclose(float(V(t[k])), float(v), rel_tol=1e-12, abs_tol=1e-12):
            return f"{k}: numpy {v} vs truth {t[k]}"
    return None


def check_probability(case: dict) -> str | None:
    a = case["args"]
    p, q, x, x2 = a["params"], a["query"], float(a["x"]), a.get("x2")
    d = a["distribution"]
    rv = {
        "binomial": lambda: stats.binom(int(p["n"]), float(p["p"])),
        "poisson": lambda: stats.poisson(float(p["lambda"])),
        "normal": lambda: stats.norm(float(p["mean"]), float(p["std"])),
        "uniform": lambda: stats.uniform(float(p["a"]), float(p["b"]) - float(p["a"])),
        "exponential": lambda: stats.expon(scale=1 / float(p["rate"])),
    }[d]()
    discrete = d in ("binomial", "poisson")
    if q == "pmf":
        ref = rv.pmf(x)
    elif q == "cdf":
        ref = rv.cdf(x)
    elif q == "greater":
        ref = rv.sf(x - 1) if discrete else rv.sf(x)
    else:
        ref = rv.cdf(float(x2)) - rv.cdf(x)
    truth = float(sp.N(P(case["truth"]["value"]).doit(), 30))
    return None if math.isclose(truth, float(ref), rel_tol=1e-9, abs_tol=1e-14) else f"scipy {ref} vs truth {truth}"


def check_combinatorics(case: dict) -> str | None:
    a = case["args"]
    n, k = int(a["n"]), a.get("k")
    ref = {
        "factorial": lambda: math.factorial(n),
        "combinations": lambda: math.comb(n, int(k)),
        "permutations": lambda: math.perm(n, int(k)),
    }[a["operation"]]()
    return None if ref == int(P(case["truth"]["value"])) else f"math gives {ref}"


def validate(case: dict) -> str | None:
    """Return None if the case's truth is independently confirmed, else a reason."""
    t, tool = case["truth"]["type"], case["tool"]
    if t == "reject":
        return None
    if t == "roots":
        return check_roots(case)
    if t == "roots_periodic":
        return check_periodic(case)
    if t == "system":
        return check_system(case)
    if t == "expr" and tool in ("simplify_expression", "factor_expression", "expand_expression"):
        return check_expr_equiv(case)
    if tool == "differentiate":
        return check_derivative(case)
    if t == "antiderivative":
        return check_antiderivative(case)
    if tool == "integrate":
        return check_definite(case)
    if tool == "limit":
        return check_limit(case)
    if tool == "matrix_operation":
        return check_matrix(case)
    if t == "stats":
        return check_stats(case)
    if tool == "probability_distribution":
        return check_probability(case)
    if tool == "combinatorics":
        return check_combinatorics(case)
    return f"no validator for {t}/{tool}"


def main() -> int:
    failures = 0
    checked = 0
    for case in CASES:
        if case["truth"]["type"] == "reject":
            continue
        checked += 1
        try:
            problem = validate(case)
        except Exception as exc:  # report, never hide
            problem = f"validator error: {type(exc).__name__}: {exc}"
        if problem:
            failures += 1
            print(f"FAIL {case['id']}: {problem}")
    print(f"ground truth: {checked - failures}/{checked} cases independently confirmed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
