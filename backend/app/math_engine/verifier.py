"""Independent verification of tool results.

The verifier never trusts the solver's own reasoning. For every tool it uses a
*different* method to confirm the answer:

* equations      -> substitute each solution back (symbolic + numeric residual),
                    and cross-check real roots with SciPy's ``brentq`` root finder
* derivatives    -> central finite differences at random sample points
* integrals      -> differentiate the antiderivative back; SciPy ``quad`` for
                    definite integrals
* limits         -> numeric evaluation approaching the point
* simplify etc.  -> numeric equivalence at random points
* linear algebra -> NumPy recomputation / A·A⁻¹ = I / A·x = b residual
* statistics     -> NumPy / SciPy recomputation

Status semantics:

* ``verified``     an exact (symbolic) check passed, plus numeric checks
* ``approximate``  only numeric checks were possible and they passed, or the
                   solver itself returned an approximate value
* ``unverified``   no independent check could be carried out
* ``failed``       an independent check disagreed with the solver
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import sympy as sp
from scipy import integrate as sci_integrate
from scipy import optimize, stats

from app.math_engine.tools import ToolOutput
from app.schemas import Verification, VerificationCheck, VerificationStatus

REL_TOL = 1e-6
ABS_TOL = 1e-8


def _close(a: complex | float, b: complex | float, rel: float = REL_TOL, abs_: float = ABS_TOL) -> bool:
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def _sample_points(n: int = 6, seed: int = 7) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(0.3, 2.7) * rng.choice([-1, 1]) for _ in range(n)]


def _lambdify(expr: sp.Expr, var: sp.Symbol) -> Callable[[float], complex]:
    f = sp.lambdify(var, expr, modules=["numpy", "scipy"])

    def wrapped(x: float) -> complex:
        with np.errstate(all="ignore"):
            return complex(f(x))

    return wrapped


def _finite(z: complex) -> bool:
    return math.isfinite(z.real) and math.isfinite(z.imag)


def _symbolic_zero(expr: sp.Expr) -> bool | None:
    try:
        simplified = sp.simplify(expr)
    except Exception:
        return None
    if simplified == 0:
        return True
    return None if simplified.free_symbols else bool(sp.N(simplified) == 0) or None


def _finish(
    checks: list[VerificationCheck], exact_ok: bool, approximate: bool, verified_summary: str | None = None
) -> Verification:
    if not checks:
        return Verification(status=VerificationStatus.UNVERIFIED, summary="No independent check was possible.")
    if any(not c.passed for c in checks):
        failed = [c.method for c in checks if not c.passed]
        return Verification(
            status=VerificationStatus.FAILED,
            checks=checks,
            summary="Independent check disagreed with the solver: " + ", ".join(failed) + ".",
        )
    if exact_ok and not approximate:
        return Verification(
            status=VerificationStatus.VERIFIED,
            checks=checks,
            summary=verified_summary or "Confirmed by an exact symbolic check and independent numeric checks.",
        )
    return Verification(
        status=VerificationStatus.APPROXIMATE,
        checks=checks,
        summary="Confirmed numerically within tolerance; no exact symbolic proof.",
    )


# --------------------------------------------------------------------------- per-tool checks


def _verify_equation(out: ToolOutput) -> Verification:
    expr, var, sols = out.data["expr"], out.data["var"], out.data["solutions"]
    checks: list[VerificationCheck] = []
    exact_all = True
    f = _lambdify(expr, var)
    for s in sols:
        residual = expr.subs(var, s)
        exact = _symbolic_zero(residual) if not out.approximate else None
        try:
            r = f(complex(sp.N(s, 30)))
        except Exception:
            r = complex("nan")
        numeric_ok = _finite(r) and abs(r) < 1e-6 * max(1.0, abs(complex(sp.N(s))))
        if exact:
            checks.append(
                VerificationCheck(
                    method="substitution (symbolic)", passed=True, detail=f"f({sp.sstr(s)}) simplifies to 0"
                )
            )
        else:
            exact_all = False
            checks.append(
                VerificationCheck(
                    method="substitution (numeric)",
                    passed=numeric_ok,
                    detail=f"|f({sp.N(s, 8)})| = {abs(r):.2e}",
                )
            )
    # Independent root count: look for real sign changes on [-50, 50] and
    # make sure every real root found by brentq is among the solutions.
    real_found = _brentq_roots(f)
    if real_found is not None:
        claimed = [complex(sp.N(s)) for s in sols]
        missing = [r for r in real_found if not any(_close(r, c, 1e-5, 1e-6) for c in claimed)]
        checks.append(
            VerificationCheck(
                method="independent root search (scipy brentq on [-50, 50])",
                passed=not missing,
                detail="all sign-change roots accounted for"
                if not missing
                else f"solver missed real root(s) near {', '.join(f'{m.real:.6g}' for m in missing)}",
            )
        )
    return _finish(checks, exact_all, out.approximate)


def _brentq_roots(f: Callable[[float], complex]) -> list[float] | None:
    xs = np.linspace(-50, 50, 4001)
    try:
        ys = [f(float(x)) for x in xs]
    except Exception:
        return None
    roots: list[float] = []
    for i in range(len(xs) - 1):
        a, b = ys[i], ys[i + 1]
        if not (_finite(a) and _finite(b)) or abs(a.imag) > 1e-9 or abs(b.imag) > 1e-9:
            continue
        if a.real == 0:
            roots.append(float(xs[i]))
        elif a.real * b.real < 0:
            # Skip poles: a genuine root has small |f| near the bracket.
            try:
                r = optimize.brentq(lambda x: f(x).real, xs[i], xs[i + 1], xtol=1e-12)
            except Exception:
                continue
            if abs(f(r)) < 1e-6:
                roots.append(r)
    return roots


def _verify_system(out: ToolOutput) -> Verification:
    exprs, sols = out.data["exprs"], out.data["solutions"]
    checks = []
    exact_all = True
    for sol in sols:
        for e in exprs:
            residual = sp.simplify(e.subs(sol))
            ok = residual == 0
            exact_all &= ok
            if not ok and not residual.free_symbols:
                ok = abs(complex(sp.N(residual))) < 1e-9
            checks.append(
                VerificationCheck(method="substitution", passed=bool(ok), detail=f"{sp.sstr(e)} → {sp.sstr(residual)}")
            )
    return _finish(checks, exact_all, False)


def _verify_equivalence(out: ToolOutput) -> Verification:
    a, b = out.data["input"], out.data["result"]
    checks = []
    exact = _symbolic_zero(sp.expand(a - b)) or _symbolic_zero(a - b)
    if exact:
        checks.append(
            VerificationCheck(method="symbolic difference", passed=True, detail="input − result simplifies to 0")
        )
    free = sorted(a.free_symbols | b.free_symbols, key=lambda s: s.name)
    rng = random.Random(3)
    ok, tested = True, 0
    for _ in range(8):
        subs = {s: rng.uniform(0.2, 2.5) for s in free}
        try:
            va, vb = complex(sp.N(a.subs(subs))), complex(sp.N(b.subs(subs)))
        except Exception:
            continue
        if _finite(va) and _finite(vb):
            tested += 1
            ok &= _close(va, vb)
    if tested:
        checks.append(
            VerificationCheck(
                method="numeric equivalence at random points", passed=ok, detail=f"{tested} points tested"
            )
        )
    return _finish(checks, bool(exact), False)


def _verify_derivative(out: ToolOutput) -> Verification:
    expr, var, order, result = out.data["expr"], out.data["var"], out.data["order"], out.data["result"]
    if len(expr.free_symbols - {var}) > 0:
        return Verification(
            status=VerificationStatus.UNVERIFIED, summary="Contains other symbols; numeric check skipped."
        )
    f = _lambdify(expr, var)
    g = _lambdify(result, var)
    checks = []
    ok, tested, worst = True, 0, 0.0
    for x0 in _sample_points():
        try:
            numeric = _finite_difference(f, x0, order)
            claimed = g(x0)
        except Exception:
            continue
        if not (_finite(numeric) and _finite(claimed)):
            continue
        tested += 1
        err = abs(numeric - claimed) / max(1.0, abs(claimed))
        worst = max(worst, err)
        ok &= err < 1e-4 * (10 ** (order - 1))
    if tested:
        checks.append(
            VerificationCheck(
                method=f"central finite differences (order {order})",
                passed=ok,
                detail=f"{tested} random points, max relative error {worst:.1e}",
            )
        )
    # Exact check: integrating the derivative back must recover f up to a constant (order 1).
    exact = False
    if order == 1:
        try:
            back = sp.integrate(result, var)
            if not back.has(sp.Integral):
                diff = sp.simplify(sp.diff(expr - back, var))
                exact = diff == 0
                checks.append(
                    VerificationCheck(
                        method="integrate back (symbolic)",
                        passed=exact,
                        detail="∫f' dx − f is constant" if exact else f"residual {sp.sstr(diff)}",
                    )
                )
        except Exception:
            pass
    if "value" in out.data and order >= 1:
        expected = sp.N(sp.diff(expr, var, order).subs(var, out.data["point"]))
        checks.append(
            VerificationCheck(
                method="re-evaluate at point",
                passed=_close(complex(expected), complex(sp.N(out.data["value"]))),
                detail=f"{sp.N(out.data['value'], 8)}",
            )
        )
    return _finish(checks, exact, False)


def _finite_difference(f: Callable[[float], complex], x0: float, order: int) -> complex:
    h = 1e-3 if order > 1 else 1e-5
    if order == 1:
        return (f(x0 + h) - f(x0 - h)) / (2 * h)
    # Repeated central differences via the binomial formula.
    total = 0j
    for i in range(order + 1):
        total += (-1) ** i * math.comb(order, i) * f(x0 + (order / 2 - i) * h)
    return total / h**order


def _verify_integral(out: ToolOutput) -> Verification:
    expr, var = out.data["expr"], out.data["var"]
    antideriv = out.data["antiderivative"]
    checks: list[VerificationCheck] = []
    exact = False
    if not antideriv.has(sp.Integral):
        residual = sp.simplify(sp.diff(antideriv, var) - expr)
        exact = residual == 0
        if not exact and not (residual.free_symbols - {var}):
            # numeric fall-back for forms simplify cannot reduce
            r = _lambdify(residual, var)
            vals = [r(x) for x in _sample_points()]
            vals = [v for v in vals if _finite(v)]
            if vals and all(abs(v) < 1e-8 for v in vals):
                checks.append(
                    VerificationCheck(
                        method="differentiate antiderivative (numeric)", passed=True, detail="F'(x) − f(x) ≈ 0"
                    )
                )
            else:
                checks.append(
                    VerificationCheck(
                        method="differentiate antiderivative",
                        passed=False,
                        detail=f"F'(x) − f(x) = {sp.sstr(residual)}",
                    )
                )
        elif exact:
            checks.append(
                VerificationCheck(
                    method="differentiate antiderivative (symbolic)", passed=True, detail="d/dx F(x) − f(x) = 0"
                )
            )

    if "value" in out.data:
        a, b = out.data["lower"], out.data["upper"]
        if not (expr.free_symbols - {var}):
            f = _lambdify(expr, var)
            try:
                lo = -np.inf if a == -sp.oo else float(a)
                hi = np.inf if b == sp.oo else float(b)
                num, err = sci_integrate.quad(lambda x: f(x).real, lo, hi, limit=200)
                claimed = complex(sp.N(out.data["value"]))
                ok = _close(num, claimed, 1e-6, max(1e-8, 10 * err))
                checks.append(
                    VerificationCheck(
                        method="scipy.integrate.quad",
                        passed=ok,
                        detail=f"quad = {num:.10g} (±{err:.1e}), solver = {claimed.real:.10g}",
                    )
                )
            except Exception:
                pass  # quadrature not applicable (e.g. symbolic limits); other checks still apply
    return _finish(checks, exact, out.approximate)


def _verify_limit(out: ToolOutput) -> Verification:
    expr, var, pt, value = out.data["expr"], out.data["var"], out.data["point"], out.data["value"]
    if expr.free_symbols - {var}:
        return Verification(status=VerificationStatus.UNVERIFIED, summary="Contains other symbols.")
    f = _lambdify(sp.simplify(expr), var)
    checks = []
    sides = {"+": [1], "-": [-1], "+-": [1, -1]}[out.data["direction"]]
    target = complex(sp.N(value)) if value.is_finite else None
    for side in sides:
        if pt.is_infinite:
            xs = [side * 10**k for k in (3, 4, 5)] if pt == sp.oo else [-(10**k) for k in (3, 4, 5)]
        else:
            p = float(pt)
            xs = [p + side * 10 ** (-k) for k in (3, 4, 5)]
        try:
            ys = [f(x) for x in xs]
        except Exception:
            continue
        if target is not None:
            ok = all(_finite(y) for y in ys) and abs(ys[-1] - target) < 1e-3 * max(1, abs(target))
            detail = f"f → {ys[-1].real:.8g} (side {'+' if side > 0 else '-'})"
        else:
            ok = abs(ys[-1]) > abs(ys[0]) and abs(ys[-1]) > 1e3
            detail = f"|f| grows: {abs(ys[0]):.3g} → {abs(ys[-1]):.3g}"
        checks.append(VerificationCheck(method="numeric approach", passed=bool(ok), detail=detail))
    return _finish(checks, False, False)


def _verify_matrix(out: ToolOutput) -> Verification:
    op, A, result = out.data["operation"], out.data["A"], out.data["result"]
    if A.free_symbols:
        return Verification(
            status=VerificationStatus.UNVERIFIED, summary="Symbolic matrix entries; numeric check skipped."
        )
    An = np.array(A.evalf(), dtype=float)
    checks: list[VerificationCheck] = []
    exact = False
    if op == "determinant":
        nd = float(np.linalg.det(An))
        checks.append(
            VerificationCheck(
                method="numpy.linalg.det", passed=_close(nd, float(result), 1e-8, 1e-8), detail=f"{nd:.10g}"
            )
        )
        exact = sp.Matrix(A).det(method="berkowitz") == result
        checks.append(VerificationCheck(method="Berkowitz determinant (exact, different algorithm)", passed=exact))
    elif op == "inverse":
        ident = sp.simplify(A * result)
        exact = ident == sp.eye(A.rows)
        checks.append(VerificationCheck(method="A · A⁻¹ = I (exact)", passed=exact))
        ni = np.linalg.inv(An)
        checks.append(
            VerificationCheck(
                method="numpy.linalg.inv",
                passed=bool(np.allclose(ni, np.array(result.evalf(), dtype=float))),
                detail="entries agree",
            )
        )
    elif op == "eigenvalues":
        lam = sp.Symbol("lambda")
        cp = A.charpoly(lam).as_expr()
        exact = all(sp.simplify(cp.subs(lam, ev)) == 0 for ev in result)
        checks.append(VerificationCheck(method="eigenvalues are roots of det(A − λI) (exact)", passed=exact))
        npv = np.linalg.eigvals(An)
        claimed = [complex(sp.N(ev)) for ev, m in result.items() for _ in range(m)]
        ok = len(claimed) == len(npv) and all(any(_close(c, v, 1e-6, 1e-6) for v in npv) for c in claimed)
        checks.append(
            VerificationCheck(method="numpy.linalg.eigvals", passed=ok, detail=", ".join(f"{v:.6g}" for v in npv))
        )
    elif op == "rank":
        nr = int(np.linalg.matrix_rank(An))
        checks.append(VerificationCheck(method="numpy.linalg.matrix_rank (SVD)", passed=nr == result, detail=str(nr)))
        exact = A.rank() == result
    elif op == "multiply":
        B = out.data["B"]
        if B.free_symbols:
            return Verification(status=VerificationStatus.UNVERIFIED, summary="Symbolic entries.")
        Bn = np.array(B.evalf(), dtype=float)
        ok = np.allclose(An @ Bn, np.array(result.evalf(), dtype=float))
        checks.append(VerificationCheck(method="numpy matmul", passed=bool(ok)))
        exact = ok
    elif op == "solve":
        b = out.data["b"]
        residual = sp.simplify(A * result - b)
        exact = residual == sp.zeros(*b.shape)
        checks.append(VerificationCheck(method="A·x − b = 0 (exact substitution)", passed=exact))
        if not result.free_symbols and A.rows == A.cols and A.det() != 0:
            ns = np.linalg.solve(An, np.array(b.evalf(), dtype=float))
            checks.append(
                VerificationCheck(
                    method="numpy.linalg.solve", passed=bool(np.allclose(ns, np.array(result.evalf(), dtype=float)))
                )
            )
    elif op == "transpose":
        exact = result.T == A
        checks.append(VerificationCheck(method="(Aᵀ)ᵀ = A", passed=exact))
    elif op == "rref":
        exact = result.rank() == A.rank() and result == result.rref()[0]
        checks.append(VerificationCheck(method="result is in RREF with same rank", passed=exact))
    return _finish(checks, exact, False)


def _verify_stats(out: ToolOutput) -> Verification:
    values = np.array(out.data["values"], dtype=float)
    s = out.data["stats"]
    checks = []
    pairs = [
        ("mean", float(np.mean(values))),
        ("median", float(np.median(values))),
        ("population_variance", float(np.var(values))),
    ]
    if len(values) > 1:
        pairs.append(("sample_std", float(np.std(values, ddof=1))))
    for key, ref in pairs:
        claimed = float(sp.N(s[key]))
        checks.append(
            VerificationCheck(method=f"numpy {key}", passed=_close(claimed, ref, 1e-9, 1e-12), detail=f"{ref:.10g}")
        )
    return _finish(checks, True, False, "Exact SymPy statistics match an independent NumPy recomputation.")


def _verify_distribution(out: ToolOutput) -> Verification:
    d = out.data
    p, x, x2, q = d["params"], d["x"], d["x2"], d["query"]
    dist_name = d["distribution"]
    if dist_name == "binomial":
        rv: Any = stats.binom(int(p["n"]), p["p"])
    elif dist_name == "poisson":
        rv = stats.poisson(p.get("lambda", p.get("lam", p.get("mean"))))
    elif dist_name == "normal":
        rv = stats.norm(p.get("mean", p.get("mu", 0.0)), p.get("std", p.get("sigma", 1.0)))
    elif dist_name == "uniform":
        a, b = p.get("a", 0.0), p.get("b", 1.0)
        rv = stats.uniform(a, b - a)
    else:
        rv = stats.expon(scale=1 / p.get("rate", p.get("lambda", 1.0)))
    discrete = dist_name in ("binomial", "poisson")
    if q in ("pmf", "pdf"):
        ref = rv.pmf(x) if discrete else rv.pdf(x)
    elif q == "cdf":
        ref = rv.cdf(x)
    elif q == "greater":
        ref = rv.sf(math.ceil(x) - 1) if discrete else rv.sf(x)
    else:
        ref = (rv.cdf(math.floor(x2)) - rv.cdf(math.ceil(x) - 1)) if discrete else rv.cdf(x2) - rv.cdf(x)
    ok = _close(float(ref), d["numeric"], 1e-7, 1e-12)
    check = VerificationCheck(method=f"scipy.stats.{rv.dist.name}", passed=ok, detail=f"{float(ref):.10g}")
    return _finish([check], True, False, "Exact SymPy probability matches an independent SciPy computation.")


def _verify_combinatorics(out: ToolOutput) -> Verification:
    d = out.data
    n = int(d["n"])
    if d["operation"] == "factorial":
        ref = math.factorial(n)
    elif d["operation"] == "combinations":
        ref = math.comb(n, int(d["k"]))
    else:
        ref = math.perm(n, int(d["k"]))
    ok = int(d["result"]) == ref
    return _finish(
        [VerificationCheck(method="Python math (integer arithmetic)", passed=ok, detail=str(ref)[:60])],
        True,
        False,
        "Exact result matches independent integer arithmetic.",
    )


VERIFIERS: dict[str, Callable[[ToolOutput], Verification]] = {
    "solve_equation": _verify_equation,
    "solve_system": _verify_system,
    "simplify_expression": _verify_equivalence,
    "factor_expression": _verify_equivalence,
    "expand_expression": _verify_equivalence,
    "differentiate": _verify_derivative,
    "integrate": _verify_integral,
    "limit": _verify_limit,
    "matrix_operation": _verify_matrix,
    "descriptive_statistics": _verify_stats,
    "probability_distribution": _verify_distribution,
    "combinatorics": _verify_combinatorics,
}


def verify(tool: str, output: ToolOutput) -> Verification:
    fn = VERIFIERS.get(tool)
    if fn is None:
        return Verification(status=VerificationStatus.UNVERIFIED, summary="No verifier for this tool.")
    try:
        return fn(output)
    except Exception as exc:  # the verifier must never crash the pipeline
        return Verification(
            status=VerificationStatus.UNVERIFIED,
            summary=f"Verification could not be completed: {type(exc).__name__}: {exc}"[:300],
        )
