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
import warnings
from collections.abc import Callable
from typing import Any

import mpmath as mp
import numpy as np
import sympy as sp
from scipy import integrate as sci_integrate
from scipy import optimize, stats

from app.math_engine.tools import ToolOutput
from app.schemas import Verification, VerificationCheck, VerificationStatus

REL_TOL = 1e-6
ABS_TOL = 1e-8


def _close(a: complex | float, b: complex | float, rel: float = REL_TOL, abs_: float = ABS_TOL) -> bool:
    a, b = complex(a), complex(b)
    if not (_finite(a) and _finite(b)):
        # inf - inf is nan and |inf| <= rel*inf is True: never compare infinities numerically.
        return _finite(a) == _finite(b) and a == b
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
    period = out.data.get("period")
    checks: list[VerificationCheck] = []
    exact_all = True
    f = _lambdify(expr, var)

    def df(z: complex) -> complex:  # central difference: robust for |x|, piecewise, etc.
        h = 1e-6 * max(1.0, abs(z))
        return (f(z + h) - f(z - h)) / (2 * h)

    for s in sols:
        residual = expr.subs(var, s)
        exact = _symbolic_zero(residual) if not out.approximate else None
        try:
            z = complex(sp.N(s, 30))
            r = f(z)
            # A genuine zero has a tiny Newton step |f/f'|, not merely a small |f|
            # (rejects "roots" where f only decays towards 0, e.g. 1/(x-1) at x = -1e7).
            slope = df(z)
            step = abs(r / slope) if _finite(slope) and slope != 0 else 0.0
        except Exception:
            z, r, step = complex("nan"), complex("nan"), float("inf")
        numeric_ok = _finite(r) and abs(r) < 1e-6 * max(1.0, abs(z)) and step < 1e-6 * max(1.0, abs(z))
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
                    method="substitution (numeric, residual and Newton step)",
                    passed=bool(numeric_ok),
                    detail=f"|f({sp.N(s, 8)})| = {abs(r):.2e}, |f/f'| = {step:.2e}",
                )
            )
    claimed = [complex(sp.N(s)) for s in sols]
    domain = out.data.get("domain", "complex")

    def in_domain(z: complex) -> bool:
        return domain == "complex" or (abs(z.imag) < 1e-9 and (domain == "real" or z.real > 1e-12))

    for c in claimed:
        if not in_domain(c):
            checks.append(
                VerificationCheck(method=f"solutions lie in the {domain} domain", passed=False, detail=f"{c:.6g}")
            )

    def accounted(root: complex) -> bool:
        if not in_domain(root):
            return True
        for c in claimed:
            if period is not None:
                t = float(sp.N(period))
                k = round((root.real - c.real) / t)
                if abs(c.imag) < 1e-9 and _close(root, c + k * t, 1e-5, 1e-6):
                    return True
            elif _close(root, c, 1e-5, 1e-6):
                return True
        return False

    # Independent real-root search: sign changes on [-50, 50] refined with brentq.
    real_found = _brentq_roots(f)
    if real_found is not None:
        missing = [r for r in real_found if not accounted(complex(r))]
        checks.append(
            VerificationCheck(
                method="independent root search (scipy brentq on [-50, 50])",
                passed=not missing,
                detail="all sign-change roots accounted for"
                if not missing
                else f"solver missed real root(s) near {', '.join(f'{m:.6g}' for m in missing[:5])}",
            )
        )
    poly_check = _polynomial_root_check(expr, var, claimed, in_domain)
    if poly_check is not None:
        checks.append(poly_check)
    return _finish(checks, exact_all, out.approximate)


def _polynomial_root_check(
    expr: sp.Expr, var: sp.Symbol, claimed: list[complex], in_domain: Callable[[complex], bool] = lambda z: True
) -> VerificationCheck | None:
    """For polynomials, compare against *all* complex roots from numpy.roots
    (companion-matrix eigenvalues) of the square-free part: catches missing
    complex roots and extra values that brentq cannot see."""
    try:
        num, _den = sp.fraction(sp.together(expr))
        if num.free_symbols != {var} or not num.is_polynomial(var):
            return None
        poly = sp.Poly(num, var)
        if poly.degree() < 1 or poly.degree() > 30:
            return None
        sqf = sp.Poly(sp.quo(poly, sp.gcd(poly, poly.diff(var))), var)
        coeffs = [complex(sp.N(c)) for c in sqf.all_coeffs()]
    except Exception:
        return None
    roots = np.roots(coeffs)
    # roots of the numerator that are poles of the full expression are excluded
    f = _lambdify(expr, var)
    roots = [r for r in roots if _finite(f(complex(r))) and in_domain(complex(r))]
    missing = [r for r in roots if not any(_close(r, c, 1e-6, 1e-7) for c in claimed)]
    extra = [c for c in claimed if not any(_close(r, c, 1e-6, 1e-7) for r in roots)]
    detail = f"{len(roots)} distinct root(s) expected"
    if missing:
        detail += "; missing " + ", ".join(f"{m:.6g}" for m in missing[:5])
    if extra:
        detail += "; not a root: " + ", ".join(f"{e:.6g}" for e in extra[:5])
    return VerificationCheck(
        method="complete root set (numpy.roots on square-free polynomial)",
        passed=not missing and not extra,
        detail=detail,
    )


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

    if "value" not in out.data:
        return _finish(checks, exact, out.approximate)
    if expr.free_symbols - {var}:
        return Verification(status=VerificationStatus.UNVERIFIED, checks=checks, summary="Contains other symbols.")
    value = out.data["value"]
    f = _lambdify(expr, var)
    a, b = out.data["lower"], out.data["upper"]
    sing = [float(sp.N(p)) for p in out.data.get("singularities", [])]
    if value.is_infinite:
        checks.append(_divergence_check(f, a, b, sing, 1 if value == sp.oo else -1))
        return _finish(checks, False, True)
    numeric = _reliable_quadrature(f, a, b, sing)
    if numeric is None:
        return Verification(
            status=VerificationStatus.UNVERIFIED,
            checks=checks,
            summary="The definite value could not be confirmed: numerical quadrature did not converge reliably.",
        )
    num, method = numeric
    claimed = complex(sp.N(value))
    checks.append(
        VerificationCheck(
            method=method,
            passed=_close(num, claimed, 1e-7, 1e-9),
            detail=f"numeric = {num:.12g}, solver = {claimed.real:.12g}",
        )
    )
    return _finish(checks, exact, out.approximate)


def _bounds(a: sp.Expr, b: sp.Expr) -> tuple[float, float]:
    lo = -np.inf if a == -sp.oo else np.inf if a == sp.oo else float(a)
    hi = np.inf if b == sp.oo else -np.inf if b == -sp.oo else float(b)
    return lo, hi


def _reliable_quadrature(
    f: Callable[[float], complex], a: sp.Expr, b: sp.Expr, sing: list[float]
) -> tuple[float, str] | None:
    """Definite integral by SciPy quad, falling back to mpmath tanh-sinh. Returns
    None unless one of them reports a small error estimate without warnings."""
    lo, hi = _bounds(a, b)
    inner = sorted(p for p in sing if min(lo, hi) < p < max(lo, hi))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            kw = {"points": inner} if inner and np.isfinite(lo) and np.isfinite(hi) else {}
            if inner and not kw:
                raise ValueError("singular point with infinite bound")
            num, err = sci_integrate.quad(lambda x: f(x).real, lo, hi, limit=400, **kw)
        if np.isfinite(num) and err <= 1e-8 * max(1.0, abs(num)):
            return float(num), "scipy.integrate.quad"
    except Exception:
        pass
    try:
        mp.mp.dps = 30
        nodes = [mp.mpf(lo) if np.isfinite(lo) else (mp.inf if lo > 0 else -mp.inf)]
        nodes += [mp.mpf(p) for p in inner]
        nodes += [mp.mpf(hi) if np.isfinite(hi) else (mp.inf if hi > 0 else -mp.inf)]
        val, err = mp.quad(lambda t: mp.re(mp.mpc(f(float(t)))), nodes, error=True, maxdegree=8)
        if mp.isfinite(val) and err <= 1e-9 * max(1, abs(val)):
            return float(val), "mpmath.quad (tanh-sinh)"
    except Exception:
        pass
    return None


def _divergence_check(
    f: Callable[[float], complex], a: sp.Expr, b: sp.Expr, sing: list[float], sign: int
) -> VerificationCheck:
    """Integrate over the interval with shrinking neighbourhoods of the singular
    points removed (and growing truncations of infinite bounds). A divergent
    integral keeps growing by non-shrinking increments with the claimed sign."""
    lo, hi = _bounds(a, b)
    flip = -1 if lo > hi else 1
    lo, hi = min(lo, hi), max(lo, hi)
    bad = sorted({p for p in sing if lo <= p <= hi})
    values = []
    for eps in (1e-2, 1e-4, 1e-6, 1e-8):
        left = lo if np.isfinite(lo) else -1 / eps
        right = hi if np.isfinite(hi) else 1 / eps
        cuts = [left]
        for p in bad:
            cuts += [p - eps, p + eps]
        cuts.append(right)
        total = 0.0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i in range(0, len(cuts), 2):
                    x0, x1 = max(cuts[i], left), min(cuts[i + 1], right)
                    if x1 > x0:
                        total += sci_integrate.quad(lambda x: f(x).real, x0, x1, limit=400)[0]
        except Exception:
            return VerificationCheck(method="truncated-integral growth", passed=False, detail="evaluation failed")
        values.append(flip * total)
    inc = np.diff(values)
    ok = all(d * sign > 0 for d in inc) and abs(inc[-1]) >= 0.5 * abs(inc[0])
    return VerificationCheck(
        method="truncated-integral growth (divergence test)",
        passed=bool(ok),
        detail="truncated values " + ", ".join(f"{v:.4g}" for v in values),
    )


def _verify_limit(out: ToolOutput) -> Verification:
    """Evaluate f with 80-digit mpmath arithmetic along x_k -> point (avoids the
    cancellation that double precision suffers in forms like sqrt(x^2+x) - x)."""
    expr, var, pt, value = out.data["expr"], out.data["var"], out.data["point"], out.data["value"]
    if expr.free_symbols - {var}:
        return Verification(status=VerificationStatus.UNVERIFIED, summary="Contains other symbols.")
    f = sp.lambdify(var, expr, modules="mpmath")
    checks = []
    exponents = (4, 8, 12, 16)
    with mp.workdps(80):
        if pt.is_infinite:
            sign_pt = 1 if pt == sp.oo else -1
            paths = {"→∞" if sign_pt > 0 else "→-∞": [sign_pt * mp.mpf(10) ** k for k in exponents]}
        else:
            p = mp.mpf(str(sp.N(pt, 50)))
            sides = {"+": [1], "-": [-1], "+-": [1, -1]}[out.data["direction"]]
            paths = {("+" if s > 0 else "-"): [p + s * mp.mpf(10) ** -k for k in exponents] for s in sides}
        for side, xs in paths.items():
            try:
                ys = [mp.mpc(f(x)) for x in xs]
            except Exception:
                continue
            real = all(abs(y.imag) < mp.mpf(10) ** -30 for y in ys)
            if value.is_finite:
                target = mp.mpc(complex(sp.N(value, 30)))
                err = abs(ys[-1] - target)
                ok = err < 1e-8 * max(1, abs(target))
                detail = f"side {side}: f = {mp.nstr(ys[-1], 12)} (|error| = {mp.nstr(err, 3)})"
            else:
                sign = 1 if value == sp.oo else -1
                inc = [sign * (b.real - a.real) for a, b in zip(ys, ys[1:], strict=False)]
                # divergence: keeps moving in the claimed direction by non-shrinking steps
                ok = real and all(d > 0 for d in inc) and inc[-1] >= 0.5 * inc[0]
                detail = f"side {side}: f = " + ", ".join(mp.nstr(y.real, 4) for y in ys)
            checks.append(
                VerificationCheck(method="numeric approach (80-digit mpmath)", passed=bool(ok), detail=detail)
            )
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
