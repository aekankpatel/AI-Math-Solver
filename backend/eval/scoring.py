"""Compare application outputs with benchmark ground truth.

Comparisons are numerical and assumption-agnostic: symbols are matched by
name (the app uses real-valued symbols for calculus), values are compared at
30 significant digits with a relative tolerance of 1e-9.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

from eval.ground_truth import P

TOL = 1e-9
_PTS = (0.37, 0.81, 1.23, 1.67, 2.11)


def _plain(e: Any) -> sp.Basic:
    e = sp.sympify(e)
    return e.subs({s: sp.Symbol(s.name) for s in e.free_symbols}) if hasattr(e, "free_symbols") else e


def _c(v: Any) -> complex:
    v = sp.sympify(v)
    if v in (sp.oo, -sp.oo, sp.zoo):
        return complex(float(v) if v != sp.zoo else float("nan"), 0)
    return complex(sp.N(v, 30))


def num_equal(a: Any, b: Any, tol: float = TOL) -> bool:
    try:
        sa, sb = sp.sympify(a), sp.sympify(b)
        if sa.is_infinite or sb.is_infinite:
            return bool(sa == sb)
        ca, cb = _c(sa), _c(sb)
    except (TypeError, ValueError, AttributeError):
        return False
    return abs(ca - cb) <= tol * max(1.0, abs(ca), abs(cb))


def _expr_equal(a: Any, b: Any, up_to_constant: bool = False) -> bool:
    a, b = _plain(a), _plain(b)
    symbols = sorted(a.free_symbols | b.free_symbols, key=lambda s: s.name)
    diffs = []
    for _i, x0 in enumerate(_PTS):
        subs = {s: sp.Rational(str(x0)) + sp.Rational(j, 7) for j, s in enumerate(symbols)}
        try:
            diffs.append(_c(a.subs(subs)) - _c(b.subs(subs)))
        except (TypeError, ValueError):
            return False
    if up_to_constant:
        return all(abs(d - diffs[0]) <= TOL * max(1.0, abs(diffs[0])) for d in diffs)
    return all(abs(d) <= TOL * max(1.0, abs(_c(b.subs({s: 1 for s in symbols})))) for d in diffs)


def _periodic_family(bases: list[Any], period: Any, lo: float = -20, hi: float = 20) -> list[float]:
    t = float(sp.N(period))
    out = []
    for b in bases:
        b = float(sp.N(b))
        k0, k1 = int((lo - b) // t) - 1, int((hi - b) // t) + 1
        out += [b + k * t for k in range(k0, k1 + 1) if lo <= b + k * t <= hi]
    return sorted(out)


def _same_multiset(a: list[Any], b: list[Any]) -> bool:
    if len(a) != len(b):
        return False
    rest = list(b)
    for x in a:
        idx = next((i for i, y in enumerate(rest) if num_equal(x, y)), None)
        if idx is None:
            return False
        rest.pop(idx)
    return True


def score(tool: str, data: dict[str, Any], truth: dict[str, Any]) -> tuple[bool, str]:
    """Return (correct, detail) for a tool output against the case truth."""
    t = truth["type"]
    try:
        if t == "roots":
            if tool != "solve_equation" or data.get("period") is not None:
                return False, f"expected a finite root set, got tool={tool} period={data.get('period')}"
            got, want = list(data["solutions"]), [P(v) for v in truth["values"]]
            return _same_multiset(got, want), f"got {got}"
        if t == "roots_periodic":
            if data.get("period") is None:
                return False, f"expected a periodic family, got {data.get('solutions')}"
            got = _periodic_family(data["solutions"], data["period"])
            want = _periodic_family([P(b) for b in truth["bases"]], P(truth["period"]))
            ok = len(got) == len(want) and all(abs(a - b) < 1e-9 for a, b in zip(got, want, strict=True))
            return ok, f"got bases {data['solutions']} period {data['period']}"
        if t == "system":
            got = [{str(k): v for k, v in sol.items()} for sol in data["solutions"]]
            want = truth["solutions"]
            ok = len(got) == len(want) and all(
                any(set(g) == set(w) and all(num_equal(g[k], P(w[k])) for k in w) for g in got) for w in want
            )
            return ok, f"got {got}"
        if t == "expr":
            got = data.get("value") if "point" in data else data["result"]
            ok = _expr_equal(got, P(truth["value"]))
            form = truth.get("form")
            if ok and form == "factored":
                ok = sp.sympify(got).is_Mul or sp.sympify(got).is_Pow or sp.sympify(got).is_number
            if ok and form == "expanded":
                ok = sp.expand(got) == got
            return bool(ok), f"got {got}"
        if t == "antiderivative":
            got = data["antiderivative"]
            return _expr_equal(got, P(truth["value"]), up_to_constant=True), f"got {got}"
        if t == "number":
            if tool == "matrix_operation":
                got = data["result"]
            elif tool in ("differentiate", "integrate", "limit", "probability_distribution"):
                got = data["value"]
            else:
                got = data["result"]
            return num_equal(got, P(truth["value"])), f"got {got}"
        if t in ("matrix", "vector"):
            got = sp.Matrix(data["result"])
            want = (
                sp.Matrix([[P(v) for v in row] for row in truth["value"]])
                if t == "matrix"
                else sp.Matrix([P(v) for v in truth["value"]])
            )
            ok = got.shape == want.shape and all(num_equal(a, b) for a, b in zip(got, want, strict=True))
            return ok, f"got {got.tolist()}"
        if t == "multiset":
            got = [k for k, m in data["result"].items() for _ in range(m)]
            return _same_multiset(got, [P(v) for v in truth["values"]]), f"got {got}"
        if t == "stats":
            s = data["stats"]
            bad = [k for k in ("mean", "median", "population_std", "sample_std") if not num_equal(s[k], P(truth[k]))]
            return not bad, "mismatch: " + ", ".join(f"{k}={s[k]}" for k in bad) if bad else "ok"
    except Exception as exc:  # malformed output counts as incorrect, never crashes the benchmark
        return False, f"scoring error: {type(exc).__name__}: {exc}"
    return False, f"unknown truth type {t}"
