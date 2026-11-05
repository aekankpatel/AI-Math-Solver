"""Whitelisted mathematical tools.

Every tool takes plain, JSON-like arguments, parses them through the safe
parser and computes with SymPy (exact) or NumPy/SciPy (numeric). Tools never
evaluate free-form code. Each tool returns a :class:`ToolOutput` that carries
the human-facing answer, the deterministic solution steps and the raw objects
needed by the independent verifier and the visualiser.
"""

from __future__ import annotations

import concurrent.futures
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from app.math_engine.parser import (
    ExpressionError,
    parse_equation,
    parse_expression,
    parse_matrix,
    parse_number_list,
    parse_symbol,
)
from app.schemas import Step

COMPUTE_TIMEOUT_SECONDS = 10.0
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)


class ToolError(ValueError):
    """The tool could not produce a result (bad input, unsolvable, ...)."""


@dataclass
class ToolOutput:
    answer_text: str
    answer_latex: str
    steps: list[Step] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    approximate: bool = False


def _with_timeout(fn: Callable[[], Any], what: str) -> Any:
    future = _EXECUTOR.submit(fn)
    try:
        return future.result(timeout=COMPUTE_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError as exc:
        raise ToolError(f"{what} timed out after {COMPUTE_TIMEOUT_SECONDS:.0f}s.") from exc
    except (ToolError, ExpressionError):
        raise
    except NotImplementedError as exc:
        raise ToolError(f"{what}: SymPy has no method for this problem.") from exc
    except Exception as exc:
        raise ToolError(f"{what} failed: {exc}") from exc


def _pick_variable(expr: sp.Basic, variable: str | None) -> sp.Symbol:
    if variable:
        return parse_symbol(variable)
    free = sorted(expr.free_symbols, key=lambda s: s.name)
    if not free:
        return sp.Symbol("x")
    if len(free) == 1:
        return free[0]
    for preferred in ("x", "t", "y", "z"):
        for s in free:
            if s.name == preferred:
                return s
    raise ToolError(
        "Ambiguous problem: several variables (" + ", ".join(s.name for s in free) + ") and no variable specified."
    )


def _as_real(expr: sp.Expr, var: sp.Symbol) -> tuple[sp.Expr, sp.Symbol]:
    """Calculus here is real calculus: give the variable a real assumption so that
    e.g. d/dx |x| = sign(x) instead of an expression in re(x) and im(x)."""
    real_var = sp.Symbol(var.name, real=True)
    return expr.subs(var, real_var), real_var


def _fmt(value: Any) -> str:
    return sp.sstr(value)


def _num(value: sp.Basic, digits: int = 10) -> str:
    return _fmt(sp.N(value, digits))


# --------------------------------------------------------------------------- algebra


DOMAINS = ("complex", "real", "positive")


def _in_domain(value: sp.Basic, domain: str) -> bool:
    if domain == "complex":
        return True
    z = complex(sp.N(value, 30))
    return abs(z.imag) < 1e-12 and (domain == "real" or z.real > 0)


def solve_equation(equation: str, variable: str | None = None, domain: str = "complex") -> ToolOutput:
    """Solve one equation. ``domain`` restricts the reported solutions to
    complex (default), real or positive values, as the question requires."""
    if domain not in DOMAINS:
        raise ToolError(f"domain must be one of {', '.join(DOMAINS)}.")
    eq = parse_equation(equation)
    expr = sp.simplify(eq.lhs - eq.rhs) if isinstance(eq, sp.Eq) else eq
    if expr == 0 or expr is sp.true:
        raise ToolError("The equation is an identity: it holds for every value of the variable.")
    var = _pick_variable(expr, variable)
    if var not in expr.free_symbols:
        raise ToolError(f"The equation has no solutions (it does not depend on {var}).")

    steps = [
        Step(title="Write the equation", latex=sp.latex(eq)),
        Step(
            title="Move every term to one side",
            detail="Subtract the right-hand side so the equation reads f = 0.",
            latex=sp.latex(sp.Eq(expr, 0)),
        ),
    ]
    approximate = False
    if expr.is_polynomial(var):
        poly = sp.Poly(expr, var)
        factored = sp.factor(expr)
        steps.append(Step(title=f"Recognise a polynomial of degree {poly.degree()}"))
        if factored != sp.expand(expr) and (factored.is_Mul or factored.is_Pow):
            steps.append(
                Step(
                    title="Factor",
                    detail="Each factor equal to zero gives a solution.",
                    latex=sp.latex(sp.Eq(factored, 0)),
                )
            )
        elif poly.degree() == 2:
            a, b, c = poly.all_coeffs()
            disc = sp.simplify(b**2 - 4 * a * c)
            steps.append(
                Step(
                    title="Apply the quadratic formula",
                    detail=f"a = {a}, b = {b}, c = {c}; discriminant b^2 - 4ac = {disc}.",
                    latex=r"x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}",
                )
            )

    # 1) Exact emptiness check: e.g. 1/(x - 1) = 0 has no solution at all.
    complex_set = _safe_solveset(expr, var, sp.S.Complexes)
    if complex_set is sp.S.EmptySet:
        raise ToolError("The equation has no solutions (the solution set is empty).")

    # 2) Periodic equations (sin x = 0, tan x = 1): report the general solution.
    periodic = _periodic_solutions(expr, var)
    if periodic is not None:
        bases, period = periodic
        k = sp.Symbol("k", integer=True)
        general = [b + period * k for b in bases]
        steps.append(
            Step(
                title="Use periodicity",
                detail=f"f is periodic with period {_fmt(period)}: solve on one period, then add "
                "integer multiples of the period. Only real solutions are listed.",
            )
        )
        latex = ", ".join(f"{sp.latex(var)} = {sp.latex(g)}" for g in general) + r",\ k \in \mathbb{Z}"
        steps.append(Step(title="General solution", latex=latex))
        if domain == "positive":
            raise ToolError("Listing only the positive members of a periodic solution family is not supported.")
        return ToolOutput(
            answer_text=", ".join(f"{var} = {_fmt(g)}" for g in general) + " (k any integer)",
            answer_latex=latex,
            steps=steps,
            data={
                "expr": expr,
                "var": var,
                "solutions": bases,
                "real_solutions": bases,
                "period": period,
                "domain": "real",
            },
        )

    # 3) Symbolic solve; retry with a real variable (helps |x| = 3, real roots of odd powers).
    approximate = False
    solutions = _symbolic_solve(expr, var)
    if not solutions:
        real_var = sp.Symbol(var.name, real=True)
        solutions = _symbolic_solve(expr.subs(var, real_var), real_var)
        if solutions:
            solutions = [sp.sympify(s).subs(real_var, var) for s in solutions]
            steps.append(Step(title="Solve over the real numbers", detail="No complex closed form; real solutions."))
    if not solutions:
        solutions = _numeric_roots(expr, var)
        if not solutions:
            raise ToolError("No solutions were found symbolically or numerically.")
        approximate = True
        steps.append(
            Step(
                title="Solve numerically",
                detail="No closed form was found; roots were located with Newton's method "
                "from several starting points in [-10, 10]. Roots outside that range may be missed.",
            )
        )
    solutions = _dedupe(solutions + _other_lambertw_branches(solutions))
    if domain != "complex":
        kept = [s for s in solutions if _in_domain(s, domain)]
        dropped = [s for s in solutions if s not in kept]
        if dropped:
            steps.append(
                Step(
                    title=f"Keep only {domain} solutions",
                    detail=f"The question asks for {domain} values, so these are discarded: "
                    + ", ".join(_fmt(d) for d in dropped)
                    + ".",
                )
            )
        if not kept:
            raise ToolError(f"The equation has no {domain} solutions.")
        solutions = kept
    real = [s for s in solutions if s.is_real]
    if complex_set is not None and not isinstance(complex_set, sp.FiniteSet) and not approximate:
        if isinstance(complex_set, (sp.ImageSet, sp.Union)):
            steps.append(
                Step(
                    title="Note: infinitely many complex solutions",
                    detail="Only the principal solutions are listed; the full complex solution set is shown below.",
                    latex=sp.latex(complex_set),
                )
            )
    steps.append(
        Step(
            title="Solutions",
            latex=", ".join(f"{sp.latex(var)} = {sp.latex(s)}" for s in solutions),
            detail=_numeric_note(solutions),
        )
    )
    text = ", ".join(f"{var} = {_fmt(s)}" for s in solutions)
    latex = ", ".join(f"{sp.latex(var)} = {sp.latex(s)}" for s in solutions)
    return ToolOutput(
        answer_text=text,
        answer_latex=latex,
        steps=steps,
        approximate=approximate,
        data={"expr": expr, "var": var, "solutions": solutions, "real_solutions": real, "domain": domain},
    )


def _other_lambertw_branches(solutions: list[sp.Basic]) -> list[sp.Basic]:
    """SymPy's solve returns only the principal branch W0 of Lambert W. For a real
    argument z in (-1/e, 0) the branch W_{-1} is also real and gives another real
    solution (e.g. e^x = 3x has x = -W0(-1/3) and x = -W_{-1}(-1/3))."""
    extra = []
    for s in solutions:
        if not isinstance(s, sp.Basic):
            continue
        for w in s.atoms(sp.LambertW):
            z = w.args[0]
            if len(w.args) == 1 and z.is_real and z.is_number and -1 / sp.E < z < 0:
                extra.append(s.subs(w, sp.LambertW(z, -1)))
    return extra


def _safe_solveset(expr: sp.Expr, var: sp.Symbol, domain: sp.Set) -> sp.Set | None:
    try:
        return _with_timeout(lambda: sp.solveset(expr, var, domain), "solveset")
    except ToolError:
        return None


def _symbolic_solve(expr: sp.Expr, var: sp.Symbol) -> list[sp.Basic]:
    try:
        result = _with_timeout(lambda: sp.solve(expr, var), "Symbolic solve")
    except ToolError:
        return []
    return [r for r in result if not (isinstance(r, sp.Basic) and r.has(sp.nan, sp.zoo))]


def _periodic_solutions(expr: sp.Expr, var: sp.Symbol) -> tuple[list[sp.Basic], sp.Expr] | None:
    """Real solutions on one period [0, T) if ``expr`` is periodic in ``var``."""
    if expr.free_symbols != {var} or expr.is_polynomial(var):
        return None
    try:
        period = _with_timeout(lambda: sp.periodicity(expr, var), "Periodicity")
    except ToolError:
        return None
    if period is None or period == 0 or not period.is_positive:
        return None
    one_period = _safe_solveset(expr, var, sp.Interval.Ropen(0, period))
    if one_period is sp.S.EmptySet:
        raise ToolError("The equation has no real solutions.")
    if not isinstance(one_period, sp.FiniteSet) or not one_period:
        return None
    return sorted(one_period, key=lambda v: float(sp.N(v))), period


def _numeric_note(values: list[sp.Basic]) -> str:
    inexact = [v for v in values if not v.is_Integer]
    if not inexact:
        return ""
    return "Numerically: " + ", ".join(_num(v, 8) for v in inexact)


def _dedupe(values: list[sp.Basic]) -> list[sp.Basic]:
    out: list[sp.Basic] = []
    for v in values:
        if not any(sp.simplify(v - o) == 0 for o in out):
            out.append(v)
    return out


def _numeric_roots(expr: sp.Expr, var: sp.Symbol) -> list[sp.Basic]:
    """Multi-start Newton search. A candidate is accepted only if it is a genuine
    zero: tiny residual AND a tiny Newton step |f/f'| (this rejects points where
    f merely decays towards 0, e.g. 1/(x - 1) for large |x|)."""
    if expr.free_symbols != {var}:
        return []
    f = sp.lambdify(var, expr, modules=["mpmath"])
    df = sp.lambdify(var, sp.diff(expr, var), modules=["mpmath"])
    roots: list[sp.Basic] = []
    for start in np.linspace(-10, 10, 21):
        try:
            r = sp.nsolve(expr, var, float(start), tol=1e-14, prec=30)
            fr, dfr = complex(f(r)), complex(df(r))
        except Exception:
            continue
        if abs(r) > 1e4 or abs(fr) > 1e-10:
            continue
        if dfr != 0 and abs(fr / dfr) > 1e-8 * max(1.0, abs(complex(r))):
            continue
        r = sp.Float(sp.re(r), 15) if abs(sp.im(r)) < 1e-12 else sp.N(r, 15)
        if not any(abs(complex(r) - complex(o)) < 1e-8 for o in roots):
            roots.append(r)
    return sorted(roots, key=lambda v: (sp.re(v), sp.im(v)))


def solve_system(equations: list[str], variables: list[str] | None = None) -> ToolOutput:
    if not equations or len(equations) > 6:
        raise ToolError("Provide between 1 and 6 equations.")
    eqs = [parse_equation(e) for e in equations]
    exprs = [e.lhs - e.rhs for e in eqs]
    if variables:
        vars_ = [parse_symbol(v) for v in variables]
    else:
        vars_ = sorted(set().union(*(e.free_symbols for e in exprs)), key=lambda s: s.name)
    if not vars_:
        raise ToolError("The system has no variables.")
    solutions = _with_timeout(lambda: sp.solve(exprs, vars_, dict=True), "System solve")
    if not solutions:
        raise ToolError("The system has no solution (it is inconsistent).")
    steps = [
        Step(title="Write the system", latex=r" \\ ".join(sp.latex(e) for e in eqs)),
        Step(
            title="Eliminate variables",
            detail="Solve one equation for a variable and substitute into the others "
            "(Gaussian elimination for linear systems).",
        ),
    ]
    parts_text, parts_latex = [], []
    for sol in solutions:
        parts_text.append(", ".join(f"{k} = {_fmt(v)}" for k, v in sol.items()))
        parts_latex.append(", ".join(f"{sp.latex(k)} = {sp.latex(v)}" for k, v in sol.items()))
    steps.append(Step(title="Solution", latex=r"\quad\text{or}\quad ".join(parts_latex)))
    underdetermined = any(len(sol) < len(vars_) for sol in solutions)
    if underdetermined:
        steps.append(Step(title="Note", detail="Infinitely many solutions: some variables are free."))
    return ToolOutput(
        answer_text=" or ".join(f"({p})" for p in parts_text),
        answer_latex=r"\quad\text{or}\quad ".join(parts_latex),
        steps=steps,
        data={"exprs": exprs, "vars": vars_, "solutions": solutions},
    )


def _transform(expression: str, op: str) -> ToolOutput:
    expr = parse_expression(expression)
    fn = {"simplify": sp.simplify, "factor": sp.factor, "expand": sp.expand}[op]
    result = _with_timeout(lambda: fn(expr), op.capitalize())
    verb = {"simplify": "Simplify", "factor": "Factor", "expand": "Expand"}[op]
    steps = [
        Step(title="Start from the expression", latex=sp.latex(expr)),
        Step(
            title=f"{verb}",
            detail={
                "simplify": "Combine like terms, cancel common factors and apply identities.",
                "factor": "Write the expression as a product of irreducible factors.",
                "expand": "Distribute products and collect like terms.",
            }[op],
            latex=sp.latex(result),
        ),
    ]
    return ToolOutput(
        answer_text=_fmt(result),
        answer_latex=sp.latex(result),
        steps=steps,
        data={"input": expr, "result": result},
    )


def simplify_expression(expression: str) -> ToolOutput:
    return _transform(expression, "simplify")


def factor_expression(expression: str) -> ToolOutput:
    return _transform(expression, "factor")


def expand_expression(expression: str) -> ToolOutput:
    return _transform(expression, "expand")


# --------------------------------------------------------------------------- calculus

_RULE_HINTS = [
    (sp.Pow, "power rule: d/dx xⁿ = n·xⁿ⁻¹"),
    (sp.Mul, "product rule: (uv)' = u'v + uv'"),
    (sp.exp, "exponential rule: d/dx eᵘ = eᵘ·u'"),
    (sp.log, "logarithm rule: d/dx ln u = u'/u"),
    (sp.sin, "trig rule: d/dx sin u = cos u·u'"),
    (sp.cos, "trig rule: d/dx cos u = -sin u·u'"),
    (sp.tan, "trig rule: d/dx tan u = sec²u·u'"),
]


def _rules_used(expr: sp.Expr) -> list[str]:
    hints = []
    for cls, hint in _RULE_HINTS:
        if expr.has(cls) or any(isinstance(a, cls) for a in sp.preorder_traversal(expr)):
            hints.append(hint)
    if any(isinstance(a, sp.Function) and a.args and not a.args[0].is_Symbol for a in sp.preorder_traversal(expr)):
        hints.append("chain rule: (f∘g)' = f'(g)·g'")
    return hints


def differentiate(expression: str, variable: str | None = None, order: int = 1, point: str | None = None) -> ToolOutput:
    expr = parse_expression(expression)
    var = _pick_variable(expr, variable)
    expr, var = _as_real(expr, var)
    order = int(order)
    if not 1 <= order <= 10:
        raise ToolError("Derivative order must be between 1 and 10.")
    result = _with_timeout(lambda: sp.simplify(sp.diff(expr, var, order)), "Differentiation")
    notation = "d" if order == 1 else f"d^{{{order}}}"
    steps = [
        Step(title="Function", latex=f"f({sp.latex(var)}) = {sp.latex(expr)}"),
    ]
    hints = _rules_used(expr)
    if hints:
        steps.append(Step(title="Rules applied", detail="; ".join(hints)))
    if expr.is_Add:
        steps.append(
            Step(
                title="Differentiate term by term",
                latex=" + ".join(
                    sp.latex(sp.Derivative(t, var, order)) + " = " + sp.latex(sp.diff(t, var, order)) for t in expr.args
                ),
            )
        )
    steps.append(
        Step(
            title="Result" if order == 1 else f"Derivative of order {order}",
            latex=rf"\frac{{{notation}}}{{{notation[0]}{sp.latex(var)}{'' if order == 1 else f'^{{{order}}}'}}}"
            rf" {sp.latex(expr)} = {sp.latex(result)}",
        )
    )
    data: dict[str, Any] = {"expr": expr, "var": var, "order": order, "result": result}
    text = _fmt(result)
    latex = sp.latex(result)
    if point is not None:
        pt = parse_expression(str(point))
        value = sp.simplify(result.subs(var, pt))
        data["point"], data["value"] = pt, value
        steps.append(
            Step(
                title=f"Evaluate at {var} = {_fmt(pt)}",
                latex="f" + "'" * min(order, 3) + f"({sp.latex(pt)}) = {sp.latex(value)}",
            )
        )
        text = f"{text}; at {var} = {_fmt(pt)}: {_fmt(value)}"
        latex = f"{latex}\\;\\Big|_{{{sp.latex(var)}={sp.latex(pt)}}} = {sp.latex(value)}"
    return ToolOutput(answer_text=text, answer_latex=latex, steps=steps, data=data)


def integrate(
    expression: str,
    variable: str | None = None,
    lower: str | None = None,
    upper: str | None = None,
) -> ToolOutput:
    expr = parse_expression(expression)
    var = _pick_variable(expr, variable)
    expr, var = _as_real(expr, var)
    definite = lower is not None and upper is not None
    if (lower is None) != (upper is None):
        raise ToolError("A definite integral needs both a lower and an upper limit.")
    steps = [Step(title="Integrand", latex=f"f({sp.latex(var)}) = {sp.latex(expr)}")]
    antideriv = _with_timeout(lambda: sp.integrate(expr, var), "Integration")
    has_closed_form = not antideriv.has(sp.Integral)
    if has_closed_form:
        if expr.is_Add:
            steps.append(
                Step(
                    title="Integrate term by term (linearity)",
                    latex=" + ".join(sp.latex(sp.integrate(t, var)) for t in expr.args),
                )
            )
        steps.append(
            Step(
                title="Antiderivative",
                latex=rf"\int {sp.latex(expr)}\,d{sp.latex(var)} = {sp.latex(antideriv)} + C",
            )
        )
    data: dict[str, Any] = {"expr": expr, "var": var, "antiderivative": antideriv}

    if not definite:
        if not has_closed_form:
            raise ToolError("No elementary antiderivative was found for this integrand.")
        return ToolOutput(
            answer_text=f"{_fmt(antideriv)} + C",
            answer_latex=f"{sp.latex(antideriv)} + C",
            steps=steps,
            data=data,
        )

    a, b = parse_expression(str(lower)), parse_expression(str(upper))
    data.update(lower=a, upper=b)
    if isinstance(expr, sp.Abs) and a.is_finite and b.is_finite and not (a.free_symbols or b.free_symbols):
        split = _integrate_abs(expr, var, a, b, steps)
        if split is not None:
            value, info = split
            data.update(value=value, abs_split=info, antiderivative=sp.Integral(expr, var))
            return ToolOutput(answer_text=_fmt(value), answer_latex=sp.latex(value), steps=steps, data=data)
    approximate = False
    if a.free_symbols or b.free_symbols:
        raise ToolError("Integration limits must be numbers.")
    if (b - a).is_negative:
        steps.append(Step(title="Reversed limits", detail="The lower limit exceeds the upper one, so the sign flips."))
    singular = _interior_singularities(expr, var, a, b)
    data["singularities"] = singular
    try:
        value = _with_timeout(lambda: sp.integrate(expr, (var, a, b)), "Definite integration")
    except ToolError:
        value = sp.nan
    if value.has(sp.nan, sp.zoo) or (value.is_infinite and value not in (sp.oo, -sp.oo)):
        if singular:
            raise ToolError(
                "The integral diverges: the integrand has a non-integrable singularity at "
                + ", ".join(f"{var} = {_fmt(p)}" for p in singular)
                + " inside the interval."
            )
        value = sp.Integral(expr, (var, a, b))
    if value.has(sp.Integral):
        try:
            numeric = sp.N(sp.Integral(expr, (var, a, b)), 15)
        except Exception as exc:
            raise ToolError("The integral could not be evaluated, even numerically.") from exc
        if not numeric.is_number or not numeric.is_finite:
            raise ToolError("The integral could not be evaluated: it appears to diverge.")
        value = numeric
        approximate = True
        steps.append(Step(title="Evaluate numerically", detail="No closed form; adaptive quadrature."))
    elif value.is_infinite:
        data["divergent"] = True
        steps.append(
            Step(
                title="The integral diverges",
                detail="The area grows without bound near "
                + (", ".join(f"{var} = {_fmt(p)}" for p in singular) if singular else "an endpoint")
                + ".",
            )
        )
    else:
        if has_closed_form and a.is_finite and b.is_finite:
            steps.append(
                Step(
                    title="Apply the Fundamental Theorem of Calculus",
                    latex=f"F({sp.latex(b)}) - F({sp.latex(a)})",
                )
            )
        value = sp.simplify(value)
    integral_latex = rf"\int_{{{sp.latex(a)}}}^{{{sp.latex(b)}}} {sp.latex(expr)}\,d{sp.latex(var)}"
    steps.append(
        Step(
            title="Value",
            latex=f"{integral_latex} = {sp.latex(value)}",
            detail=_numeric_note([value]) if value.is_number and value.is_finite else "",
        )
    )
    data["value"] = value
    text = f"diverges ({_fmt(value)})" if value.is_infinite else _fmt(value)
    latex = sp.latex(value) + (r"\ \text{(diverges)}" if value.is_infinite else "")
    return ToolOutput(answer_text=text, answer_latex=latex, steps=steps, data=data, approximate=approximate)


def _integrate_abs(
    expr: sp.Abs, var: sp.Symbol, a: sp.Expr, b: sp.Expr, steps: list[Step]
) -> tuple[sp.Expr, dict[str, Any]] | None:
    """∫_a^b |g| exactly: split [a, b] where g changes sign and add |∫ g| over each piece."""
    g = expr.args[0]
    lo, hi = (a, b) if (b - a).is_nonnegative else (b, a)
    G = _with_timeout(lambda: sp.integrate(g, var), "Integration")
    if G.has(sp.Integral):
        return None
    roots = _with_timeout(lambda: sp.solveset(g, var, sp.Interval.open(lo, hi)), "Sign changes")
    if roots == sp.S.EmptySet:
        inner: list[sp.Expr] = []
    elif isinstance(roots, sp.FiniteSet) and len(roots) <= 20:
        inner = sorted(roots, key=lambda r: float(sp.N(r)))
    else:
        return None
    points = [lo, *inner, hi]
    pieces = [sp.simplify(G.subs(var, q) - G.subs(var, p)) for p, q in zip(points, points[1:], strict=False)]
    value = sp.simplify(sum(sp.Abs(v) for v in pieces))
    if (b - a).is_negative:
        value = -value
    steps.append(
        Step(
            title="Split where the inside changes sign",
            detail="|g| = g where g ≥ 0 and −g where g < 0, so integrate g on each piece and add the absolute values.",
            latex=rf"g({sp.latex(var)}) = {sp.latex(g)} = 0 \Rightarrow {sp.latex(var)} \in \{{"
            + ", ".join(sp.latex(r) for r in points[1:-1])
            + r"\}",
        )
    )
    steps.append(
        Step(
            title="Add the pieces",
            latex=" + ".join(rf"\left|{sp.latex(v)}\right|" for v in pieces) + f" = {sp.latex(value)}",
            detail=_numeric_note([value]),
        )
    )
    return value, {"inner": g, "antiderivative": G, "points": points}


def _interior_singularities(expr: sp.Expr, var: sp.Symbol, a: sp.Expr, b: sp.Expr) -> list[sp.Basic]:
    """Points in the closed integration interval where the integrand is singular."""
    lo, hi = (a, b) if not (b - a).is_negative else (b, a)
    try:
        sing = _with_timeout(lambda: sp.singularities(expr, var, sp.Interval(lo, hi)), "Singularities")
    except ToolError:
        return []
    return sorted(sing, key=lambda v: float(sp.N(v))) if isinstance(sing, sp.FiniteSet) else []


def limit(expression: str, variable: str | None = None, point: str = "0", direction: str = "+-") -> ToolOutput:
    expr = parse_expression(expression)
    var = _pick_variable(expr, variable)
    expr, var = _as_real(expr, var)
    pt = parse_expression(str(point))
    if direction not in ("+", "-", "+-"):
        raise ToolError("direction must be '+', '-' or '+-'.")
    if pt.is_infinite:
        direction = "-" if pt == sp.oo else "+"
    try:
        value = _with_timeout(lambda: sp.limit(expr, var, pt, dir=direction), "Limit")
    except ToolError as exc:
        if "does not exist" in str(exc) or "left hand limit" in str(exc):
            raise ToolError(_limit_dne_message(expr, var, pt)) from exc
        raise
    if isinstance(value, sp.AccumBounds):
        raise ToolError(
            f"The limit does not exist: the function oscillates between {_fmt(value.min)} and {_fmt(value.max)}."
        )
    if value.has(sp.zoo, sp.nan) or (value.is_infinite and value not in (sp.oo, -sp.oo)):
        raise ToolError(_limit_dne_message(expr, var, pt))
    steps = [
        Step(title="Limit", latex=rf"\lim_{{{sp.latex(var)} \to {sp.latex(pt)}}} {sp.latex(expr)}"),
    ]
    direct = None
    if pt.is_finite:
        try:
            direct = expr.subs(var, pt)
        except Exception:
            direct = None
    if direct is not None and direct.is_finite and direct is not sp.nan:
        steps.append(Step(title="Direct substitution works", latex=sp.latex(direct)))
    else:
        steps.append(
            Step(
                title="Indeterminate or singular form",
                detail="Direct substitution fails, so the limit is computed with series "
                "expansion (Gruntz algorithm), equivalent to applying L'Hôpital's rule.",
            )
        )
    steps.append(Step(title="Result", latex=sp.latex(value)))
    return ToolOutput(
        answer_text=_fmt(value),
        answer_latex=sp.latex(value),
        steps=steps,
        data={"expr": expr, "var": var, "point": pt, "direction": direction, "value": value},
    )


def _limit_dne_message(expr: sp.Expr, var: sp.Symbol, pt: sp.Expr) -> str:
    try:
        left = sp.limit(expr, var, pt, dir="-")
        right = sp.limit(expr, var, pt, dir="+")
        return (
            f"The two-sided limit does not exist: the left-hand limit is {_fmt(left)} and the right-hand limit "
            f"is {_fmt(right)}."
        )
    except Exception:
        return "The limit does not exist."


# --------------------------------------------------------------------------- linear algebra

MATRIX_OPERATIONS = (
    "determinant",
    "inverse",
    "eigenvalues",
    "rank",
    "transpose",
    "multiply",
    "solve",
    "rref",
)


def matrix_operation(
    operation: str,
    matrix: list[list[Any]],
    other: list[list[Any]] | None = None,
    vector: list[Any] | None = None,
) -> ToolOutput:
    if operation not in MATRIX_OPERATIONS:
        raise ToolError(f"Unsupported matrix operation {operation!r}.")
    A = parse_matrix(matrix)
    steps = [Step(title="Matrix", latex="A = " + sp.latex(A))]
    data: dict[str, Any] = {"operation": operation, "A": A}
    square = A.rows == A.cols

    if operation in ("determinant", "inverse", "eigenvalues") and not square:
        raise ToolError(f"The {operation} is only defined for square matrices.")

    if operation == "determinant":
        result = A.det()
        if A.rows == 2:
            steps.append(Step(title="2×2 formula", latex=r"\det A = ad - bc"))
        else:
            steps.append(Step(title="Cofactor / LU expansion", detail="Expand along rows to reduce size."))
        latex = sp.latex(result)
        text = _fmt(result)
    elif operation == "inverse":
        det = A.det()
        steps.append(Step(title="Check invertibility", latex=rf"\det A = {sp.latex(det)}"))
        if det == 0:
            raise ToolError("The matrix is singular (determinant 0), so it has no inverse.")
        result = A.inv()
        steps.append(Step(title="Gauss–Jordan elimination on [A | I]", latex="A^{-1} = " + sp.latex(result)))
        latex, text = sp.latex(result), _fmt(result.tolist())
    elif operation == "eigenvalues":
        lam = sp.Symbol("lambda")
        charpoly = A.charpoly(lam).as_expr()
        steps.append(
            Step(title="Characteristic polynomial", latex=rf"\det(A - \lambda I) = {sp.latex(sp.factor(charpoly))}")
        )
        ev = A.eigenvals()
        result = ev
        data["eigenvalues"] = ev
        latex = ", ".join(rf"\lambda = {sp.latex(k)}" + (rf" \;(\times{m})" if m > 1 else "") for k, m in ev.items())
        text = ", ".join(f"{_fmt(k)}" + (f" (multiplicity {m})" if m > 1 else "") for k, m in ev.items())
        steps.append(Step(title="Roots of the characteristic polynomial", latex=latex))
    elif operation == "rank":
        rref, pivots = A.rref()
        result = len(pivots)
        steps.append(Step(title="Row reduce", latex=sp.latex(rref)))
        steps.append(Step(title="Count pivot columns", detail=f"Pivots in columns {list(pivots)}."))
        latex, text = str(result), str(result)
    elif operation == "rref":
        rref, pivots = A.rref()
        result = rref
        steps.append(Step(title="Reduced row echelon form", latex=sp.latex(rref)))
        latex, text = sp.latex(rref), _fmt(rref.tolist())
    elif operation == "transpose":
        result = A.T
        latex, text = sp.latex(result), _fmt(result.tolist())
        steps.append(Step(title="Swap rows and columns", latex="A^T = " + latex))
    elif operation == "multiply":
        if other is None:
            raise ToolError("Matrix multiplication needs a second matrix.")
        B = parse_matrix(other)
        if A.cols != B.rows:
            raise ToolError(f"Cannot multiply a {A.rows}×{A.cols} by a {B.rows}×{B.cols} matrix.")
        data["B"] = B
        result = A * B
        steps.append(Step(title="Second matrix", latex="B = " + sp.latex(B)))
        steps.append(Step(title="Row-by-column products", latex="AB = " + sp.latex(result)))
        latex, text = sp.latex(result), _fmt(result.tolist())
    else:  # solve
        if vector is None:
            raise ToolError("Solving Ax = b needs a right-hand-side vector b.")
        b = parse_matrix([[v] for v in vector])
        if b.rows != A.rows:
            raise ToolError("Vector b must have one entry per row of A.")
        data["b"] = b
        steps.append(Step(title="Augmented matrix", latex=sp.latex(A.row_join(b))))
        try:
            sol, params = A.gauss_jordan_solve(b)
        except ValueError as exc:
            raise ToolError("The linear system is inconsistent: no solution exists.") from exc
        if params.shape[0] > 0:
            steps.append(Step(title="Free parameters", detail="The system has infinitely many solutions."))
        result = sol
        steps.append(Step(title="Gauss–Jordan elimination", latex="x = " + sp.latex(sol)))
        latex, text = sp.latex(sol), _fmt(list(sol))

    data["result"] = result
    return ToolOutput(answer_text=text, answer_latex=latex, steps=steps, data=data)


# --------------------------------------------------------------------------- statistics


def descriptive_statistics(data: list[Any]) -> ToolOutput:
    values = parse_number_list(data)
    if len(values) < 1:
        raise ToolError("Provide at least one data value.")
    if len(values) > 10_000:
        raise ToolError("At most 10,000 data values are supported.")
    exact = [sp.nsimplify(v, rational=True) for v in values]
    n = len(exact)
    mean = sp.Rational(sum(exact), n) if all(v.is_Rational for v in exact) else sum(exact) / n
    srt = sorted(exact)
    median = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2
    pop_var = sum((v - mean) ** 2 for v in exact) / n
    sample_var = sum((v - mean) ** 2 for v in exact) / (n - 1) if n > 1 else sp.nan
    counts: dict[Any, int] = {}
    for v in exact:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts.values())
    modes = [k for k, c in counts.items() if c == top] if top > 1 else []
    stats = {
        "n": n,
        "mean": mean,
        "median": median,
        "mode": modes,
        "min": srt[0],
        "max": srt[-1],
        "range": srt[-1] - srt[0],
        "population_variance": pop_var,
        "population_std": sp.sqrt(pop_var),
        "sample_variance": sample_var,
        "sample_std": sp.sqrt(sample_var) if n > 1 else sp.nan,
    }
    steps = [
        Step(
            title="Data",
            detail=f"n = {n} values, sorted: {', '.join(_fmt(v) for v in srt[:30])}{' …' if n > 30 else ''}",
        ),
        Step(
            title="Mean",
            latex=rf"\bar{{x}} = \frac{{\sum x_i}}{{n}} = \frac{{{_fmt(sum(exact))}}}{{{n}}} = {sp.latex(mean)}",
        ),
        Step(
            title="Median",
            detail="Middle value of the sorted data" + (" (average of the two middle values)." if n % 2 == 0 else "."),
            latex=sp.latex(median),
        ),
        Step(
            title="Variance",
            latex=rf"s^2 = \frac{{\sum (x_i-\bar{{x}})^2}}{{n-1}} = {sp.latex(sample_var)},\quad \sigma^2 = {sp.latex(pop_var)}",
        ),
        Step(
            title="Standard deviation",
            latex=rf"s = {sp.latex(stats['sample_std'])} \approx {_num(stats['sample_std'], 6)}"
            if n > 1
            else r"\sigma = 0",
        ),
    ]
    summary = {
        k: (
            _num(v, 8)
            if isinstance(v, sp.Basic) and not v.is_Integer
            else (str(v) if not isinstance(v, list) else [str(m) for m in v])
        )
        for k, v in stats.items()
    }
    text = (
        f"mean = {_fmt(mean)} (≈{_num(mean, 6)}), median = {_fmt(median)}, "
        f"sample std = {_num(stats['sample_std'], 6) if n > 1 else 'n/a'}, "
        f"population std = {_num(stats['population_std'], 6)}"
    )
    latex = (
        rf"\bar{{x}} = {sp.latex(mean)},\ \tilde{{x}} = {sp.latex(median)},\ "
        rf"s \approx {_num(stats['sample_std'], 6) if n > 1 else r'\text{n/a}'}"
    )
    return ToolOutput(
        answer_text=text,
        answer_latex=latex,
        steps=steps,
        data={"values": values, "stats": stats, "summary": summary},
    )


DISTRIBUTIONS = ("binomial", "poisson", "hypergeometric", "normal", "uniform", "exponential")
QUERIES = ("pmf", "pdf", "cdf", "greater", "between")


def _p(value: Any, name: str) -> sp.Expr:
    expr = parse_expression(str(value))
    if not expr.is_number:
        raise ToolError(f"Parameter {name} must be a number.")
    return sp.nsimplify(expr, rational=True) if expr.is_Float else expr


def probability_distribution(
    distribution: str,
    params: dict[str, Any],
    query: str = "cdf",
    x: Any = None,
    x2: Any = None,
) -> ToolOutput:
    """Probability for a named distribution.

    Queries: ``pmf`` P(X = x), ``pdf`` density at x, ``cdf`` P(X <= x),
    ``greater`` P(X > x) (P(X >= x) for discrete), ``between`` P(x <= X <= x2).
    """
    if distribution not in DISTRIBUTIONS:
        raise ToolError(f"Unsupported distribution {distribution!r}.")
    if query not in QUERIES:
        raise ToolError(f"Unsupported query {query!r}.")
    if x is None:
        raise ToolError("A value x is required.")
    xv = _p(x, "x")
    x2v = _p(x2, "x2") if x2 is not None else None
    if query == "between" and x2v is None:
        raise ToolError("'between' needs both x and x2.")
    if query == "between" and (x2v - xv).is_negative:
        raise ToolError(f"Invalid interval: the lower bound {xv} is greater than the upper bound {x2v}.")
    k = sp.Symbol("k", integer=True)
    t = sp.Symbol("t", real=True)
    steps: list[Step] = []
    discrete = distribution in ("binomial", "poisson", "hypergeometric")

    if distribution == "binomial":
        n, p = _p(params.get("n"), "n"), _p(params.get("p"), "p")
        if not (n.is_integer and n >= 0 and 0 <= p <= 1):
            raise ToolError("Binomial needs integer n >= 0 and 0 <= p <= 1.")
        pmf = sp.binomial(n, k) * p**k * (1 - p) ** (n - k)
        support = (0, n)
        steps.append(
            Step(
                title="Binomial PMF",
                latex=rf"P(X=k) = \binom{{{n}}}{{k}} ({sp.latex(p)})^k ({sp.latex(1 - p)})^{{{n}-k}}",
            )
        )
    elif distribution == "hypergeometric":
        good, bad = _p(params.get("success"), "success"), _p(params.get("failure"), "failure")
        draws = _p(params.get("draws"), "draws")
        if not all(v.is_integer and v >= 0 for v in (good, bad, draws)) or draws > good + bad:
            raise ToolError("Drawing without replacement needs whole numbers, and no more draws than items.")
        pmf = sp.binomial(good, k) * sp.binomial(bad, draws - k) / sp.binomial(good + bad, draws)
        support = (sp.Max(0, draws - bad), sp.Min(draws, good))
        steps.append(
            Step(
                title="Drawing without replacement (hypergeometric)",
                detail=f"{good} items of the kind asked for, {bad} others, {draws} drawn.",
                latex=rf"P(X=k) = \frac{{\binom{{{good}}}{{k}}\binom{{{bad}}}{{{draws}-k}}}}{{\binom{{{good + bad}}}{{{draws}}}}}",
            )
        )
    elif distribution == "poisson":
        lam = _p(params.get("lambda", params.get("lam", params.get("mean"))), "lambda")
        if not lam > 0:
            raise ToolError("Poisson needs lambda > 0.")
        pmf = sp.exp(-lam) * lam**k / sp.factorial(k)
        support = (0, sp.oo)
        steps.append(
            Step(title="Poisson PMF", latex=rf"P(X=k) = \frac{{e^{{-{sp.latex(lam)}}} {sp.latex(lam)}^k}}{{k!}}")
        )
    elif distribution == "normal":
        mu = _p(params.get("mean", params.get("mu", 0)), "mean")
        sigma = _p(params.get("std", params.get("sigma", 1)), "std")
        if not sigma > 0:
            raise ToolError("Normal needs std > 0.")
        pdf = sp.exp(-((t - mu) ** 2) / (2 * sigma**2)) / (sigma * sp.sqrt(2 * sp.pi))
        cdf = lambda v: (1 + sp.erf((v - mu) / (sigma * sp.sqrt(2)))) / 2  # noqa: E731
        steps.append(Step(title="Standardise", latex=rf"Z = \frac{{X - {sp.latex(mu)}}}{{{sp.latex(sigma)}}}"))
    elif distribution == "uniform":
        a, b = _p(params.get("a", 0), "a"), _p(params.get("b", 1), "b")
        if not b > a:
            raise ToolError("Uniform needs b > a.")
        pdf = sp.Piecewise((1 / (b - a), (t >= a) & (t <= b)), (0, True))
        cdf = lambda v: sp.Min(sp.Max((v - a) / (b - a), 0), 1)  # noqa: E731
        steps.append(Step(title="Uniform density", latex=rf"f(x) = \frac{{1}}{{{sp.latex(b - a)}}}"))
    else:  # exponential
        rate = _p(params.get("rate", params.get("lambda", 1)), "rate")
        if not rate > 0:
            raise ToolError("Exponential needs rate > 0.")
        pdf = sp.Piecewise((rate * sp.exp(-rate * t), t >= 0), (0, True))
        cdf = lambda v: sp.Piecewise((1 - sp.exp(-rate * v), v >= 0), (0, True))  # noqa: E731
        steps.append(Step(title="Exponential CDF", latex=rf"F(x) = 1 - e^{{-{sp.latex(rate)} x}}"))

    if discrete:
        lo, hi = support

        def psum(a: sp.Expr, b: sp.Expr) -> sp.Expr:
            a, b = sp.Max(a, lo), sp.Min(b, hi)
            if (b - a).is_negative:
                return sp.Integer(0)
            return sp.summation(pmf, (k, a, b))

        if query in ("pmf", "pdf"):
            if not xv.is_integer:
                raise ToolError("For a discrete distribution x must be an integer.")
            value, desc = pmf.subs(k, xv), f"P(X = {xv})"
        elif query == "cdf":
            value, desc = psum(lo, sp.floor(xv)), f"P(X ≤ {xv})"
        elif query == "greater":
            value, desc = 1 - psum(lo, sp.ceiling(xv) - 1), f"P(X ≥ {xv})"
        else:
            value, desc = psum(sp.ceiling(xv), sp.floor(x2v)), f"P({xv} ≤ X ≤ {x2v})"
    else:
        if query == "pmf":
            raise ToolError("Continuous distributions have no PMF; P(X = x) = 0. Use 'pdf' or 'cdf'.")
        if query == "pdf":
            value, desc = pdf.subs(t, xv), f"f({xv})"
        elif query == "cdf":
            value, desc = cdf(xv), f"P(X ≤ {xv})"
        elif query == "greater":
            value, desc = 1 - cdf(xv), f"P(X > {xv})"
        else:
            value, desc = cdf(x2v) - cdf(xv), f"P({xv} ≤ X ≤ {x2v})"

    value = sp.simplify(value)
    numeric = float(sp.N(value, 15))
    steps.append(Step(title=f"Compute {desc}", latex=f"{sp.latex(value)} \\approx {numeric:.6g}"))
    return ToolOutput(
        answer_text=f"{desc} = {_fmt(value)} ≈ {numeric:.6g}",
        answer_latex=f"{sp.latex(value)} \\approx {numeric:.6g}",
        steps=steps,
        data={
            "distribution": distribution,
            "params": {k_: float(_p(v, k_)) for k_, v in params.items()},
            "query": query,
            "x": float(xv),
            "x2": float(x2v) if x2v is not None else None,
            "value": value,
            "numeric": numeric,
        },
    )


def combinatorics(operation: str, n: Any, k: Any = None) -> ToolOutput:
    nv = _p(n, "n")
    if not (nv.is_integer and nv >= 0) or nv > 10_000:
        raise ToolError("n must be an integer between 0 and 10000.")
    if operation == "factorial":
        result = sp.factorial(nv)
        latex_expr = f"{nv}!"
    else:
        if k is None:
            raise ToolError(f"{operation} needs k.")
        kv = _p(k, "k")
        if not (kv.is_integer and 0 <= kv <= nv):
            raise ToolError("k must be an integer with 0 <= k <= n.")
        if operation == "combinations":
            result = sp.binomial(nv, kv)
            latex_expr = rf"\binom{{{nv}}}{{{kv}}} = \frac{{{nv}!}}{{{kv}!\,({nv}-{kv})!}}"
        elif operation == "permutations":
            result = sp.factorial(nv) / sp.factorial(nv - kv)
            latex_expr = rf"P({nv},{kv}) = \frac{{{nv}!}}{{({nv}-{kv})!}}"
        else:
            raise ToolError(f"Unsupported combinatorics operation {operation!r}.")
    steps = [Step(title="Formula", latex=latex_expr), Step(title="Result", latex=sp.latex(result))]
    return ToolOutput(
        answer_text=_fmt(result),
        answer_latex=sp.latex(result),
        steps=steps,
        data={"operation": operation, "n": nv, "k": k, "result": result},
    )


# --------------------------------------------------------------------------- evaluation


def evaluate_expression(expression: str, value: str, variable: str | None = None) -> ToolOutput:
    """The exact value of an expression at a point, e.g. x^3 - 3x at x = 2."""
    expr = parse_expression(expression)
    var = _pick_variable(expr, variable)
    if expr.free_symbols - {var}:
        raise ToolError("The expression has more than one variable; give a value for each.")
    point = parse_expression(str(value))
    if point.free_symbols:
        raise ToolError("The value must be a number.")
    result = _with_timeout(lambda: sp.simplify(expr.subs(var, point)), "Evaluation")
    if result.has(sp.zoo, sp.nan) or result.is_infinite:
        raise ToolError(f"The expression is undefined at {var} = {_fmt(point)}.")
    steps = [
        Step(title="Expression", latex=f"f({sp.latex(var)}) = {sp.latex(expr)}"),
        Step(
            title=f"Substitute {var} = {_fmt(point)}",
            latex=f"f({sp.latex(point)}) = {sp.latex(expr.subs(var, sp.UnevaluatedExpr(point)))}",
        ),
        Step(title="Simplify", latex=f"f({sp.latex(point)}) = {sp.latex(result)}", detail=_numeric_note([result])),
    ]
    return ToolOutput(
        answer_text=_fmt(result),
        answer_latex=sp.latex(result),
        steps=steps,
        data={"expr": expr, "var": var, "point": point, "result": result},
    )


# --------------------------------------------------------------------------- sums, sequences, extrema, percentages


def summation(expression: str, lower: str, upper: str, variable: str | None = None) -> ToolOutput:
    """Σ expression for variable = lower..upper (upper may be oo or a symbol such as n)."""
    expr = parse_expression(expression)
    var = sp.Symbol(variable) if variable else _pick_variable(expr, None)
    lo, hi = parse_expression(str(lower)), parse_expression(str(upper))
    if lo.free_symbols or (hi.free_symbols and hi.free_symbols != {sp.Symbol(str(next(iter(hi.free_symbols))))}):
        raise ToolError("The lower limit must be a number.")
    if hi.free_symbols & {var}:
        raise ToolError("The upper limit cannot be the summation variable.")
    if not expr.free_symbols <= {var}:
        raise ToolError("The terms may only depend on the summation variable.")
    if hi.is_number and hi.is_finite and (hi - lo) > 10**6:
        raise ToolError("Too many terms.")
    value = _with_timeout(lambda: sp.simplify(sp.summation(expr, (var, lo, hi))), "Summation")
    if value.has(sp.Sum):
        raise ToolError("No closed form was found for this sum.")
    sigma = rf"\sum_{{{sp.latex(var)}={sp.latex(lo)}}}^{{{sp.latex(hi)}}} {sp.latex(expr)}"
    steps = [Step(title="The sum", latex=sigma)]
    if hi.free_symbols:
        steps.append(
            Step(title="Closed form", detail="Standard sum formulas (or telescoping) in terms of the upper limit.")
        )
    elif hi.is_infinite:
        steps.append(
            Step(
                title="Infinite series",
                detail="Converges to a closed form." if value.is_finite else "The series diverges.",
            )
        )
    steps.append(
        Step(
            title="Value",
            latex=f"{sigma} = {sp.latex(sp.factor(value) if hi.free_symbols else value)}",
            detail=_numeric_note([value]) if value.is_number and value.is_finite else "",
        )
    )
    if value.is_infinite:
        text = "diverges (∞)"
    else:
        text = _fmt(sp.factor(value) if hi.free_symbols else value)
    return ToolOutput(
        answer_text=text,
        answer_latex=sp.latex(sp.factor(value) if hi.free_symbols else value),
        steps=steps,
        data={"expr": expr, "var": var, "lower": lo, "upper": hi, "value": value},
    )


def sequence(terms: list[Any], n: Any, find: str = "term") -> ToolOutput:
    """nth term or sum of the first n terms of an arithmetic or geometric sequence given by its first terms."""
    if find not in ("term", "sum"):
        raise ToolError("find must be 'term' or 'sum'.")
    vals = [parse_expression(str(t)) for t in terms]
    if len(vals) < 3 or any(v.free_symbols for v in vals):
        raise ToolError("Give at least three numeric terms of the sequence.")
    N = parse_expression(str(n))
    if not (N.is_integer and N >= 1):
        raise ToolError("n must be a positive whole number.")
    diffs = [sp.nsimplify(b - a) for a, b in zip(vals, vals[1:], strict=False)]
    a1 = sp.nsimplify(vals[0])
    steps = [Step(title="Given terms", latex=", ".join(sp.latex(v) for v in vals) + r", \dots")]
    if all(d == diffs[0] for d in diffs):
        kind, d = "arithmetic", diffs[0]
        steps.append(Step(title="Common difference", latex=f"d = {sp.latex(d)}"))
        if find == "term":
            value = a1 + (N - 1) * d
            steps.append(
                Step(
                    title="nth term",
                    latex=rf"a_n = a_1 + (n-1)d = {sp.latex(a1)} + ({N}-1)\cdot {sp.latex(d)} = {sp.latex(value)}",
                )
            )
        else:
            value = N * (2 * a1 + (N - 1) * d) / 2
            steps.append(
                Step(
                    title="Sum of n terms",
                    latex=rf"S_n = \frac{{n}}{{2}}\left(2a_1 + (n-1)d\right) = {sp.latex(value)}",
                )
            )
        info = {"kind": kind, "a1": a1, "d": d}
    elif all(v != 0 for v in vals) and len({sp.nsimplify(b / a) for a, b in zip(vals, vals[1:], strict=False)}) == 1:
        kind, r = "geometric", sp.nsimplify(vals[1] / vals[0])
        steps.append(Step(title="Common ratio", latex=f"r = {sp.latex(r)}"))
        if find == "term":
            value = a1 * r ** (N - 1)
            steps.append(
                Step(
                    title="nth term",
                    latex=rf"a_n = a_1 r^{{n-1}} = {sp.latex(a1)}\cdot {sp.latex(r)}^{{{N}-1}} = {sp.latex(value)}",
                )
            )
        else:
            value = N * a1 if r == 1 else a1 * (1 - r**N) / (1 - r)
            steps.append(Step(title="Sum of n terms", latex=rf"S_n = a_1\frac{{1-r^n}}{{1-r}} = {sp.latex(value)}"))
        info = {"kind": kind, "a1": a1, "r": r}
    else:
        raise ToolError("The terms are neither arithmetic (constant difference) nor geometric (constant ratio).")
    value = sp.nsimplify(sp.simplify(value))
    label = f"a_{N}" if find == "term" else f"S_{N}"
    return ToolOutput(
        answer_text=f"{label} = {_fmt(value)} ({kind})",
        answer_latex=f"{label.replace('_', '_{') + '}'} = {sp.latex(value)}",
        steps=steps,
        data={"terms": vals, "n": int(N), "find": find, "value": value, **info},
    )


def critical_points(expression: str, variable: str | None = None) -> ToolOutput:
    """Real critical points (f' = 0) with their classification and function values."""
    expr = parse_expression(expression)
    var = _pick_variable(expr, variable)
    expr, var = _as_real(expr, var)
    d1 = sp.diff(expr, var)
    d2 = sp.diff(d1, var)
    sols = _with_timeout(lambda: sp.solveset(d1, var, sp.S.Reals), "Solving f'(x) = 0")
    if not isinstance(sols, sp.FiniteSet) and sols != sp.S.EmptySet:
        raise ToolError("f'(x) = 0 has infinitely many or no closed-form solutions; this case is not supported.")
    points = sorted(sols, key=lambda c: float(sp.N(c))) if sols != sp.S.EmptySet else []
    steps = [
        Step(title="Differentiate", latex=f"f'({sp.latex(var)}) = {sp.latex(sp.factor(d1))}"),
        Step(
            title="Solve f'(x) = 0",
            latex=(r",\ ".join(f"{sp.latex(var)} = {sp.latex(c)}" for c in points)) or r"\text{no real solutions}",
        ),
    ]
    found = []
    for c in points:
        curv = sp.simplify(d2.subs(var, c))
        if curv.is_positive:
            kind = "local minimum"
        elif curv.is_negative:
            kind = "local maximum"
        else:  # second-derivative test inconclusive: first-derivative sign test
            h = sp.Rational(1, 10**4)
            left, right = sp.N(d1.subs(var, c - h)), sp.N(d1.subs(var, c + h))
            kind = (
                "local minimum" if left < 0 < right else "local maximum" if left > 0 > right else "neither (inflection)"
            )
        fval = sp.simplify(expr.subs(var, c))
        found.append((c, kind, fval))
        steps.append(
            Step(
                title=f"Classify {var} = {_fmt(c)}",
                detail=f"f''({_fmt(c)}) = {_fmt(curv)}: {kind}; f({_fmt(c)}) = {_fmt(fval)}",
            )
        )
    if not found:
        text, latex = "no real critical points", r"\text{no real critical points}"
    else:
        text = "; ".join(f"{var} = {_fmt(c)}: {k}, f = {_fmt(v)}" for c, k, v in found)
        latex = r",\quad ".join(
            rf"{sp.latex(var)} = {sp.latex(c)}\ \text{{({k}}},\ f = {sp.latex(v)}\text{{)}}" for c, k, v in found
        )
    return ToolOutput(
        answer_text=text,
        answer_latex=latex,
        steps=steps,
        data={"expr": expr, "var": var, "d1": d1, "points": found},
    )


PERCENT_OPERATIONS = ("of", "what_percent", "change", "increase", "decrease")


def percentage(operation: str, value: Any, whole: Any) -> ToolOutput:
    """of: value% of whole · what_percent: value is what % of whole · change: % change from whole to value ·
    increase / decrease: whole changed by value%."""
    if operation not in PERCENT_OPERATIONS:
        raise ToolError(f"Unsupported percentage operation {operation!r}.")
    v, w = sp.nsimplify(parse_expression(str(value))), sp.nsimplify(parse_expression(str(whole)))
    if v.free_symbols or w.free_symbols:
        raise ToolError("Percentages need numbers.")
    if operation in ("what_percent", "change") and w == 0:
        raise ToolError("The reference value cannot be zero.")
    if operation == "of":
        result, latex, text = (
            v / 100 * w,
            rf"\frac{{{sp.latex(v)}}}{{100}}\times {sp.latex(w)}",
            f"{_fmt(v)}% of {_fmt(w)}",
        )
    elif operation == "what_percent":
        result, latex, text = (
            v / w * 100,
            rf"\frac{{{sp.latex(v)}}}{{{sp.latex(w)}}}\times 100\%",
            f"{_fmt(v)} as a percentage of {_fmt(w)}",
        )
    elif operation == "change":
        result = (v - w) / w * 100
        latex, text = (
            rf"\frac{{{sp.latex(v)} - {sp.latex(w)}}}{{{sp.latex(w)}}}\times 100\%",
            f"percentage change from {_fmt(w)} to {_fmt(v)}",
        )
    elif operation == "increase":
        result, latex, text = (
            w * (1 + v / 100),
            rf"{sp.latex(w)}\times\left(1 + \frac{{{sp.latex(v)}}}{{100}}\right)",
            f"{_fmt(w)} increased by {_fmt(v)}%",
        )
    else:
        result, latex, text = (
            w * (1 - v / 100),
            rf"{sp.latex(w)}\times\left(1 - \frac{{{sp.latex(v)}}}{{100}}\right)",
            f"{_fmt(w)} decreased by {_fmt(v)}%",
        )
    result = sp.nsimplify(result)
    unit = "%" if operation in ("what_percent", "change") else ""
    steps = [
        Step(
            title=text[:1].upper() + text[1:],
            latex=f"{latex} = {sp.latex(result)}{r'\%' if unit else ''}",
            detail=_numeric_note([result]),
        )
    ]
    return ToolOutput(
        answer_text=f"{_fmt(result)}{unit}",
        answer_latex=f"{sp.latex(result)}{r'\%' if unit else ''}",
        steps=steps,
        data={"operation": operation, "value": v, "whole": w, "result": result},
    )


# --------------------------------------------------------------------------- spectral facts


def matrix_spectrum(
    queries: list[str],
    size: int | None = None,
    eigenvalues: list[Any] | None = None,
    annihilating: str | None = None,
    charpoly: str | None = None,
    trace: Any = None,
    determinant: Any = None,
    real: bool = False,
) -> ToolOutput:
    """Eigenvalues, traces, determinants and characteristic polynomials of (functions of) a matrix that is
    described only by spectral facts. See ``app/math_engine/spectral.py``."""
    from app.math_engine import spectral

    def run() -> dict[str, Any]:
        try:
            return spectral.matrix_spectrum(
                queries, size, eigenvalues, annihilating, charpoly, trace, determinant, bool(real)
            )
        except spectral.SpectrumError as exc:
            raise ToolError(str(exc)) from exc

    r = _with_timeout(run, "Spectral computation")
    if len(r["lines"]) == 1:
        text, latex = r["lines"][0], r["latex_lines"][0]
    else:
        text = "; ".join(f"{i}. {line}" for i, line in enumerate(r["lines"], 1))
        latex = r"\begin{aligned}" + r" \\ ".join(f"&{line}" for line in r["latex_lines"]) + r"\end{aligned}"
    return ToolOutput(answer_text=text, answer_latex=latex, steps=r["steps"], data=r)


TOOLS: dict[str, Callable[..., ToolOutput]] = {
    "solve_equation": solve_equation,
    "solve_system": solve_system,
    "simplify_expression": simplify_expression,
    "factor_expression": factor_expression,
    "expand_expression": expand_expression,
    "differentiate": differentiate,
    "integrate": integrate,
    "limit": limit,
    "matrix_operation": matrix_operation,
    "descriptive_statistics": descriptive_statistics,
    "probability_distribution": probability_distribution,
    "combinatorics": combinatorics,
    "matrix_spectrum": matrix_spectrum,
    "evaluate_expression": evaluate_expression,
    "summation": summation,
    "sequence": sequence,
    "critical_points": critical_points,
    "percentage": percentage,
}


def run_tool(name: str, args: dict[str, Any]) -> ToolOutput:
    """Dispatch to a whitelisted tool; unknown tools or arguments are rejected.

    Every failure surfaces as ``ToolError``/``ExpressionError`` so a bad input
    can never crash the workflow.
    """
    if name not in TOOLS:
        raise ToolError(f"Unknown tool {name!r}.")
    fn = TOOLS[name]
    sig = inspect.signature(fn)
    unknown = set(args) - set(sig.parameters)
    if unknown:
        raise ToolError(f"Unexpected arguments for {name}: {sorted(unknown)}")
    clean = {k: v for k, v in args.items() if v is not None}
    try:
        sig.bind(**clean)
    except TypeError as exc:
        raise ToolError(f"Invalid arguments for {name}: {exc}") from exc
    try:
        return fn(**clean)
    except (ToolError, ExpressionError):
        raise
    except Exception as exc:  # unexpected library error: report, don't crash
        raise ToolError(f"{name} could not be completed: {type(exc).__name__}: {exc}"[:300]) from exc
