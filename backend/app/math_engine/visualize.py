"""Plotly figure specs (plain JSON) generated from tool outputs.

Figures are produced server-side from the *computed* objects, so plots always
reflect the verified math rather than anything an LLM wrote.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp
from scipy import stats

from app.math_engine.tools import ToolOutput

PRIMARY = "#6366f1"
SECONDARY = "#f59e0b"
ACCENT = "#10b981"
MAX_POINTS = 600


def _layout(title: str, x_title: str = "x", y_title: str = "y") -> dict[str, Any]:
    return {
        "title": {"text": title},
        "xaxis": {"title": {"text": x_title}, "zeroline": True},
        "yaxis": {"title": {"text": y_title}, "zeroline": True},
        "hovermode": "x unified",
        "margin": {"l": 55, "r": 20, "t": 50, "b": 40},
        "legend": {"orientation": "h", "y": -0.3},
    }


def _sample(expr: sp.Expr, var: sp.Symbol, lo: float, hi: float) -> tuple[list[float], list[float | None]]:
    f = sp.lambdify(var, expr, modules=["numpy", "scipy"])
    xs = np.linspace(lo, hi, MAX_POINTS)
    with np.errstate(all="ignore"):
        ys = np.asarray(f(xs), dtype=complex) * np.ones_like(xs)
    out: list[float | None] = []
    finite = np.abs(ys.imag) < 1e-9
    real = ys.real
    good = real[finite & np.isfinite(real)]
    # Clip extreme values (poles) so the interesting region stays visible.
    cap = float(np.percentile(np.abs(good), 95) * 4 + 1) if good.size else 1.0
    for y, ok in zip(real, finite, strict=True):
        out.append(float(y) if ok and np.isfinite(y) and abs(y) <= cap else None)
    return xs.tolist(), out


def _window(points: list[float], default: tuple[float, float] = (-10.0, 10.0)) -> tuple[float, float]:
    if not points:
        return default
    lo, hi = min(points), max(points)
    span = max(hi - lo, 2.0)
    return lo - 0.6 * span, hi + 0.6 * span


def _plottable(expr: sp.Expr, var: sp.Symbol) -> bool:
    return expr.free_symbols <= {var}


def _fn_trace(
    expr: sp.Expr, var: sp.Symbol, lo: float, hi: float, name: str, color: str, **extra: Any
) -> dict[str, Any]:
    xs, ys = _sample(expr, var, lo, hi)
    return {
        "type": "scatter",
        "mode": "lines",
        "x": xs,
        "y": ys,
        "name": name,
        "line": {"color": color, "width": 2.5},
        **extra,
    }


def visualize(tool: str, out: ToolOutput) -> dict[str, Any] | None:
    try:
        builder = _BUILDERS.get(tool)
        return builder(out) if builder else None
    except Exception:
        return None  # a plot is optional and must never break a solution


def _equation(out: ToolOutput) -> dict[str, Any] | None:
    expr, var = out.data["expr"], out.data["var"]
    if not _plottable(expr, var):
        return None
    roots = (
        [float(sp.N(r)) for r in out.data["real_solutions"] if sp.N(r).is_real]
        if out.data["real_solutions"]
        else [float(sp.re(sp.N(s))) for s in out.data["solutions"] if abs(complex(sp.N(s)).imag) < 1e-9]
    )
    lo, hi = _window(roots)
    traces = [_fn_trace(expr, var, lo, hi, f"f({var}) = {sp.sstr(expr)}", PRIMARY)]
    if roots:
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "x": roots,
                "y": [0] * len(roots),
                "name": "roots",
                "marker": {"color": SECONDARY, "size": 11, "line": {"color": "white", "width": 2}},
            }
        )
    title = "Roots are where the curve crosses y = 0" if roots else "No real roots: the curve never crosses y = 0"
    return {"data": traces, "layout": _layout(title, str(var), f"f({var})")}


def _system(out: ToolOutput) -> dict[str, Any] | None:
    exprs, vars_ = out.data["exprs"], out.data["vars"]
    if len(vars_) != 2 or len(exprs) != 2:
        return None
    x, y = vars_
    traces = []
    pts = []
    for sol in out.data["solutions"]:
        if x in sol and y in sol and sol[x].is_real and sol[y].is_real:
            pts.append((float(sol[x]), float(sol[y])))
    lo, hi = _window([p[0] for p in pts])
    colors = [PRIMARY, ACCENT]
    for i, e in enumerate(exprs):
        solved = sp.solve(e, y)
        if not solved:
            return None
        for branch in solved:
            traces.append(_fn_trace(branch, x, lo, hi, f"{sp.sstr(e)} = 0", colors[i % 2]))
    if pts:
        traces.append(
            {
                "type": "scatter",
                "mode": "markers",
                "x": [p[0] for p in pts],
                "y": [p[1] for p in pts],
                "name": "intersection",
                "marker": {"color": SECONDARY, "size": 12},
            }
        )
    return {"data": traces, "layout": _layout("Solutions are the intersection points", str(x), str(y))}


def _equivalence(out: ToolOutput) -> dict[str, Any] | None:
    expr = out.data["input"]
    if len(expr.free_symbols) != 1:
        return None
    (var,) = expr.free_symbols
    return {
        "data": [
            _fn_trace(out.data["result"], var, -5, 5, "result", PRIMARY),
            _fn_trace(expr, var, -5, 5, "original", SECONDARY, line={"color": SECONDARY, "dash": "dot", "width": 3}),
        ],
        "layout": _layout("Original and transformed expressions coincide", str(var)),
    }


def _derivative(out: ToolOutput) -> dict[str, Any] | None:
    expr, var, result = out.data["expr"], out.data["var"], out.data["result"]
    if not _plottable(expr, var):
        return None
    center = float(out.data["point"]) if "point" in out.data else 0.0
    lo, hi = center - 5, center + 5
    label = "f'" if out.data["order"] == 1 else f"f^({out.data['order']})"
    traces = [
        _fn_trace(expr, var, lo, hi, f"f({var})", PRIMARY),
        _fn_trace(
            result, var, lo, hi, f"{label}({var})", SECONDARY, line={"color": SECONDARY, "dash": "dash", "width": 2.5}
        ),
    ]
    if "point" in out.data and out.data["order"] == 1:
        p = out.data["point"]
        y0, slope = float(sp.N(expr.subs(var, p))), float(sp.N(out.data["value"]))
        xs = [center - 2, center + 2]
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "x": xs,
                "y": [y0 + slope * (x - center) for x in xs],
                "name": f"tangent at {var} = {sp.sstr(p)}",
                "line": {"color": ACCENT, "width": 2},
            }
        )
    return {"data": traces, "layout": _layout("Function and its derivative", str(var))}


def _integral(out: ToolOutput) -> dict[str, Any] | None:
    expr, var = out.data["expr"], out.data["var"]
    if not _plottable(expr, var):
        return None
    if "value" not in out.data:
        anti = out.data["antiderivative"]
        return {
            "data": [
                _fn_trace(expr, var, -5, 5, f"f({var})", PRIMARY),
                _fn_trace(
                    anti,
                    var,
                    -5,
                    5,
                    f"F({var}) (C = 0)",
                    SECONDARY,
                    line={"color": SECONDARY, "dash": "dash", "width": 2.5},
                ),
            ],
            "layout": _layout("Integrand and antiderivative", str(var)),
        }
    a, b = out.data["lower"], out.data["upper"]
    af = float(a) if a.is_finite else -6.0
    bf = float(b) if b.is_finite else (af + 8.0)
    pad = 0.3 * max(bf - af, 1.0)
    curve = _fn_trace(expr, var, af - pad, bf + pad, f"f({var})", PRIMARY)
    xs, ys = _sample(expr, var, af, bf)
    area = {
        "type": "scatter",
        "mode": "lines",
        "x": xs,
        "y": ys,
        "fill": "tozeroy",
        "name": f"area = {sp.N(out.data['value'], 6)}",
        "line": {"color": "rgba(99,102,241,0)"},
        "fillcolor": "rgba(99,102,241,0.25)",
    }
    return {"data": [area, curve], "layout": _layout("Signed area under the curve", str(var))}


def _limit(out: ToolOutput) -> dict[str, Any] | None:
    expr, var, pt = out.data["expr"], out.data["var"], out.data["point"]
    if not _plottable(expr, var):
        return None
    if pt.is_infinite:
        lo, hi = (1.0, 60.0) if pt == sp.oo else (-60.0, -1.0)
    else:
        lo, hi = float(pt) - 3, float(pt) + 3
    traces = [_fn_trace(expr, var, lo, hi, f"f({var})", PRIMARY)]
    val = out.data["value"]
    if val.is_finite and val.is_real:
        if pt.is_finite:
            traces.append(
                {
                    "type": "scatter",
                    "mode": "markers",
                    "x": [float(pt)],
                    "y": [float(val)],
                    "name": "limit",
                    "marker": {"color": SECONDARY, "size": 11, "symbol": "circle-open", "line": {"width": 3}},
                }
            )
        else:
            traces.append(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": [lo, hi],
                    "y": [float(val)] * 2,
                    "name": f"y = {sp.sstr(val)}",
                    "line": {"color": SECONDARY, "dash": "dot"},
                }
            )
    return {"data": traces, "layout": _layout(f"Behaviour near {var} → {sp.sstr(pt)}", str(var))}


def _matrix(out: ToolOutput) -> dict[str, Any] | None:
    A = out.data["A"]
    if A.free_symbols:
        return None
    z = np.array(A.evalf(), dtype=float).tolist()
    fig = {
        "data": [
            {
                "type": "heatmap",
                "z": z,
                "colorscale": "RdBu",
                "reversescale": True,
                "showscale": True,
                "text": [[sp.sstr(v) for v in row] for row in A.tolist()],
                "texttemplate": "%{text}",
                "x": [f"c{j + 1}" for j in range(A.cols)],
                "y": [f"r{i + 1}" for i in range(A.rows)],
            }
        ],
        "layout": {**_layout("Matrix A", "column", "row"), "yaxis": {"autorange": "reversed"}, "hovermode": "closest"},
    }
    if out.data["operation"] == "eigenvalues":
        evs = [complex(sp.N(e)) for e in out.data["eigenvalues"]]
        return {
            "data": [
                {
                    "type": "scatter",
                    "mode": "markers+text",
                    "x": [e.real for e in evs],
                    "y": [e.imag for e in evs],
                    "text": [sp.sstr(e) for e in out.data["eigenvalues"]],
                    "textposition": "top center",
                    "marker": {"size": 13, "color": PRIMARY},
                    "name": "eigenvalues",
                }
            ],
            "layout": {**_layout("Eigenvalues in the complex plane", "Re(λ)", "Im(λ)"), "hovermode": "closest"},
        }
    return fig


def _stats(out: ToolOutput) -> dict[str, Any]:
    values = out.data["values"]
    s = out.data["stats"]
    mean = float(sp.N(s["mean"]))
    median = float(sp.N(s["median"]))
    return {
        "data": [
            {
                "type": "histogram",
                "x": values,
                "name": "data",
                "marker": {"color": PRIMARY, "line": {"color": "white", "width": 1}},
                "opacity": 0.85,
            },
        ],
        "layout": {
            **_layout("Distribution of the data", "value", "count"),
            "hovermode": "closest",
            "shapes": [
                {
                    "type": "line",
                    "x0": mean,
                    "x1": mean,
                    "yref": "paper",
                    "y0": 0,
                    "y1": 1,
                    "line": {"color": SECONDARY, "width": 3},
                },
                {
                    "type": "line",
                    "x0": median,
                    "x1": median,
                    "yref": "paper",
                    "y0": 0,
                    "y1": 1,
                    "line": {"color": ACCENT, "width": 3, "dash": "dash"},
                },
            ],
            "annotations": [
                {
                    "x": mean,
                    "y": 1.04,
                    "yref": "paper",
                    "text": f"mean {mean:.4g}",
                    "showarrow": False,
                    "font": {"color": SECONDARY},
                },
                {
                    "x": median,
                    "y": 1.10,
                    "yref": "paper",
                    "text": f"median {median:.4g}",
                    "showarrow": False,
                    "font": {"color": ACCENT},
                },
            ],
        },
    }


def _distribution(out: ToolOutput) -> dict[str, Any] | None:
    d = out.data
    p, q, x, x2 = d["params"], d["query"], d["x"], d["x2"]
    name = d["distribution"]

    def in_region(v: float) -> bool:
        if q in ("pmf", "pdf"):
            return abs(v - x) < 1e-9
        if q == "cdf":
            return v <= x
        if q == "greater":
            return v >= x
        return x <= v <= (x2 if x2 is not None else x)

    if name in ("binomial", "poisson", "hypergeometric"):
        if name == "binomial":
            rv, ks = stats.binom(int(p["n"]), p["p"]), np.arange(0, int(p["n"]) + 1)
        elif name == "hypergeometric":
            good, bad, draws = int(p["success"]), int(p["failure"]), int(p["draws"])
            rv, ks = stats.hypergeom(good + bad, good, draws), np.arange(0, draws + 1)
        else:
            lam = p.get("lambda", p.get("lam", p.get("mean")))
            rv, ks = stats.poisson(lam), np.arange(0, int(lam + 5 * np.sqrt(lam) + 5))
        pm = rv.pmf(ks)
        colors = [SECONDARY if in_region(float(k)) else PRIMARY for k in ks]
        return {
            "data": [
                {"type": "bar", "x": ks.tolist(), "y": pm.tolist(), "marker": {"color": colors}, "name": "P(X = k)"}
            ],
            "layout": {
                **_layout(f"{name.title()} distribution — highlighted region ≈ {d['numeric']:.4g}", "k", "P(X = k)"),
                "hovermode": "closest",
            },
        }
    if name == "normal":
        mu, sd = p.get("mean", p.get("mu", 0.0)), p.get("std", p.get("sigma", 1.0))
        rv, lo, hi = stats.norm(mu, sd), mu - 4 * sd, mu + 4 * sd
    elif name == "uniform":
        a, b = p.get("a", 0.0), p.get("b", 1.0)
        rv, lo, hi = stats.uniform(a, b - a), a - 0.25 * (b - a), b + 0.25 * (b - a)
    else:
        rate = p.get("rate", p.get("lambda", 1.0))
        rv, lo, hi = stats.expon(scale=1 / rate), 0.0, 6 / rate
    xs = np.linspace(lo, hi, 400)
    ys = rv.pdf(xs)
    mask = np.array([in_region(float(v)) for v in xs])
    traces = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": xs.tolist(),
            "y": ys.tolist(),
            "name": "density",
            "line": {"color": PRIMARY, "width": 2.5},
        }
    ]
    if q != "pdf" and mask.any():
        traces.insert(
            0,
            {
                "type": "scatter",
                "mode": "lines",
                "x": xs[mask].tolist(),
                "y": ys[mask].tolist(),
                "fill": "tozeroy",
                "fillcolor": "rgba(245,158,11,0.35)",
                "line": {"color": "rgba(0,0,0,0)"},
                "name": f"probability ≈ {d['numeric']:.4g}",
            },
        )
    return {"data": traces, "layout": _layout(f"{name.title()} distribution", "x", "density")}


def _critical(out: ToolOutput) -> dict[str, Any] | None:
    expr, var, points = out.data["expr"], out.data["var"], out.data["points"]
    if not _plottable(expr, var):
        return None
    xs = [float(sp.N(c)) for c, _, _ in points]
    lo, hi = _window(xs, (-5.0, 5.0))
    traces = [_fn_trace(expr, var, lo, hi, f"f({var})", PRIMARY)]
    if points:
        traces.append(
            {
                "type": "scatter",
                "mode": "markers+text",
                "x": xs,
                "y": [float(sp.N(v)) for _, _, v in points],
                "text": [k.replace("local ", "") for _, k, _ in points],
                "textposition": "top center",
                "marker": {"size": 10, "color": SECONDARY},
                "name": "critical points",
            }
        )
    return {"data": traces, "layout": _layout("Critical points", str(var), f"f({var})")}


def _sequence(out: ToolOutput) -> dict[str, Any] | None:
    d = out.data
    n = min(d["n"], 40)
    step = (lambda a: a + d["d"]) if d["kind"] == "arithmetic" else (lambda a: a * d["r"])
    vals, a = [], d["a1"]
    for _ in range(n):
        vals.append(float(sp.N(a)))
        a = step(a)
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "markers+lines",
                "x": list(range(1, n + 1)),
                "y": vals,
                "marker": {"color": PRIMARY},
                "line": {"color": PRIMARY, "width": 1},
                "name": "aₙ",
            }
        ],
        "layout": {**_layout(f"First {n} terms ({d['kind']})", "n", "aₙ"), "hovermode": "closest"},
    }


_BUILDERS = {
    "critical_points": _critical,
    "sequence": _sequence,
    "solve_equation": _equation,
    "solve_system": _system,
    "simplify_expression": _equivalence,
    "factor_expression": _equivalence,
    "expand_expression": _equivalence,
    "differentiate": _derivative,
    "integrate": _integral,
    "limit": _limit,
    "matrix_operation": _matrix,
    "descriptive_statistics": _stats,
    "probability_distribution": _distribution,
}
