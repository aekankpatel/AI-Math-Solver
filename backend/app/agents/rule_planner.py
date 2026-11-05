"""Deterministic, offline problem classifier and tool planner.

Used when no LLM is configured, and as a fallback when the LLM's plan is
invalid. It recognises common phrasings for each supported category and
produces a :class:`ToolCall`. Anything it cannot recognise confidently is
reported as unsupported rather than guessed.
"""

from __future__ import annotations

import ast
import re

from app.math_engine.parser import (
    ALLOWED_CONSTANTS,
    ALLOWED_FUNCTIONS,
    ExpressionError,
    validate_expression_text,
)
from app.schemas import ProblemCategory, ToolCall


class PlanningError(ValueError):
    """The problem could not be mapped to a supported tool."""


_NUM = r"-?\d+(?:\.\d+)?(?:/\d+)?"


def _strip_prefix(text: str, words: list[str]) -> str:
    pattern = r"^\s*(?:" + "|".join(words) + r")\b[\s:,]*(?:the\s+)?(?:expression\s+|function\s+)?(?:of\s+)?"
    return re.sub(pattern, "", text, flags=re.I).strip()


def _clean_expr(text: str) -> str:
    text = text.strip().strip(".?!").strip()
    text = re.sub(r"^\s*(?:f|g|y)\s*\(\s*[a-z]\s*\)\s*=\s*", "", text)  # f(x) = ...
    text = re.sub(r"^\s*y\s*=\s*", "", text)
    text = re.sub(r"\s*d([a-z])\s*$", "", text)  # trailing "dx" in integrals
    return text.strip()


def _var_from(text: str) -> str | None:
    m = re.search(r"\b(?:with respect to|wrt|for|in terms of)\s+([a-z])\b", text, re.I)
    if m:
        return m.group(1)
    m = re.search(r"d\s*/\s*d([a-z])", text)
    return m.group(1) if m else None


def _remove_var_clause(text: str) -> str:
    return re.sub(r"\s*\b(?:with respect to|wrt|for|in terms of)\s+[a-z]\b", "", text, flags=re.I).strip()


def _parse_list_literal(text: str) -> list | None:
    try:
        value = ast.literal_eval(text)  # literals only: never executes code
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, list) else None


def _find_matrices(text: str) -> list[list[list[float]]]:
    found = []
    for m in re.finditer(r"\[\s*\[.*?\]\s*\]", text):
        val = _parse_list_literal(m.group(0))
        if val and all(isinstance(r, list) for r in val):
            found.append(val)
    return found


def _numbers(text: str) -> list[str]:
    return re.findall(_NUM, text)


def _looks_mathematical(text: str) -> bool:
    """Heuristic guard so plain English is not parsed as a product of variables."""
    if not re.search(r"[\d+\-*/^()=√π²³!]", text):
        return False
    words = re.findall(r"[A-Za-z]+", text)
    return all(len(w) <= 2 or w in ALLOWED_FUNCTIONS or w in ALLOWED_CONSTANTS for w in words)


def plan_with_rules(problem: str) -> ToolCall:
    text = problem.strip()
    low = text.lower()
    if not text:
        raise PlanningError("Empty problem.")

    from app.agents.spectral_planner import plan_spectrum

    spectral = plan_spectrum(text)
    if spectral is not None:
        return spectral
    for planner in (
        _plan_area_between,
        _plan_matrix,
        _plan_probability,
        _plan_combinatorics,
        _plan_statistics,
        _plan_limit,
        _plan_integral,
        _plan_derivative,
        _plan_transform,
        _plan_evaluate,
        _plan_equation,
    ):
        call = planner(text, low)
        if call is not None:
            return call
    raise PlanningError(
        "I couldn't recognise this as a supported problem (algebra, calculus, integrals, "
        "linear algebra, probability or statistics). Try rephrasing, e.g. 'solve x^2 - 4 = 0'."
    )


# --------------------------------------------------------------------------- category planners


_AREA = re.compile(r"\barea\s+(?:between|enclosed\s+by|bounded\s+by|of\s+the\s+region\s+between)\b", re.I)


def area_curves(text: str) -> list[str]:
    """The two curves of an 'area between' question: 'y = x^2 and y = x', 'y = sin(x) and y = cos(x)'."""
    curves = re.findall(
        r"\b(?:y|[fg]\s*\(\s*x\s*\))\s*=\s*(.+?)(?=\s*(?:,|;|\band\b|\bfrom\b|\bbetween\b|\bfor\b|\bon\b|\bover\b|$|\?|\.\s|\.$))",
        text,
    )
    return [c.strip() for c in curves][:2]


def area_bounds(text: str) -> tuple[str, str] | None:
    m = (
        re.search(rf"\bfrom\s+x\s*=\s*({_NUM}|pi|-?\w+)\s+to\s+x\s*=\s*({_NUM}|pi|-?\w+)", text)
        or re.search(rf"\bfrom\s+({_NUM}|pi)\s+to\s+({_NUM}|pi)\b", text)
        or re.search(rf"\bbetween\s+x\s*=\s*({_NUM}|pi)\s+and\s+x\s*=\s*({_NUM}|pi)", text)
        or re.search(rf"\[\s*({_NUM}|pi)\s*,\s*({_NUM}|pi)\s*\]", text)
    )
    return (m.group(1), m.group(2)) if m else None


def _plan_area_between(text: str, low: str) -> ToolCall | None:
    """'area between y = f and y = g from a to b' -> ∫_a^b |f - g| dx (written from the question, nothing computed)."""
    if not _AREA.search(text):
        return None
    curves = area_curves(text)
    if len(curves) != 2:
        return None
    bounds = area_bounds(text)
    if bounds is None:
        raise PlanningError("Between which x-values? Please give the interval (e.g. 'from x = 0 to x = 1').")
    return ToolCall(
        tool="integrate",
        category=ProblemCategory.INTEGRALS,
        args={
            "expression": f"abs(({curves[0]}) - ({curves[1]}))",
            "variable": "x",
            "lower": bounds[0],
            "upper": bounds[1],
        },
        rationale="Area between two curves: integrate the absolute difference.",
    )


def _plan_matrix(text: str, low: str) -> ToolCall | None:
    mats = _find_matrices(text)
    if not mats:
        return None
    ops = [
        ("eigen", "eigenvalues"),
        ("determinant", "determinant"),
        ("det", "determinant"),
        ("inverse", "inverse"),
        ("invert", "inverse"),
        ("rank", "rank"),
        ("transpose", "transpose"),
        ("rref", "rref"),
        ("row echelon", "rref"),
        ("row reduce", "rref"),
        ("multiply", "multiply"),
        ("product", "multiply"),
        ("times", "multiply"),
        ("*", "multiply"),
    ]

    # Whole-word matching: "determine the rank" must not match "det".
    def has(keyword: str) -> bool:
        return (
            keyword == "*"
            and "*" in low
            or bool(re.search(rf"\b{re.escape(keyword)}", low))
            and not (keyword == "det" and re.search(r"\bdetermin(e|ing)\b", low))
        )

    op = next((o for k, o in ops if has(k)), None)
    args: dict = {"matrix": mats[0]}
    if op == "multiply" or (op is None and len(mats) >= 2):
        if len(mats) < 2:
            raise PlanningError("Matrix multiplication needs two matrices.")
        op, args["other"] = "multiply", mats[1]
    vec_text = re.sub(r"\[\s*\[.*?\]\s*\]", " ", text)
    vec_m = re.search(r"\[[^\[\]]+\]", vec_text)
    if (op is None or "solve" in low) and vec_m:
        vec = _parse_list_literal(vec_m.group(0))
        if vec:
            op, args["vector"] = "solve", vec
    if op is None:
        raise PlanningError(
            "Which matrix operation? (determinant, inverse, eigenvalues, rank, transpose, rref, multiply, or solve Ax = b)"
        )
    args["operation"] = op
    return ToolCall(
        tool="matrix_operation",
        category=ProblemCategory.LINEAR_ALGEBRA,
        args=args,
        rationale=f"Matrix literal detected; operation: {op}.",
    )


def _kv(low: str, *names: str) -> str | None:
    for name in names:
        m = re.search(rf"\b{name}\s*(?:=|is|of|:)?\s*({_NUM})", low)
        if m:
            return m.group(1)
    return None


def _prob_query(low: str) -> tuple[str, str | None, str | None] | None:
    m = re.search(rf"p\s*\(\s*({_NUM})\s*(<=|<|≤)\s*x\s*(<=|<|≤)\s*({_NUM})\s*\)", low)
    if m:
        return "between", m.group(1), m.group(4)
    m = re.search(r"p\s*\(\s*x\s*(=|<=|>=|<|>|≤|≥)\s*(" + _NUM + r")\s*\)", low)
    if m:
        op, val = m.group(1), m.group(2)
        if op == "=":
            return "pmf", val, None
        if op in ("<=", "≤"):
            return "cdf", val, None
        if op == "<":
            return "cdf_strict", val, None
        if op in (">=", "≥"):
            return "greater", val, None
        return "greater_strict", val, None
    m = re.search(rf"between\s+({_NUM})\s+and\s+({_NUM})", low)
    if m:
        return "between", m.group(1), m.group(2)
    m = re.search(rf"(?:at most|no more than)\s+({_NUM})", low)
    if m:
        return "cdf", m.group(1), None
    m = re.search(rf"(?:at least|no less than)\s+({_NUM})", low)
    if m:
        return "greater", m.group(1), None
    m = re.search(rf"(?:exactly)\s+({_NUM})", low)
    if m:
        return "pmf", m.group(1), None
    return None


def _plan_probability(text: str, low: str) -> ToolCall | None:
    dist = next((d for d in ("binomial", "poisson", "normal", "uniform", "exponential") if d in low), None)
    if dist is None and "gaussian" in low:
        dist = "normal"
    if dist is None:
        if re.search(r"\bflip|coin", low) and "head" in low:
            dist = "binomial"
        else:
            return None
    params: dict[str, str] = {}
    if dist == "binomial":
        n = _kv(low, "n", "trials") or (re.search(r"(\d+)\s+(?:times|flips|trials|tosses)", low) or [None, None])[1]
        p = _kv(low, "p", "probability") or ("0.5" if "fair" in low or "coin" in low else None)
        if n is None or p is None:
            raise PlanningError("A binomial problem needs n (trials) and p (success probability).")
        params = {"n": n, "p": p}
    elif dist == "poisson":
        lam = _kv(low, "lambda", "λ", "rate", "mean", "mu")
        if lam is None:
            raise PlanningError("A Poisson problem needs a rate λ.")
        params = {"lambda": lam}
    elif dist == "normal":
        mu, sd = _kv(low, "mean", "mu", "μ"), _kv(low, "std", "sd", "sigma", "σ", "standard deviation")
        var = _kv(low, "variance")
        if sd is None and var is not None:
            sd = f"sqrt({var})"
        params = {"mean": mu or "0", "std": sd or "1"}
        if mu is None and sd is None and "standard normal" not in low:
            raise PlanningError("A normal problem needs a mean and standard deviation (or say 'standard normal').")
    elif dist == "uniform":
        m = re.search(rf"(?:from|on|between|\[)\s*({_NUM})\s*(?:to|and|,)\s*({_NUM})", low)
        a, b = (m.group(1), m.group(2)) if m else (_kv(low, "a"), _kv(low, "b"))
        if a is None or b is None:
            raise PlanningError("A uniform problem needs bounds a and b.")
        params = {"a": a, "b": b}
        # remove the bounds so they aren't confused with the query
        low = low.replace(m.group(0), " ") if m else low
    else:
        rate = _kv(low, "rate", "lambda", "λ")
        mean = _kv(low, "mean")
        if rate is None and mean is not None:
            rate = f"1/({mean})"
        if rate is None:
            raise PlanningError("An exponential problem needs a rate λ (or a mean).")
        params = {"rate": rate}

    q = _prob_query(low)
    if q is None:
        heads = re.search(rf"(?:exactly\s+)?({_NUM})\s+heads", low)
        if heads:
            q = ("pmf", heads.group(1), None)
        else:
            raise PlanningError("What probability should be computed? e.g. P(X <= 3) or P(1 < X < 2).")
    query, x, x2 = q
    discrete = dist in ("binomial", "poisson")
    if query == "cdf_strict":
        query, x = ("cdf", str(int(float(x)) - 1)) if discrete else ("cdf", x)
    if query == "greater_strict":
        query, x = ("greater", str(int(float(x)) + 1)) if discrete else ("greater", x)
    if query == "pmf" and not discrete:
        query = "pdf"
    return ToolCall(
        tool="probability_distribution",
        category=ProblemCategory.PROBABILITY_STATISTICS,
        args={"distribution": dist, "params": params, "query": query, "x": x, "x2": x2},
        rationale=f"{dist} distribution, query {query}.",
    )


def _plan_combinatorics(text: str, low: str) -> ToolCall | None:
    m = re.search(r"(\d+)\s*(?:choose|c)\s*(\d+)", low) or re.search(
        r"\b(?:c|ncr|combinations?)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", low
    )
    if m:
        return ToolCall(
            tool="combinatorics",
            category=ProblemCategory.PROBABILITY_STATISTICS,
            args={"operation": "combinations", "n": m.group(1), "k": m.group(2)},
            rationale="n choose k",
        )
    m = re.search(r"\b(?:p|npr|permutations?)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", low) or re.search(
        r"(?:arrange|permutations? of|ways to order)\s+(\d+)\D+?(?:taken\s+)?(\d+)\s+at a time", low
    )
    if m:
        return ToolCall(
            tool="combinatorics",
            category=ProblemCategory.PROBABILITY_STATISTICS,
            args={"operation": "permutations", "n": m.group(1), "k": m.group(2)},
            rationale="permutations",
        )
    m = re.search(r"(?:ways to choose|choose|select|pick)\s+(\d+)\s+(?:\w+\s+)?(?:from|out of)\s+(\d+)", low)
    if m:
        return ToolCall(
            tool="combinatorics",
            category=ProblemCategory.PROBABILITY_STATISTICS,
            args={"operation": "combinations", "n": m.group(2), "k": m.group(1)},
            rationale="choose k from n",
        )
    m = re.fullmatch(r"\s*(?:what is\s+|compute\s+|calculate\s+)?(\d+)\s*!\s*\??\s*", low) or re.search(
        r"factorial of (\d+)", low
    )
    if m:
        return ToolCall(
            tool="combinatorics",
            category=ProblemCategory.PROBABILITY_STATISTICS,
            args={"operation": "factorial", "n": m.group(1)},
            rationale="factorial",
        )
    return None


_STAT_WORDS = ("mean", "median", "average", "standard deviation", "variance", "statistics", "std", "mode", "summary")


def _plan_statistics(text: str, low: str) -> ToolCall | None:
    if not any(w in low for w in _STAT_WORDS):
        return None
    m = re.search(r"\[([^\[\]]+)\]", text)
    chunk = m.group(1) if m else text.split(":", 1)[-1] if ":" in text else re.sub(r"^[^\d\-]*", "", text)
    nums = re.findall(r"-?\d+(?:\.\d+)?", chunk)
    if len(nums) < 2:
        return None
    return ToolCall(
        tool="descriptive_statistics",
        category=ProblemCategory.PROBABILITY_STATISTICS,
        args={"data": [float(n) if "." in n else int(n) for n in nums]},
        rationale="Descriptive statistics of a data set.",
    )


def _plan_limit(text: str, low: str) -> ToolCall | None:
    if "lim" not in low:
        return None
    m = re.search(
        r"(?:as|when)\s+([a-z])\s*(?:->|→|approaches|tends to|goes to)\s*([^\s,]+?)(?:\s*(\+|-|from the right|from the left|from above|from below))?\s*(?:[,.]|$|\s+of\b|\s)",
        low + " ",
    )
    m2 = re.search(r"lim(?:it)?\s*_?\s*\{?\(?\s*([a-z])\s*(?:->|→)\s*([^\s}),]+)\s*\}?\)?", low)
    if not (m or m2):
        raise PlanningError("For a limit, say which point the variable approaches, e.g. 'limit of sin(x)/x as x -> 0'.")
    if m:
        var, point, side = m.group(1), m.group(2), (m.group(3) or "")
        expr = re.sub(
            r"(?:as|when)\s+[a-z]\s*(?:->|→|approaches|tends to|goes to)\s*\S+(\s*(from the right|from the left|from above|from below))?",
            "",
            text,
            flags=re.I,
        )
    else:
        var, point, side = m2.group(1), m2.group(2), ""
        expr = text[m2.end() :]
    if point.endswith("+") or point.endswith("-"):
        side, point = point[-1], point[:-1]
    point = {"infinity": "oo", "∞": "oo", "inf": "oo", "-infinity": "-oo", "-∞": "-oo"}.get(point, point)
    direction = (
        "+"
        if side in ("+", "from the right", "from above")
        else "-"
        if side in ("-", "from the left", "from below")
        else "+-"
    )
    expr = _clean_expr(
        _strip_prefix(expr, ["find", "compute", "evaluate", "calculate", "what is", "the", r"lim(?:it)?(?:_\S+)?"])
    )
    expr = _clean_expr(_strip_prefix(expr, [r"lim(?:it)?"]))
    return ToolCall(
        tool="limit",
        category=ProblemCategory.CALCULUS,
        args={"expression": expr, "variable": var, "point": point, "direction": direction},
        rationale="Limit problem.",
    )


def _plan_integral(text: str, low: str) -> ToolCall | None:
    if not re.search(r"integra|∫|antiderivative|area under", low):
        return None
    var = _var_from(text)
    body = _remove_var_clause(text)
    m = re.search(r"\b(?:from|between)\s+(\S+)\s+(?:to|and)\s+(\S+)", body, re.I)
    lower = upper = None
    if m:
        lower, upper = m.group(1).strip(",.?"), m.group(2).strip(",.?")
        body = body[: m.start()] + body[m.end() :]
    body = re.sub(r"∫", " ", body)
    expr = _strip_prefix(body, ["find", "compute", "evaluate", "calculate", "what is", "determine"])
    expr = _strip_prefix(
        expr,
        [
            "the definite integral",
            "the indefinite integral",
            "the integral",
            "definite integral",
            "indefinite integral",
            "integral",
            "integrate",
            "the antiderivative",
            "antiderivative",
            "the area under",
            "area under",
        ],
    )
    m_d = re.search(r"\bd([a-z])\s*$", expr.strip(" .?"))
    if m_d and var is None:
        var = m_d.group(1)
    expr = _clean_expr(expr)
    if not expr:
        raise PlanningError("Which function should be integrated?")
    norm = {"infinity": "oo", "∞": "oo", "-infinity": "-oo", "-∞": "-oo"}
    if lower is not None:
        lower, upper = norm.get(lower, lower), norm.get(upper, upper)
    return ToolCall(
        tool="integrate",
        category=ProblemCategory.INTEGRALS,
        args={"expression": expr, "variable": var, "lower": lower, "upper": upper},
        rationale="Definite integral." if lower is not None else "Indefinite integral.",
    )


def _plan_derivative(text: str, low: str) -> ToolCall | None:
    if not re.search(r"derivative|differentiate|d\s*/\s*d[a-z]|\bf\s*'|\bslope of the tangent", low):
        return None
    order = 1
    if re.search(r"second|2nd|d\^?2|f\s*''", low):
        order = 2
    elif re.search(r"third|3rd|d\^?3|f\s*'''", low):
        order = 3
    var = _var_from(text)
    body = _remove_var_clause(text)
    point = None
    m = re.search(r"\b(?:at|when|where)\s+(?:[a-z]\s*=\s*)?(" + _NUM + r"|pi(?:/\d+)?)\b", body, re.I)
    if m:
        point = m.group(1)
        body = body[: m.start()] + body[m.end() :]
    body = re.sub(r"d\^?\d?\s*/\s*d[a-z]\^?\d?", " ", body)
    expr = _strip_prefix(body, ["find", "compute", "evaluate", "calculate", "what is", "determine"])
    expr = _strip_prefix(
        expr,
        [
            "the second derivative",
            "the third derivative",
            "the 2nd derivative",
            "the 3rd derivative",
            "second derivative",
            "third derivative",
            "the derivative",
            "derivative",
            "differentiate",
        ],
    )
    expr = _clean_expr(expr)
    if not expr:
        raise PlanningError("Which function should be differentiated?")
    return ToolCall(
        tool="differentiate",
        category=ProblemCategory.CALCULUS,
        args={"expression": expr, "variable": var, "order": order, "point": point},
        rationale=f"Derivative of order {order}.",
    )


def _plan_transform(text: str, low: str) -> ToolCall | None:
    for word, tool in (
        ("factor", "factor_expression"),
        ("expand", "expand_expression"),
        ("simplify", "simplify_expression"),
    ):
        if re.match(rf"^\s*(?:please\s+)?{word}\b", low):
            expr = _clean_expr(_strip_prefix(text, [word]))
            return ToolCall(
                tool=tool, category=ProblemCategory.ALGEBRA, args={"expression": expr}, rationale=f"{word} request."
            )
    return None


_EVAL = re.compile(
    rf"^\s*(?:evaluate|find|compute|calculate|what\s+is|what's|work\s+out)\s+(?:the\s+value\s+of\s+)?(?P<expr>.+?)\s+"
    rf"(?:at|when|for|if|where)\s+(?P<var>[a-z])\s*=\s*(?P<val>{_NUM}|pi|e)\s*[.?]?\s*$",
    re.I,
)


def _plan_evaluate(text: str, low: str) -> ToolCall | None:
    """'Evaluate x^3 - 3x at x = 2' / 'What is x^2 + 1 when x = 3?' -> exact substitution."""
    m = _EVAL.match(text)
    if not m or "=" in m.group("expr"):
        return None
    expr = (
        m.group("expr").strip().strip("()")
        if m.group("expr").strip().startswith("(") and m.group("expr").strip().endswith(")")
        else m.group("expr").strip()
    )
    return ToolCall(
        tool="evaluate_expression",
        category=ProblemCategory.ALGEBRA,
        args={"expression": expr, "variable": m.group("var"), "value": m.group("val")},
        rationale="Evaluate the expression at the given point.",
    )


def _plan_equation(text: str, low: str) -> ToolCall | None:
    var = _var_from(text)
    body = _remove_var_clause(text)
    body = _strip_prefix(
        body,
        [
            "solve",
            "find the roots of",
            "find the solutions of",
            "find the zeros of",
            "find",
            "roots of",
            "zeros of",
            "what is",
            "calculate",
            "compute",
            "evaluate",
        ],
    )
    body = re.sub(r"^\s*(?:the\s+)?(?:equation|system)\s*:?\s*", "", body, flags=re.I)
    if "=" in body or re.search(r"roots|zeros|solve", low):
        parts = [p.strip() for p in re.split(r"\s*(?:;|,\s*and\s+|\band\b|,)\s*", body.strip(" .?")) if p.strip()]
        eq_parts = [p for p in parts if "=" in p]
        if len(eq_parts) >= 2:
            return ToolCall(
                tool="solve_system",
                category=ProblemCategory.ALGEBRA,
                args={"equations": eq_parts},
                rationale="System of equations.",
            )
        eq = body.strip(" .?")
        if re.match(r"^\s*[fgy]\s*\(\s*[a-z]\s*\)\s*=", eq):  # "roots of f(x) = x^2-4"
            eq = _clean_expr(eq)
        return ToolCall(
            tool="solve_equation",
            category=ProblemCategory.ALGEBRA,
            args={"equation": eq, "variable": var},
            rationale="Single equation.",
        )
    # A bare arithmetic/expression with no instruction: evaluate/simplify it.
    if re.fullmatch(r"[\w\s+\-*/^().,!√π²³]+", body) and body.strip():
        if not _looks_mathematical(body):
            return None
        try:
            validate_expression_text(_clean_expr(body))
        except ExpressionError:
            return None
        return ToolCall(
            tool="simplify_expression",
            category=ProblemCategory.ALGEBRA,
            args={"expression": _clean_expr(body)},
            rationale="Bare expression: simplify/evaluate.",
        )
    return None
