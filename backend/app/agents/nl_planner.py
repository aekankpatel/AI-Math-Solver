"""Intent-based planner for natural-language questions (offline, deterministic).

The rule planner (``rule_planner.py``) expects fairly formulaic input
("integrate x^2 from 0 to 3"). This planner builds a tool call from the
structured intent extracted by ``intent.extract_intent`` (task cues, slots,
math spans and probability parameters), so word problems such as "Weights are
normal with mean 60 and standard deviation 5. What fraction weigh more than
70?" can be handled without an LLM.

Because the same extraction is used to *check* intent, plans from this planner
are never reported as intent-"confirmed": the workflow caps them at "assumed"
and states that the interpretation should be double-checked.

Missing information raises ``ClarificationNeeded`` (a ``PlanningError``) with a
question for the user instead of a guessed default.
"""

from __future__ import annotations

import re

from app.agents.intent import (
    Intent,
    _expr_of,
    _math_spans,
    _normalise,
    _numbers_in,
    _substantial,
    count_roles,
    extract_intent,
    has_inequality,
)
from app.agents.rule_planner import PlanningError
from app.schemas import ProblemCategory, ToolCall

_NUM = r"-?\d+(?:\.\d+)?"


class ClarificationNeeded(PlanningError):
    """The question lacks information that a plan would otherwise have to invent."""


def _call(tool: str, category: ProblemCategory, args: dict, rationale: str) -> ToolCall:
    return ToolCall(
        tool=tool,
        category=category,
        args={k: v for k, v in args.items() if v is not None},
        rationale=rationale,
        source="nl",
    )


def plan_from_intent(problem: str) -> ToolCall:
    it = extract_intent(problem)
    task = it.task
    if it.task_conflict:
        raise ClarificationNeeded("The question asks for several operations; please ask one at a time.")
    if task is None:
        raise PlanningError("No recognisable mathematical task in the question.")
    if task.startswith("matrix"):
        return _plan_matrix(it, task)
    if task == "probability":
        return _plan_probability(problem, it)
    if task == "counting":
        return _plan_counting(problem, it)
    if task == "statistics":
        return _plan_statistics(problem)
    if task == "solve":
        return _plan_solve(problem, it)
    return _plan_expression_task(problem, it, task)


# --------------------------------------------------------------------------- algebra / calculus


def _largest_expression(problem: str) -> str | None:
    spans = [s for s in _math_spans(problem, strip_definitions=True) if "=" not in s and _substantial(s)]
    spans = [s for s in spans if _expr_of(s) is not None]
    return max(spans, key=len) if spans else None


def _plan_solve(problem: str, it: Intent) -> ToolCall:
    if has_inequality(problem):
        raise PlanningError("Inequalities are not supported yet.")
    spans = _math_spans(problem, strip_definitions=False)
    eqs = [s for s in spans if "=" in s and _expr_of(s) is not None]
    if len(eqs) >= 2:
        return _call(
            "solve_system", ProblemCategory.ALGEBRA, {"equations": eqs}, "Several equations: solve the system."
        )
    if not eqs:
        expr = _largest_expression(problem)
        if expr is None:
            raise PlanningError("I could not find an equation in the question.")
        eqs = [expr]
    args = {"equation": eqs[0], "variable": it.variable, "domain": it.domain}
    return _call("solve_equation", ProblemCategory.ALGEBRA, args, "Single equation.")


def _plan_expression_task(problem: str, it: Intent, task: str) -> ToolCall:
    expr = _largest_expression(problem)
    if expr is None:
        raise ClarificationNeeded(f"Which expression should I use for the {task}?")
    if task in ("factor", "expand", "simplify"):
        tool = {"factor": "factor_expression", "expand": "expand_expression", "simplify": "simplify_expression"}[task]
        return _call(tool, ProblemCategory.ALGEBRA, {"expression": expr}, f"{task} request.")
    if task == "derivative":
        args = {"expression": expr, "variable": it.variable, "order": it.order or 1, "point": it.point}
        return _call("differentiate", ProblemCategory.CALCULUS, args, "Derivative (rate of change / slope).")
    if task == "integral":
        if it.partial_bounds:
            raise ClarificationNeeded("Only one integration limit is given; what is the other?")
        args = {"expression": expr, "variable": it.variable, "lower": it.lower, "upper": it.upper}
        return _call("integrate", ProblemCategory.INTEGRALS, args, "Integral / area under a curve.")
    if task == "limit":
        if it.limit_point is None:
            raise ClarificationNeeded("Which point does the variable approach?")
        args = {"expression": expr, "variable": it.variable, "point": it.limit_point, "direction": it.direction or "+-"}
        return _call("limit", ProblemCategory.CALCULUS, args, "Limit.")
    raise PlanningError(f"Unsupported task {task}.")


# --------------------------------------------------------------------------- matrices / statistics


def _plan_matrix(it: Intent, task: str) -> ToolCall:
    if task == "matrix":
        raise ClarificationNeeded("Which matrix operation should I perform (determinant, inverse, eigenvalues, ...)?")
    op = task.split(":", 1)[1]
    args: dict = {"operation": op, "matrix": it.matrices[0]}
    if op == "multiply":
        if len(it.matrices) < 2:
            raise ClarificationNeeded("Matrix multiplication needs a second matrix.")
        args["other"] = it.matrices[1]
    if op == "solve":
        if not it.vectors:
            raise ClarificationNeeded("Solving Ax = b needs the right-hand side vector b.")
        args["vector"] = it.vectors[0]
    return _call("matrix_operation", ProblemCategory.LINEAR_ALGEBRA, args, f"Matrix {op}.")


def _plan_statistics(problem: str) -> ToolCall:
    text = _normalise(problem)
    lists = re.findall(r"-?\d+(?:\.\d+)?(?:\s*(?:,|;|\band\b)\s*-?\d+(?:\.\d+)?)+", text)
    if not lists:
        raise ClarificationNeeded("Which data values should I use?")
    longest = max(lists, key=lambda chunk: len(re.findall(_NUM, chunk)))
    data = [float(v) if "." in v else int(v) for v in re.findall(_NUM, longest)]
    return _call(
        "descriptive_statistics", ProblemCategory.PROBABILITY_STATISTICS, {"data": data}, "Summary statistics."
    )


# --------------------------------------------------------------------------- counting


def _plan_counting(problem: str, it: Intent) -> ToolCall:
    low = _normalise(problem).lower()
    m = re.search(r"(\d+)\s*(?:choose|c)\s*(\d+)", low)
    if m:  # explicit "n choose k": keep the order as written
        return _call(
            "combinatorics",
            ProblemCategory.PROBABILITY_STATISTICS,
            {"operation": "combinations", "n": m.group(1), "k": m.group(2)},
            "n choose k.",
        )
    m = re.search(r"\b([pc])\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", low) or re.search(
        r"\b(permutations?|combinations?)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", low
    )
    if m:
        op = "permutations" if m.group(1).startswith("p") else "combinations"
        return _call(
            "combinatorics",
            ProblemCategory.PROBABILITY_STATISTICS,
            {"operation": op, "n": m.group(2), "k": m.group(3)},
            "Counting formula.",
        )
    m = re.search(r"(\d+)\s*!", low) or re.search(r"factorial of (\d+)", low)
    if m:
        return _call(
            "combinatorics",
            ProblemCategory.PROBABILITY_STATISTICS,
            {"operation": "factorial", "n": m.group(1)},
            "Factorial.",
        )
    nums = [int(n) for n in re.findall(r"(?<![\w.])\d+(?![\w.])", low)]
    if not nums:
        raise ClarificationNeeded("How many objects are there?")
    roles = count_roles(low)
    if len(nums) == 1 and roles >= 2:  # "12 people elect a president and a treasurer": P(12, 2)
        return _call(
            "combinatorics",
            ProblemCategory.PROBABILITY_STATISTICS,
            {"operation": "permutations", "n": nums[0], "k": roles},
            f"{roles} distinct roles from {nums[0]}.",
        )
    if len(nums) == 1:
        # n! only when the question says that ALL objects are arranged; otherwise a part
        # of the information (how many are chosen / ranked) is missing - ask, don't guess.
        if it.counting != "combinations" and re.search(
            r"\barrange|\bline(?:d)? up\b|\bin a (?:row|line|queue)\b|\bon a shelf\b|\border(?:ed|ings?)? (?:all|the)\b",
            low,
        ):
            return _call(
                "combinatorics",
                ProblemCategory.PROBABILITY_STATISTICS,
                {"operation": "factorial", "n": nums[0]},
                "Arrangements of all objects: n!.",
            )
        raise ClarificationNeeded("How many are chosen (or ranked), and from how many in total?")
    n, k = max(nums[:2]), min(nums[:2])
    op = it.counting or "combinations"
    return _call(
        "combinatorics",
        ProblemCategory.PROBABILITY_STATISTICS,
        {"operation": op, "n": n, "k": k},
        f"{op} of {k} from {n}.",
    )


# --------------------------------------------------------------------------- probability


def _find(pattern: str, low: str) -> str | None:
    m = re.search(pattern, low)
    return m.group(1) if m else None


def _plan_probability(problem: str, it: Intent) -> ToolCall:
    low = _normalise(problem).lower()
    dist = it.distribution
    if dist is None:
        raise ClarificationNeeded("Which probability distribution describes the situation?")
    params: dict[str, str | float] = {}
    if dist == "normal":
        if re.search(r"standard normal|\bz\b", low):
            params = {"mean": 0, "std": 1}
        mean = _find(rf"\b(?:mean|average|mu|μ)\s*(?:of|is|=|:)?\s*({_NUM})", low)
        std = _find(rf"(?:standard deviation|\bsd\b|\bstd\b|sigma|σ)\s*(?:of|is|=|:)?\s*({_NUM})", low)
        var = _find(rf"\bvariance\s*(?:of|is|=|:)?\s*({_NUM})", low)
        if mean is not None:
            params["mean"] = mean
        if std is not None:
            params["std"] = std
        elif var is not None:
            params["std"] = f"sqrt({var})"
        if "mean" not in params or "std" not in params:
            missing = "standard deviation" if "mean" in params else "mean and standard deviation"
            raise ClarificationNeeded(f"What is the {missing} of the normal distribution?")
    elif dist == "binomial":
        n = (
            _find(r"\bn\s*=\s*(\d+)", low)
            or _find(
                r"(\d+)\s+(?:independent\s+)?(?:\w+\s+)?(?:trials|times|flips|tosses|shots|attempts|throws|items|questions|"
                r"patients|people|bulbs|parts|games|children|births)\b",
                low,
            )
            or _find(r"\b(?:takes|of|in)\s+(\d+)\s+(?:independent\s+)?\w+", low)
        )
        pct = _find(r"(\d+(?:\.\d+)?)\s*%", low)
        p = _find(rf"\bp\s*=\s*({_NUM})", low) or _find(r"probability (?:of )?(0?\.\d+|1|0)\b", low)
        if p is None and pct is not None:
            p = str(float(pct) / 100)
        if p is None and re.search(r"\bfair\b|\bcoin\b", low):
            p = "0.5"
        if n is None or p is None:
            raise ClarificationNeeded("A binomial question needs the number of trials and the success probability.")
        params = {"n": n, "p": p}
    elif dist == "poisson":
        lam = (
            _find(rf"\blambda\s*=?\s*({_NUM})|λ\s*=?\s*({_NUM})", low)
            or _find(rf"(?:rate|mean|average)(?: rate)?(?: of)?\s*({_NUM})", low)
            or _find(rf"({_NUM})\s+(?:\w+\s+)?per\s+(?:minute|hour|day|second|week|page|month|year)", low)
        )
        if lam is None:
            raise ClarificationNeeded("What is the average rate λ of the Poisson distribution?")
        params = {"lambda": lam}
    elif dist == "uniform":
        m = re.search(rf"(?:between|from)\s+({_NUM})\s+(?:and|to)\s+({_NUM})", low)
        if not m:
            raise ClarificationNeeded("What interval is the uniform distribution on?")
        params = {"a": m.group(1), "b": m.group(2)}
        low = low[: m.start()] + " " + low[m.end() :]
    elif dist == "exponential":
        rate = _find(rf"\brate\s*(?:of|is|=|:)?\s*({_NUM})", low)
        mean = _find(rf"\bmean\s*(?:of|is|=|:)?\s*({_NUM})", low)
        if rate is None and mean is not None:
            rate = f"1/({mean})"
        if rate is None:
            raise ClarificationNeeded("What is the rate of the exponential distribution?")
        params = {"rate": rate}

    from app.agents.intent import _relation

    rel = _relation(low)
    if rel is None:
        rel = _within_relation(low)
    if rel is None:
        raise ClarificationNeeded("Which probability should be computed, e.g. P(X ≤ k) or P(X > k)?")
    q, thr, thr2 = rel
    discrete = dist in ("binomial", "poisson", "hypergeometric")
    t = float(thr)
    if q == "between":
        query, x, x2 = "between", thr, thr2
    elif q == "pmf":
        query, x, x2 = ("pmf" if discrete else "pdf"), thr, None
    elif q == "cdf":
        query, x, x2 = "cdf", thr, None
    elif q == "cdf<":
        query, x, x2 = "cdf", (str(int(t) - 1) if discrete else thr), None
    elif q == "greater>=":
        query, x, x2 = "greater", thr, None
    else:
        query, x, x2 = "greater", (str(int(t) + 1) if discrete else thr), None
    args = {"distribution": dist, "params": params, "query": query, "x": x, "x2": x2}
    return _call("probability_distribution", ProblemCategory.PROBABILITY_STATISTICS, args, f"{dist} model.")


def _within_relation(low: str) -> tuple[str, str, str | None] | None:
    m = re.search(rf"\bwithin\s+({_NUM})", low)
    return ("cdf", m.group(1), None) if m else None


__all__ = ["ClarificationNeeded", "plan_from_intent", "_numbers_in"]
