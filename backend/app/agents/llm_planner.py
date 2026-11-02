"""LLM-based classifier/planner using structured output (tool calling).

The LLM only *chooses a tool and fills in string arguments*. It never computes
the answer and never produces code. Its output is validated by Pydantic and by
the safe expression parser inside each tool.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.llm import get_chat_model
from app.math_engine.tools import TOOLS
from app.schemas import ProblemCategory, ToolCall

PROMPT_VERSION = "2026-11-02.p3"

PLANNER_SYSTEM_PROMPT = """You are the planning component of MathForge AI, a math solver.
Your ONLY job is to translate the user's question into a call to exactly one whitelisted tool.
You must NOT solve the problem or do any arithmetic yourself: a computer algebra system does that.

Write all expressions in plain math syntax: use ^ or ** for powers, * for multiplication,
functions like sin, cos, tan, exp, log (natural), sqrt, abs. Constants: pi, e, oo (infinity).
Variables are single letters. Never output Python code, imports, quotes or attribute access.

Tools and the fields they use:
- solve_equation: equation ("lhs = rhs"), variable (optional), domain ("complex" default, "real" or
  "positive" when the question asks for real / positive solutions)
- solve_system: equations (list of "lhs = rhs"), variables (optional)
- simplify_expression / factor_expression / expand_expression: expression
- differentiate: expression, variable, order (int, default 1), point (only if the question gives one)
- integrate: expression, variable, lower and upper (both or neither; use oo / -oo for infinity)
- limit: expression, variable, point, direction ("+", "-" or "+-")
- matrix_operation: operation (determinant|inverse|eigenvalues|rank|transpose|multiply|solve|rref),
  matrix (list of rows), other (second matrix, for multiply), vector (b for solve Ax=b)
- descriptive_statistics: data (list of numbers)
- probability_distribution: distribution (binomial|poisson|normal|uniform|exponential),
  params (binomial: n, p; poisson: lambda; normal: mean, std; uniform: a, b; exponential: rate),
  query (pmf: P(X=x), pdf: density, cdf: P(X<=x), greater: P(X>x) for continuous and P(X>=x) for
  discrete, between: P(x<=X<=x2)), x, x2.
- combinatorics: operation (combinations|permutations|factorial), n, k

Faithfulness rules (your plan is checked against the question; violations are rejected):
1. Every number you put in the arguments must appear in the question (or be implied by a word
   such as "fair coin" = 0.5, "standard normal" = mean 0 / std 1, "squared" = 2, "70%" = 0.7).
   NEVER invent a missing parameter (e.g. a standard deviation, n, an integration limit or a
   limit point). If one is missing, set supported=false and ask for it in `clarification`.
2. Use every condition in the question: "positive"/"real" solutions -> domain; "at x = a" -> point;
   "second derivative" -> order 2; "from the right" -> direction "+"; "from a to b" -> lower=a, upper=b.
3. Discrete distributions (binomial, poisson): "fewer than k" / "< k" -> cdf with x = k-1;
   "at most k" -> cdf x = k; "at least k" -> greater x = k; "more than k" / "> k" -> greater x = k+1.
   Continuous distributions: "below k" -> cdf x = k; "above k" -> greater x = k.
4. Counting: order matters (arrangements, rankings, line-ups, 1st/2nd/3rd) -> permutations;
   order does not matter (committees, hands, selections) -> combinations.
5. If an equation has several unknowns and the question does not say which to solve for (and gives
   no second equation), list this in `ambiguities`.
6. Fill `interpretation` with a one-sentence restatement of exactly what will be computed, and list
   any reading you had to assume in `assumptions` (e.g. "log means natural log").

Examples:
- "What fraction of scores are above 91 if scores ~ normal(mean 75, sd 8)?" ->
  probability_distribution, distribution=normal, params={mean: 75, std: 8}, query=greater, x=91
- "Probability of fewer than 2 calls when calls arrive at 4 per hour (Poisson)" ->
  probability_distribution, distribution=poisson, params={lambda: 4}, query=cdf, x=1
- "How fast is x^3 changing at x = 2?" -> differentiate, expression="x^3", point="2"
- "Normal distribution with mean 10, P(X < 12)?" -> supported=false,
  clarification="What is the standard deviation?"

If the problem is not mathematics, is outside these tools (e.g. proofs, geometry constructions,
differential equations, inequalities), or is too ambiguous to translate faithfully, set
supported=false and explain why in `clarification`. Prefer asking over guessing."""


class PlannerOutput(BaseModel):
    supported: bool = Field(description="False if no tool can faithfully solve the problem.")
    clarification: str | None = Field(
        default=None, description="Why unsupported/ambiguous, or a question for the user."
    )
    category: Literal["algebra", "calculus", "integrals", "linear_algebra", "probability_statistics", "unsupported"]
    tool: Literal[
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
        "none",
    ]
    rationale: str = Field(default="", description="One short sentence on why this tool was chosen.")
    interpretation: str = Field(default="", description="One-sentence restatement of what will be computed.")
    assumptions: list[str] = Field(default_factory=list, description="Readings you had to assume.")
    ambiguities: list[str] = Field(default_factory=list, description="Information that is missing or unclear.")
    domain: str | None = None
    expression: str | None = None
    equation: str | None = None
    equations: list[str] | None = None
    variable: str | None = None
    variables: list[str] | None = None
    order: int | None = None
    point: str | None = None
    lower: str | None = None
    upper: str | None = None
    direction: str | None = None
    operation: str | None = None
    matrix: list[list[float | str]] | None = None
    other: list[list[float | str]] | None = None
    vector: list[float | str] | None = None
    data: list[float] | None = None
    distribution: str | None = None
    params: dict[str, float | str] | None = None
    query: str | None = None
    x: float | str | None = None
    x2: float | str | None = None
    n: int | str | None = None
    k: int | str | None = None


class LLMPlanUnsupported(ValueError):
    """The LLM judged the problem unsupported or ambiguous."""


def plan_output_to_tool_call(out: PlannerOutput) -> ToolCall:
    if not out.supported or out.tool == "none":
        raise LLMPlanUnsupported(out.clarification or "The problem is not supported.")
    allowed = set(inspect.signature(TOOLS[out.tool]).parameters)
    fields = out.model_dump(exclude_none=True)
    args: dict[str, Any] = {k: v for k, v in fields.items() if k in allowed}
    return ToolCall(
        tool=out.tool,
        category=ProblemCategory(out.category),
        args=args,
        rationale=out.rationale,
        interpretation=out.interpretation,
        assumptions=out.assumptions,
        ambiguities=out.ambiguities,
    )


# ---- optional response cache (used by the benchmark to make Groq runs reproducible)
_CACHE_LOCK = threading.Lock()


def _cache_key(model_name: str, problem: str, context: str | None) -> str:
    raw = json.dumps([PROMPT_VERSION, model_name, problem, context or ""])
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> dict[str, Any] | None:
    path = os.environ.get("MATHFORGE_LLM_CACHE")
    if not path or not Path(path).exists():
        return None
    with _CACHE_LOCK, open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("key") == key:
                return row["output"]
    return None


def _cache_put(key: str, problem: str, output: dict[str, Any]) -> None:
    path = os.environ.get("MATHFORGE_LLM_CACHE")
    if not path:
        return
    with _CACHE_LOCK, open(path, "a", encoding="utf-8") as fh:
        fh.write(
            json.dumps({"key": key, "prompt_version": PROMPT_VERSION, "problem": problem, "output": output}) + "\n"
        )


def plan_with_llm(problem: str, context: str | None = None) -> ToolCall | None:
    """Return a ToolCall, ``None`` if no LLM is configured, or raise on unsupported."""
    model = get_chat_model()
    if model is None:
        return None
    model_name = str(getattr(model, "model_name", getattr(model, "model", "unknown")))
    key = _cache_key(model_name, problem, context)
    cached = _cache_get(key)
    if cached is not None:
        return plan_output_to_tool_call(PlannerOutput.model_validate(cached))
    structured = model.with_structured_output(PlannerOutput)
    user = problem if not context else f"Previous context:\n{context}\n\nNew request: {problem}"
    result = structured.invoke([("system", PLANNER_SYSTEM_PROMPT), ("human", user)])
    if not isinstance(result, PlannerOutput):
        result = PlannerOutput.model_validate(result)
    _cache_put(key, problem, result.model_dump())
    return plan_output_to_tool_call(result)
