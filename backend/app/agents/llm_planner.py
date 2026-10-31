"""LLM-based classifier/planner using structured output (tool calling).

The LLM only *chooses a tool and fills in string arguments*. It never computes
the answer and never produces code. Its output is validated by Pydantic and by
the safe expression parser inside each tool.
"""

from __future__ import annotations

import inspect
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.llm import get_chat_model
from app.math_engine.tools import TOOLS
from app.schemas import ProblemCategory, ToolCall

PLANNER_SYSTEM_PROMPT = """You are the planning component of MathForge AI, a math solver.
Your ONLY job is to classify the user's problem and translate it into a call to exactly one
whitelisted tool. You must NOT solve the problem or do any arithmetic yourself.

Write all expressions in plain math syntax: use ^ or ** for powers, * for multiplication,
functions like sin, cos, tan, exp, log (natural), sqrt, abs. Constants: pi, e, oo (infinity).
Variables are single letters. Never output Python code, imports, quotes or attribute access.

Tools and the fields they use:
- solve_equation: equation ("lhs = rhs"), variable (optional)
- solve_system: equations (list of "lhs = rhs"), variables (optional)
- simplify_expression / factor_expression / expand_expression: expression
- differentiate: expression, variable, order (int, default 1), point (optional)
- integrate: expression, variable, lower and upper (both or neither; use oo / -oo for infinity)
- limit: expression, variable, point, direction ("+", "-" or "+-")
- matrix_operation: operation (determinant|inverse|eigenvalues|rank|transpose|multiply|solve|rref),
  matrix (list of rows), other (second matrix, for multiply), vector (b for solve Ax=b)
- descriptive_statistics: data (list of numbers)
- probability_distribution: distribution (binomial|poisson|normal|uniform|exponential),
  params (binomial: n, p; poisson: lambda; normal: mean, std; uniform: a, b; exponential: rate),
  query (pmf: P(X=x), pdf: density, cdf: P(X<=x), greater: P(X>x) or P(X>=x) for discrete,
  between: P(x<=X<=x2)), x, x2. For discrete P(X<k) use cdf with x=k-1; P(X>k) use greater with x=k+1.
- combinatorics: operation (combinations|permutations|factorial), n, k

If the problem is not mathematics, is outside these tools (e.g. proofs, geometry constructions,
differential equations), or is too ambiguous to translate faithfully, set supported=false and
explain why in `clarification`. Prefer asking over guessing."""


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
    return ToolCall(tool=out.tool, category=ProblemCategory(out.category), args=args, rationale=out.rationale)


def plan_with_llm(problem: str, context: str | None = None) -> ToolCall | None:
    """Return a ToolCall, ``None`` if no LLM is configured, or raise on unsupported."""
    model = get_chat_model()
    if model is None:
        return None
    structured = model.with_structured_output(PlannerOutput)
    user = problem if not context else f"Previous context:\n{context}\n\nNew request: {problem}"
    result = structured.invoke([("system", PLANNER_SYSTEM_PROMPT), ("human", user)])
    if not isinstance(result, PlannerOutput):
        result = PlannerOutput.model_validate(result)
    return plan_output_to_tool_call(result)
