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
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agents.llm import charge_usage, get_chat_model
from app.config import get_settings
from app.math_engine.tools import TOOLS
from app.schemas import ProblemCategory, ToolCall

PROMPT_VERSION = "2026-11-04.p3c"  # p3c: product renamed to Mathora in the prompts (p3b: tool/category optional)

PLANNER_SYSTEM_PROMPT = """You are the planning component of Mathora, a math solver.
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
    supported: bool = Field(default=True, description="False if no tool can faithfully solve the problem.")
    clarification: str | None = Field(
        default=None, description="Why unsupported/ambiguous, or a question for the user."
    )
    # Optional: it is determined by the tool. Some models (e.g. gpt-oss on Groq) omit it, and Groq
    # rejects tool calls that miss a *required* property with HTTP 400.
    # Free text (not an enum): gpt-oss sometimes writes e.g. "combinatorics" and Groq rejects enum violations
    # with HTTP 400. Unknown values are replaced by the category implied by the tool.
    category: str | None = Field(
        default=None,
        description="One of algebra, calculus, integrals, linear_algebra, probability_statistics (optional).",
    )
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
    ] = Field(
        default="none",
        description="The tool to call; 'none' (or omitted) when supported is false.",
    )
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


class LLMPlanError(RuntimeError):
    """The LLM call failed or returned no usable structured plan."""


_TOOL_CATEGORY = {
    "solve_equation": "algebra",
    "solve_system": "algebra",
    "simplify_expression": "algebra",
    "factor_expression": "algebra",
    "expand_expression": "algebra",
    "differentiate": "calculus",
    "limit": "calculus",
    "integrate": "integrals",
    "matrix_operation": "linear_algebra",
    "descriptive_statistics": "probability_statistics",
    "probability_distribution": "probability_statistics",
    "combinatorics": "probability_statistics",
}


_KNOWN_CATEGORIES = {*_TOOL_CATEGORY.values(), "unsupported"}


def _category_for(out: PlannerOutput) -> str:
    """A recognisable category (any case / spacing) is kept so that a contradiction with the tool is
    detected by the intent check; a missing or unrecognisable one is derived from the tool."""
    raw = (out.category or "").strip().lower().replace(" ", "_").replace("-", "_")
    return raw if raw in _KNOWN_CATEGORIES else _TOOL_CATEGORY[out.tool]


def plan_output_to_tool_call(out: PlannerOutput) -> ToolCall:
    if not out.supported:
        raise LLMPlanUnsupported(out.clarification or "The problem is not supported.")
    if out.tool == "none":  # claims to be supported but names no tool: not a usable plan
        raise LLMPlanError("the plan names no tool")
    allowed = set(inspect.signature(TOOLS[out.tool]).parameters)
    fields = out.model_dump(exclude_none=True)
    args: dict[str, Any] = {k: v for k, v in fields.items() if k in allowed}
    return ToolCall(
        tool=out.tool,
        category=ProblemCategory(_category_for(out)),
        args=args,
        rationale=out.rationale,
        interpretation=out.interpretation,
        assumptions=out.assumptions,
        ambiguities=out.ambiguities,
    )


# ---- per-call usage log (read by the benchmark runner; keyed by problem text)
_USAGE_LOCK = threading.Lock()
_USAGE: dict[str, dict[str, Any]] = {}


def pop_call_stats(problem: str) -> dict[str, Any] | None:
    """Token usage / latency of the last real planner call for ``problem`` (None if cached or no call)."""
    with _USAGE_LOCK:
        return _USAGE.pop(problem, None)


_TOOL_CALL_RETRIES = 1


def _record_usage(problem: str, raw: Any, seconds: float, retries: int = 0) -> None:
    usage = getattr(raw, "usage_metadata", None) or {}
    meta = getattr(raw, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or {}
    stats = {
        "input_tokens": usage.get("input_tokens", token_usage.get("prompt_tokens")),
        "output_tokens": usage.get("output_tokens", token_usage.get("completion_tokens")),
        "total_tokens": usage.get("total_tokens", token_usage.get("total_tokens")),
        "seconds": round(seconds, 3),
        "tool_call_retries": retries,
        "model": meta.get("model_name"),
    }
    with _USAGE_LOCK:
        _USAGE[problem] = stats


# ---- optional response cache (used by the benchmark to make Groq runs reproducible)
_CACHE_LOCK = threading.Lock()


def _cache_key(model_name: str, problem: str, context: str | None, method: str = "function_calling") -> str:
    parts: list[str] = [PROMPT_VERSION, model_name, problem, context or ""]
    if method != "function_calling":  # keeps keys of existing function-calling caches unchanged
        parts.append(method)
    raw = json.dumps(parts)
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> dict[str, Any] | None:
    path = os.environ.get("MATHORA_LLM_CACHE")
    if not path or not Path(path).exists():
        return None
    with _CACHE_LOCK, open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("key") == key:
                return row["output"]
    return None


def _cache_put(key: str, problem: str, output: dict[str, Any]) -> None:
    path = os.environ.get("MATHORA_LLM_CACHE")
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
    method = get_settings().llm_structured_method
    key = _cache_key(model_name, problem, context, method)
    cached = _cache_get(key)
    if cached is not None:
        return plan_output_to_tool_call(PlannerOutput.model_validate(cached))
    # include_raw keeps the provider response so token usage can be measured.
    # function_calling: the plan is a forced tool call; json_schema: the plan is the JSON reply
    # (response_format=json_schema), which avoids Groq's tool-call validation failures.
    if method == "json_schema":
        structured = model.with_structured_output(PlannerOutput, method="json_schema", include_raw=True)
    else:
        structured = model.with_structured_output(PlannerOutput, include_raw=True)
    user = problem if not context else f"Previous context:\n{context}\n\nNew request: {problem}"
    t0 = time.perf_counter()
    retries = 0
    while True:
        try:
            result = structured.invoke([("system", PLANNER_SYSTEM_PROMPT), ("human", user)])
            break
        except Exception as exc:  # network, rate limit after SDK retries, malformed tool call (400)
            # Groq rejects malformed tool calls with HTTP 400 "Tool call validation failed" (e.g. gpt-oss
            # occasionally calls a non-existent tool named 'json'). The SDK does not retry 400s; retry once.
            low = str(exc).lower()
            if retries < _TOOL_CALL_RETRIES and (
                "tool call validation failed" in low
                or "failed to generate json" in low
                or "json_validate_failed" in low
            ):
                retries += 1
                continue
            raise LLMPlanError(
                f"{type(exc).__name__}: {str(exc)[:600]}" + (f" (after {retries} retry)" if retries else "")
            ) from exc
    if isinstance(result, dict) and "parsed" in result:
        _record_usage(problem, result.get("raw"), time.perf_counter() - t0, retries)
        charge_usage(result.get("raw"))
        if result.get("parsing_error") is not None:
            raise LLMPlanError(f"invalid structured output: {str(result['parsing_error'])[:200]}")
        result = result.get("parsed")
    if result is None:  # the model answered in prose instead of calling the planning tool
        raise LLMPlanError("the model returned no structured plan")
    if not isinstance(result, PlannerOutput):
        try:
            result = PlannerOutput.model_validate(result)
        except Exception as exc:
            raise LLMPlanError(f"invalid structured output: {str(exc)[:600]}") from exc
    _cache_put(key, problem, result.model_dump())
    return plan_output_to_tool_call(result)
