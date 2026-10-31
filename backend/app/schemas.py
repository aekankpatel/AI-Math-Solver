"""Pydantic models shared by the API, the agent graph and the math tools."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ProblemCategory(StrEnum):
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    INTEGRALS = "integrals"
    LINEAR_ALGEBRA = "linear_algebra"
    PROBABILITY_STATISTICS = "probability_statistics"
    UNSUPPORTED = "unsupported"


class VerificationStatus(StrEnum):
    VERIFIED = "verified"  # independently confirmed (symbolic or exact check)
    APPROXIMATE = "approximate"  # confirmed numerically within tolerance only
    UNVERIFIED = "unverified"  # no independent check was possible
    FAILED = "failed"  # the independent check disagreed with the solver


ToolName = Literal[
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
]


class ToolCall(BaseModel):
    """A structured, validated request to run one whitelisted math tool.

    This is the *only* thing the LLM is allowed to produce for computation:
    a tool name from a fixed list plus string/number arguments. Arguments are
    parsed by the safe expression parser before any computation happens.
    """

    tool: ToolName
    category: ProblemCategory
    args: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class Step(BaseModel):
    title: str
    detail: str = ""
    latex: str | None = None


class VerificationCheck(BaseModel):
    method: str
    passed: bool
    detail: str = ""


class Verification(BaseModel):
    status: VerificationStatus
    checks: list[VerificationCheck] = Field(default_factory=list)
    summary: str = ""


class SolveRequest(BaseModel):
    problem: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)


class FollowUpRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    question: str = Field(..., min_length=1, max_length=2000)


class SolveResponse(BaseModel):
    session_id: str
    problem: str
    status: Literal["solved", "unsupported", "error", "answered"]
    category: ProblemCategory
    tool_call: ToolCall | None = None
    answer_text: str | None = None
    answer_latex: str | None = None
    steps: list[Step] = Field(default_factory=list)
    explanation: str | None = None
    verification: Verification | None = None
    visualization: dict[str, Any] | None = None
    error: str | None = None
    llm_used: bool = False
    planner: Literal["llm", "rules"] = "rules"


class HealthResponse(BaseModel):
    status: Literal["ok"]
    llm_configured: bool
    model: str
