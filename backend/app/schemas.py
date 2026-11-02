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
    # Filled by the LLM planner: its own restatement, assumptions and the
    # ambiguities it noticed. Ambiguities lead to a clarification request.
    interpretation: str = ""
    assumptions: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    source: Literal["llm", "rules", "nl"] | None = None  # which planner produced the call


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


class IntentStatus(StrEnum):
    CONFIRMED = "confirmed"  # plan matches the question; nothing assumed
    ASSUMED = "assumed"  # consistent with the question, but relies on stated assumptions
    AMBIGUOUS = "ambiguous"  # information is missing or the question has several readings
    MISMATCH = "mismatch"  # the planned tool/arguments contradict the question
    UNCHECKED = "unchecked"


class OverallStatus(StrEnum):
    """End-to-end trust: mathematical verification AND intent verification."""

    VERIFIED = "verified"  # math verified + intent confirmed
    VERIFIED_WITH_ASSUMPTIONS = "verified_with_assumptions"  # math verified + intent assumed
    APPROXIMATE = "approximate"
    UNVERIFIED = "unverified"
    FAILED = "failed"
    NEEDS_CLARIFICATION = "needs_clarification"


class IntentCheck(BaseModel):
    name: str
    passed: bool
    detail: str = ""
    severity: Literal["error", "missing", "warning", "info"] = "info"


class IntentVerification(BaseModel):
    status: IntentStatus
    interpretation: str = ""
    checks: list[IntentCheck] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    clarification: str | None = None
    repaired: bool = False


class SolveRequest(BaseModel):
    problem: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)


class FollowUpRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    question: str = Field(..., min_length=1, max_length=2000)


class SolveResponse(BaseModel):
    session_id: str
    problem: str
    status: Literal["solved", "unsupported", "error", "answered", "needs_clarification"]
    category: ProblemCategory
    tool_call: ToolCall | None = None
    answer_text: str | None = None
    answer_latex: str | None = None
    steps: list[Step] = Field(default_factory=list)
    explanation: str | None = None
    verification: Verification | None = None
    intent: IntentVerification | None = None
    overall_status: OverallStatus | None = None
    visualization: dict[str, Any] | None = None
    error: str | None = None
    llm_used: bool = False
    planner: Literal["llm", "rules", "nl"] = "rules"


class HealthResponse(BaseModel):
    status: Literal["ok"]
    llm_configured: bool
    model: str
