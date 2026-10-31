"""FastAPI entry point for MathForge AI."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.schemas import FollowUpRequest, HealthResponse, SolveRequest, SolveResponse
from app.service import handle_followup, solve_problem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("mathforge")

settings = get_settings()
app = FastAPI(
    title="MathForge AI",
    version="0.1.0",
    description="Solver–Verifier–Tutor math reasoning API built with LangGraph, SymPy, NumPy and SciPy.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    s = get_settings()
    return HealthResponse(status="ok", llm_configured=s.llm_configured, model=s.llm_model)


@app.post("/api/solve", response_model=SolveResponse)
async def solve(req: SolveRequest) -> SolveResponse:
    return await run_in_threadpool(solve_problem, req.problem, req.session_id)


@app.post("/api/followup", response_model=SolveResponse)
async def followup(req: FollowUpRequest) -> SolveResponse:
    try:
        return await run_in_threadpool(handle_followup, req.session_id, req.question)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown or expired session.") from exc


@app.get("/api/examples")
def examples() -> dict[str, list[str]]:
    return {
        "Algebra": ["Solve x^2 - 5x + 6 = 0", "x + 2y = 7, 3x - y = 7", "factor x^3 - 8"],
        "Calculus": [
            "Find the derivative of x^3 sin(x)",
            "limit of sin(x)/x as x -> 0",
            "second derivative of exp(x^2)",
        ],
        "Integrals": ["Integrate x^2 from 0 to 3", "integral of x e^x dx", "integrate exp(-x^2) from -oo to oo"],
        "Linear algebra": ["eigenvalues of [[2,1],[1,2]]", "inverse of [[4,7],[2,6]]", "solve [[2,1],[1,3]] x = [3,5]"],
        "Probability & statistics": [
            "mean and standard deviation of 2, 4, 4, 4, 5, 5, 7, 9",
            "Binomial n=10 p=0.5, P(X <= 3)",
            "normal mean 100 std 15, P(X > 130)",
        ],
    }
