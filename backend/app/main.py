"""FastAPI entry point for MathForge AI."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.ratelimit import client_key
from app.sandbox import SandboxBusy, get_runner, shutdown_runner
from app.schemas import FollowUpRequest, HealthResponse, SolveRequest, SolveResponse
from app.security import HardeningMiddleware, install_log_redaction
from app.service import get_client_limits, handle_followup, llm_state, solve_problem

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
install_log_redaction()
log = logging.getLogger("mathforge")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await run_in_threadpool(get_runner)  # start the computation workers before serving
    install_log_redaction()  # uvicorn configures its handlers after import
    yield
    await run_in_threadpool(shutdown_runner)


app = FastAPI(
    title="MathForge AI",
    version="0.2.0",
    description="Solver–Verifier–Tutor math reasoning API built with LangGraph, SymPy, NumPy and SciPy.",
    lifespan=lifespan,
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.enable_docs else None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    max_age=600,
)
app.add_middleware(HardeningMiddleware, max_body=settings.max_request_bytes, docs_paths=("/docs", "/openapi.json"))


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.error("Unhandled error: %s", type(exc).__name__, exc_info=exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.exception_handler(RequestValidationError)
async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
    # concise messages; never echo the submitted input back
    errors = [
        {"field": ".".join(str(p) for p in e.get("loc", ())[1:]) or "body", "message": e.get("msg", "invalid")}
        for e in exc.errors()
    ][:5]
    return JSONResponse(status_code=422, content={"detail": "Invalid request.", "errors": errors})


@app.exception_handler(SandboxBusy)
async def busy(_: Request, exc: SandboxBusy) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "The server is busy with other computations. Please try again in a few seconds."},
        headers={"Retry-After": "5"},
    )


def client_id(request: Request) -> str:
    return client_key(
        request.client.host if request.client else None,
        request.headers.get("x-forwarded-for"),
        get_settings().trusted_proxy_hops,
    )


def rate_limited(request: Request) -> str:
    """Dependency: per-client request limits for the compute endpoints."""
    key = client_id(request)
    ok, retry = get_client_limits().check_request(key)
    if not ok:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a moment before trying again.",
            headers={"Retry-After": str(int(retry) + 1)},
        )
    return key


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    s = get_settings()
    state = llm_state()
    return HealthResponse(
        status="ok",
        llm_configured=s.llm_configured,
        model=s.llm_model,
        llm_available=state == "available",
        llm_state=state,  # type: ignore[arg-type]
    )


@app.post("/api/solve", response_model=SolveResponse)
async def solve(req: SolveRequest, client: str = Depends(rate_limited)) -> SolveResponse:
    return await run_in_threadpool(solve_problem, req.problem, req.session_id, None, client)


@app.post("/api/followup", response_model=SolveResponse)
async def followup(req: FollowUpRequest, client: str = Depends(rate_limited)) -> SolveResponse:
    try:
        return await run_in_threadpool(handle_followup, req.session_id, req.question, client)
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
