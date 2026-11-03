"""Regression tests for the public-deployment protections (all offline; no real Groq calls).

* global token budget, Groq rate-limit / quota handling and the circuit breaker
* per-client request limits and the per-client share of the Groq budget
* the offline-planner notice in API responses
* request validation, body-size limit, security headers, docs toggle
* secrets never reach responses or logs; sandbox failures become clear errors
"""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from app import service
from app.agents.llm_planner import LLMPlanError, LLMPlanUnsupported
from app.budget import TokenBudget, parse_duration, retry_after_from_error
from app.config import get_settings
from app.ratelimit import SlidingWindowLimiter, client_key
from app.sandbox import SandboxBusy, SandboxTimeout
from app.schemas import ProblemCategory, ToolCall
from app.security import RedactingFilter, redact

GOOD_PLAN = ToolCall(
    tool="integrate",
    category=ProblemCategory.INTEGRALS,
    args={"expression": "x^2", "variable": "x", "lower": "0", "upper": "3"},
    interpretation="Integrate x^2 from 0 to 3.",
)
TPD_429 = (
    "RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` "
    "in organization `org_abcdefgh12345678` service tier `on_demand` on tokens per day (TPD): Limit 200000, "
    "Used 199500, Requested 1900. Please try again in 7m12.5s.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)


class Clock:
    def __init__(self) -> None:
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def llm_on(monkeypatch):
    """Pretend Groq is configured; the planner call itself is replaced per test."""
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_unitTestNotARealKey000")
    get_settings.cache_clear()
    from app.budget import get_budget

    get_budget.cache_clear()
    service.get_client_limits.cache_clear()
    calls = []

    def planner(problem, context=None):
        calls.append(problem)
        return GOOD_PLAN

    monkeypatch.setattr(service, "plan_with_llm", planner)
    return calls


@pytest.fixture
def client():
    from app.main import app

    return TestClient(app)


# --------------------------------------------------------------------- budget unit tests


def test_parse_groq_durations():
    assert parse_duration("7m12.5s") == pytest.approx(432.5)
    assert parse_duration("2m59.56s") == pytest.approx(179.56)
    assert parse_duration("1h2m") == pytest.approx(3720)
    assert parse_duration("450ms") == pytest.approx(0.45)
    assert parse_duration("12") == 12
    assert parse_duration("soon") is None
    assert retry_after_from_error(TPD_429) == pytest.approx(432.5)


def test_budget_counts_real_tokens_over_a_rolling_day():
    clock = Clock()
    b = TokenBudget(daily_tokens=10_000, daily_requests=100, reserve_tokens=2_000, clock=clock)
    assert b.check().allowed
    b.record(4_000)
    b.record(3_500)
    assert b.tokens_used() == 7_500
    d = b.check()  # 7.5k + 2k estimate <= 10k
    assert d.allowed
    b.record(1_000)
    d = b.check()
    assert not d.allowed and d.reason == "budget" and d.retry_after > 0
    clock.t += 24 * 3600  # the window rolls over
    assert b.check().allowed and b.tokens_used() == 0


def test_budget_request_cap():
    b = TokenBudget(daily_tokens=10**9, daily_requests=2, reserve_tokens=1, clock=Clock())
    b.record(1)
    b.record(1)
    assert b.check().reason == "budget"


def test_groq_daily_quota_429_pauses_until_reset():
    clock = Clock()
    b = TokenBudget(daily_tokens=10**6, daily_requests=1000, reserve_tokens=100, clock=clock)
    assert b.failure(TPD_429) == "quota"
    d = b.check()
    assert not d.allowed and d.reason == "cooldown" and d.retry_after == pytest.approx(432.5)
    clock.t += 433
    assert b.check().allowed


def test_circuit_breaker_after_repeated_errors():
    clock = Clock()
    b = TokenBudget(10**6, 1000, 100, failure_threshold=3, failure_cooldown=60, clock=clock)
    assert b.failure("APIConnectionError: boom") == ""
    assert b.failure("APIConnectionError: boom") == ""
    assert b.failure("APIConnectionError: boom") == "errors"
    assert b.check().reason == "cooldown"
    clock.t += 61
    assert b.check().allowed


def test_rate_limit_headers_with_no_requests_left_pause_the_llm():
    clock = Clock()
    b = TokenBudget(10**6, 1000, 100, clock=clock)
    b.observe_headers({"x-ratelimit-remaining-requests": "5"})
    assert b.check().allowed
    b.observe_headers({"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "2m0s"})
    d = b.check()
    assert d.reason == "cooldown" and d.retry_after == pytest.approx(120)


def test_budget_state_persists_without_secrets(tmp_path):
    path = tmp_path / "budget.json"
    clock = Clock()
    b = TokenBudget(10_000, 100, 100, state_path=str(path), clock=clock)
    b.record(1234)
    b.failure(TPD_429)
    again = TokenBudget(10_000, 100, 100, state_path=str(path), clock=clock)
    assert again.tokens_used() == 1234
    assert again.check().reason == "cooldown"
    text = path.read_text()
    assert "org_" not in text and "gsk_" not in text
    assert set(json.loads(text)) == {"ledger", "paused_until", "pause_reason"}


def test_corrupt_budget_file_is_ignored(tmp_path):
    path = tmp_path / "budget.json"
    path.write_text("{not json")
    assert TokenBudget(10_000, 100, 100, state_path=str(path)).check().allowed


# --------------------------------------------------------------------- rate limiter unit tests


def test_sliding_window_limiter():
    clock = Clock()
    lim = SlidingWindowLimiter(limit=3, window=60, clock=clock)
    assert all(lim.hit("a")[0] for _ in range(3))
    ok, retry = lim.hit("a")
    assert not ok and 0 < retry <= 60
    assert lim.hit("b")[0]  # other clients are unaffected
    clock.t += 60
    assert lim.hit("a")[0]


def test_client_key_ignores_spoofed_forwarded_for_by_default():
    assert client_key("203.0.113.9", "1.2.3.4", trusted_hops=0) == "203.0.113.9"
    # behind one trusted proxy the right-most entry (added by the proxy) is used, not the client's own
    assert client_key("10.0.0.2", "6.6.6.6, 198.51.100.7", trusted_hops=1) == "198.51.100.7"
    assert client_key("2001:db8::1", None, 0) == client_key("2001:db8::ffff", None, 0)  # IPv6 /64
    assert client_key("not-an-ip", None, 0) == "invalid"


# --------------------------------------------------------------------- service: Groq gating and notices


def test_groq_plan_is_used_and_reported(llm_on):
    r = service.solve_problem("Integrate x^2 from 0 to 3")
    assert r.status == "solved" and r.answer_text == "9"
    assert r.planner == "llm" and r.llm_status == "used" and r.llm_used
    assert r.planner_notice is None
    from app.budget import get_budget

    assert get_budget().requests_used() == 0  # the fake planner reported no usage; nothing charged


def test_budget_exhausted_switches_to_offline_planner_with_notice(llm_on):
    from app.budget import get_budget

    get_budget().record(get_settings().llm_daily_token_budget)
    r = service.solve_problem("Integrate x^2 from 0 to 3")
    assert llm_on == []  # Groq was not called at all
    assert r.status == "solved" and r.answer_text == "9"
    assert r.planner != "llm" and r.llm_status == "budget"
    assert "budget" in r.planner_notice and "offline planner" in r.planner_notice


def test_groq_429_pauses_planner_and_later_requests_skip_it(llm_on, monkeypatch):
    calls = []

    def rate_limited(problem, context=None):
        calls.append(problem)
        raise LLMPlanError(TPD_429)

    monkeypatch.setattr(service, "plan_with_llm", rate_limited)
    first = service.solve_problem("Integrate x^2 from 0 to 3")
    assert first.status == "solved" and first.llm_status == "cooldown" and first.planner_notice
    second = service.solve_problem("Solve x^2 - 5x + 6 = 0")
    assert second.status == "solved" and second.llm_status == "cooldown"
    assert len(calls) == 1  # paused: no second Groq call
    dumped = first.model_dump_json()
    assert "org_abcdefgh" not in dumped and "rate_limit_exceeded" not in dumped and "RateLimitError" not in dumped


def test_groq_failure_falls_back_offline(llm_on, monkeypatch):
    monkeypatch.setattr(
        service, "plan_with_llm", lambda p, c=None: (_ for _ in ()).throw(LLMPlanError("the plan names no tool"))
    )
    r = service.solve_problem("Integrate x^2 from 0 to 3")
    assert r.status == "solved" and r.answer_text == "9"
    assert r.llm_status == "error" and r.planner != "llm" and "offline planner" in r.planner_notice


def test_groq_plan_that_fails_intent_checks_is_reported_as_rejected(llm_on, monkeypatch):
    wrong = ToolCall(
        tool="differentiate", category=ProblemCategory.CALCULUS, args={"expression": "x^2", "variable": "x"}
    )
    monkeypatch.setattr(service, "plan_with_llm", lambda p, c=None: wrong)
    r = service.solve_problem("Integrate x^2 from 0 to 3")
    assert r.answer_text == "9" and r.planner != "llm"
    assert r.llm_status == "rejected" and r.planner_notice


def test_groq_unsupported_judgement_is_kept(llm_on, monkeypatch):
    monkeypatch.setattr(
        service, "plan_with_llm", lambda p, c=None: (_ for _ in ()).throw(LLMPlanUnsupported("Not a math question."))
    )
    r = service.solve_problem("write a poem")
    assert r.status == "unsupported" and r.llm_status == "unsupported"


def test_per_client_llm_share(llm_on, monkeypatch):
    monkeypatch.setenv("LLM_CALLS_PER_CLIENT_PER_DAY", "2")
    get_settings.cache_clear()
    service.get_client_limits.cache_clear()
    statuses = [service.solve_problem("Integrate x^2 from 0 to 3", client="198.51.100.1").llm_status for _ in range(3)]
    assert statuses == ["used", "used", "client_quota"]
    assert service.solve_problem("Integrate x^2 from 0 to 3", client="198.51.100.2").llm_status == "used"


def test_offline_mode_has_no_warning_notice():
    r = service.solve_problem("Integrate x^2 from 0 to 3")
    assert r.llm_status == "disabled" and r.planner_notice is None and r.planner != "llm"


def test_followup_without_budget_uses_templates(llm_on, monkeypatch):
    from app.agents import tutor
    from app.budget import get_budget

    first = service.solve_problem("Integrate x^2 from 0 to 3")
    get_budget().record(get_settings().llm_daily_token_budget)
    monkeypatch.setattr(tutor, "get_chat_model", lambda: (_ for _ in ()).throw(AssertionError("no LLM call")))
    r = service.handle_followup(first.session_id, "how was this verified?")
    assert r.status == "answered" and not r.llm_used and r.llm_status == "budget"


def test_unknown_session_id_gets_a_fresh_server_id():
    r = service.solve_problem("Integrate x^2 from 0 to 3", session_id="attacker-chosen-id")
    assert r.session_id != "attacker-chosen-id" and len(r.session_id) == 32
    again = service.solve_problem("derivative of x^3", session_id=r.session_id)
    assert again.session_id == r.session_id


# --------------------------------------------------------------------- API: limits, validation, hardening


def test_request_rate_limit_returns_429_with_retry_after(monkeypatch, client):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    get_settings.cache_clear()
    service.get_client_limits.cache_clear()
    codes = [client.post("/api/solve", json={"problem": "derivative of x^3"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = client.post("/api/solve", json={"problem": "derivative of x^3"})
    assert int(r.headers["retry-after"]) >= 1 and "Too many requests" in r.json()["detail"]
    # a spoofed X-Forwarded-For does not get a fresh allowance
    r = client.post("/api/solve", json={"problem": "derivative of x^3"}, headers={"X-Forwarded-For": "9.9.9.9"})
    assert r.status_code == 429


def test_daily_request_limit(monkeypatch, client):
    monkeypatch.setenv("RATE_LIMIT_PER_DAY", "1")
    get_settings.cache_clear()
    service.get_client_limits.cache_clear()
    assert client.post("/api/solve", json={"problem": "derivative of x^3"}).status_code == 200
    assert client.post("/api/solve", json={"problem": "derivative of x^3"}).status_code == 429


def test_body_size_limit(client):
    big = json.dumps({"problem": "x" * 50_000})
    r = client.post("/api/solve", content=big, headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_request_validation(client):
    bad = [
        {"problem": ""},
        {"problem": "   "},
        {"problem": "x" * 1001},
        {"problem": "x\x00+1"},
        {"problem": "x+1", "session_id": "../../etc"},
        {"problem": "x+1", "extra": 1},
        {},
    ]
    for body in bad:
        r = client.post("/api/solve", json=body)
        assert r.status_code == 422, body
        assert "x\x00" not in r.text and "etc" not in r.text  # input is not echoed back
    assert client.post("/api/followup", json={"session_id": "abc", "question": ""}).status_code == 422
    assert client.post("/api/followup", json={"session_id": "abc", "question": "why?"}).status_code == 404


def test_security_headers_and_docs_disabled(client):
    r = client.get("/api/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert "default-src 'none'" in r.headers["content-security-policy"]
    assert r.headers["cache-control"] == "no-store"
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_cors_only_allows_configured_origins(client):
    r = client.options(
        "/api/solve",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert r.headers.get("access-control-allow-origin") is None


def test_health_reports_llm_state_without_secrets(llm_on, client):
    body = client.get("/api/health").json()
    assert body["llm_configured"] and body["llm_state"] == "available"
    assert "gsk_" not in json.dumps(body) and "key" not in json.dumps(body).lower()


def test_responses_never_contain_the_api_key(llm_on, client):
    r = client.post("/api/solve", json={"problem": "Integrate x^2 from 0 to 3"})
    assert r.status_code == 200 and "gsk_" not in r.text


def test_timeout_and_busy_are_clear_errors(monkeypatch, client):
    class Slow:
        def run(self, *a, **k):
            raise SandboxTimeout("slow")

    monkeypatch.setattr(service, "get_runner", lambda: Slow())
    body = client.post("/api/solve", json={"problem": "expand (x+y)^900"}).json()
    assert body["status"] == "error" and body["error_code"] == "timeout"
    assert "time limit" in body["error"]

    class Busy:
        def run(self, *a, **k):
            raise SandboxBusy("busy")

    monkeypatch.setattr(service, "get_runner", lambda: Busy())
    r = client.post("/api/solve", json={"problem": "derivative of x^3"})
    assert r.status_code == 503 and r.headers["retry-after"] == "5"


def test_explosive_input_returns_quickly_through_the_api(client):
    import time

    t = time.monotonic()
    body = client.post("/api/solve", json={"problem": "simplify 9^9^9^9"}).json()
    assert time.monotonic() - t < 2
    assert body["status"] in ("unsupported", "error", "needs_clarification") and body["answer_text"] is None


def test_internal_errors_are_generic(monkeypatch):
    from app.main import app

    monkeypatch.setattr(service, "_compute", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gsk_secretinternal")))
    c = TestClient(app, raise_server_exceptions=False)
    r = c.post("/api/solve", json={"problem": "derivative of x^3"})
    assert r.status_code == 500 and r.json() == {"detail": "Internal server error."}


# --------------------------------------------------------------------- logs


def test_secrets_are_redacted_from_logs():
    text = (
        "auth Bearer gsk_abcdefghijklmnopqrstuvwxyz012345 key=gsk_ZZZZZZZZZZZZZZZZ "
        "api_key='sk-live-12345678' org org_01hxyzabcdefghij"
    )
    out = redact(text)
    assert "abcdefghijklmnop" not in out and "ZZZZZZZZ" not in out and "sk-live" not in out
    assert "01hxyzabcdefghij" not in out


def test_log_records_are_redacted_globally(caplog):
    from app.security import install_log_redaction

    install_log_redaction()
    with caplog.at_level(logging.INFO):
        logging.getLogger("some.library").warning(
            "token gsk_abcdefghijklmnopqrstuvwxyz in organization org_abcdefgh123"
        )
    assert "gsk_abcdefghijklmnop" not in caplog.text and "org_abcdefgh123" not in caplog.text


def test_log_filter_redacts_messages_and_exceptions(caplog):
    logger = logging.getLogger("mathora.test")
    logger.addFilter(RedactingFilter())
    with caplog.at_level(logging.INFO, logger="mathora.test"):
        logger.warning("Groq call failed: %s", "Authorization: Bearer gsk_abcdefghijklmnopqrstuvwxyz")
        try:
            raise RuntimeError("bad key gsk_abcdefghijklmnopqrstuvwxyz")
        except RuntimeError as exc:
            logger.error("boom", exc_info=exc)
    assert "gsk_abcdefghijklmnop" not in caplog.text
    assert "[REDACTED]" in caplog.text


# --------------------------------------------------------------------- structured-output method


def test_json_schema_method_is_requested_and_validated(monkeypatch):
    from app.agents import llm_planner

    seen = {}

    class Model:
        model_name = "fake"

        def with_structured_output(self, schema, **kwargs):
            seen.update(kwargs)
            outer = self

            class Runner:
                def invoke(self, messages):
                    return {
                        "raw": None,
                        "parsed": schema(
                            tool="integrate",
                            category="integrals",
                            expression="x^2",
                            variable="x",
                            lower="0",
                            upper="3",
                        ),
                        "parsing_error": None,
                    }

            return Runner() if outer else None

    monkeypatch.setenv("LLM_STRUCTURED_METHOD", "json_schema")
    get_settings.cache_clear()
    monkeypatch.setattr(llm_planner, "get_chat_model", lambda: Model())
    call = llm_planner.plan_with_llm("Integrate x^2 from 0 to 3")
    assert seen.get("method") == "json_schema" and seen.get("include_raw") is True
    assert call.tool == "integrate" and call.args["upper"] == "3"
    # different method -> different cache key (recorded function-calling caches stay valid)
    assert llm_planner._cache_key("m", "p", None) != llm_planner._cache_key("m", "p", None, "json_schema")


def test_json_reply_without_a_tool_is_rejected(monkeypatch):
    from app.agents import llm_planner

    class Model:
        model_name = "fake"

        def with_structured_output(self, schema, **kwargs):
            class Runner:
                def invoke(self, messages):
                    return {"raw": None, "parsed": schema(tool="none"), "parsing_error": None}

            return Runner()

    monkeypatch.setenv("LLM_STRUCTURED_METHOD", "json_schema")
    get_settings.cache_clear()
    monkeypatch.setattr(llm_planner, "get_chat_model", lambda: Model())
    with pytest.raises(LLMPlanError):
        llm_planner.plan_with_llm("Integrate x^2 from 0 to 3")


def test_cors_origins_are_exact_and_never_wildcard(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", " https://mathora.vercel.app/ , *, not-an-origin ")
    get_settings.cache_clear()
    assert get_settings().cors_origin_list == ["https://mathora.vercel.app"]


def test_root_endpoint_for_the_space_page(client):
    r = client.get("/")
    assert r.status_code == 200 and r.json()["health"] == "/api/health"
