"""Global Groq usage budget: protects the free-tier quota of the whole deployment.

Groq's free tier limits tokens and requests per rolling 24 hours (and tokens per
minute). This module keeps a rolling 24-hour ledger of the tokens Groq actually
reported for every call (an estimate is charged when a call fails without a usage
report), refuses new LLM calls once the configured daily budget is reached, and
pauses the LLM when Groq itself says the quota is exhausted (HTTP 429), when its
rate-limit headers show no requests left, or after repeated failures (circuit
breaker). While the LLM is paused the app keeps working with its offline planners.

The ledger is in memory; set ``LLM_BUDGET_STATE_PATH`` to persist it across
restarts (best effort, a small JSON file with timestamps and token counts only).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings

log = logging.getLogger(__name__)

WINDOW_SECONDS = 24 * 3600

_DURATION = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?")


def parse_duration(text: str | None) -> float | None:
    """Parse Groq durations such as '7m12.5s', '2m59.56s', '1h2m', '450ms' or '12' (seconds)."""
    if not text:
        return None
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    m = _DURATION.fullmatch(text)
    if not m or not any(m.groups()):
        return None
    h, mi, s, ms = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000


_RETRY_IN = re.compile(r"try again in\s+([0-9hms.]+)", re.I)


def retry_after_from_error(message: str) -> float | None:
    """Wait time stated in a Groq 429 error message ("Please try again in 7m12.5s")."""
    m = _RETRY_IN.search(message or "")
    return parse_duration(m.group(1).rstrip(".")) if m else None


def is_rate_limit_error(message: str) -> bool:
    low = (message or "").lower()
    return "429" in low or "rate_limit" in low or "rate limit" in low or "ratelimit" in low


def is_daily_quota_error(message: str) -> bool:
    low = (message or "").lower()
    return "per day" in low or "(tpd)" in low or "(rpd)" in low


@dataclass
class Decision:
    allowed: bool
    reason: str = "ok"  # ok | disabled | budget | cooldown
    retry_after: float = 0.0


class TokenBudget:
    """Thread-safe rolling 24-hour token and request ledger with a pause switch."""

    def __init__(
        self,
        daily_tokens: int,
        daily_requests: int,
        reserve_tokens: int,
        state_path: str | None = None,
        failure_threshold: int = 3,
        failure_cooldown: float = 120.0,
        clock: Any = time.time,
    ) -> None:
        self.daily_tokens = daily_tokens
        self.daily_requests = daily_requests
        self.reserve_tokens = reserve_tokens
        self.failure_threshold = failure_threshold
        self.failure_cooldown = failure_cooldown
        self._clock = clock
        self._lock = threading.Lock()
        self._ledger: deque[tuple[float, int]] = deque()
        self._paused_until = 0.0
        self._pause_reason = ""
        self._failures = 0
        self._path = Path(state_path) if state_path else None
        self._load()

    # ---- persistence (best effort; never stores anything but timestamps and counts)
    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            now = self._clock()
            self._ledger = deque(
                (float(t), int(n)) for t, n in data.get("ledger", []) if now - float(t) < WINDOW_SECONDS
            )
            self._paused_until = float(data.get("paused_until", 0.0))
            self._pause_reason = str(data.get("pause_reason", ""))[:40]
        except Exception as exc:  # a corrupt file must not stop the app
            log.warning("Ignoring unreadable budget state file: %s", type(exc).__name__)

    def _save(self) -> None:
        if not self._path:
            return
        try:
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "ledger": list(self._ledger),
                        "paused_until": self._paused_until,
                        "pause_reason": self._pause_reason,
                    }
                ),
                encoding="utf-8",
            )
            os.replace(tmp, self._path)
        except Exception as exc:
            log.warning("Could not persist budget state: %s", type(exc).__name__)

    def _prune(self, now: float) -> None:
        while self._ledger and now - self._ledger[0][0] >= WINDOW_SECONDS:
            self._ledger.popleft()

    # ---- queries
    def tokens_used(self) -> int:
        with self._lock:
            self._prune(self._clock())
            return sum(n for _, n in self._ledger)

    def requests_used(self) -> int:
        with self._lock:
            self._prune(self._clock())
            return len(self._ledger)

    def check(self, estimate: int | None = None) -> Decision:
        """May one more LLM call of about ``estimate`` tokens be made now?"""
        need = self.reserve_tokens if estimate is None else estimate
        with self._lock:
            now = self._clock()
            self._prune(now)
            if now < self._paused_until:
                return Decision(False, "cooldown", self._paused_until - now)
            used = sum(n for _, n in self._ledger)
            if used + need > self.daily_tokens or len(self._ledger) + 1 > self.daily_requests:
                retry = WINDOW_SECONDS - (now - self._ledger[0][0]) if self._ledger else 60.0
                return Decision(False, "budget", max(retry, 1.0))
            return Decision(True)

    # ---- updates
    def record(self, tokens: int | None, *, success: bool = True) -> None:
        """Charge one call. ``tokens`` is Groq's reported total (None -> the reserve estimate)."""
        n = self.reserve_tokens if tokens is None else max(int(tokens), 0)
        with self._lock:
            now = self._clock()
            self._ledger.append((now, n))
            self._prune(now)
            if success:
                self._failures = 0
            self._save()

    def failure(self, message: str) -> str:
        """Register a failed call; returns the reason the LLM is now paused for ('' if not paused)."""
        with self._lock:
            now = self._clock()
            if is_rate_limit_error(message):
                wait = retry_after_from_error(message)
                daily = is_daily_quota_error(message)
                wait = wait if wait is not None else (900.0 if daily else 60.0)
                self._pause(now, wait, "quota" if daily else "rate_limit")
                self._failures = 0
                return self._pause_reason
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._failures = 0
                self._pause(now, self.failure_cooldown, "errors")
                return self._pause_reason
            return ""

    def observe_headers(self, headers: dict[str, str]) -> None:
        """Pause when Groq's x-ratelimit headers show the daily request quota is used up."""
        remaining = headers.get("x-ratelimit-remaining-requests")
        if remaining is None:
            return
        try:
            left = int(float(remaining))
        except ValueError:
            return
        if left <= 0:
            wait = parse_duration(headers.get("x-ratelimit-reset-requests")) or 900.0
            with self._lock:
                self._pause(self._clock(), wait, "quota")

    def _pause(self, now: float, seconds: float, reason: str) -> None:
        seconds = min(max(seconds, 1.0), WINDOW_SECONDS)
        if now + seconds > self._paused_until:
            self._paused_until = now + seconds
            self._pause_reason = reason
            log.warning("LLM paused for %.0fs (%s); offline planners in use", seconds, reason)
            self._save()

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            self._prune(now)
            used = sum(n for _, n in self._ledger)
            paused = now < self._paused_until
            return {
                "tokens_used_24h": used,
                "token_budget_24h": self.daily_tokens,
                "requests_used_24h": len(self._ledger),
                "request_budget_24h": self.daily_requests,
                "paused": paused,
                "pause_reason": self._pause_reason if paused else "",
                "resumes_in_seconds": round(self._paused_until - now) if paused else 0,
            }


@lru_cache(maxsize=1)
def get_budget() -> TokenBudget:
    s = get_settings()
    return TokenBudget(
        daily_tokens=s.llm_daily_token_budget,
        daily_requests=s.llm_daily_request_budget,
        reserve_tokens=s.llm_tokens_per_call_estimate,
        state_path=s.llm_budget_state_path,
        failure_threshold=s.llm_failure_threshold,
        failure_cooldown=s.llm_failure_cooldown_seconds,
    )
