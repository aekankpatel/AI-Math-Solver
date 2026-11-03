"""Per-client sliding-window rate limits (in memory, single process).

Two independent limits protect a public deployment:

* request limits on the solve/follow-up endpoints (per minute and per day), and
* a per-client share of the Groq budget (LLM calls per day), so a single visitor
  cannot spend the whole deployment's free quota. Beyond that share the client is
  served by the offline planners instead of being refused.

Client identity is the socket peer address. Behind a reverse proxy set
``TRUSTED_PROXY_HOPS`` to the number of proxies that append to ``X-Forwarded-For``;
the address is then read that many entries from the right, so a client cannot
spoof it by sending its own header.
"""

from __future__ import annotations

import ipaddress
import threading
import time
from collections import OrderedDict, deque
from typing import Any


class SlidingWindowLimiter:
    """Allow at most ``limit`` events per ``window`` seconds for each key."""

    def __init__(self, limit: int, window: float, max_keys: int = 20_000, clock: Any = time.monotonic) -> None:
        self.limit = limit
        self.window = window
        self._max_keys = max_keys
        self._clock = clock
        self._events: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def hit(self, key: str) -> tuple[bool, float]:
        """Record an event; returns (allowed, retry_after_seconds). Refused events are not recorded."""
        if self.limit <= 0:
            return True, 0.0
        with self._lock:
            now = self._clock()
            q = self._events.pop(key, None) or deque()
            while q and now - q[0] >= self.window:
                q.popleft()
            self._events[key] = q
            while len(self._events) > self._max_keys:
                self._events.popitem(last=False)
            if len(q) >= self.limit:
                return False, max(self.window - (now - q[0]), 1.0)
            q.append(now)
            return True, 0.0

    def remaining(self, key: str) -> int:
        if self.limit <= 0:
            return 1 << 30
        with self._lock:
            now = self._clock()
            q = self._events.get(key) or deque()
            return max(self.limit - sum(1 for t in q if now - t < self.window), 0)


def client_key(peer: str | None, forwarded_for: str | None, trusted_hops: int) -> str:
    """Stable client identifier (normalised IP; IPv6 grouped by /64)."""
    ip = peer or "unknown"
    if trusted_hops > 0 and forwarded_for:
        parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
        if parts:
            ip = parts[-trusted_hops] if len(parts) >= trusted_hops else parts[0]
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return "invalid"
    if addr.version == 6:
        return str(ipaddress.ip_network(f"{addr}/64", strict=False))
    return str(addr)


class ClientLimits:
    def __init__(self, per_minute: int, per_day: int, llm_per_day: int) -> None:
        self.minute = SlidingWindowLimiter(per_minute, 60)
        self.day = SlidingWindowLimiter(per_day, 24 * 3600)
        self.llm_day = SlidingWindowLimiter(llm_per_day, 24 * 3600)

    def check_request(self, key: str) -> tuple[bool, float]:
        ok, retry = self.minute.hit(key)
        if not ok:
            return False, retry
        return self.day.hit(key)

    def take_llm_call(self, key: str | None) -> bool:
        return True if key is None else self.llm_day.hit(key)[0]
