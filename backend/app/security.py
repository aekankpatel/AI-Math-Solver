"""HTTP hardening: request-size limit, security headers and secret redaction in logs."""

from __future__ import annotations

import logging
import re
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Anything that looks like a credential is masked before a log line is written.
_SECRET_PATTERNS = [
    (re.compile(r"gsk_[A-Za-z0-9]{8,}"), "gsk_[REDACTED]"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1[REDACTED]"),
    (re.compile(r"(?i)((?:api[_-]?key|authorization|x-api-key)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]{6,}"), r"\1[REDACTED]"),
    (re.compile(r"\borg_[A-Za-z0-9]{8,}"), "org_[REDACTED]"),
]


def redact(text: str) -> str:
    for pattern, repl in _SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        clean = redact(message)
        if clean != message:
            record.msg, record.args = clean, None
        if record.exc_info and record.exc_info[1] is not None:
            # exception text is formatted later; strip it into a redacted message instead
            exc = record.exc_info[1]
            record.msg = f"{record.msg} | {type(exc).__name__}: {redact(str(exc))}"
            record.args, record.exc_info, record.exc_text = None, None, None
        return True


_FACTORY_INSTALLED = False


def _install_record_factory() -> None:
    """Redact every log record when it is created, whichever logger or handler it goes to."""
    global _FACTORY_INSTALLED
    if _FACTORY_INSTALLED:
        return
    previous = logging.getLogRecordFactory()
    flt = RedactingFilter()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        flt.filter(record)
        return record

    logging.setLogRecordFactory(factory)
    _FACTORY_INSTALLED = True


def install_log_redaction() -> None:
    """Redact secrets from all log records (record factory) and, as a second layer, on the
    handlers of the root and uvicorn loggers."""
    _install_record_factory()
    flt = RedactingFilter()
    for name in ("", "uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "groq"):
        logger = logging.getLogger(name)
        logger.addFilter(flt)
        for handler in logger.handlers:
            handler.addFilter(flt)


SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
    (b"cross-origin-resource-policy", b"same-site"),
    (b"cache-control", b"no-store"),
]
API_CSP = (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'")


class HardeningMiddleware:
    """Pure ASGI middleware: rejects oversized bodies (413) and adds security headers."""

    def __init__(self, app: ASGIApp, max_body: int, docs_paths: tuple[str, ...] = ()) -> None:
        self.app = app
        self.max_body = max_body
        self.docs_paths = docs_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope.get("path", "")
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                too_big = int(declared) > self.max_body
            except ValueError:
                too_big = True
            if too_big:
                await _reject(send, 413, b'{"detail":"Request body too large."}')
                return

        received = 0
        overflow = False

        async def limited_receive() -> Message:
            nonlocal received, overflow
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body:
                    overflow = True
                    raise _BodyTooLarge
            return message

        is_docs = any(path.startswith(p) for p in self.docs_paths)

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                hdrs = list(message.get("headers") or [])
                present = {k.lower() for k, _ in hdrs}
                extra: list[tuple[bytes, bytes]] = list(SECURITY_HEADERS)
                if not is_docs:
                    extra.append(API_CSP)
                hdrs.extend((k, v) for k, v in extra if k not in present)
                message = {**message, "headers": hdrs}
            await send(message)

        try:
            await self.app(scope, limited_receive, send_with_headers)
        except _BodyTooLarge:
            if overflow:
                await _reject(send, 413, b'{"detail":"Request body too large."}')


class _BodyTooLarge(Exception):
    pass


async def _reject(send: Send, status: int, body: bytes, extra: list[Any] | None = None) -> None:
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    headers.extend(SECURITY_HEADERS)
    headers.extend(extra or [])
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
