"""Process isolation: runaway computations are stopped and the server keeps working."""

import os
import time

import pytest

from app.math_engine.parser import ExpressionError, parse_expression
from app.sandbox import Sandbox, SandboxBusy, SandboxCrashed, SandboxTimeout

pytestmark = pytest.mark.skipif(os.name != "posix", reason="worker limits need POSIX")


@pytest.fixture(scope="module")
def box():
    sb = Sandbox(workers=1, timeout=2.0, memory_mb=1024, queue_timeout=1.0)
    yield sb
    sb.close()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie still answers kill(0); check its state
    with open(f"/proc/{pid}/stat") as fh:
        return fh.read().split()[2] != "Z"


def test_solves_in_worker(box):
    out = box.run("solve", {"problem": "Integrate x^2 from 0 to 3", "context": None, "llm_result": {"call": None}})
    assert out["response"].status == "solved" and out["response"].answer_text == "9"


def test_timeout_kills_worker_and_recovers(box):
    pid = box._idle.queue[0].proc.pid
    t = time.monotonic()
    with pytest.raises(SandboxTimeout):
        box.run("sleep", {"seconds": 30})
    assert time.monotonic() - t < 4
    time.sleep(0.2)
    assert not os.path.exists(f"/proc/{pid}") or not _alive(pid)  # terminated completely
    assert box.run("sleep", {"seconds": 0}) == "slept"  # replacement worker serves the next job


def test_memory_limit(box):
    with pytest.raises(SandboxCrashed):
        box.run("allocate", {"megabytes": 4096})
    assert box.run("allocate", {"megabytes": 50}) == 50 * 1024 * 1024


def test_runaway_sympy_is_stopped(box):
    # bypasses the static guards on purpose: the sandbox itself must stop it
    from app.schemas import ToolCall

    call = ToolCall(tool="expand_expression", category="algebra", args={"expression": "(x+y+z+1)^900"})
    t = time.monotonic()
    with pytest.raises(SandboxTimeout):
        box.run("solve", {"problem": "expand (x+y+z+1)^900", "context": None, "llm_result": {"call": call}})
    assert time.monotonic() - t < 4


def test_busy_when_all_workers_are_taken():
    import threading

    sb = Sandbox(workers=1, timeout=3.0, queue_timeout=0.2)
    try:
        th = threading.Thread(target=sb.run, args=("sleep", {"seconds": 1}))
        th.start()
        time.sleep(0.2)
        with pytest.raises(SandboxBusy):
            sb.run("sleep", {"seconds": 0})
        th.join()
    finally:
        sb.close()


def test_worker_has_no_secrets(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_testsecretvalue123")
    sb = Sandbox(workers=1, timeout=5.0)
    try:
        out = sb.run("env", {"name": "GROQ_API_KEY"})
        assert out is None
        assert sb.run("env", {"name": "LLM_ENABLED"}) == "false"
    finally:
        sb.close()


@pytest.mark.parametrize("text", ["9^9^9^9", "10^10^10", "9^(9^9)", "(x+1)^5000", "factorial(99999)", "1" * 80 + "+x"])
def test_static_guards_reject_explosive_inputs(text):
    t = time.monotonic()
    with pytest.raises(ExpressionError):
        parse_expression(text)
    assert time.monotonic() - t < 0.5


@pytest.mark.parametrize("text", ["e^(x^2)", "(x^2)^3", "(2^3)^4", "2^3^2", "x^1000", "factorial(20)"])
def test_static_guards_keep_normal_inputs(text):
    parse_expression(text)
