"""Process isolation for all symbolic/numeric computation.

SymPy can be driven into near-unbounded work by short inputs (``9^9^9^9``,
``expand((x+1)^5000)``) and a Python thread cannot be stopped once it is inside
such a computation. Every computation (parsing, planning, intent checks, tools,
verification, plots) therefore runs in a pool of worker processes:

* each job has a wall-clock timeout; on timeout the worker is killed (SIGKILL,
  together with anything it started) and replaced, so no runaway work survives;
* each worker runs under an address-space limit (``RLIMIT_AS``), so a memory bomb
  fails inside the worker instead of taking the server down; a worker that dies is
  replaced and the request gets a clear error;
* workers are recycled after a number of jobs, which bounds SymPy cache growth;
* workers never see API keys and never call the LLM (network calls stay in the
  API process, where the token budget is enforced);
* the pool size bounds concurrent computations; requests wait briefly for a free
  worker and are refused with "busy" instead of piling up.

Workers start from a fork server that has the math modules preloaded, so a
replacement worker is ready in well under a second. With ``SANDBOX_ENABLED=false``
jobs run inline (tests and debugging only: no isolation).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from multiprocessing.connection import Connection
from typing import Any

log = logging.getLogger(__name__)

# One thread per numeric library: predictable memory and CPU per worker.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

_SECRET_ENV = ("GROQ_API_KEY",)


class SandboxError(Exception):
    """Base class: the computation did not produce a result."""


class SandboxTimeout(SandboxError):
    pass


class SandboxCrashed(SandboxError):
    """The worker died (usually the memory limit) or returned an unusable result."""


class SandboxBusy(SandboxError):
    pass


def _limit_memory(mem_mb: int) -> None:
    if mem_mb <= 0:
        return
    try:
        import resource

        limit = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ImportError, ValueError, OSError) as exc:  # not available on this platform
        log.warning("Could not set the worker memory limit: %s", exc)


def _worker_main(conn: Connection, mem_mb: int) -> None:
    """Worker loop: receive (job, kwargs), send ("ok", result) or ("error", kind, message)."""
    for name in _SECRET_ENV:
        os.environ.pop(name, None)
    os.environ["LLM_ENABLED"] = "false"  # belt and braces: workers never reach the network
    try:
        os.setpgid(0, 0)  # own process group, so a kill also reaches anything the job spawned
    except OSError:
        pass
    from app import jobs
    from app.config import get_settings

    get_settings.cache_clear()
    _limit_memory(mem_mb)
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg is None:
            return
        name, kwargs = msg
        try:
            result = jobs.run_job(name, kwargs)
            conn.send(("ok", result))
        except MemoryError:
            conn.send(("error", "memory", "The computation needed more memory than allowed."))
        except Exception as exc:  # report, never crash silently
            try:
                conn.send(("error", "exception", f"{type(exc).__name__}: {str(exc)[:300]}"))
            except Exception:
                return


class _Worker:
    def __init__(self, ctx: Any, mem_mb: int) -> None:
        self.conn, child = ctx.Pipe(duplex=True)
        self.proc = ctx.Process(target=_worker_main, args=(child, mem_mb), daemon=True)
        self.proc.start()
        child.close()
        self.jobs = 0

    def kill(self) -> None:
        try:
            if self.proc.pid:
                try:
                    os.killpg(self.proc.pid, 9)
                except (ProcessLookupError, PermissionError, OSError):
                    self.proc.kill()
        finally:
            self.proc.join(timeout=5)
            self.conn.close()

    def stop(self) -> None:
        try:
            self.conn.send(None)
        except (OSError, BrokenPipeError):
            pass
        self.proc.join(timeout=2)
        if self.proc.is_alive():
            self.kill()
        else:
            self.conn.close()


class Sandbox:
    def __init__(
        self,
        workers: int = 2,
        timeout: float = 10.0,
        memory_mb: int = 1536,
        queue_timeout: float = 15.0,
        max_jobs_per_worker: int = 200,
    ) -> None:
        self.size = max(workers, 1)
        self.timeout = timeout
        self.memory_mb = memory_mb
        self.queue_timeout = queue_timeout
        self.max_jobs = max_jobs_per_worker
        method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        self._ctx = mp.get_context(method)
        if method == "forkserver":
            os.environ.setdefault("MATHENZA_SANDBOX_WARMUP", "1")  # read by app.jobs in the fork server
            self._ctx.set_forkserver_preload(["app.jobs"])
        self._idle: queue.Queue[_Worker] = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        self.stats = {"jobs": 0, "timeouts": 0, "crashes": 0, "busy": 0, "restarts": 0}
        for _ in range(self.size):
            self._idle.put(self._spawn())

    def _spawn(self) -> _Worker:
        return _Worker(self._ctx, self.memory_mb)

    def _replace(self, worker: _Worker, *, kill: bool) -> None:
        if kill:
            worker.kill()
        else:
            worker.stop()
        with self._lock:
            self.stats["restarts"] += 1
            closed = self._closed
        if not closed:
            self._idle.put(self._spawn())

    def run(self, name: str, kwargs: dict[str, Any], timeout: float | None = None) -> Any:
        limit = self.timeout if timeout is None else timeout
        try:
            worker = self._idle.get(timeout=self.queue_timeout)
        except queue.Empty as exc:
            with self._lock:
                self.stats["busy"] += 1
            raise SandboxBusy("All computation workers are busy.") from exc
        with self._lock:
            self.stats["jobs"] += 1
        if not worker.proc.is_alive():  # died while idle (should not happen); start fresh
            self._replace(worker, kill=True)
            worker = self._idle.get(timeout=self.queue_timeout)
        started = time.monotonic()
        try:
            worker.conn.send((name, kwargs))
            ready = worker.conn.poll(limit)
            if not ready:
                with self._lock:
                    self.stats["timeouts"] += 1
                log.warning("Computation %s exceeded %.1fs; worker killed", name, limit)
                self._replace(worker, kill=True)
                worker = None
                raise SandboxTimeout(f"The computation took longer than {limit:g} seconds and was stopped.")
            try:
                reply = worker.conn.recv()
            except (EOFError, OSError) as exc:
                with self._lock:
                    self.stats["crashes"] += 1
                log.warning("Computation worker died during %s (exit code %s)", name, worker.proc.exitcode)
                self._replace(worker, kill=True)
                worker = None
                raise SandboxCrashed(
                    "The computation stopped unexpectedly (it may have exceeded the memory limit)."
                ) from exc
        except (BrokenPipeError, OSError) as exc:
            if worker is not None:
                self._replace(worker, kill=True)
                worker = None
            raise SandboxCrashed("The computation worker failed.") from exc
        finally:
            if worker is not None:
                worker.jobs += 1
                if worker.jobs >= self.max_jobs:
                    self._replace(worker, kill=False)
                else:
                    self._idle.put(worker)
        log.debug("job %s finished in %.3fs", name, time.monotonic() - started)
        if reply[0] == "ok":
            return reply[1]
        _, kind, message = reply
        if kind == "memory":
            raise SandboxCrashed(message)
        raise SandboxCrashed(f"The computation failed unexpectedly ({message}).")

    def close(self) -> None:
        with self._lock:
            self._closed = True
        while True:
            try:
                self._idle.get_nowait().stop()
            except queue.Empty:
                break


class InlineRunner:
    """No isolation: runs jobs in-process (tests / debugging)."""

    stats: dict[str, int] = {}
    size = 0

    def run(self, name: str, kwargs: dict[str, Any], timeout: float | None = None) -> Any:
        from app import jobs

        return jobs.run_job(name, kwargs)

    def close(self) -> None:
        pass


_runner: Sandbox | InlineRunner | None = None
_runner_lock = threading.Lock()


def get_runner() -> Sandbox | InlineRunner:
    global _runner
    with _runner_lock:
        if _runner is None:
            from app.config import get_settings

            s = get_settings()
            if s.sandbox_enabled:
                _runner = Sandbox(
                    workers=s.sandbox_workers,
                    timeout=s.sandbox_timeout_seconds,
                    memory_mb=s.sandbox_memory_mb,
                    queue_timeout=s.sandbox_queue_timeout_seconds,
                    max_jobs_per_worker=s.sandbox_max_jobs_per_worker,
                )
            else:
                _runner = InlineRunner()
        return _runner


def shutdown_runner() -> None:
    global _runner
    with _runner_lock:
        if _runner is not None:
            _runner.close()
            _runner = None
