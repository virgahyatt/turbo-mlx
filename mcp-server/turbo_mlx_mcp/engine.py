"""Drives `turbo-engine serve` over its JSON-lines protocol (Engine/README.md, "Protocol").

One engine process, one job at a time: jobs wait in a queue here, so a queued job can be cancelled
before the engine ever sees it. The process is restarted if it dies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("turbo-mlx-mcp.engine")

TERMINAL = ("done", "failed", "cancelled")


@dataclass
class Job:
    id: str
    model_id: str
    kind: str
    model: dict[str, Any]
    params: dict[str, Any]
    status: str = "queued"  # queued, running, done, failed, cancelled
    phase: str | None = None
    step: int | None = None
    total: int | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    finished_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    changed: asyncio.Condition = field(default_factory=asyncio.Condition, repr=False)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "job_id": self.id, "kind": self.kind, "model": self.model_id, "status": self.status,
        }
        if self.phase:
            out["phase"] = self.phase
        if self.step is not None and self.total:
            out["progress"] = f"{self.step}/{self.total}"
        if self.started:
            out["elapsed_seconds"] = round((self.finished or time.time()) - self.started, 1)
        if self.error:
            out["error"] = self.error
        return out


class EngineManager:
    def __init__(self, binary: str, max_history: int = 200):
        self.binary = binary
        self.proc: asyncio.subprocess.Process | None = None
        self.info: dict[str, Any] | None = None
        self.jobs: dict[str, Job] = {}
        self._history: deque[str] = deque()
        self._max_history = max_history
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._current: Job | None = None
        self._ready = asyncio.Event()
        self._tasks: list[asyncio.Task] = []
        self.log_tail: deque[str] = deque(maxlen=200)
        self._start_lock = asyncio.Lock()

    # ---- process ----
    def _ensure_worker(self) -> None:
        if not self._tasks:
            self._tasks.append(asyncio.create_task(self._worker()))

    async def ensure_ready(self) -> None:
        """Starts the engine process (and the job worker) if they are not running."""
        self._ensure_worker()
        await self._ensure_process()

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        if self.proc and self.proc.returncode is None:
            try:
                self.proc.stdin.write(b'{"cmd": "shutdown"}\n')
                await self.proc.stdin.drain()
                await asyncio.wait_for(self.proc.wait(), 10)
            except Exception:
                self.proc.kill()

    async def _ensure_process(self) -> None:
        async with self._start_lock:
            await self._start_process()

    async def _start_process(self) -> None:
        if self.proc and self.proc.returncode is None and self._ready.is_set():
            return
        self._ready.clear()
        log.info("starting %s serve", self.binary)
        self.proc = await asyncio.create_subprocess_exec(
            self.binary, "serve",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            limit=16 * 1024 * 1024,
        )
        asyncio.create_task(self._read_stdout(self.proc))
        asyncio.create_task(self._read_stderr(self.proc))
        await asyncio.wait_for(self._ready.wait(), 60)

    async def _send(self, obj: dict[str, Any]) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())
        await self.proc.stdin.drain()

    async def _read_stderr(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stderr
        while line := await proc.stderr.readline():
            text = line.decode(errors="replace").rstrip()
            self.log_tail.append(text)
            log.info("engine: %s", text)

    async def _read_stdout(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout
        while line := await proc.stdout.readline():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            await self._handle(event)
        # The process ended: fail whatever was running.
        self._ready.clear()
        code = await proc.wait()
        log.warning("turbo-engine exited with code %s", code)
        job = self._current
        if job and job.status == "running":
            tail = "\n".join(list(self.log_tail)[-5:])
            await self._finish(job, "failed", error=f"turbo-engine exited (code {code}). Last log lines:\n{tail}")

    async def _handle(self, event: dict[str, Any]) -> None:
        kind = event.get("event")
        if kind == "ready":
            self.info = {k: v for k, v in event.items() if k != "event"}
            self._ready.set()
            return
        job = self.jobs.get(event.get("id") or "") or (self._current if kind == "phase" else None)
        if job is None:
            return
        if kind == "phase":
            job.phase = event.get("phase")
            job.step, job.total = None, event.get("total")
        elif kind == "progress":
            job.step, job.total = event.get("step"), event.get("total")
        elif kind == "done":
            await self._finish(job, "done", result={k: v for k, v in event.items() if k not in ("event", "id")})
            return
        elif kind == "failed":
            await self._finish(job, "failed", error=event.get("message") or "generation failed")
            return
        elif kind == "cancelled":
            await self._finish(job, "cancelled")
            return
        async with job.changed:
            job.changed.notify_all()

    async def _finish(self, job: Job, status: str, *, result: dict | None = None, error: str | None = None) -> None:
        if job.status in TERMINAL:
            return
        job.status, job.result, job.error = status, result, error
        job.finished = time.time()
        job.finished_event.set()
        async with job.changed:
            job.changed.notify_all()

    # ---- jobs ----
    def submit(self, *, model_id: str, kind: str, model: dict[str, Any], params: dict[str, Any], job_id: str | None = None) -> Job:
        self._ensure_worker()
        job = Job(id=job_id or uuid.uuid4().hex, model_id=model_id, kind=kind, model=model, params=params)
        self.jobs[job.id] = job
        self._history.append(job.id)
        while len(self._history) > self._max_history:
            old = self._history.popleft()
            if self.jobs.get(old) and self.jobs[old].status in TERMINAL:
                del self.jobs[old]
        self._queue.put_nowait(job)
        return job

    async def cancel(self, job_id: str) -> Job:
        job = self.jobs[job_id]
        if job.status == "queued":
            await self._finish(job, "cancelled")
        elif job.status == "running":
            await self._send({"cmd": "cancel", "id": job_id})
        return job

    async def unload(self) -> None:
        if self.proc and self.proc.returncode is None and self._current is None:
            await self._send({"cmd": "unload"})

    def queue_position(self, job: Job) -> int | None:
        if job.status != "queued":
            return None
        queued = [j for j in self._queue._queue if j.status == "queued"]  # noqa: SLF001
        return queued.index(job) + 1 if job in queued else None

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            if job.status != "queued":
                continue
            self._current = job
            try:
                await self._ensure_process()
                job.status, job.started = "running", time.time()
                await self._send({"cmd": "generate", "id": job.id, "model": job.model, "params": job.params})
                await job.finished_event.wait()
            except Exception as e:  # noqa: BLE001
                log.exception("job %s", job.id)
                await self._finish(job, "failed", error=f"{type(e).__name__}: {e}")
            finally:
                self._current = None

    async def wait(self, job: Job, timeout: float, on_change=None) -> bool:
        """Waits up to `timeout` seconds for the job to end; calls on_change(job) at each update."""
        deadline = time.monotonic() + timeout
        while job.status not in TERMINAL:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            async with job.changed:
                try:
                    await asyncio.wait_for(job.changed.wait(), min(remaining, 15))
                except asyncio.TimeoutError:
                    pass
            if on_change:
                await on_change(job)
        return True
