"""Background job lanes with real progress and cancellation."""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .events import bus

log = logging.getLogger("rotobrush")


class Cancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    lane: str
    kind: str
    title: str
    status: str = "waiting"  # waiting | running | completed | failed | cancelled
    progress: float = 0.0
    message: str = ""
    output: str | None = None
    error: dict | None = None
    result: dict = field(default_factory=dict)
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    group: str | None = None                      # e.g. the layer a background job belongs to
    finished: threading.Event = field(default_factory=threading.Event)
    _last_pub: float = 0.0

    def public(self) -> dict:
        return {
            "id": self.id, "lane": self.lane, "kind": self.kind, "title": self.title,
            "status": self.status, "progress": round(self.progress, 4), "message": self.message,
            "output": self.output, "error": self.error, "result": self.result, "group": self.group,
        }

    def update(self, progress: float | None = None, message: str | None = None, force: bool = False) -> None:
        if self.cancel_flag.is_set():
            raise Cancelled()
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        if message is not None:
            self.message = message
        now = time.time()
        if force or now - self._last_pub > 0.1:
            self._last_pub = now
            self.publish()

    def publish(self) -> None:
        ev = "export_progress" if self.lane == "export" else "processing_progress"
        bus.publish(ev, job=self.public())

    @property
    def cancelled(self) -> bool:
        return self.cancel_flag.is_set()


class JobManager:
    LANES = ("cache", "process", "export", "auto")  # "auto" = background AI updates, never blocks the others

    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self._queues: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        for lane in self.LANES:
            q: queue.Queue = queue.Queue()
            self._queues[lane] = q
            threading.Thread(target=self._worker, args=(lane, q), daemon=True, name=f"job-{lane}").start()

    def submit(self, lane: str, kind: str, title: str, fn: Callable[[Job], Any], group: str | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], lane=lane, kind=kind, title=title, group=group)
        with self._lock:
            self.jobs[job.id] = job
        self._queues[lane].put((job, fn))
        job.publish()
        return job

    def _worker(self, lane: str, q: queue.Queue) -> None:
        while True:
            job, fn = q.get()
            if job.cancelled:
                job.status = "cancelled"
                job.finished.set()
                job.publish()
                continue
            job.status = "running"
            job.publish()
            try:
                res = fn(job)
                if isinstance(res, dict):
                    job.result.update(res)
                job.progress = 1.0
                job.status = "completed"
            except Cancelled:
                job.status = "cancelled"
            except Exception as e:  # noqa: BLE001 - jobs must never kill the worker
                log.exception("job %s (%s) failed", job.id, job.kind)
                msg = getattr(e, "message", None) or str(e)
                job.error = {"message": msg, "hints": list(getattr(e, "hints", []) or []),
                             "stage": getattr(e, "stage", None), "exit_code": getattr(e, "exit_code", None),
                             "detail": getattr(e, "detail", None), "log": getattr(e, "log", None)}
                job.status = "failed"
            job.finished.set()
            job.publish()
            bus.publish("job_finished", job=job.public())

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job:
            return False
        job.cancel_flag.set()
        if job.status == "waiting":
            job.status = "cancelled"
            job.publish()
        return True

    def cancel_group(self, kinds: tuple[str, ...], group: str, wait: float = 1.5) -> int:
        """Cancel this group's background jobs and wait (briefly) until they have really stopped, so they can never
        overwrite an edit the user is about to make. Returns how many were cancelled."""
        hit = [x for x in list(self.jobs.values()) if x.group == group and x.kind in kinds and x.status in ("waiting", "running")]
        for x in hit:
            self.cancel(x.id)
        deadline = time.time() + wait
        for x in hit:
            x.finished.wait(max(0.0, deadline - time.time()))
        return len(hit)

    def list(self) -> list[dict]:
        return [j.public() for j in self.jobs.values()][-50:]

    def running(self, kind: str | None = None) -> list[Job]:
        return [j for j in self.jobs.values() if j.status in ("waiting", "running") and (kind is None or j.kind == kind)]
