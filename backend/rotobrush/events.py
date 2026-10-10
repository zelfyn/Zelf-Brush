"""Thread-safe event bus feeding a Server-Sent-Events stream."""
from __future__ import annotations

import asyncio
import threading
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subs: set[asyncio.Queue] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    @staticmethod
    def _put(q: asyncio.Queue, msg: dict) -> None:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    def publish(self, type_: str, **data: Any) -> None:
        msg = {"type": type_, **data}
        with self._lock:
            subs = list(self._subs)
        if not self._loop:
            return
        for q in subs:
            self._loop.call_soon_threadsafe(self._put, q, msg)


bus = EventBus()
