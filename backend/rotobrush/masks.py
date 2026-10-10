"""Per-frame alpha mask store with on-disk PNGs and bounded undo/redo history."""
from __future__ import annotations

import copy
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np


class Op:
    def __init__(self, label: str, meta: dict) -> None:
        self.label = label
        self.meta = meta
        self.touched: dict[int, bytes | None] = {}


class MaskStore:
    def __init__(self, directory: Path, size: tuple[int, int],
                 get_meta: Callable[[], dict], set_meta: Callable[[dict], None], max_ops: int = 40) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.w, self.h = size
        self._get_meta, self._set_meta, self.max_ops = get_meta, set_meta, max_ops
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._ver: dict[int, int] = {}
        self._undo: list[Op] = []
        self._redo: list[Op] = []
        self._op: Op | None = None
        self._lock = threading.RLock()
        self._present = {int(p.stem) for p in self.dir.glob("*.png") if p.stem.isdigit()}

    def _path(self, i: int) -> Path:
        return self.dir / f"{i:06d}.png"

    def has(self, i: int) -> bool:
        return i in self._present

    def frames(self) -> list[int]:
        return sorted(self._present)

    def version(self, i: int) -> int:
        return self._ver.get(i, 0)

    def get(self, i: int) -> np.ndarray:
        with self._lock:
            if i in self._cache:
                self._cache.move_to_end(i)
                return self._cache[i].copy()
            if i in self._present:
                m = cv2.imread(str(self._path(i)), cv2.IMREAD_GRAYSCALE)
                if m is not None and m.shape == (self.h, self.w):
                    self._remember(i, m)
                    return m.copy()
            return np.zeros((self.h, self.w), np.uint8)

    def _remember(self, i: int, m: np.ndarray) -> None:
        self._cache[i] = m
        self._cache.move_to_end(i)
        while len(self._cache) > 48:
            self._cache.popitem(last=False)

    def _raw(self, i: int) -> bytes | None:
        p = self._path(i)
        return p.read_bytes() if i in self._present and p.exists() else None

    def _write_bytes(self, i: int, data: bytes | None) -> None:
        p = self._path(i)
        if data is None:
            if p.exists():
                p.unlink()
            self._present.discard(i)
        else:
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, p)
            self._present.add(i)
        self._cache.pop(i, None)
        self._ver[i] = self._ver.get(i, 0) + 1

    def set(self, i: int, arr: np.ndarray) -> None:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        with self._lock:
            if self._op is not None and i not in self._op.touched:
                self._op.touched[i] = self._raw(i)
            ok, buf = cv2.imencode(".png", arr)
            if not ok:
                raise RuntimeError("PNG encode failed")
            self._write_bytes(i, buf.tobytes())
            self._remember(i, arr.copy())

    def delete(self, i: int) -> None:
        with self._lock:
            if self._op is not None and i not in self._op.touched:
                self._op.touched[i] = self._raw(i)
            self._write_bytes(i, None)

    @contextmanager
    def operation(self, label: str) -> Iterator[Op]:
        with self._lock:
            op = Op(label, copy.deepcopy(self._get_meta()))
            self._op = op
            try:
                yield op
            finally:
                self._op = None
                if op.touched:
                    self._undo.append(op)
                    del self._undo[: -self.max_ops]
                    self._redo.clear()

    def _swap(self, src: list[Op], dst: list[Op]) -> tuple[str, list[int]] | None:
        with self._lock:
            if not src:
                return None
            op = src.pop()
            inv = Op(op.label, copy.deepcopy(self._get_meta()))
            for i, before in op.touched.items():
                inv.touched[i] = self._raw(i)
                self._write_bytes(i, before)
            self._set_meta(copy.deepcopy(op.meta))
            dst.append(inv)
            return op.label, sorted(op.touched)

    def undo(self) -> tuple[str, list[int]] | None:
        return self._swap(self._undo, self._redo)

    def redo(self) -> tuple[str, list[int]] | None:
        return self._swap(self._redo, self._undo)

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)
