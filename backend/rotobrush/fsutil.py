"""Crash-safe file writes that survive concurrent writers and Windows' transient file locks."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

_write_lock = threading.Lock()  # one writer at a time: autosave, tracking jobs and the UI all touch the same files


def unique_tmp(path: Path) -> Path:
    """A temp name no other thread or process will use."""
    return path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def replace(src: Path, dst: Path, attempts: int = 10) -> None:
    """os.replace with retries. On Windows an antivirus scan, the search indexer or a media player can hold the
    destination open for a moment (WinError 32 / PermissionError); waiting a few ms almost always clears it."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.03 * (i + 1))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    with _write_lock:
        tmp = unique_tmp(path)
        try:
            tmp.write_bytes(data)
            replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))
