"""Log setup: logs/app.log, ai.log, ffmpeg.log, export.log (diagnostics live here, not in the UI)."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .config import log_dir

_done = False


def setup_logging() -> None:
    global _done
    if _done:
        return
    _done = True
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    files = {
        "rotobrush": "app.log",
        "rotobrush.ai": "ai.log",
        "rotobrush.ffmpeg": "ffmpeg.log",
        "rotobrush.export": "export.log",
    }
    for name, fname in files.items():
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        h = RotatingFileHandler(log_dir() / fname, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        h.setFormatter(fmt)
        lg.addHandler(h)
    root = logging.getLogger("rotobrush")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)
