"""Paths and global settings."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

APP_NAME = "RotoBrush"
VERSION = "0.1.0"

PROXY_MAX_SIDE = int(os.environ.get("ROTOBRUSH_PROXY_SIDE", "960"))
LOW_CONFIDENCE = 0.55
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def app_data_dir() -> Path:
    base = os.environ.get("ROTOBRUSH_DATA") or os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path.home() / ".local" / "share"
    p = root / APP_NAME if not os.environ.get("ROTOBRUSH_DATA") else root
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_dir() -> Path:
    p = app_data_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = app_data_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def workspace_dir() -> Path:
    env = os.environ.get("ROTOBRUSH_WORKSPACE")
    p = Path(env) if env else Path.home() / "RotoBrushProjects"
    p.mkdir(parents=True, exist_ok=True)
    return p


def bundled_bin(name: str) -> str | None:
    """Look for ffmpeg/ffprobe shipped next to the backend (backend/bin)."""
    exe = name + (".exe" if os.name == "nt" else "")
    roots = [Path(__file__).resolve().parent.parent / "bin"]
    if getattr(sys, "frozen", False):
        roots.insert(0, Path(sys.executable).parent / "bin")
    for r in roots:
        cand = r / exe
        if cand.exists():
            return str(cand)
    return None


def find_binary(name: str) -> str | None:
    env = os.environ.get(f"ROTOBRUSH_{name.upper()}")
    if env and Path(env).exists():
        return env
    return bundled_bin(name) or shutil.which(name)


def binary_candidates(name: str) -> list[str]:
    """Every place a tool might live, in priority order. Callers must still validate that each one runs."""
    out: list[str] = []
    env = os.environ.get(f"ROTOBRUSH_{name.upper()}")
    if env:
        out.append(env)
    b = bundled_bin(name)
    if b:
        out.append(b)
    if not os.environ.get("ROTOBRUSH_NO_PATH_FALLBACK"):
        w = shutil.which(name)
        if w:
            out.append(w)
    seen: set[str] = set()
    return [c for c in out if not (c in seen or seen.add(c))]
