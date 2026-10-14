"""Depth map estimation (Depth Anything V2 Small, Apache-2.0) on ONNX Runtime: DirectML / CUDA / CPU.

The network outputs *relative inverse depth* (larger = closer). We normalise it per frame for preview, and with a
smoothed range + optical-flow blending for video export so the result does not pump or flicker.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from . import matte
from .engines.onnx_engine import ModelSpec, OrtRuntime, _IMNET_STD, _M, find_model_file
from .video import MediaError

log = logging.getLogger("rotobrush.ai")
DEPTH_FILE = "depth_anything_v2_vits.onnx"
SPEC = ModelSpec("depth", DEPTH_FILE, 518, _M, _IMNET_STD, "Depth Anything V2 Small")


def depth_model_path() -> Path | None:
    return find_model_file(DEPTH_FILE, "ROTOBRUSH_MODEL_DEPTH")


class DepthEstimator:
    _inst: "DepthEstimator | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        path = depth_model_path()
        if path is None:
            raise MediaError("The depth model is not installed.", ["Run scripts\\setup.ps1", "Or see README (depth_anything_v2_vits.onnx)"],
                             stage="depth-model")
        self.rt = OrtRuntime(path, spec=SPEC)

    @classmethod
    def get(cls) -> "DepthEstimator":
        with cls._lock:
            if cls._inst is None:
                cls._inst = cls()
            return cls._inst

    @classmethod
    def reset(cls) -> None:
        import gc
        with cls._lock:
            cls._inst = None
        gc.collect()

    @property
    def provider(self) -> str:
        return {"CUDAExecutionProvider": "CUDA", "DmlExecutionProvider": "DirectML"}.get(self.rt.provider, "CPU")

    def raw(self, bgr: np.ndarray) -> np.ndarray:
        """Relative inverse depth at the frame size (float32, larger = nearer, arbitrary scale)."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        im = (im - SPEC.mean) / SPEC.std
        out = self.rt.infer(np.ascontiguousarray(im.transpose(2, 0, 1)[None]))
        d = np.asarray(out).reshape(518, 518).astype(np.float32)
        return cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)


def percentile_range(d: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(d[::4, ::4], (2, 98))
    return float(lo), float(max(hi, lo + 1e-6))


def normalise(d: np.ndarray, lo: float, hi: float, invert: bool = False) -> np.ndarray:
    n = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    return 1.0 - n if invert else n


class DepthSmoother:
    """Video-stable depth: smoothed normalisation range + flow-warped blend with the previous frame."""

    def __init__(self, strength: float = 0.5, range_alpha: float = 0.15) -> None:
        self.strength, self.ra = strength, range_alpha
        self.lo = self.hi = None
        self.prev_d = self.prev_bgr = None

    def step(self, raw: np.ndarray, bgr: np.ndarray, invert: bool = False) -> np.ndarray:
        lo, hi = percentile_range(raw)
        if self.lo is None:
            self.lo, self.hi = lo, hi
        else:
            self.lo += self.ra * (lo - self.lo)
            self.hi += self.ra * (hi - self.hi)
        d = normalise(raw, self.lo, self.hi, invert=False)
        if self.prev_d is not None and self.strength > 0:
            d = matte.stabilize(self.prev_d, self.prev_bgr, bgr, d, self.strength)
        self.prev_d, self.prev_bgr = d, bgr
        return 1.0 - d if invert else d


def colorize(d01: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap((np.clip(d01, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def frame_depth(p, i: int) -> np.ndarray:
    """Per-frame normalised depth (float 0..1, 1 = nearest) for the viewport, cached on disk as 16-bit PNG."""
    cache = p.root / "previews" / "depth"
    f = cache / f"{i:06d}.png"
    if f.exists():
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[:2] == (p.proxy[1], p.proxy[0]):
            return img.astype(np.float32) / 65535.0
    bgr = p.frame_bgr(i)
    raw = DepthEstimator.get().raw(bgr)
    lo, hi = percentile_range(raw)
    d = normalise(raw, lo, hi)
    cache.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(f), (d * 65535).astype(np.uint16))
    return d
