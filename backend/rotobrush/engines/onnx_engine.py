"""GPU engine on ONNX Runtime: DirectML (any DX12 GPU incl. NVIDIA GTX/RTX) or CUDA, with automatic CPU fallback.

Needs no PyTorch and no CUDA toolkit: `onnxruntime-directml` is a ~20 MB wheel and the U²-Net model (Apache-2.0) is
~176 MB. The network does three jobs, all on the GPU when one is available:
  * detection      - salient-subject map -> connected components -> boxes
  * segmentation   - subject mask inside a box / from a click
  * tracking       - optical flow warps the previous mask, then the network re-decides ONLY the uncertain edge band
                     (replaces GrabCut), so masks are re-estimated per frame instead of copied.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

from ..config import models_dir
from . import cvseg
from .base import Detection, EngineError, InteractiveSegmentation, ObjectDetection, SegmentationEngine
from .opencv_engine import CvInteractive, CvTemporal, _iou

log = logging.getLogger("rotobrush.ai")

MODEL_NAME = "u2net.onnx"
GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def model_candidates() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("ROTOBRUSH_MODEL_U2NET")
    if env:
        out.append(Path(env))
    roots = []
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).parent / "models")
    roots += [Path(__file__).resolve().parents[2] / "models", models_dir()]
    return out + [r / MODEL_NAME for r in roots]


def find_model() -> Path | None:
    return next((p for p in model_candidates() if p.exists() and p.stat().st_size > 1_000_000), None)


def onnx_status() -> tuple[bool, str]:
    try:
        import onnxruntime  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"onnxruntime is not installed ({e.__class__.__name__})"
    if find_model() is None:
        return False, "AI model u2net.onnx was not found (run scripts\\setup.ps1, or see README)"
    return True, ""


class OrtRuntime:
    """One ONNX session. Prefers GPU providers; validates them with a warm-up run; falls back to CPU on any failure."""

    def __init__(self, model: Path, force_cpu: bool = False) -> None:
        import onnxruntime as ort

        self.ort, self.model, self.lock = ort, model, threading.Lock()
        self.fell_back, self.error = False, ""
        avail = ort.get_available_providers()
        want = [] if (force_cpu or os.environ.get("ROTOBRUSH_FORCE_CPU")) else [p for p in GPU_PROVIDERS if p in avail]
        self.session = None
        for prov in [*want, None]:
            try:
                self.session = self._make(prov)
                break
            except Exception as e:  # noqa: BLE001 - driver/DLL problems must never prevent the app from working
                self.error = f"{prov}: {e}"
                log.warning("ONNX provider %s unusable (%s); trying next", prov, e)
        if self.session is None:
            raise EngineError("The AI model could not be loaded.", ["See ai.log for details"])
        self.input_name = self.session.get_inputs()[0].name
        log.info("ONNX engine ready: provider=%s model=%s (available: %s)", self.provider, model.name, avail)

    def _make(self, provider: str | None):
        ort = self.ort
        so = ort.SessionOptions()
        if provider == "DmlExecutionProvider":  # required by the DirectML provider
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        provs = [provider, "CPUExecutionProvider"] if provider else ["CPUExecutionProvider"]
        s = ort.InferenceSession(str(self.model), sess_options=so, providers=provs)
        s.run(None, {s.get_inputs()[0].name: np.zeros((1, 3, 320, 320), np.float32)})  # warm-up proves the provider works
        return s

    @property
    def provider(self) -> str:
        return self.session.get_providers()[0]

    @property
    def accelerated(self) -> bool:
        return self.provider != "CPUExecutionProvider"

    def infer(self, x: np.ndarray) -> np.ndarray:
        with self.lock:
            try:
                return self.session.run(None, {self.input_name: x})[0]
            except Exception as e:  # noqa: BLE001
                if not self.accelerated:
                    raise
                log.warning("GPU inference failed (%s); continuing on CPU", e)
                self.session, self.fell_back = self._make(None), True
                return self.session.run(None, {self.input_name: x})[0]

    def saliency(self, bgr: np.ndarray) -> np.ndarray:
        """Subject probability map (float32 0..1) at the input size. All zeros when nothing salient is present."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = cv2.resize(rgb, (320, 320), interpolation=cv2.INTER_AREA if max(h, w) > 320 else cv2.INTER_LINEAR).astype(np.float32)
        im = (im / max(float(im.max()), 1e-6) - MEAN) / STD
        pred = self.infer(np.ascontiguousarray(im.transpose(2, 0, 1)[None]))[0, 0]
        lo, hi = float(pred.min()), float(pred.max())
        if hi < 0.35:
            return np.zeros((h, w), np.float32)
        return cv2.resize((pred - lo) / (hi - lo + 1e-8), (w, h), interpolation=cv2.INTER_LINEAR)


class OnnxDetection(ObjectDetection):
    def __init__(self, rt: OrtRuntime) -> None:
        self.rt = rt

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        prob = self.rt.saliency(bgr)
        m = cv2.morphologyEx((prob > 0.5).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        area = float(m.size)
        dets: list[Detection] = []
        for i in sorted(range(1, n), key=lambda k: -stats[k, cv2.CC_STAT_AREA])[:4]:
            if stats[i, cv2.CC_STAT_AREA] < 0.01 * area:
                continue
            x, y, w, h = (int(v) for v in stats[i, :4])
            dets.append(Detection(f"d{len(dets)}", "Subject", round(float(prob[lab == i].mean()), 3), (x, y, w, h),
                                  (lab == i).astype(np.uint8) * 255))
        if dets:
            people = cvseg.hog_people(bgr)
            for d in dets:
                if any(_iou(d.bbox, pb) > 0.3 for pb, _ in people):
                    d.label = "Person"
        return dets


class OnnxInteractive(InteractiveSegmentation):
    def __init__(self, rt: OrtRuntime) -> None:
        self.rt, self._cv = rt, CvInteractive()

    def segment(self, bgr, *, box=None, fg_points=None, bg_points=None) -> np.ndarray:
        h, w = bgr.shape[:2]
        if box is not None:
            x, y, bw, bh = (int(v) for v in box)
            pad = int(0.12 * max(bw, bh)) + 4
            x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)
            prob = self.rt.saliency(bgr[y0:y1, x0:x1])
            roi = np.zeros(prob.shape, bool)
            roi[max(0, y - y0 - 6):y - y0 + bh + 6, max(0, x - x0 - 6):x - x0 + bw + 6] = True
            m = cvseg._clean(((prob > 0.5) & roi).astype(np.uint8) * 255)
            out = np.zeros((h, w), np.uint8)
            out[y0:y1, x0:x1] = m
            return cvseg._soften(out)
        if fg_points:
            prob = self.rt.saliency(bgr)
            n, lab, _, _ = cv2.connectedComponentsWithStats((prob > 0.5).astype(np.uint8))
            keep = {int(lab[int(np.clip(py, 0, h - 1)), int(np.clip(px, 0, w - 1))]) for px, py in fg_points} - {0}
            if not keep:
                return self._cv.segment(bgr, box=None, fg_points=fg_points, bg_points=bg_points)
            out = np.isin(lab, list(keep)).astype(np.uint8) * 255
            for px, py in bg_points or []:
                cv2.circle(out, (int(px), int(py)), max(6, int(0.02 * max(h, w))), 0, -1)
            return cvseg._soften(out)
        return np.zeros((h, w), np.uint8)

    def refine_prior(self, bgr, prior, band) -> np.ndarray:
        """Keep the confident core, remove the confident outside, let the network decide the edge band."""
        h, w = prior.shape
        b = prior > 127
        ys, xs = np.where(b)
        if len(xs) < 20:
            return prior
        pad = max(band * 3, int(0.08 * max(h, w))) + 4
        x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)
        y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
        prob = self.rt.saliency(bgr[y0:y1, x0:x1])
        pb = b[y0:y1, x0:x1].astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
        core, outer = cv2.erode(pb, k) > 0, cv2.dilate(pb, k) > 0
        soft = np.clip((prob - 0.5) * 3.0 + 0.5, 0, 1)
        alpha = np.where(core, 1.0, np.where(outer, soft, 0.0))
        if alpha.sum() < 0.5 * pb.sum():  # the network disagrees with the prior (subject not salient): use GrabCut instead
            return cvseg.grabcut_prior(bgr, prior, band)
        out = np.zeros((h, w), np.uint8)
        out[y0:y1, x0:x1] = (alpha * 255).astype(np.uint8)
        return cvseg._soften(out)


class OnnxEngine(SegmentationEngine):
    name = "onnx"

    def __init__(self) -> None:
        ok, why = onnx_status()
        if not ok:
            raise EngineError(why)
        self.rt = OrtRuntime(find_model())  # type: ignore[arg-type]
        self.detection = OnnxDetection(self.rt)
        self.interactive = OnnxInteractive(self.rt)
        # The network re-decides every tracked frame only when a GPU runs it (~tens of ms). On CPU it costs ~1 s/frame,
        # so tracking uses GrabCut there (detection + initial segmentation still use the network).
        force = bool(os.environ.get("ROTOBRUSH_ONNX_TRACK"))

        def refine(bgr: np.ndarray, prior: np.ndarray, band: int) -> np.ndarray:
            if self.rt.accelerated or force:
                return self.interactive.refine_prior(bgr, prior, band)
            return cvseg.grabcut_prior(bgr, prior, band)

        self.temporal = CvTemporal(refine=refine)

    @property
    def provider(self) -> str:  # type: ignore[override]
        return {"CUDAExecutionProvider": "CUDA", "DmlExecutionProvider": "DirectML"}.get(self.rt.provider, "CPU")

    @property
    def accelerated(self) -> bool:  # type: ignore[override]
        return self.rt.accelerated

    @property
    def description(self) -> str:  # type: ignore[override]
        return f"ONNX Runtime on {self.provider}: U²-Net + optical-flow tracking"
