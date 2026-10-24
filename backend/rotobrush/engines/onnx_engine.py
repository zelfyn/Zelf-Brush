"""GPU engine on ONNX Runtime: DirectML (any DX12 GPU incl. NVIDIA GTX/RTX) or CUDA, with automatic CPU fallback.

No PyTorch and no CUDA toolkit needed. Three interchangeable subject models (all Apache-2.0):
  general  IS-Net general use, 1024 px - sharpest edges, best default for real footage
  anime    IS-Net anime,       1024 px - trained on illustrations: keeps hair/outfit of characters
  fast     U²-Net,              320 px - ~10x cheaper, rougher edges

The network does three jobs, on the GPU when one is available:
  * detection      - subject map -> connected components -> boxes
  * segmentation   - subject mask inside a box / from a click
  * tracking       - optical flow warps the previous mask, then the network re-decides ONLY the uncertain edge band
"""
from __future__ import annotations

import contextlib
import gc
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import models_dir
from . import cvseg
from .base import Detection, EngineError, InteractiveSegmentation, ObjectDetection, SegmentationEngine
from .opencv_engine import CvInteractive, CvTemporal, _iou

log = logging.getLogger("rotobrush.ai")

GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")
_M = np.array([0.485, 0.456, 0.406], np.float32)
_ONES = np.ones(3, np.float32)
_IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    file: str
    size: int
    mean: np.ndarray
    std: np.ndarray
    label: str


SPECS: dict[str, ModelSpec] = {
    "general": ModelSpec("general", "isnet-general-use.onnx", 1024, _M, _ONES, "General (best edges)"),
    "anime": ModelSpec("anime", "isnet-anime.onnx", 1024, _M, _ONES, "Anime / illustration"),
    "fast": ModelSpec("fast", "u2net.onnx", 320, _M, _IMNET_STD, "Fast (lower quality)"),
}
DEFAULT_ORDER = ("general", "anime", "fast")


def model_dirs() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("ROTOBRUSH_MODEL_DIR")
    if env:
        out.append(Path(env))
    if getattr(sys, "frozen", False):
        out.append(Path(sys.executable).parent / "models")
    out += [Path(__file__).resolve().parents[2] / "models", models_dir()]
    return out


def find_model_file(filename: str, env_var: str | None = None) -> Path | None:
    cands = ([Path(os.environ[env_var])] if env_var and os.environ.get(env_var) else []) + [d / filename for d in model_dirs()]
    return next((p for p in cands if p.exists() and p.stat().st_size > 1_000_000), None)


def find_model(key: str = "fast") -> Path | None:
    return find_model_file(SPECS[key].file, "ROTOBRUSH_MODEL_U2NET" if key == "fast" else None)


def available_models() -> list[str]:
    return [k for k in DEFAULT_ORDER if find_model(k)]


def onnx_status() -> tuple[bool, str]:
    try:
        import onnxruntime  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, f"onnxruntime is not installed ({e.__class__.__name__})"
    if not available_models():
        return False, "No AI model found (isnet-general-use.onnx / isnet-anime.onnx / u2net.onnx). Run scripts\\setup.ps1"
    return True, ""


def spec_for_path(model: Path) -> ModelSpec:
    return next((s for s in SPECS.values() if s.file == model.name), SPECS["fast"])


def decide_model(p_general: np.ndarray, p_anime: np.ndarray) -> str:
    """Heuristic for 'Auto': pick the anime model only when it finds clearly MORE subject (e.g. hair a general model
    drops) without being less certain. Real photos usually give near-identical areas, so general wins."""
    a_g, a_a = float((p_general > 0.5).mean()), float((p_anime > 0.5).mean())

    def unc(p: np.ndarray) -> float:
        return float(((p > 0.1) & (p < 0.9)).mean())

    return "anime" if a_a > 1.2 * max(a_g, 1e-6) and unc(p_anime) <= unc(p_general) + 0.03 else "general"


_prio = threading.local()


@contextlib.contextmanager
def interactive():
    """Mark this thread's AI calls as 'the user is waiting for this'. Background jobs yield to them, so a long tracking
    run can never make a click, a lasso refine or a depth preview wait behind it."""
    prev = getattr(_prio, "on", False)
    _prio.on = True
    try:
        yield
    finally:
        _prio.on = prev


class OrtRuntime:
    """One ONNX session. Prefers GPU providers; validates them with a warm-up run; falls back to CPU on any failure."""

    def __init__(self, model: Path, force_cpu: bool = False, spec: ModelSpec | None = None) -> None:
        import onnxruntime as ort

        self.ort, self.model, self.lock = ort, model, threading.Lock()
        self.spec = spec or spec_for_path(model)
        self.fell_back, self.error = False, ""
        self._waiting, self._wlock = 0, threading.Lock()   # interactive callers currently queued for this session
        avail = ort.get_available_providers()
        want = [] if (force_cpu or os.environ.get("ROTOBRUSH_FORCE_CPU")) else [p for p in GPU_PROVIDERS if p in avail]
        self.session = None
        for prov in [*want, None]:
            try:
                self.session = self._make(prov)
                break
            except Exception as e:  # noqa: BLE001 - driver/DLL/out-of-memory problems must never prevent the app from working
                self.error = f"{prov}: {e}"
                log.warning("ONNX provider %s unusable for %s (%s); trying next", prov, model.name, e)
        if self.session is None:
            raise EngineError("The AI model could not be loaded.", ["See ai.log for details"])
        self.input_name = self.session.get_inputs()[0].name
        log.info("ONNX model %s ready: provider=%s (available: %s)", model.name, self.provider, avail)

    def _warm_shape(self, s) -> tuple[int, ...]:
        shp = s.get_inputs()[0].shape
        return tuple(d if isinstance(d, int) and d > 0 else (3 if i == 1 else self.spec.size) for i, d in enumerate(shp))

    def _make(self, provider: str | None):
        ort = self.ort
        so = ort.SessionOptions()
        if provider == "DmlExecutionProvider":  # required by the DirectML provider
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        provs = [provider, "CPUExecutionProvider"] if provider else ["CPUExecutionProvider"]
        s = ort.InferenceSession(str(self.model), sess_options=so, providers=provs)
        s.run(None, {s.get_inputs()[0].name: np.zeros(self._warm_shape(s), np.float32)})  # warm-up proves the provider works
        return s

    @property
    def provider(self) -> str:
        return self.session.get_providers()[0]

    @property
    def accelerated(self) -> bool:
        return self.provider != "CPUExecutionProvider"

    def infer(self, x: np.ndarray) -> np.ndarray:
        inter = getattr(_prio, "on", False)
        if inter:
            with self._wlock:
                self._waiting += 1
        else:
            while self._waiting > 0:  # background work steps aside for anything the user is waiting on
                time.sleep(0.004)
        try:
            with self.lock:
                try:
                    return self.session.run(None, {self.input_name: x})[0]
                except Exception as e:  # noqa: BLE001
                    if not self.accelerated:
                        raise
                    log.warning("GPU inference failed (%s); continuing on CPU", e)
                    self.session, self.fell_back = self._make(None), True
                    return self.session.run(None, {self.input_name: x})[0]
        finally:
            if inter:
                with self._wlock:
                    self._waiting -= 1

    def saliency(self, bgr: np.ndarray) -> np.ndarray:
        """Subject probability map (float32 0..1) at the input size. All zeros when nothing salient is present."""
        h, w = bgr.shape[:2]
        n = self.spec.size
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = cv2.resize(rgb, (n, n), interpolation=cv2.INTER_AREA if max(h, w) > n else cv2.INTER_LINEAR).astype(np.float32)
        im = (im / max(float(im.max()), 1e-6) - self.spec.mean) / self.spec.std
        pred = self.infer(np.ascontiguousarray(im.transpose(2, 0, 1)[None]))[0, 0]
        lo, hi = float(pred.min()), float(pred.max())
        if hi < 0.35:
            return np.zeros((h, w), np.float32)
        return cv2.resize((pred - lo) / (hi - lo + 1e-8), (w, h), interpolation=cv2.INTER_LINEAR)


class OnnxDetection(ObjectDetection):
    def __init__(self, engine: "OnnxEngine") -> None:
        self.engine = engine

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        prob = self.engine.runtime().saliency(bgr)
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
    def __init__(self, engine: "OnnxEngine") -> None:
        self.engine, self._cv = engine, CvInteractive()

    def segment(self, bgr, *, box=None, fg_points=None, bg_points=None) -> np.ndarray:
        rt = self.engine.runtime()
        h, w = bgr.shape[:2]
        if box is not None:
            x, y, bw, bh = (int(v) for v in box)
            pad = int(0.12 * max(bw, bh)) + 4
            x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)
            prob = rt.saliency(bgr[y0:y1, x0:x1])
            roi = np.zeros(prob.shape, bool)
            roi[max(0, y - y0 - 6):y - y0 + bh + 6, max(0, x - x0 - 6):x - x0 + bw + 6] = True
            m = cvseg._clean(((prob > 0.5) & roi).astype(np.uint8) * 255)
            if int((m > 127).sum()) < 20:  # nothing salient inside the box: let GrabCut take the box literally
                return self._cv.segment(bgr, box=(x, y, bw, bh))
            out = np.zeros((h, w), np.uint8)
            out[y0:y1, x0:x1] = m
            return cvseg._soften(out)
        if fg_points:
            prob = rt.saliency(bgr)
            out = None
            for thr in (0.5, 0.35, 0.2):  # the click may sit on a part the network is less sure about (hair, shadows)
                n, lab, _, _ = cv2.connectedComponentsWithStats((prob > thr).astype(np.uint8))
                keep = {int(lab[int(np.clip(py, 0, h - 1)), int(np.clip(px, 0, w - 1))]) for px, py in fg_points} - {0}
                if keep:
                    out = np.isin(lab, list(keep)).astype(np.uint8) * 255
                    break
            if out is None:
                out = self._cv.segment(bgr, box=None, fg_points=fg_points, bg_points=bg_points)
            clicked = max(float(prob[int(np.clip(py, 0, h - 1)), int(np.clip(px, 0, w - 1))]) for px, py in fg_points)
            if int((out > 127).sum()) < 20 and clicked < 0.08:  # clearly background: say so instead of inventing a region
                return np.zeros((h, w), np.uint8)
            if int((out > 127).sum()) < 20:  # last resort for uncertain spots: a small region the network then refines
                out = np.zeros((h, w), np.uint8)
                for px, py in fg_points:
                    cv2.circle(out, (int(px), int(py)), max(8, int(0.03 * max(h, w))), 255, -1)
                out = self.refine_prior(bgr, out, max(6, int(0.01 * max(h, w))))
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
        prob = self.engine.runtime().saliency(bgr[y0:y1, x0:x1])
        pb = b[y0:y1, x0:x1].astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
        core, outer = cv2.erode(pb, k) > 0, cv2.dilate(pb, k) > 0
        alpha = np.where(core, 1.0, np.where(outer, (prob > 0.5).astype(np.float32), 0.0))  # hard decision, no soft band
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
        self._rts: dict[str, OrtRuntime] = {}
        self._lock = threading.Lock()
        self._tl = threading.local()
        self.detection = OnnxDetection(self)
        self.interactive = OnnxInteractive(self)
        force = bool(os.environ.get("ROTOBRUSH_ONNX_TRACK"))

        def refine(bgr: np.ndarray, prior: np.ndarray, band: int) -> np.ndarray:
            # The network re-decides tracked frames only when a GPU runs it (~tens of ms); on CPU it costs seconds per
            # frame, so tracking uses GrabCut there (detection + initial segmentation still use the network).
            if self.runtime().accelerated or force:
                return self.interactive.refine_prior(bgr, prior, band)
            return cvseg.grabcut_prior(bgr, prior, band)

        self.temporal = CvTemporal(refine=refine)
        if not os.environ.get("ROTOBRUSH_NO_WARMUP"):
            threading.Thread(target=self._warm, daemon=True, name="onnx-warmup").start()

    def _warm(self) -> None:
        try:
            self.runtime()
        except Exception:  # noqa: BLE001
            log.exception("model warm-up failed")

    # ---- model selection (per thread, so two layers can use different models at the same time)
    @contextlib.contextmanager
    def model_context(self, choice: str = "auto", resolved: str | None = None):
        prev = getattr(self._tl, "model", None)
        self._tl.model = choice if choice in SPECS else (resolved if resolved in SPECS else None)
        try:
            yield self
        finally:
            self._tl.model = prev

    def runtime(self, key: str | None = None) -> OrtRuntime:
        key = key or getattr(self._tl, "model", None) or DEFAULT_ORDER[0]
        for k in [key] + [k for k in DEFAULT_ORDER if k != key]:
            path = find_model(k)
            if path is None:
                continue
            with self._lock:
                if k not in self._rts:
                    self._rts[k] = OrtRuntime(path, spec=SPECS[k])
                return self._rts[k]
        raise EngineError("No AI model is installed.", ["Run scripts\\setup.ps1 or see README"])

    def choose_model(self, frame_bgr: np.ndarray) -> str:
        """'Auto': compare General and Anime on one frame. Needs both models; otherwise whatever exists."""
        have = available_models()
        if "general" not in have or "anime" not in have:
            return have[0]
        pick = decide_model(self.runtime("general").saliency(frame_bgr), self.runtime("anime").saliency(frame_bgr))
        with self._lock:  # free the model that was not chosen (each session holds ~0.7-1.5 GB)
            self._rts.pop("anime" if pick == "general" else "general", None)
        gc.collect()
        try:  # on a small GPU the winner may have fallen back to CPU while both models were loaded: give it the GPU back
            import onnxruntime as ort
            gpu_possible = any(p in ort.get_available_providers() for p in GPU_PROVIDERS) and not os.environ.get("ROTOBRUSH_FORCE_CPU")
        except Exception:  # noqa: BLE001
            gpu_possible = False
        rt = self._rts.get(pick)
        if gpu_possible and rt is not None and not rt.accelerated:
            log.info("re-creating %s now that the other model is released", pick)
            with self._lock:
                self._rts.pop(pick, None)
            del rt
            gc.collect()
            self.runtime(pick)
        log.info("Auto model choice: %s", pick)
        return pick

    @property
    def _primary(self) -> OrtRuntime | None:
        return next(iter(self._rts.values()), None)

    @property
    def provider(self) -> str:  # type: ignore[override]
        rt = self._primary
        if rt is not None:
            return {"CUDAExecutionProvider": "CUDA", "DmlExecutionProvider": "DirectML"}.get(rt.provider, "CPU")
        try:
            import onnxruntime as ort
            av = ort.get_available_providers()
            return "CUDA" if "CUDAExecutionProvider" in av else "DirectML" if "DmlExecutionProvider" in av else "CPU"
        except Exception:  # noqa: BLE001
            return "CPU"

    @property
    def accelerated(self) -> bool:  # type: ignore[override]
        rt = self._primary
        return bool(rt and rt.accelerated)

    @property
    def description(self) -> str:  # type: ignore[override]
        loaded = ", ".join(self._rts) or "loading"
        return f"ONNX Runtime on {self.provider}: IS-Net/U²-Net ({loaded}) + optical-flow tracking"
