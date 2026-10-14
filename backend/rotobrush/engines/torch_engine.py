"""Optional deep-learning engine (PyTorch). NVIDIA CUDA when available, otherwise CPU.

Detection + instance masks : torchvision Mask R-CNN (BSD-3)             - weights auto-download on first use
Interactive segmentation   : SAM 2 image predictor (Apache-2.0)         - needs `sam2` package + checkpoint
Video propagation          : SAM 2 video predictor  (Apache-2.0)         - chunked so long clips stay bounded

If SAM 2 is not installed the engine falls back per-module to Mask R-CNN / the OpenCV tracker.
CUDA out-of-memory is handled by cache-clear -> smaller inference size -> CPU fallback.

NOTE: this module has not been executed in the authoring environment (no PyTorch/GPU there); see README.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..config import models_dir
from .base import Detection, EngineError, GetFrame, InteractiveSegmentation, ObjectDetection, SegmentationEngine, TemporalPropagation
from .opencv_engine import CvInteractive, CvTemporal

log = logging.getLogger("rotobrush.ai")

COCO = ["__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]


def torch_available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"PyTorch is not installed ({e.__class__.__name__}). Run scripts/setup.ps1 -Gpu"


def sam2_paths() -> tuple[str, str] | None:
    ckpt = os.environ.get("SAM2_CHECKPOINT") or str(models_dir() / "sam2.1_hiera_small.pt")
    cfg = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_s.yaml")
    try:
        import sam2  # noqa: F401
    except Exception:  # noqa: BLE001
        return None
    return (cfg, ckpt) if Path(ckpt).exists() else None


class _Runtime:
    """Device selection, VRAM-aware inference size and OOM recovery."""

    def __init__(self) -> None:
        import torch
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_side = self._pick_side()

    def _pick_side(self) -> int:
        if self.device != "cuda":
            return 640
        free, _ = self.torch.cuda.mem_get_info()
        gb = free / 1024 ** 3
        return 1280 if gb > 8 else 960 if gb > 4 else 720 if gb > 2 else 512

    def run(self, fn: Callable[[int], np.ndarray | list]) -> np.ndarray | list:
        side = self.max_side
        while True:
            try:
                return fn(side)
            except self.torch.cuda.OutOfMemoryError:
                self.torch.cuda.empty_cache()
                if side > 384:
                    side = int(side * 0.75)
                    self.max_side = side
                    log.warning("CUDA OOM: retrying at %d px", side)
                else:
                    log.warning("CUDA OOM at minimum size: falling back to CPU")
                    self.device, self.max_side = "cpu", 512
                    raise EngineError("GPU memory ran out. Switched to CPU mode for this session.",
                                      ["Close other GPU-heavy apps", "Lower the preview quality"])


class TorchDetection(ObjectDetection):
    def __init__(self, rt: _Runtime) -> None:
        self.rt, self._model = rt, None

    def _load(self):
        if self._model is None:
            import torchvision
            m = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
            self._model = m.eval().to(self.rt.device)
        return self._model

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        torch = self.rt.torch

        def go(side: int) -> list[Detection]:
            model = self._load()
            h, w = bgr.shape[:2]
            s = min(1.0, side / max(h, w))
            small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
            t = torch.from_numpy(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255).to(self.rt.device)
            with torch.no_grad():
                out = model([t])[0]
            dets: list[Detection] = []
            for box, lab, score, m in zip(out["boxes"], out["labels"], out["scores"], out["masks"]):
                if float(score) < 0.5:
                    continue
                x0, y0, x1, y1 = [float(v) / s for v in box]
                mask = (m[0].cpu().numpy() > 0.5).astype(np.uint8) * 255
                if s < 1:
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
                name = COCO[int(lab)] if int(lab) < len(COCO) else f"class {int(lab)}"
                dets.append(Detection(f"d{len(dets)}", name.capitalize(), round(float(score), 3),
                                      (int(x0), int(y0), int(x1 - x0), int(y1 - y0)), mask))
            return dets

        return self.rt.run(go)  # type: ignore[return-value]


class TorchInteractive(InteractiveSegmentation):
    def __init__(self, rt: _Runtime, det: TorchDetection, sam: tuple[str, str] | None) -> None:
        self.rt, self.det, self.sam, self._pred = rt, det, sam, None
        self._fallback = CvInteractive()

    def _predictor(self):
        if self._pred is None and self.sam:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            cfg, ckpt = self.sam
            self._pred = SAM2ImagePredictor(build_sam2(cfg, ckpt, device=self.rt.device))
        return self._pred

    def segment(self, bgr, *, box=None, fg_points=None, bg_points=None) -> np.ndarray:
        pred = self._predictor()
        if pred is None:
            return self._fallback.segment(bgr, box=box, fg_points=fg_points, bg_points=bg_points)
        torch = self.rt.torch
        pts = list(fg_points or []) + list(bg_points or [])
        labels = [1] * len(fg_points or []) + [0] * len(bg_points or [])

        def go(side: int) -> np.ndarray:
            with torch.inference_mode(), torch.autocast(self.rt.device, dtype=torch.bfloat16, enabled=self.rt.device == "cuda"):
                pred.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                masks, scores, _ = pred.predict(
                    point_coords=np.array(pts, np.float32) if pts else None,
                    point_labels=np.array(labels, np.int32) if pts else None,
                    box=np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], np.float32) if box else None,
                    multimask_output=False)
            return (masks[0] > 0).astype(np.uint8) * 255

        return self.rt.run(go)  # type: ignore[return-value]

    def refine_prior(self, bgr, prior, band) -> np.ndarray:
        ys, xs = np.where(prior > 127)
        if len(xs) < 20 or self._predictor() is None:
            return self._fallback.refine_prior(bgr, prior, band)
        box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))
        core = cv2.erode((prior > 127).astype(np.uint8), np.ones((2 * band + 1, 2 * band + 1), np.uint8))
        cy, cx = np.where(core > 0)
        pts = [(float(cx[i]), float(cy[i])) for i in np.linspace(0, len(cx) - 1, min(5, len(cx))).astype(int)] if len(cx) else None
        return self.segment(bgr, box=box, fg_points=pts)


class TorchTemporal(TemporalPropagation):
    """SAM 2 video predictor over bounded chunks; the last mask of a chunk seeds the next one."""
    CHUNK = 200

    def __init__(self, rt: _Runtime, sam: tuple[str, str] | None) -> None:
        self.rt, self.sam, self._pred = rt, sam, None
        self._fallback = CvTemporal()

    def _predictor(self):
        if self._pred is None and self.sam:
            from sam2.build_sam import build_sam2_video_predictor
            cfg, ckpt = self.sam
            self._pred = build_sam2_video_predictor(cfg, ckpt, device=self.rt.device)
        return self._pred

    def propagate(self, get_frame, start, mask, step, last, stop_at, cancelled, on_frame) -> None:
        pred = self._predictor()
        if pred is None:
            return self._fallback.propagate(get_frame, start, mask, step, last, stop_at, cancelled, on_frame)
        torch = self.rt.torch
        end = (stop_at - step) if stop_at is not None else (last if step > 0 else 0)
        anchor, anchor_mask = start, mask
        while not cancelled() and (anchor < end if step > 0 else anchor > end):
            far = min(anchor + self.CHUNK, end) if step > 0 else max(anchor - self.CHUNK, end)
            lo, hi = min(anchor, far), max(anchor, far)
            tmp = Path(tempfile.mkdtemp(prefix="roto_sam2_"))
            try:
                for n, i in enumerate(range(lo, hi + 1)):
                    cv2.imwrite(str(tmp / f"{n:05d}.jpg"), get_frame(i))
                rel = anchor - lo
                with torch.inference_mode(), torch.autocast(self.rt.device, dtype=torch.bfloat16, enabled=self.rt.device == "cuda"):
                    state = pred.init_state(video_path=str(tmp), offload_video_to_cpu=True, offload_state_to_cpu=True)
                    pred.add_new_mask(state, frame_idx=rel, obj_id=1, mask=torch.from_numpy(anchor_mask > 127))
                    for f, _ids, logits in pred.propagate_in_video(state, start_frame_idx=rel, reverse=step < 0):
                        if cancelled():
                            break
                        if f == rel:
                            continue
                        lg = logits[0, 0].float().cpu().numpy()
                        m = (lg > 0).astype(np.uint8) * 255
                        edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((7, 7), np.uint8)) > 0
                        conf = float(np.mean(np.abs(lg)[edge] > 1.5)) if edge.any() else (1.0 if m.any() else 0.0)
                        on_frame(lo + f, m, conf)
                        anchor_mask, last_f = m, lo + f
                    pred.reset_state(state)
                if cancelled():
                    return
                anchor = far
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
                if self.rt.device == "cuda":
                    torch.cuda.empty_cache()


class TorchEngine(SegmentationEngine):
    name = "torch"

    def __init__(self) -> None:
        ok, why = torch_available()
        if not ok:
            raise EngineError(why)
        self.rt = _Runtime()
        sam = sam2_paths()
        self.detection = TorchDetection(self.rt)
        self.interactive = TorchInteractive(self.rt, self.detection, sam)
        self.temporal = TorchTemporal(self.rt, sam)
        self.provider = "CUDA" if self.rt.device == "cuda" else "CPU"
        self.accelerated = self.rt.device == "cuda"
        self.description = (f"PyTorch on {self.rt.device.upper()}: Mask R-CNN detection"
                            + (" + SAM 2 segmentation/tracking" if sam else " (SAM 2 not installed: using GrabCut/flow for masks)"))
