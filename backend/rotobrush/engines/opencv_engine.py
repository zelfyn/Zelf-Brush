"""Always-available engine: HOG/saliency detection, GrabCut segmentation, optical-flow + GrabCut tracking.

Needs no model weights, so the app works on any Windows PC. Quality is below deep models (SAM 2 etc.),
which plug in through torch_engine.py.
"""
from __future__ import annotations

import logging
from typing import Callable

import cv2
import numpy as np

from .. import matte
from . import cvseg
from .base import Detection, GetFrame, InteractiveSegmentation, ObjectDetection, SegmentationEngine, TemporalPropagation

log = logging.getLogger("rotobrush.ai")


class CvDetection(ObjectDetection):
    def detect(self, bgr: np.ndarray) -> list[Detection]:
        out: list[Detection] = []
        for box, conf in cvseg.hog_people(bgr):
            out.append(Detection(f"d{len(out)}", "Person", round(conf, 3), box))
        sal = cvseg.salient_box(bgr)
        if sal:
            box, conf = sal
            overlaps = any(_iou(box, d.bbox) > 0.4 for d in out)
            if not overlaps:
                out.append(Detection(f"d{len(out)}", "Salient object", round(conf, 3), box))
        return out


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter + 1e-6)


class CvInteractive(InteractiveSegmentation):
    def segment(self, bgr, *, box=None, fg_points=None, bg_points=None) -> np.ndarray:
        if fg_points:
            return cvseg.grabcut_points(bgr, list(fg_points), list(bg_points or []), box)
        if box:
            return cvseg.grabcut_box(bgr, box)
        return np.zeros(bgr.shape[:2], np.uint8)

    def refine_prior(self, bgr, prior, band) -> np.ndarray:
        return cvseg.grabcut_prior(bgr, prior, band)


def track_step(prev_bgr: np.ndarray, prev_mask: np.ndarray, cur_bgr: np.ndarray,
               refine: Callable[[np.ndarray, np.ndarray, int], np.ndarray] | None = None) -> tuple[np.ndarray, float]:
    """Warp the previous mask with dense optical flow, then re-decide the uncertain band with GrabCut.

    Confidence combines forward/backward flow consistency inside the object with agreement between the
    warped prediction and the re-estimated mask. It is NOT a pixel copy: colour models are re-learned on
    every frame from the confident core and background.
    """
    h, w = prev_mask.shape
    g0, g1 = matte.small_gray(prev_bgr), matte.small_gray(cur_bgr)
    flow_bw_s = matte.flow_between(g1, g0)   # cur -> prev
    flow_fw_s = matte.flow_between(g0, g1)   # prev -> cur
    flow_bw = matte.upscale_flow(flow_bw_s, (w, h))
    flow_fw = matte.upscale_flow(flow_fw_s, (w, h))
    warped = matte.warp_backward(prev_mask.astype(np.float32), flow_bw)
    fw_at_prev = matte.warp_backward(flow_fw, flow_bw)          # forward flow sampled at the source location
    err = np.linalg.norm(flow_bw + fw_at_prev, axis=2)
    inside = warped > 127
    area = int(inside.sum())
    if area < max(30, 0.0003 * h * w):
        return warped.astype(np.uint8), 0.0
    flow_conf = float(np.mean(err[inside] < max(1.5, 0.004 * max(h, w))))
    band = int(np.clip(0.05 * np.sqrt(area), 3, 20))
    refined = (refine or cvseg.grabcut_prior)(cur_bgr, warped.astype(np.uint8), band)
    a, b = warped > 127, refined > 127
    iou = float((a & b).sum() / max(1, (a | b).sum()))
    growth = float(b.sum()) / max(1, area)
    penalty = 0.0 if 0.6 < growth < 1.6 else 0.25
    conf = float(np.clip(0.55 * flow_conf + 0.45 * iou - penalty, 0, 1))
    return refined, conf


class CvTemporal(TemporalPropagation):
    def __init__(self, refine: Callable[[np.ndarray, np.ndarray, int], np.ndarray] | None = None) -> None:
        self.refine = refine  # e.g. a GPU network; default is GrabCut

    def propagate(self, get_frame: GetFrame, start: int, mask: np.ndarray, step: int, last: int,
                  stop_at: int | None, cancelled: Callable[[], bool],
                  on_frame: Callable[[int, np.ndarray, float], None]) -> None:
        prev_idx, prev_mask, prev_bgr = start, mask, get_frame(start)
        idx = start + step
        while (idx <= last if step > 0 else idx >= last) and 0 <= idx and idx != stop_at:  # `last` is the far bound in the travel direction
            if cancelled():
                return
            cur = get_frame(idx)
            new, conf = track_step(prev_bgr, prev_mask, cur, self.refine)
            on_frame(idx, new, conf)
            prev_idx, prev_mask, prev_bgr = idx, new, cur
            idx += step


class OpenCVEngine(SegmentationEngine):
    name = "opencv"
    description = "Classical CV (HOG + GrabCut + optical flow). Works everywhere, no downloads."

    def __init__(self) -> None:
        self.detection = CvDetection()
        self.interactive = CvInteractive()
        self.temporal = CvTemporal()
