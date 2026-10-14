"""Matte refinement: shift, smooth, edge refine (guided filter), feather, decontamination, temporal stabilisation.

All operations are non-destructive: they run on top of the stored raw alpha at render/export time.
"""
from __future__ import annotations

import threading
from typing import Any

import cv2
import numpy as np

DEFAULT_MATTE: dict[str, Any] = {
    "edgeRefine": 0.6,    # 0..1  guided-filter strength (snaps edges to image detail)
    "detail": 0.5,        # 0..1  higher keeps thin structures (hair) at the cost of noise
    "smooth": 0.25,       # 0..1  removes jagged contours
    "feather": 0.0,       # px    softens the edge
    "shift": 0.0,         # px    +expand / -contract
    "decontaminate": True,
    "temporal": 0.4,      # 0..1  temporal stabilisation strength
}


def merge_settings(s: dict | None) -> dict:
    out = dict(DEFAULT_MATTE)
    if s:
        out.update({k: s[k] for k in DEFAULT_MATTE if k in s})
    return out


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    k = (2 * r + 1, 2 * r + 1)

    def mean(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, -1, k, normalize=True, borderType=cv2.BORDER_REPLICATE)

    mi, mp = mean(guide), mean(src)
    cov = mean(guide * src) - mi * mp
    var = mean(guide * guide) - mi * mi
    a = cov / (var + eps)
    b = mp - a * mi
    return mean(a) * guide + mean(b)


def refine_alpha(alpha_u8: np.ndarray, frame_bgr: np.ndarray, s: dict, scale: float = 1.0) -> np.ndarray:
    """Returns float32 alpha in 0..1. `scale` converts px settings from proxy to the working resolution."""
    s = merge_settings(s)
    a = alpha_u8.astype(np.float32) / 255.0
    if float(a.max()) <= 0.0:
        return a
    shift = float(s["shift"]) * scale
    if abs(shift) >= 0.5:
        r = int(round(abs(shift)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        a = cv2.dilate(a, k) if shift > 0 else cv2.erode(a, k)
    smooth = float(s["smooth"])
    if smooth > 0:
        b = cv2.GaussianBlur(a, (0, 0), max(0.6, smooth * 4.0 * scale))
        a = np.clip((b - 0.5) * (1.0 + smooth * 6.0) + 0.5, 0, 1)
    er = float(s["edgeRefine"])
    if er > 0:
        guide = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        r = max(2, int(round((2 + er * 10) * scale)))
        eps = 10.0 ** (-2.0 - float(s["detail"]) * 3.0)
        gf = np.clip(guided_filter(guide, a, r, eps), 0, 1)
        hard = (a > 0.5).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        band = (cv2.dilate(hard, k) - cv2.erode(hard, k)) > 0
        a = np.where(band, a * (1 - er) + gf * er, a)
    f = float(s["feather"]) * scale
    if f > 0.2:
        a = cv2.GaussianBlur(a, (0, 0), f)
    return np.clip(a, 0, 1).astype(np.float32)


def _blur(x: np.ndarray, r: int) -> np.ndarray:
    y = cv2.blur(x, (r, r))
    return y[..., None] if (x.ndim == 3 and y.ndim == 2) else y


def _fb_step(img: np.ndarray, F: np.ndarray, B: np.ndarray, alpha: np.ndarray, r: int):
    a3 = alpha[..., None]
    ba = _blur(alpha, r)[..., None] if alpha.ndim == 2 else _blur(alpha, r)
    bf = _blur(F * a3, r) / (ba + 1e-5)
    bb = _blur(B * (1 - a3), r) / ((1 - ba) + 1e-5)
    F2 = np.clip(bf + a3 * (img - a3 * bf - (1 - a3) * bb), 0, 1)
    return F2, bb


def decontaminate(frame_bgr: np.ndarray, alpha: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Blur-fusion foreground estimation: removes background colour spill from edge pixels. Returns float BGR 0..1."""
    img = frame_bgr.astype(np.float32) / 255.0
    F, bb = _fb_step(img, img, img, alpha, max(3, int(45 * scale)))
    F, _ = _fb_step(img, F, bb, alpha, max(2, int(6 * scale)))
    return F


def composite(frame_bgr: np.ndarray, alpha: np.ndarray, bg_bgr: tuple[int, int, int], decon: bool,
              scale: float = 1.0) -> np.ndarray:
    fg = decontaminate(frame_bgr, alpha, scale) if decon else frame_bgr.astype(np.float32) / 255.0
    bg = np.array(bg_bgr, np.float32).reshape(1, 1, 3) / 255.0
    a = alpha[..., None]
    return np.clip((fg * a + bg * (1 - a)) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def foreground_bgra(frame_bgr: np.ndarray, alpha: np.ndarray, decon: bool, scale: float = 1.0) -> np.ndarray:
    fg = decontaminate(frame_bgr, alpha, scale) if decon else frame_bgr.astype(np.float32) / 255.0
    out = np.empty(frame_bgr.shape[:2] + (4,), np.uint8)
    out[..., :3] = np.clip(fg * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(alpha * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


# ---------- optical-flow helpers (shared with temporal tracking) ----------
_tl = threading.local()


def _dis() -> Any:
    d = getattr(_tl, "dis", None)
    if d is None:
        d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        _tl.dis = d
    return d


def small_gray(bgr: np.ndarray, max_side: int = 480) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = min(1.0, max_side / max(g.shape[:2]))
    if s < 1.0:
        g = cv2.resize(g, (max(2, int(g.shape[1] * s)), max(2, int(g.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return g


def flow_between(g_from: np.ndarray, g_to: np.ndarray) -> np.ndarray:
    return _dis().calc(g_from, g_to, None)


def upscale_flow(flow: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    w, h = size_wh
    fx, fy = w / flow.shape[1], h / flow.shape[0]
    up = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    up[..., 0] *= fx
    up[..., 1] *= fy
    return up


def warp_backward(img: np.ndarray, flow_cur_to_prev: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, xs + flow_cur_to_prev[..., 0], ys + flow_cur_to_prev[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def stabilize(prev_alpha: np.ndarray, prev_bgr: np.ndarray, cur_bgr: np.ndarray, alpha: np.ndarray,
              strength: float) -> np.ndarray:
    """Blend the flow-warped previous alpha into the current one to suppress flicker.

    Where the two disagree strongly the change is treated as real motion and the current alpha wins.
    """
    if strength <= 0:
        return alpha
    gp, gc = small_gray(prev_bgr), small_gray(cur_bgr)
    flow = upscale_flow(flow_between(gc, gp), (alpha.shape[1], alpha.shape[0]))
    warped = warp_backward(prev_alpha.astype(np.float32), flow)
    w = strength * 0.6 * (1.0 - np.clip(np.abs(warped - alpha) * 2.5, 0, 1))
    return (alpha * (1 - w) + warped * w).astype(np.float32)
