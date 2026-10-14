"""Classical CV building blocks (OpenCV only, no model weights): GrabCut, HOG people, spectral-residual saliency."""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

_hog = None
_hog_broken = False
_hog_lock = threading.Lock()


def _resize_to(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    s = min(1.0, max_side / max(img.shape[:2]))
    if s < 1.0:
        img = cv2.resize(img, (max(2, int(img.shape[1] * s)), max(2, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return img, s


def _nms(boxes: list[list[int]], scores: list[float], thr: float = 0.4) -> list[int]:
    if not boxes:
        return []
    idx = cv2.dnn.NMSBoxes(boxes, [float(s) for s in scores], 0.0, thr)
    return [int(i) for i in np.array(idx).reshape(-1)]


def hog_people(bgr: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
    """Upright-person detector. Returns [] if this OpenCV build lacks HOG (saliency detection still works)."""
    global _hog, _hog_broken
    if _hog_broken:
        return []
    try:
        with _hog_lock:
            if _hog is None:
                _hog = cv2.HOGDescriptor()
                _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            small, s = _resize_to(bgr, 640)
            rects, weights = _hog.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
    except (AttributeError, cv2.error) as e:
        _hog_broken = True
        logging.getLogger("rotobrush.ai").warning(
            "HOG person detector unavailable in this OpenCV build (%s); using saliency only. cv2=%s",
            e, getattr(cv2, "__version__", "?"))
        return []
    boxes = [[int(x / s), int(y / s), int(w / s), int(h / s)] for x, y, w, h in rects]
    ws = [float(w) for w in np.array(weights).reshape(-1)]
    keep = _nms(boxes, ws)
    return [(tuple(boxes[i]), float(1 - np.exp(-max(ws[i], 0.0)))) for i in keep]  # type: ignore[arg-type]


def saliency_map(bgr: np.ndarray) -> np.ndarray:
    """Frequency-tuned saliency (Achanta et al. 2009): Lab distance to the global mean colour. Float 0..1."""
    small, _ = _resize_to(bgr, 256)
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (0, 0), 2.0), cv2.COLOR_BGR2LAB).astype(np.float32)
    mean = lab.reshape(-1, 3).mean(0)
    s = np.linalg.norm(lab - mean, axis=2)
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    return cv2.resize(s, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)


def salient_box(bgr: np.ndarray) -> tuple[tuple[int, int, int, int], float] | None:
    s = saliency_map(bgr)
    u8 = (s * 255).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (u8 > thr).astype(np.uint8)
    k = max(3, int(0.02 * max(m.shape))) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k * 2 + 1, k * 2 + 1), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = stats[k]
    frac = area / float(m.size)
    if frac < 0.01 or frac > 0.75:
        return None
    contrast = float(s[lab == k].mean() - s[lab != k].mean())
    return (int(x), int(y), int(w), int(h)), float(np.clip(0.35 + contrast, 0.3, 0.75))


def _run_grabcut(img: np.ndarray, mask: np.ndarray, rect, iters: int, mode: int) -> np.ndarray | None:
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, iters, mode)
    except cv2.error:
        return None
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def _crop_bounds(w: int, h: int, x: int, y: int, bw: int, bh: int, pad: int) -> tuple[int, int, int, int]:
    return max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)


def _clean(mask: np.ndarray, keep_points: list[tuple[float, float]] | None = None, min_frac: float = 0.0005) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8))
    out = np.zeros_like(mask)
    if n <= 1:
        return out
    keep: set[int] = set()
    if keep_points:
        for px, py in keep_points:
            xi, yi = int(np.clip(px, 0, mask.shape[1] - 1)), int(np.clip(py, 0, mask.shape[0] - 1))
            if lab[yi, xi] > 0:
                keep.add(int(lab[yi, xi]))
    if not keep:
        keep = {1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))}
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_frac * mask.size * 20:
                keep.add(i)
    for i in keep:
        out[lab == i] = 255
    return out


def _soften(mask: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(mask, (0, 0), 0.9)


def grabcut_box(bgr: np.ndarray, box: tuple[int, int, int, int], iters: int = 4, max_side: int = 640) -> np.ndarray:
    h, w = bgr.shape[:2]
    x, y, bw, bh = [int(v) for v in box]
    x0, y0, x1, y1 = _crop_bounds(w, h, x, y, bw, bh, int(0.08 * max(bw, bh)) + 4)
    crop, s = _resize_to(bgr[y0:y1, x0:x1], max_side)
    rect = (max(0, int((x - x0) * s)), max(0, int((y - y0) * s)), max(2, int(bw * s)), max(2, int(bh * s)))
    rect = (rect[0], rect[1], min(rect[2], crop.shape[1] - rect[0] - 1), min(rect[3], crop.shape[0] - rect[1] - 1))
    if rect[2] < 2 or rect[3] < 2:
        return np.zeros((h, w), np.uint8)
    res = _run_grabcut(crop, np.zeros(crop.shape[:2], np.uint8), rect, iters, cv2.GC_INIT_WITH_RECT)
    out = np.zeros((h, w), np.uint8)
    if res is None:
        return out
    out[y0:y1, x0:x1] = cv2.resize(res, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    return _soften(_clean(out))


def grabcut_points(bgr: np.ndarray, fg: list[tuple[float, float]], bg: list[tuple[float, float]],
                   box: tuple[int, int, int, int] | None = None, iters: int = 4, max_side: int = 640) -> np.ndarray:
    h, w = bgr.shape[:2]
    if box is None:
        cx, cy = np.mean(np.array(fg), axis=0)
        half = int(0.3 * max(h, w))
        box = (max(0, int(cx - half)), max(0, int(cy - half)),
               min(w, int(cx + half)) - max(0, int(cx - half)), min(h, int(cy + half)) - max(0, int(cy - half)))
    x, y, bw, bh = [int(v) for v in box]
    seed = np.zeros((h, w), np.uint8)
    if fg:
        blur = cv2.GaussianBlur(bgr, (0, 0), 1.5)
        for px, py in fg:
            ff = np.ones((h + 2, w + 2), np.uint8)
            ff[y + 1:y + bh + 1, x + 1:x + bw + 1] = 0
            cv2.floodFill(blur, ff, (int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))), 0, (14, 14, 14), (14, 14, 14),
                          4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8))
            seed |= (ff[1:-1, 1:-1] == 255).astype(np.uint8)
    m = np.full((h, w), cv2.GC_BGD, np.uint8)
    m[y:y + bh, x:x + bw] = cv2.GC_PR_BGD
    m[(seed > 0)] = cv2.GC_PR_FGD
    r = max(4, int(0.015 * max(h, w)))
    for px, py in fg:
        cv2.circle(m, (int(px), int(py)), r, int(cv2.GC_FGD), -1)
    for px, py in bg:
        cv2.circle(m, (int(px), int(py)), r, int(cv2.GC_BGD), -1)
    crop, s = _resize_to(bgr[y:y + bh, x:x + bw], max_side)
    mc = m[y:y + bh, x:x + bw]
    if s < 1.0:
        mc = cv2.resize(mc, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    mc = np.ascontiguousarray(mc)
    if (mc == cv2.GC_FGD).sum() < 3 or (mc == cv2.GC_BGD).sum() < 3:
        return np.zeros((h, w), np.uint8)
    res = _run_grabcut(crop, mc, None, iters, cv2.GC_INIT_WITH_MASK)
    out = np.zeros((h, w), np.uint8)
    if res is None:
        return out
    out[y:y + bh, x:x + bw] = cv2.resize(res, (bw, bh), interpolation=cv2.INTER_LINEAR)
    return _soften(_clean(out, fg))


def grabcut_prior(bgr: np.ndarray, prior: np.ndarray, band: int, iters: int = 2, max_side: int = 640) -> np.ndarray:
    """Re-estimate colour models from the prior's confident core/background and re-decide only the uncertain band."""
    h, w = prior.shape
    b = prior > 127
    ys, xs = np.where(b)
    if len(xs) < 20:
        return prior
    pad = band * 3 + 8
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
    img, s = _resize_to(bgr[y0:y1, x0:x1], max_side)
    pb = b[y0:y1, x0:x1].astype(np.uint8)
    if s < 1.0:
        pb = cv2.resize(pb, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    bd = max(1, int(band * s))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bd + 1, 2 * bd + 1))
    core, outer = cv2.erode(pb, k), cv2.dilate(pb, k)
    m = np.full(pb.shape, cv2.GC_BGD, np.uint8)
    m[outer > 0] = cv2.GC_PR_BGD
    m[pb > 0] = cv2.GC_PR_FGD
    m[core > 0] = cv2.GC_FGD if core.sum() >= 10 else cv2.GC_PR_FGD
    if core.sum() < 10:
        m[pb > 0] = cv2.GC_FGD
    if (m == cv2.GC_BGD).sum() < 10 or (m == cv2.GC_FGD).sum() < 10:
        return prior
    res = _run_grabcut(img, m, None, iters, cv2.GC_INIT_WITH_MASK)
    if res is None:
        return prior
    out = np.zeros((h, w), np.uint8)
    out[y0:y1, x0:x1] = cv2.resize(res, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) if s < 1.0 else res
    return _soften(out)
