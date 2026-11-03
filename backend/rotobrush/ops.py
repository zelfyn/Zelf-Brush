"""Editing operations: brush/eraser/lasso/object select, viewport rendering, analysis, auto roto, propagation."""
from __future__ import annotations

import functools
import hashlib
import json
import logging
from collections import OrderedDict
from typing import Sequence

import cv2
import numpy as np

from . import matte
from .config import LOW_CONFIDENCE
from .engines import get_engine
from .engines.onnx_engine import interactive
from .engines.base import Detection, EngineError
from .jobs import Job
from .project import Project
from .video import MediaError

log = logging.getLogger("rotobrush.ai")


def _interactive(fn):
    """The user is waiting for this call: its AI inference jumps ahead of any background tracking job."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        with interactive():
            return fn(*a, **k)
    return wrapper


def _eng(p: Project):
    """Engine + the model this layer asked for, active for the current thread only."""
    eng = get_engine(p.data["engine"])
    return eng, (lambda: eng.model_context(p.data.get("aiModel", "auto"), p.data.get("aiModelResolved")))


# ------------------------------------------------------------------ manual tools
def _stamp(size: float, hardness: float) -> np.ndarray:
    r = max(1.0, size / 2.0)
    n = int(np.ceil(r)) * 2 + 1
    yy, xx = np.mgrid[:n, :n].astype(np.float32)
    d = np.hypot(xx - n // 2, yy - n // 2) / r
    inner = float(np.clip(hardness, 0, 0.99))
    return np.clip((1.0 - d) / (1.0 - inner), 0, 1).astype(np.float32)


def stroke_coverage(shape: tuple[int, int], points: Sequence[Sequence[float]], size: float, hardness: float) -> np.ndarray:
    h, w = shape
    cov = np.zeros((h, w), np.float32)
    if not points:
        return cov
    st = _stamp(size, hardness)
    n = st.shape[0]
    half = n // 2
    spacing = max(1.0, size * 0.12)
    pts = [tuple(map(float, p[:2])) for p in points]
    dense: list[tuple[float, float]] = [pts[0]]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        dist = float(np.hypot(x1 - x0, y1 - y0))
        steps = max(1, int(dist / spacing))
        for k in range(1, steps + 1):
            t = k / steps
            dense.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    for x, y in dense:
        cx, cy = int(round(x)), int(round(y))
        x0, y0, x1, y1 = cx - half, cy - half, cx + half + 1, cy + half + 1
        sx0, sy0 = max(0, -x0), max(0, -y0)
        x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1c <= x0c or y1c <= y0c:
            continue
        patch = st[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
        np.maximum(cov[y0c:y1c, x0c:x1c], patch, out=cov[y0c:y1c, x0c:x1c])
    return cov


def apply_stroke(mask: np.ndarray, cov: np.ndarray, mode: str, strength: float) -> np.ndarray:
    m = mask.astype(np.float32)
    k = cov * float(np.clip(strength, 0, 1))
    m = m + (255.0 - m) * k if mode == "add" else m * (1.0 - k)
    return np.clip(m, 0, 255).astype(np.uint8)


def polygon_mask(shape: tuple[int, int], polygon: Sequence[Sequence[float]]) -> np.ndarray:
    h, w = shape
    ss = 3
    big = np.zeros((h * ss, w * ss), np.uint8)
    pts = (np.array(polygon, np.float32) * ss).astype(np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(big, [pts], 255)
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)


def _mark_user(p: Project, frame: int, tool: str) -> None:
    p.add_keyframe(frame, "user")
    p.data["corrections"].append({"frame": frame, "tool": tool})
    p.data["corrections"] = p.data["corrections"][-500:]
    p.set_confidence(frame, 1.0)


def brush(p: Project, frame: int, mode: str, points, size: float, hardness: float, strength: float) -> None:
    with p.lock, p.masks.operation("Brush" if mode == "add" else "Eraser"):
        cov = stroke_coverage(p.masks.get(frame).shape, points, size, hardness)
        p.masks.set(frame, apply_stroke(p.masks.get(frame), cov, mode, strength))
        _mark_user(p, frame, "brush" if mode == "add" else "eraser")


def lasso(p: Project, frame: int, polygon, mode: str, ai_refine: bool = False, defer_refine: bool = False) -> dict | None:
    """Apply the lasso. With defer_refine the polygon is applied INSTANTLY and the AI edge-snap is left for
    lasso_refine() to do in the background (no waiting); the returned dict is what that step needs."""
    prior = polygon_mask(p.masks.get(frame).shape, polygon)
    region = prior
    if ai_refine and not defer_refine:
        region = _lasso_region(p, frame, prior)
    with p.masks.operation("Lasso"):
        before = p.masks.get(frame)
        cur = before.astype(np.float32)
        r = region.astype(np.float32) / 255.0
        out = np.maximum(cur, r * 255.0) if mode == "add" else cur * (1.0 - r)
        p.masks.set(frame, out)
        _mark_user(p, frame, "lasso")
    if ai_refine and defer_refine:
        return {"before": before, "prior": prior, "mode": mode, "version": p.masks.version(frame)}
    return None


@_interactive
def _lasso_region(p: Project, frame: int, prior: np.ndarray) -> np.ndarray:
    eng, ctx = _eng(p)
    band = int(np.clip(0.02 * max(prior.shape), 3, 14))
    with ctx():
        return eng.interactive.refine_prior(p.frame_bgr(frame), prior, band)


def lasso_refine(p: Project, frame: int, pending: dict, job: Job | None = None) -> dict:
    """Background step: snap the lasso to real edges, unless the user touched this frame again in the meantime."""
    region = _lasso_region(p, frame, pending["prior"])
    if job:
        job.update(0.9, "Applying")
    if p.masks.version(frame) != pending["version"]:
        log.info("AI refine of the lasso on frame %d dropped: the frame was edited again", frame)
        return {"applied": False}
    cur = pending["before"].astype(np.float32)
    r = region.astype(np.float32) / 255.0
    out = np.maximum(cur, r * 255.0) if pending["mode"] == "add" else cur * (1.0 - r)
    with p.masks.operation("AI refine", merge=True):  # folds into the lasso's undo step: one Ctrl+Z undoes both
        p.masks.set(frame, out)
    return {"applied": True}


@_interactive
def select_object(p: Project, frame: int, mode: str, points: list | None, negative: list | None,
                  box: list | None, object_id: str | None) -> None:
    eng, ctx = _eng(p)
    bgr = p.frame_bgr(frame)
    mask: np.ndarray | None = None
    if object_id:
        obj = next((o for o in p.data["objects"] if o["id"] == object_id), None)
        if not obj:
            raise MediaError("That object is no longer in the list. Run analysis again.")
        box = obj["bbox"]
    fg = [tuple(pt) for pt in (points or [])] or None
    bg = [tuple(pt) for pt in (negative or [])] or None
    try:
        with ctx():
            mask = eng.interactive.segment(bgr, box=tuple(box) if box else None, fg_points=fg, bg_points=bg)
    except EngineError:
        raise
    if mask is None or int((mask > 127).sum()) < 20:
        raise MediaError("No object was found there.",
                         ["Drag a box around the object instead of clicking", "Or use Lasso / Brush"],
                         stage="object-selection", detail="the model returned an empty region")
    with p.lock, p.masks.operation("Object selection"):
        cur = p.masks.get(frame)
        if mode == "replace":
            out = mask
        elif mode == "subtract":
            out = (cur.astype(np.float32) * (1 - mask.astype(np.float32) / 255.0))
        else:
            out = np.maximum(cur, mask)
        p.masks.set(frame, out)
        _mark_user(p, frame, "object")


def clear_mask(p: Project, frame: int) -> None:
    with p.lock, p.masks.operation("Clear mask"):
        p.masks.set(frame, np.zeros(p.proxy[::-1], np.uint8))
        _mark_user(p, frame, "clear")


# ------------------------------------------------------------------ analysis / auto roto
@_interactive
def analyze(p: Project, frame: int | None = None) -> dict:
    """Frame-sample the clip, detect subjects on the sampled frames and pick the best anchor frame."""
    eng, ctx = _eng(p)
    n = p.n_frames
    cands = [frame] if frame is not None else sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1} if n > 4 else {0})
    best: tuple[float, int, list[Detection]] | None = None
    if p.data.get("aiModel", "auto") == "auto":  # decide General vs Anime once, on a representative frame
        wa = p.data["workArea"]
        mid = max(wa["in"], min(wa["out"], (wa["in"] + wa["out"]) // 2))
        pick = eng.choose_model(p.frame_bgr(mid))
        if pick:
            p.data["aiModelResolved"] = pick
            p.touch()
        eng, ctx = _eng(p)
    for f in cands:
        f = max(0, min(f, n - 1))
        with ctx():
            dets = eng.detection.detect(p.frame_bgr(f))
        score = max((d.confidence * min(1.0, d.bbox[2] * d.bbox[3] / (0.15 * p.proxy[0] * p.proxy[1])) for d in dets), default=0.0)
        if best is None or score > best[0]:
            best = (score, f, dets)
    assert best is not None
    _, anchor, dets = best
    area = float(p.proxy[0] * p.proxy[1])
    dets = sorted(dets, key=lambda d: d.confidence * d.bbox[2] * d.bbox[3], reverse=True)
    dominant = None
    if dets:
        top = dets[0]
        second = dets[1].confidence * dets[1].bbox[2] * dets[1].bbox[3] if len(dets) > 1 else 0.0
        big = top.bbox[2] * top.bbox[3] / area >= 0.04
        clear = top.confidence * top.bbox[2] * top.bbox[3] >= 2.0 * second
        if top.confidence >= 0.5 and big and clear:
            dominant = top.id
    objs = [{"id": d.id, "label": d.label, "confidence": d.confidence, "bbox": list(d.bbox),
             "frame": anchor, "selected": d.id == dominant} for d in dets]
    with p.lock:
        p.data["objects"] = objs
        p.touch()
    return {"frame": anchor, "objects": objs, "dominant": dominant, "engine": eng.name,
            "model": p.data.get("aiModelResolved") if p.data.get("aiModel", "auto") == "auto" else p.data.get("aiModel")}


def _propagate_segment(p: Project, eng, start: int, step: int, stop_at: int | None, job: Job | None,
                       counter: list[int], total: int, max_frames: int | None = None) -> None:
    last = p.n_frames - 1 if step > 0 else 0
    wa = p.data["workArea"]
    last = min(last, wa["out"]) if step > 0 else max(last, wa["in"])
    if max_frames is not None:
        last = min(last, start + max_frames) if step > 0 else max(last, start - max_frames)
    if (step > 0 and start >= last) or (step < 0 and start <= last):
        return
    seed = p.masks.get(start)
    if not (seed > 127).any():
        return

    def on_frame(idx: int, m: np.ndarray, conf: float) -> None:
        if idx in p.user_keyframes():  # the user corrected this frame after we started: never overwrite their work
            return
        p.masks.set(idx, m)
        p.set_confidence(idx, conf)
        counter[0] += 1
        if job:
            job.update(counter[0] / max(1, total), f"Tracking frame {idx}")

    eng.temporal.propagate(p.frame_bgr, start, seed, step, last, stop_at,
                           (lambda: job.cancelled) if job else (lambda: False), on_frame)


def propagate(p: Project, frame: int, direction: str, job: Job | None = None, recalc: bool = False,
              limit: dict | None = None, merge: bool = False) -> dict:
    """Propagate the mask at `frame` through time.

    forward/backward/both stop at the next user-corrected keyframe. `recalc` re-tracks everything after
    `frame`, treating each later user correction as the anchor for the frames after it.
    """
    eng, ctx = _eng(p)
    if not p.masks.has(frame) or not (p.masks.get(frame) > 127).any():
        raise MediaError("There is no mask on this frame to propagate.",
                         ["Run Auto Roto or paint a mask with Brush or Lasso first"])
    wa = p.data["workArea"]
    users = [k for k in p.user_keyframes() if k != frame]
    counter = [0]
    p.add_keyframe(frame, "ai" if frame not in p.user_keyframes() else "user")

    def segs(step: int) -> list[tuple[int, int | None]]:
        after = sorted(k for k in users if (k > frame if step > 0 else k < frame))
        after = after if step > 0 else after[::-1]
        if recalc and step > 0:
            return [(frame, after[0] if after else None)] + [(a, after[i + 1] if i + 1 < len(after) else None) for i, a in enumerate(after)]
        return [(frame, after[0] if after else None)]

    lim = limit or {}
    plan: list[tuple[int, int, int | None, int | None]] = []
    if direction in ("forward", "both"):
        plan += [(s, 1, stop, lim.get("forward")) for s, stop in segs(1)]
    if direction in ("backward", "both"):
        plan += [(s, -1, stop, lim.get("backward")) for s, stop in segs(-1)]
    total = 0
    for s, step, stop, maxf in plan:
        end = (stop if stop is not None else (wa["out"] + 1 if step > 0 else wa["in"] - 1))
        span = max(0, abs(end - s))
        total += min(span, maxf) if maxf is not None else span
    label = "Recalculate" if recalc else f"Propagate {direction}"
    # No lock is held while tracking: previews, edits and the timeline stay responsive for the whole run.
    with p.masks.operation(label, merge=merge), ctx():
        for s, step, stop, maxf in plan:
            _propagate_segment(p, eng, s, step, stop, job, counter, total, maxf)
    p.save_tracking()
    low = [f for f, c in p.data["tracking"]["confidence"].items() if c < LOW_CONFIDENCE]
    return {"frames": counter[0], "low_confidence": len(low), "engine": eng.name}


def auto_update(p: Project, frame: int, forward: int, backward: int, job: Job | None = None) -> dict:
    """After a manual edit: let the AI carry the correction to the neighbouring frames (limited range, cancellable)."""
    forward, backward = max(0, int(forward)), max(0, int(backward))
    if (forward <= 0 and backward <= 0) or not p.masks.has(frame) or not (p.masks.get(frame) > 127).any():
        return {"frames": 0, "skipped": True}
    direction = "both" if forward > 0 and backward > 0 else ("forward" if forward > 0 else "backward")
    return propagate(p, frame, direction, job, limit={"forward": forward, "backward": backward}, merge=True)


def auto_roto(p: Project, frame: int, object_ids: list[str] | None, job: Job | None = None) -> dict:
    """Initial segmentation of the chosen subjects on the anchor frame, then propagation both ways."""
    eng, ctx = _eng(p)
    objs = [o for o in p.data["objects"] if (o["id"] in object_ids if object_ids else o.get("selected"))]
    if not objs:
        raise MediaError("Object detection failed.", ["Select the object manually", "Use Brush", "Use Lasso"])
    bgr = p.frame_bgr(frame)
    if job:
        job.update(0.02, "Segmenting subject")
    union = np.zeros(bgr.shape[:2], np.uint8)
    with ctx():
        for o in objs:
            m = eng.interactive.segment(bgr, box=tuple(o["bbox"]))
            union = np.maximum(union, m)
    if int((union > 127).sum()) < 30:
        raise MediaError("Object detection failed.", ["Select the object manually", "Use Brush", "Use Lasso"])
    with p.lock, p.masks.operation("Auto roto"):
        p.masks.set(frame, union)
        p.set_confidence(frame, 1.0)
        p.add_keyframe(frame, "ai")
    if job:
        job.update(0.1, "Tracking")
    p.data["objects"] = [{**o, "selected": o in objs} for o in p.data["objects"]]
    sub = _SubJob(job)
    return propagate(p, frame, "both", sub)


class _SubJob:
    """Maps propagation progress into the 10%..100% range of the auto-roto job."""

    def __init__(self, job: Job | None) -> None:
        self.job = job

    @property
    def cancelled(self) -> bool:
        return bool(self.job and self.job.cancelled)

    def update(self, progress: float | None = None, message: str | None = None) -> None:
        if self.job:
            self.job.update(0.1 + 0.9 * (progress or 0.0), message)


# ------------------------------------------------------------------ viewport rendering
_render_cache: OrderedDict[str, bytes] = OrderedDict()


def _settings_hash(s: dict) -> str:
    return hashlib.md5(json.dumps(s, sort_keys=True).encode()).hexdigest()[:8]


def view_alpha(p: Project, frame: int, bgr: np.ndarray) -> np.ndarray:
    return matte.refine_alpha(p.masks.get(frame), bgr, p.data["maskSettings"])


def render(p: Project, frame: int, mode: str, quality: int) -> tuple[bytes, str]:
    settings = p.data["maskSettings"]
    # Rendered frames belong to a project. The cache is process-global, so frame/mask
    # versions alone are not unique: two projects can both have frame 10 at version 1.
    # Include the stable project root to prevent one project's image being served for another.
    project_key = str(p.root.resolve())
    key = f"{project_key}:{frame}:{mode}:{quality}:{p.masks.version(frame)}:{_settings_hash(settings)}:{int(p.masks.has(frame))}"
    if key in _render_cache:
        _render_cache.move_to_end(key)
        mt_b, data = _render_cache[key].split(b"\0", 1)
        return data, mt_b.decode()
    bgr = p.frame_bgr(frame)
    a = view_alpha(p, frame, bgr) if mode not in ("original", "depth") else None
    png = False
    if mode == "original":
        img = bgr
    elif mode == "mask":
        img = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif mode == "transparent":
        img = matte.foreground_bgra(bgr, a, bool(settings["decontaminate"]))
        png = True
    elif mode == "depth":
        from . import depth as depthmod
        with interactive():
            img = depthmod.colorize(depthmod.frame_depth(p, frame))
    elif mode == "overlay":
        tint = np.array([255, 120, 50], np.float32)  # BGR blue, contrasts with warm subjects
        base = bgr.astype(np.float32)
        img = (base * (1 - 0.45 * a[..., None]) + tint * 0.45 * a[..., None]).astype(np.uint8)
    elif mode == "edge":
        raw = p.masks.get(frame).astype(np.float32) / 255.0
        img = (bgr.astype(np.float32) * 0.55).astype(np.uint8)
        soft = (a > 0.03) & (a < 0.97)
        moved = np.abs(a - raw) > 0.25
        hard = cv2.morphologyEx((a > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        img[hard] = (255, 210, 60)
        img[soft] = (255, 255, 0)
        img[moved] = (60, 60, 255)
        if str(frame) in p.data["tracking"]["confidence"] and p.data["tracking"]["confidence"][str(frame)] < LOW_CONFIDENCE:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (60, 60, 255), 6)
    else:
        raise MediaError(f"Unknown viewport mode: {mode}")
    if quality < 100:
        s = max(0.1, quality / 100.0)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if png:
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        mt = "image/png"
    else:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        mt = "image/jpeg"
    if not ok:
        raise MediaError("The frame could not be rendered.")
    data = buf.tobytes()
    _render_cache[key] = mt.encode() + b"\0" + data
    while len(_render_cache) > 96:
        _render_cache.popitem(last=False)
    return data, mt
