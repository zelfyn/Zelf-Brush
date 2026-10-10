"""Project: folder layout, .roto JSON, autosave/recovery, proxy-frame cache, mask store."""
from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import re
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import video
from .config import PROXY_MAX_SIDE, VERSION
from .jobs import Job
from .masks import MaskStore
from .matte import DEFAULT_MATTE
from .video import MediaError, VideoInfo

log = logging.getLogger("rotobrush")
ROTO_FILE = "project.roto"
AUTOSAVE_FILE = "project.autosave.roto"

DEFAULT_EXPORT = {
    "kind": "mp4", "background": "#FFFFFF", "scale": 1.0, "codec": "h264", "crf": 18,
    "movCodec": "prores4444", "range": "full", "copySource": True, "hwEncode": "auto",
}


def proxy_size(w: int, h: int, max_side: int = PROXY_MAX_SIDE) -> tuple[int, int]:
    s = min(1.0, max_side / max(w, h))
    return max(2, int(round(w * s / 2)) * 2), max(2, int(round(h * s / 2)) * 2)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class Project:
    def __init__(self, root: Path, data: dict, recovered: bool = False) -> None:
        self.root = Path(root)
        self.data = data
        self.recovered = recovered
        self.dirty = False
        self.lock = threading.RLock()
        for sub in ("source", "cache/frames", "masks", "previews", "tracking"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        pw, ph = self.proxy
        self.masks = MaskStore(self.root / "masks", (pw, ph), self._get_meta, self._set_meta)
        self._frames: OrderedDict[int, np.ndarray] = OrderedDict()
        self._load_tracking()
        self._info: VideoInfo | None = None

    # ---------- basic accessors ----------
    @property
    def meta(self) -> dict:
        return self.data["project"]

    @property
    def proxy(self) -> tuple[int, int]:
        return self.data["proxy"]["width"], self.data["proxy"]["height"]

    @property
    def n_frames(self) -> int:
        return int(self.meta["frame_count"])

    @property
    def fps(self) -> float:
        return float(self.meta["fps"])

    @property
    def source_path(self) -> str:
        return self.meta["source"]

    @property
    def intermediate_path(self) -> Path:
        return self.root / "cache" / "intermediate.mov"

    @property
    def info(self) -> VideoInfo:
        """Where frames are DECODED from: the stable intermediate if one was made, else the original file."""
        if self._info is None:
            self._info = video.probe(self.source_path)
            self._info.frame_count = self.n_frames
        if self.intermediate_path.exists():
            return dataclasses.replace(self._info, path=str(self.intermediate_path))
        return self._info

    def touch(self) -> None:
        self.dirty = True

    # ---------- meta (undo/redo covers keyframes + confidence) ----------
    def _get_meta(self) -> dict:
        return {"keyframes": self.data["keyframes"], "confidence": self.data["tracking"]["confidence"]}

    def _set_meta(self, m: dict) -> None:
        self.data["keyframes"] = m["keyframes"]
        self.data["tracking"]["confidence"] = m["confidence"]
        self.touch()

    def add_keyframe(self, frame: int, kind: str) -> None:
        kfs = self.data["keyframes"]
        for k in kfs:
            if k["frame"] == frame:
                if kind == "user":
                    k["kind"] = "user"
                return
        kfs.append({"frame": frame, "kind": kind})
        kfs.sort(key=lambda k: k["frame"])
        self.touch()

    def user_keyframes(self) -> list[int]:
        return [k["frame"] for k in self.data["keyframes"] if k["kind"] == "user"]

    def set_confidence(self, frame: int, value: float) -> None:
        self.data["tracking"]["confidence"][str(frame)] = round(float(value), 3)
        self.touch()

    # ---------- creation / opening ----------
    @classmethod
    def create(cls, info: VideoInfo, workspace: Path, name: str | None = None) -> "Project":
        base = re.sub(r"[^\w\- ]+", "_", name or Path(info.filename).stem).strip() or "Untitled"
        root, n = workspace / base, 1
        while root.exists():
            n += 1
            root = workspace / f"{base}_{n}"
        pw, ph = proxy_size(info.width, info.height)
        data = {
            "version": 1, "app": VERSION,
            "project": {
                "name": root.name, "source": info.path, "width": info.width, "height": info.height,
                "fps": info.fps, "duration": info.duration, "frame_count": info.frame_count,
                "codec": info.codec, "size_bytes": info.size_bytes, "has_audio": info.has_audio,
                "filename": info.filename, "audio_codec": info.audio_codec, "container": info.container,
                "pix_fmt": info.pix_fmt,
            },
            "proxy": {"width": pw, "height": ph},
            "engine": "auto",
            "objects": [], "keyframes": [], "corrections": [],
            "tracking": {"confidence": {}},
            "maskSettings": dict(DEFAULT_MATTE),
            "exportSettings": dict(DEFAULT_EXPORT),
            "workArea": {"in": 0, "out": max(0, info.frame_count - 1)},
        }
        root.mkdir(parents=True)
        p = cls(root, data)
        p._info = info
        p.save()
        return p

    @classmethod
    def open(cls, path: str | Path) -> "Project":
        p = Path(path)
        root = p if p.is_dir() else p.parent
        main, auto = root / ROTO_FILE, root / AUTOSAVE_FILE
        if p.is_file() and p.suffix == ".roto" and p.name not in (ROTO_FILE, AUTOSAVE_FILE):
            main = p
        if not main.exists() and not auto.exists():
            raise MediaError("This is not a RotoBrush project.", ["Choose a folder that contains project.roto"])
        recovered = auto.exists() and (not main.exists() or auto.stat().st_mtime > main.stat().st_mtime)
        try:
            data = json.loads((auto if recovered else main).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if recovered and main.exists():
                data, recovered = json.loads(main.read_text(encoding="utf-8")), False
            else:
                raise MediaError("The project file is damaged and could not be read.")
        data.setdefault("tracking", {"confidence": {}})
        data["maskSettings"] = {**DEFAULT_MATTE, **data.get("maskSettings", {})}
        data["exportSettings"] = {**DEFAULT_EXPORT, **data.get("exportSettings", {})}
        if not Path(data["project"]["source"]).exists():
            raise MediaError(
                f"The source video is missing: {data['project']['source']}",
                ["Move the video back to its original location", "Original footage is never copied or modified"])
        proj = cls(root.resolve(), data, recovered=recovered)
        if recovered:
            log.warning("recovered project %s from autosave", root)
        return proj

    # ---------- persistence ----------
    def _write(self, fname: str) -> None:
        with self.lock:
            self.save_tracking()
            _atomic_write(self.root / fname, json.dumps(self.data, indent=2))

    def save(self) -> None:
        self._write(ROTO_FILE)
        auto = self.root / AUTOSAVE_FILE
        if auto.exists():
            auto.unlink()
        self.dirty = False
        self.recovered = False

    def autosave(self) -> bool:
        if not self.dirty:
            return False
        self._write(AUTOSAVE_FILE)
        return True

    def save_as(self, dest: str | Path) -> "Project":
        self.save()
        dest = Path(dest)
        if dest.suffix == ".roto":
            dest = dest.parent / dest.stem
        if dest.exists() and any(dest.iterdir()):
            raise MediaError("That folder is not empty.", ["Choose a new or empty folder"])
        shutil.copytree(self.root, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("cache", "previews", "*.tmp"))
        data = copy.deepcopy(self.data)
        data["project"]["name"] = dest.name
        _atomic_write(dest / ROTO_FILE, json.dumps(data, indent=2))
        return Project.open(dest)

    def save_tracking(self) -> None:
        _atomic_write(self.root / "tracking" / "tracking.json",
                      json.dumps({"engine": self.data.get("engine"), "confidence": self.data["tracking"]["confidence"]}))

    def _load_tracking(self) -> None:
        f = self.root / "tracking" / "tracking.json"
        if f.exists() and not self.data["tracking"]["confidence"]:
            try:
                self.data["tracking"]["confidence"] = json.loads(f.read_text()).get("confidence", {})
            except ValueError:
                pass

    # ---------- proxy frame cache (lazy, resumable, reusable) ----------
    @property
    def frames_dir(self) -> Path:
        return self.root / "cache" / "frames"

    def _marker(self) -> Path:
        return self.frames_dir / ".complete"

    def cache_complete(self) -> bool:
        try:
            m = json.loads(self._marker().read_text())
        except (OSError, ValueError):
            return False
        return m.get("w") == self.proxy[0] and m.get("h") == self.proxy[1] and m.get("count", 0) > 0

    def cached_frame_count(self) -> int:
        if self.cache_complete():
            return json.loads(self._marker().read_text())["count"]
        return sum(1 for _ in self.frames_dir.glob("*.jpg"))

    def build_cache(self, job: Job | None = None) -> dict:
        if self.cache_complete():
            log.info("frame cache reused (%d frames)", self.cached_frame_count())
            return {"reused": True, "frames": self.cached_frame_count()}
        w, h = self.proxy
        total = max(1, self.n_frames)

        def prog(n: int) -> None:
            if job:
                job.update(min(0.99, n / total), f"Caching frames {n}/{total}")

        from .jobs import Cancelled

        def extract() -> int:
            try:
                return video.extract_proxy_frames(self.info, self.frames_dir, w, h, prog, (lambda: job.cancelled) if job else None)
            except InterruptedError:
                raise Cancelled()

        try:
            count = extract()
            if count == 0:
                raise MediaError("No frames could be decoded.", stage="frame-extraction")
        except MediaError as first:
            if self.intermediate_path.exists():
                raise
            # The source is valid (it probed fine) but could not be decoded in one pass: normalise it into a
            # stable constant-frame-rate intermediate and extract from that instead of calling the file corrupt.
            log.warning("direct extraction failed (%s); transcoding to a stable intermediate", first.detail or first.message)
            if job:
                job.update(0.0, "Converting to a stable intermediate format")
            try:
                video.make_intermediate(self._info or self.info, self.intermediate_path,
                                        (lambda f: job.update(min(0.45, f / total * 0.45), f"Converting {f}/{total}")) if job else None,
                                        (lambda: job.cancelled) if job else None)
            except InterruptedError:
                raise Cancelled()
            except MediaError as e:
                raise MediaError("This video could not be decoded.", e.hints, stage="transcode",
                                 exit_code=e.exit_code, detail=e.detail or first.detail, log=e.log or first.log)
            count = extract()
            if count == 0:
                raise first
        if count > 0 and count != self.n_frames:
            log.info("frame count corrected %d -> %d", self.n_frames, count)
            wa = self.data["workArea"]
            if wa["out"] >= self.n_frames - 1 or wa["out"] >= count:
                wa["out"] = count - 1
            self.meta["frame_count"] = count
            self.meta["duration"] = count / self.fps
            self.touch()
        self._marker().write_text(json.dumps({"count": count, "w": w, "h": h}))
        return {"reused": False, "frames": count}

    def frame_bgr(self, i: int) -> np.ndarray:
        """Proxy-resolution frame: from the disk cache, else decoded on demand (lazy loading)."""
        i = max(0, min(i, self.n_frames - 1))
        with self.lock:
            if i in self._frames:
                self._frames.move_to_end(i)
                return self._frames[i].copy()
        w, h = self.proxy
        img = cv2.imread(str(self.frames_dir / f"{i:06d}.jpg"))
        if img is None or img.shape[1] != w or img.shape[0] != h:
            img = video.decode_frame(self.info, i, w, h)
            if img is None:
                raise MediaError(f"Frame {i} could not be decoded.", ["The video may be damaged near this frame"])
        with self.lock:
            self._frames[i] = img
            while len(self._frames) > 24:
                self._frames.popitem(last=False)
        return img.copy()

    # ---------- public state ----------
    def public_state(self) -> dict:
        d = copy.deepcopy(self.data)
        d["root"] = str(self.root)
        d["recovered"] = self.recovered
        d["proxy"] = {"width": self.proxy[0], "height": self.proxy[1]}
        return d

    def timeline(self, low: float) -> dict:
        conf = {int(k): v for k, v in self.data["tracking"]["confidence"].items()}
        frames = self.masks.frames()
        ranges: list[list[int]] = []
        for f in frames:
            if ranges and f == ranges[-1][1] + 1:
                ranges[-1][1] = f
            else:
                ranges.append([f, f])
        return {
            "n_frames": self.n_frames, "keyframes": self.data["keyframes"], "confidence": conf,
            "warnings": sorted(f for f, c in conf.items() if c < low), "masked": ranges,
            "cache": {"frames": self.cached_frame_count(), "complete": self.cache_complete()},
            "can_undo": self.masks.can_undo(), "can_redo": self.masks.can_redo(),
        }
