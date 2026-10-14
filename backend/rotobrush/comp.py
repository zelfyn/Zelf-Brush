"""Composition = a stack of video layers on one timeline (like an After Effects comp).

Each layer wraps one clip (a Project: its own source, proxy cache, roto masks, tracking). The layer adds timeline
placement: `start` (comp frame where the trimmed clip begins), `trimIn`/`trimOut` (source frames kept), plus
visibility, opacity, scale and position. Roto tools always work on the *active* layer, in that layer's source frames.
Layers are stored top -> bottom (index 0 is the top layer).
"""
from __future__ import annotations

import copy
import json
import logging
import re
import shutil
import threading
from pathlib import Path

from . import fsutil
from .config import VERSION
from .project import AUTOSAVE_FILE, DEFAULT_EXPORT, ROTO_FILE, Project, proxy_size
from .video import MediaError, VideoInfo

log = logging.getLogger("rotobrush")
COMP_FILE = "comp.roto"
LAYER_DEFAULTS = {"visible": True, "opacity": 1.0, "scale": 1.0, "x": 0.0, "y": 0.0, "useMask": True}


def fit_size(sw: int, sh: int, cw: int, ch: int) -> tuple[float, float]:
    """Largest size with the source aspect that fits inside the comp (letterbox)."""
    s = min(cw / sw, ch / sh)
    return sw * s, sh * s


class Comp:
    def __init__(self, root: Path, data: dict, projects: dict[str, Project]) -> None:
        self.root, self.data, self.projects = Path(root), data, projects
        self.lock = threading.RLock()
        self._sync_all()

    # ------------------------------------------------------------------ construction
    @classmethod
    def create(cls, info: VideoInfo, workspace: Path, name: str | None = None) -> "Comp":
        base = re.sub(r"[^\w\- ]+", "_", name or Path(info.filename).stem).strip() or "Untitled"
        root, n = workspace / base, 1
        while root.exists():
            n += 1
            root = workspace / f"{base}_{n}"
        root.mkdir(parents=True)
        data = {
            "version": 2, "app": VERSION, "name": root.name,
            "comp": {"width": info.width, "height": info.height, "fps": info.fps},
            "layers": [], "active": None, "exportSettings": dict(DEFAULT_EXPORT), "workArea": None,
        }
        comp = cls(root, data, {})
        comp.add_layer(info, start=0)
        return comp

    @classmethod
    def open(cls, path: str | Path) -> "Comp":
        p = Path(path)
        root = p if p.is_dir() else p.parent
        cf = root / COMP_FILE
        if cf.exists():
            try:
                data = json.loads(cf.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise MediaError("The composition file is damaged and could not be read.", stage="open-project")
            projects: dict[str, Project] = {}
            for L in data["layers"]:
                projects[L["id"]] = Project.open((root / L["dir"]).resolve())
            data["exportSettings"] = {**DEFAULT_EXPORT, **data.get("exportSettings", {})}
            return cls(root.resolve(), data, projects)
        # legacy / single-clip project folder: wrap it as a one-layer composition
        proj = Project.open(p)
        m = proj.meta
        data = {
            "version": 2, "app": VERSION, "name": m["name"],
            "comp": {"width": m["width"], "height": m["height"], "fps": m["fps"]},
            "layers": [{"id": "L1", "name": m.get("filename", m["name"]), "dir": ".", "start": 0, "trimIn": 0,
                        "trimOut": proj.n_frames - 1, **LAYER_DEFAULTS}],
            "active": "L1", "exportSettings": dict(proj.data.get("exportSettings", DEFAULT_EXPORT)), "workArea": None,
        }
        return cls(proj.root, data, {"L1": proj})

    # ------------------------------------------------------------------ layers
    def _new_id(self) -> str:
        n = 1
        while f"L{n}" in {L["id"] for L in self.data["layers"]} or (self.root / "layers" / f"L{n}").exists():
            n += 1
        return f"L{n}"

    def add_layer(self, info: VideoInfo, start: int | None = None, name: str | None = None) -> str:
        with self.lock:
            lid = self._new_id()
            proj = Project.create(info, self.root / "layers", name=lid)
            display = name or info.filename
            proj.data["project"]["name"] = display
            proj.save()
            n = proj.n_frames
            first = not self.data["layers"]
            L = {"id": lid, "name": display, "dir": f"layers/{lid}", "start": int(start or 0), "trimIn": 0,
                 "trimOut": n - 1, **LAYER_DEFAULTS}
            self.data["layers"].insert(0, L)  # new layers go on top, like After Effects
            self.projects[lid] = proj
            self.data["active"] = lid
            if first:
                self.data["comp"] = {"width": info.width, "height": info.height, "fps": info.fps}
            self._sync_layer(L)
            self.write()
            return lid

    def layer(self, lid: str | None = None) -> dict:
        lid = lid or self.data["active"]
        for L in self.data["layers"]:
            if L["id"] == lid:
                return L
        raise MediaError("That layer no longer exists.", stage="layer")

    def project(self, lid: str | None = None) -> Project:
        return self.projects[self.layer(lid)["id"]]

    @property
    def fps(self) -> float:
        return float(self.data["comp"]["fps"])

    @property
    def width(self) -> int:
        return int(self.data["comp"]["width"])

    @property
    def height(self) -> int:
        return int(self.data["comp"]["height"])

    def layer_frames(self, L: dict) -> int:
        """Length of the trimmed layer measured in COMP frames (handles clips with a different frame rate)."""
        p = self.projects[L["id"]]
        return max(1, int(round((L["trimOut"] - L["trimIn"] + 1) * self.fps / p.fps)))

    def layer_range(self, L: dict) -> tuple[int, int]:
        return L["start"], L["start"] + self.layer_frames(L) - 1

    def local_frame(self, L: dict, comp_frame: int) -> int | None:
        """Comp frame -> the layer's own source frame, or None if the layer is not on screen at that time."""
        a, b = self.layer_range(L)
        if not a <= comp_frame <= b:
            return None
        p = self.projects[L["id"]]
        return min(L["trimOut"], L["trimIn"] + int(round((comp_frame - L["start"]) * p.fps / self.fps)))

    def comp_frame(self, L: dict, local: int) -> int:
        p = self.projects[L["id"]]
        return L["start"] + int(round((local - L["trimIn"]) * self.fps / p.fps))

    @property
    def duration(self) -> int:
        return max((self.layer_range(L)[1] + 1 for L in self.data["layers"]), default=1)

    def work_range(self) -> tuple[int, int]:
        wa = self.data.get("workArea")
        if wa:
            return max(0, int(wa["in"])), min(self.duration - 1, int(wa["out"]))
        return 0, self.duration - 1

    def update_layer(self, lid: str, patch: dict) -> dict:
        with self.lock:
            L = self.layer(lid)
            p = self.projects[lid]
            for k in ("visible", "useMask"):
                if k in patch:
                    L[k] = bool(patch[k])
            for k, lo, hi in (("opacity", 0.0, 1.0), ("scale", 0.05, 8.0)):
                if k in patch:
                    L[k] = float(min(hi, max(lo, float(patch[k]))))
            for k in ("x", "y"):
                if k in patch:
                    L[k] = float(patch[k])
            if "name" in patch and str(patch["name"]).strip():
                L["name"] = str(patch["name"]).strip()[:80]
            if "start" in patch:
                L["start"] = max(0, int(patch["start"]))
            n = p.n_frames
            if "trimIn" in patch or "trimOut" in patch:
                a = int(patch.get("trimIn", L["trimIn"]))
                b = int(patch.get("trimOut", L["trimOut"]))
                a, b = max(0, min(a, n - 1)), max(0, min(b, n - 1))
                if a > b:
                    raise MediaError("The In point must come before the Out point.", stage="trim")
                # keep the content under the playhead steady when trimming the head: shift start with trimIn
                if "trimIn" in patch and "start" not in patch:
                    if len(self.data["layers"]) == 1:
                        L["start"] = 0  # a lone clip ripples: trimming its head must not leave an empty gap
                    else:
                        L["start"] = max(0, L["start"] + int(round((a - L["trimIn"]) * self.fps / p.fps)))
                L["trimIn"], L["trimOut"] = a, b
            self._sync_layer(L)
            self.write()
            return L

    def reorder(self, order_top_to_bottom: list[str]) -> None:
        with self.lock:
            byid = {L["id"]: L for L in self.data["layers"]}
            if sorted(order_top_to_bottom) != sorted(byid):
                raise MediaError("The layer order is not valid.", stage="layer")
            self.data["layers"] = [byid[i] for i in order_top_to_bottom]
            self.write()

    def remove_layer(self, lid: str) -> None:
        with self.lock:
            if len(self.data["layers"]) <= 1:
                raise MediaError("A composition needs at least one layer.", ["Start a new project instead"], stage="layer")
            L = self.layer(lid)
            self.data["layers"] = [x for x in self.data["layers"] if x["id"] != lid]
            self.projects.pop(lid, None)  # files stay on disk: removing a layer never deletes footage or masks
            if self.data["active"] == lid:
                self.data["active"] = self.data["layers"][0]["id"]
            self.write()
            log.info("layer %s removed (folder %s kept)", L["name"], L["dir"])

    def activate(self, lid: str) -> None:
        self.layer(lid)
        self.data["active"] = lid
        self.write()

    def set_work_area(self, a: int, b: int) -> None:
        a, b = max(0, min(a, b)), min(self.duration - 1, max(a, b))
        self.data["workArea"] = None if (a <= 0 and b >= self.duration - 1) else {"in": a, "out": b}
        self.write()

    def _sync_layer(self, L: dict) -> None:
        """Roto tracking and 'work area' exports of this clip are limited to its trim range."""
        p = self.projects[L["id"]]
        L["trimOut"] = min(L["trimOut"], p.n_frames - 1)
        L["trimIn"] = min(L["trimIn"], L["trimOut"])
        p.data["workArea"] = {"in": L["trimIn"], "out": L["trimOut"]}
        p.touch()

    def _sync_all(self) -> None:
        for L in self.data["layers"]:
            for k, v in LAYER_DEFAULTS.items():
                L.setdefault(k, v)
            if L["id"] in self.projects:
                self._sync_layer(L)
        if self.data.get("active") not in self.projects and self.data["layers"]:
            self.data["active"] = self.data["layers"][0]["id"]

    # ------------------------------------------------------------------ persistence
    def write(self) -> None:
        with self.lock:
            fsutil.atomic_write_text(self.root / COMP_FILE, json.dumps(self.data, indent=2))

    def save(self) -> None:
        with self.lock:
            for p in self.projects.values():
                p.save()
            self.write()

    def autosave(self) -> bool:
        did = False
        for p in list(self.projects.values()):
            did = p.autosave() or did
        return did

    def save_as(self, dest: str | Path) -> "Comp":
        self.save()
        dest = Path(dest)
        if dest.suffix == ".roto":
            dest = dest.parent / dest.stem
        if dest.exists() and any(dest.iterdir()):
            raise MediaError("That folder is not empty.", ["Choose a new or empty folder"], stage="save-as")
        shutil.copytree(self.root, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("cache", "previews", "*.tmp"))
        data = copy.deepcopy(self.data)
        data["name"] = dest.name
        fsutil.atomic_write_text(dest / COMP_FILE, json.dumps(data, indent=2))
        return Comp.open(dest)

    # ------------------------------------------------------------------ state for the UI
    def public_state(self) -> dict:
        layers = []
        for L in self.data["layers"]:
            p = self.projects[L["id"]]
            a, b = self.layer_range(L)
            layers.append({
                **L, "frames": p.n_frames, "end": b, "length": b - a + 1, "filename": p.meta.get("filename", ""),
                "width": p.meta["width"], "height": p.meta["height"], "fps": p.fps,
                "aiModel": p.data.get("aiModel", "auto"), "aiModelResolved": p.data.get("aiModelResolved"),
                "hasMask": bool(p.masks.frames()), "hasAudio": bool(p.meta.get("has_audio")),
            })
        wa_in, wa_out = self.work_range()
        pw, ph = proxy_size(self.width, self.height)
        return {"proxy": {"width": pw, "height": ph}, "name": self.data["name"], "root": str(self.root), "width": self.width, "height": self.height,
                "fps": self.fps, "duration": self.duration, "layers": layers, "active": self.data["active"],
                "workArea": {"in": wa_in, "out": wa_out}, "exportSettings": self.data["exportSettings"]}
