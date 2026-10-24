"""HTTP API + Server-Sent-Events for the desktop UI. Bound to 127.0.0.1 and protected by a per-launch token."""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__, compose, depth as depthmod, export as exporter, ops, video
from .config import CREATE_NO_WINDOW, LOW_CONFIDENCE, app_data_dir, log_dir, workspace_dir
from .engines import get_engine, list_engines
from .engines.onnx_engine import available_models as onnx_models
from .engines.base import EngineError
from .events import bus
from .gpu import gpu_info
from .jobs import JobManager
from .logs import setup_logging
from .matte import DEFAULT_MATTE, merge_settings
from .comp import Comp
from .project import DEFAULT_EXPORT, Project
from .video import MediaError

setup_logging()
log = logging.getLogger("rotobrush")
TOKEN = os.environ.get("ROTOBRUSH_TOKEN", "")

@asynccontextmanager
async def lifespan(_: FastAPI):
    bus.bind(asyncio.get_running_loop())
    threading.Thread(target=_autosave_loop, daemon=True, name="autosave").start()
    log.info("Zelf-Brush backend %s started", __version__)
    yield


app = FastAPI(title="Zelf-Brush backend", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["tauri://localhost", "http://tauri.localhost", "https://tauri.localhost",
                   "http://localhost:1420", "http://127.0.0.1:1420"],
    allow_methods=["*"], allow_headers=["*"],
)
jobs = JobManager()


_layer_var: contextvars.ContextVar = contextvars.ContextVar("layer", default=None)


class Session:
    """The open composition. `project` is the ACTIVE layer's clip (what the roto tools edit)."""
    comp: Comp | None = None

    @property
    def project(self) -> Project | None:
        return self.comp.project() if self.comp else None

    def require_comp(self) -> Comp:
        if self.comp is None:
            raise MediaError("No project is open.", ["Import a video or open a project first"])
        return self.comp

    def require(self, layer: str | None = None) -> Project:
        return self.require_comp().project(layer or _layer_var.get())


S = Session()


# ---------------------------------------------------------------- plumbing
@app.middleware("http")
async def auth(request: Request, call_next):
    if TOKEN and request.method != "OPTIONS" and request.url.path != "/health":
        if request.headers.get("x-token") != TOKEN and request.query_params.get("token") != TOKEN:
            return JSONResponse({"detail": {"message": "Unauthorized", "hints": []}}, status_code=401)
    tok = _layer_var.set(request.headers.get("x-layer") or request.query_params.get("layer"))
    try:
        return await call_next(request)
    finally:
        _layer_var.reset(tok)


@app.exception_handler(MediaError)
async def _media_err(_: Request, e: MediaError):
    return JSONResponse({"detail": e.to_dict()}, status_code=400)


@app.exception_handler(EngineError)
async def _engine_err(_: Request, e: EngineError):
    bus.publish("error", message=e.message, hints=e.hints)
    return JSONResponse({"detail": {"message": e.message, "hints": e.hints}}, status_code=400)


@app.exception_handler(Exception)
async def _any_err(_: Request, e: Exception):
    log.exception("unhandled error")
    bus.publish("error", message="Something went wrong.", hints=["Details are in app.log"])
    return JSONResponse({"detail": {"message": "Something went wrong.", "hints": [f"See {log_dir() / 'app.log'}"]}}, status_code=500)




def _autosave_loop() -> None:
    while True:
        time.sleep(20)
        try:
            c = S.comp
            if c and c.autosave():
                bus.publish("autosaved", time=time.time())
        except Exception:  # noqa: BLE001
            log.exception("autosave failed")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "version": __version__}


@app.get("/events")
async def events(request: Request):
    q = bus.subscribe()

    async def gen():
        try:
            yield "retry: 2000\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), 15)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield f"data: {json.dumps(msg)}\n\n"
        finally:
            bus.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- system
@app.get("/system/status")
def system_status() -> dict:
    p = S.project
    eng = get_engine(p.data["engine"] if p else "auto")
    return {
        "version": __version__, "gpu": gpu_info(), "engines": list_engines(), "active_engine": eng.info(),
        "cache": {"frames": p.cached_frame_count(), "complete": p.cache_complete()} if p else None,
        "workspace": str(workspace_dir()), "logs": str(log_dir()),
        "models": {"segmentation": onnx_models(), "depth": depthmod.depth_model_path() is not None},
    }


class PickReq(BaseModel):
    kind: Literal["video", "project", "folder", "save-video"]
    default_name: str | None = None
    ext: str | None = None


@app.post("/system/pick")
def pick(req: PickReq) -> dict:
    """Native file dialog (Tk, ships with Python on Windows). Runs in a subprocess so it can't block the API."""
    frozen = getattr(sys, "frozen", False)
    cmd = [sys.executable, "--pick"] if frozen else [sys.executable, "-m", "rotobrush.dialogs"]
    try:
        r = subprocess.run([*cmd, req.kind, req.default_name or "", req.ext or ""],
                           capture_output=True, text=True, timeout=600, creationflags=CREATE_NO_WINDOW)
        return {"path": json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else ""}
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        raise MediaError("The file dialog could not be opened.", ["Drag a file into the window instead"])


# ---------------------------------------------------------------- project / import
def _state(comp: Comp) -> dict:
    """Active layer state (what the editor panels use) + the whole composition (layers, timing)."""
    return {**comp.project().public_state(), "comp": comp.public_state()}


def _cache_jobs(comp: Comp, only: str | None = None) -> None:
    for L in comp.data["layers"]:
        p = comp.projects[L["id"]]
        if (only is None or L["id"] == only) and not p.cache_complete():
            def run(job, p=p, lid=L["id"]):
                r = p.build_cache(job)
                comp._sync_layer(comp.layer(lid)) if any(x["id"] == lid for x in comp.data["layers"]) else None
                comp.write()
                return r
            jobs.submit("cache", "cache", f"Caching {L['name']}", run)


def _open_comp(comp: Comp) -> dict:
    S.comp = comp
    _cache_jobs(comp)
    bus.publish("project_opened", name=comp.data["name"])
    return _state(comp)


class PathReqEarly(BaseModel):
    path: str


class ImportReq(BaseModel):
    path: str
    trimIn: int | None = None       # first source frame to keep (from the import dialog)
    trimOut: int | None = None      # last source frame to keep
    name: str | None = None
    aiModel: str | None = None


def _trim_of(req: "ImportReq") -> tuple[int, int] | None:
    if req.trimIn is None and req.trimOut is None:
        return None
    return int(req.trimIn or 0), int(req.trimOut if req.trimOut is not None else 10 ** 9)


@app.post("/video/probe")
def video_probe(req: PathReqEarly) -> dict:
    """Metadata for the import dialog (nothing is imported yet)."""
    return video.probe_cached(req.path).to_dict()


@app.get("/video/frame")
def video_frame(path: str, index: int = 0, w: int = 640) -> Response:
    """One frame of a video that has not been imported yet (import dialog preview and trim)."""
    return Response(video.preview_jpeg(path, index, w), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"})


@app.post("/video/stage")
async def video_stage(file: UploadFile = File(...)) -> dict:
    """Browser dev mode: save an uploaded video so the import dialog can preview it. Tauri passes real paths instead."""
    dest_dir = app_data_dir() / "uploads"
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / Path(file.filename or "upload.mp4").name
    with dest.open("wb") as f:
        while chunk := await file.read(8 * 1024 * 1024):
            f.write(chunk)
    info = await asyncio.get_running_loop().run_in_executor(None, lambda: video.probe_cached(dest))
    return {"path": str(dest), "info": info.to_dict()}


@app.post("/video/import")
def video_import(req: ImportReq) -> dict:
    """New composition from one video."""
    info = video.probe(req.path)
    comp = Comp.create(info, workspace_dir(), name=req.name, trim=_trim_of(req), layer_name=req.name, ai_model=req.aiModel)
    log.info("imported %s (%dx%d @ %.3f fps, %d frames)", info.filename, info.width, info.height, info.fps, info.frame_count)
    return _open_comp(comp)


@app.post("/video/upload")
async def video_upload(file: UploadFile = File(...), as_layer: bool = False) -> dict:
    """Browser dev mode only (Tauri passes real file paths). Streams the upload to disk in chunks."""
    dest_dir = app_data_dir() / "uploads"
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / Path(file.filename or "upload.mp4").name
    with dest.open("wb") as f:
        while chunk := await file.read(8 * 1024 * 1024):
            f.write(chunk)
    fn = (lambda: layer_add(AddLayerReq(path=str(dest)))) if as_layer and S.comp else (lambda: video_import(ImportReq(path=str(dest))))
    return await asyncio.get_running_loop().run_in_executor(None, fn)


class PathReq(BaseModel):
    path: str


@app.post("/project/open")
def project_open(req: PathReq) -> dict:
    return _open_comp(Comp.open(req.path))


@app.get("/project")
def project_get() -> dict:
    return _state(S.require_comp())


@app.post("/project/save")
def project_save() -> dict:
    c = S.require_comp()
    c.save()
    return {"saved": str(c.root / "comp.roto")}


@app.post("/project/save-as")
def project_save_as(req: PathReq) -> dict:
    return _open_comp(S.require_comp().save_as(req.path))


@app.post("/project/close")
def project_close() -> dict:
    if S.comp:
        S.comp.save()
    S.comp = None
    return {"closed": True}


# ---------------------------------------------------------------- composition / layers
@app.get("/comp")
def comp_get() -> dict:
    return S.require_comp().public_state()


class AddLayerReq(BaseModel):
    path: str
    start: int | None = None
    trimIn: int | None = None
    trimOut: int | None = None
    name: str | None = None
    aiModel: str | None = None


@app.post("/layers/add")
def layer_add(req: AddLayerReq) -> dict:
    """Add a video as a new layer on top of the stack (starts at `start`, default 0)."""
    comp = S.require_comp()
    info = video.probe(req.path)
    trim = None if req.trimIn is None and req.trimOut is None else (int(req.trimIn or 0), int(req.trimOut if req.trimOut is not None else 10 ** 9))
    lid = comp.add_layer(info, start=req.start, name=req.name, trim=trim, ai_model=req.aiModel)
    _cache_jobs(comp, lid)
    return _state(comp)


class LayerPatch(BaseModel):
    name: str | None = None
    start: int | None = None
    trimIn: int | None = None
    trimOut: int | None = None
    visible: bool | None = None
    opacity: float | None = None
    scale: float | None = None
    x: float | None = None
    y: float | None = None
    useMask: bool | None = None


@app.patch("/layers/{layer_id}")
def layer_patch(layer_id: str, req: LayerPatch) -> dict:
    comp = S.require_comp()
    comp.update_layer(layer_id, req.model_dump(exclude_none=True))
    if req.trimIn is not None or req.trimOut is not None:
        _cache_jobs(comp, layer_id)  # no-op when the range is already cached
    return _state(comp)


@app.post("/layers/{layer_id}/activate")
def layer_activate(layer_id: str) -> dict:
    comp = S.require_comp()
    comp.activate(layer_id)
    return _state(comp)


@app.delete("/layers/{layer_id}")
def layer_delete(layer_id: str) -> dict:
    comp = S.require_comp()
    comp.remove_layer(layer_id)
    return _state(comp)


class ReorderReq(BaseModel):
    order: list[str]


@app.post("/layers/reorder")
def layer_reorder(req: ReorderReq) -> dict:
    comp = S.require_comp()
    comp.reorder(req.order)
    return _state(comp)


class CompWorkArea(BaseModel):
    start: int
    end: int


@app.post("/comp/work-area")
def comp_work_area(req: CompWorkArea) -> dict:
    comp = S.require_comp()
    comp.set_work_area(req.start, req.end)
    return comp.public_state()


@app.get("/comp/render/{frame}")
def comp_render(frame: int, q: int = 100) -> Response:
    comp = S.require_comp()
    data, mt = compose.render_comp(comp, max(0, min(frame, comp.duration - 1)), max(10, min(100, q)))
    return Response(data, media_type=mt, headers={"Cache-Control": "private, max-age=600"})


class SettingsReq(BaseModel):
    maskSettings: dict | None = None
    exportSettings: dict | None = None
    workArea: dict | None = None
    engine: str | None = None
    aiModel: str | None = None


@app.patch("/project/settings")
def project_settings(req: SettingsReq) -> dict:
    p = S.require()
    with p.lock:
        if req.maskSettings is not None:
            p.data["maskSettings"] = merge_settings({**p.data["maskSettings"], **req.maskSettings})
        if req.exportSettings is not None:
            p.data["exportSettings"] = {**DEFAULT_EXPORT, **p.data["exportSettings"], **req.exportSettings}
        if req.workArea is not None:
            a, b = int(req.workArea["in"]), int(req.workArea["out"])
            a, b = max(0, min(a, b)), min(p.n_frames - 1, max(a, b))
            p.data["workArea"] = {"in": a, "out": b}
            c = S.comp
            lid = next((L["id"] for L in c.data["layers"] if c.projects[L["id"]] is p), None) if c else None
            if lid:
                c.update_layer(lid, {"trimIn": a, "trimOut": b})
        if req.engine is not None:
            if req.engine not in ("auto", "opencv", "onnx", "torch"):
                raise MediaError("Unknown engine.")
            p.data["engine"] = req.engine
        if req.aiModel is not None:
            if req.aiModel not in ("auto", "general", "anime", "fast"):
                raise MediaError("Unknown AI model.")
            p.data["aiModel"], p.data["aiModelResolved"] = req.aiModel, None
        p.touch()
    return {"maskSettings": p.data["maskSettings"], "exportSettings": p.data["exportSettings"],
            "workArea": p.data["workArea"], "engine": p.data["engine"], "aiModel": p.data["aiModel"]}


@app.get("/timeline")
def timeline() -> dict:
    return S.require().timeline(LOW_CONFIDENCE)


# ---------------------------------------------------------------- frames / rendering
@app.get("/render/{frame}")
def render(frame: int, mode: str = "original", q: int = 100) -> Response:
    p = S.require()
    frame = max(0, min(frame, p.n_frames - 1))
    data, mt = ops.render(p, frame, mode, max(10, min(100, q)))
    return Response(data, media_type=mt, headers={"Cache-Control": "private, max-age=600"})


@app.get("/thumbnail")
def thumbnail() -> Response:
    import cv2
    p = S.require()
    img = p.frame_bgr(p.n_frames // 10)
    s = 320 / max(img.shape[:2])
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), media_type="image/jpeg")


# ---------------------------------------------------------------- segmentation
class AnalyzeReq(BaseModel):
    frame: int | None = None


@app.post("/segmentation/analyze")
def seg_analyze(req: AnalyzeReq) -> dict:
    return ops.analyze(S.require(), req.frame)


class SelectReq(BaseModel):
    frame: int
    mode: Literal["replace", "add", "subtract"] = "add"
    points: list[list[float]] | None = None
    negative: list[list[float]] | None = None
    box: list[int] | None = None
    object_id: str | None = None


@app.post("/segmentation/select")
def seg_select(req: SelectReq) -> dict:
    p = S.require()
    _cancel_auto(p)
    ops.select_object(p, req.frame, req.mode, req.points, req.negative, req.box, req.object_id)
    return {"ok": True, "frame": req.frame}


class AutoReq(BaseModel):
    frame: int
    object_ids: list[str] | None = None


@app.post("/rotoscope/auto")
def rotoscope_auto(req: AutoReq) -> dict:
    p = S.require()
    _cancel_auto(p)
    job = jobs.submit("process", "auto-roto", "Auto roto", lambda j: ops.auto_roto(p, req.frame, req.object_ids, j))
    return job.public()


def _cancel_auto(p: Project) -> None:
    """A manual edit always wins: stop this layer's background AI updates (and wait until they have really stopped,
    so they cannot overwrite what the user is about to do)."""
    jobs.cancel_group(("auto-update", "ai-refine"), group=str(p.root))


# ---------------------------------------------------------------- manual mask tools
class StrokeReq(BaseModel):
    frame: int
    mode: Literal["add", "remove"]
    points: list[list[float]] = Field(min_length=1)
    size: float = 40
    hardness: float = 0.7
    strength: float = 1.0


@app.post("/mask/stroke")
def mask_stroke(req: StrokeReq) -> dict:
    _cancel_auto(S.require())
    ops.brush(S.require(), req.frame, req.mode, req.points, req.size, req.hardness, req.strength)
    return {"ok": True}


class LassoReq(BaseModel):
    frame: int
    polygon: list[list[float]] = Field(min_length=3)
    mode: Literal["add", "remove"] = "add"
    ai_refine: bool = True
    wait: bool = False  # True = refine before answering (scripts/tests); the UI uses the instant + background path


@app.post("/mask/lasso")
def mask_lasso(req: LassoReq) -> dict:
    p = S.require()
    _cancel_auto(p)
    mode = "add" if req.mode == "add" else "subtract"
    if req.ai_refine and not req.wait:
        # the polygon is applied instantly; snapping it to the real edges happens in the background
        pending = ops.lasso(p, req.frame, req.polygon, mode, True, defer_refine=True)
        job = jobs.submit("auto", "ai-refine", "Snapping to edges", lambda j: ops.lasso_refine(p, req.frame, pending, j), group=str(p.root))
        return {"ok": True, "refining": job.id}
    ops.lasso(p, req.frame, req.polygon, mode, req.ai_refine)
    return {"ok": True}


class AutoUpdateReq(BaseModel):
    frame: int
    forward: int = 30
    backward: int = 0


@app.post("/mask/auto-update")
def mask_auto_update(req: AutoUpdateReq) -> dict:
    """After a manual edit: carry the correction to the neighbouring frames in the background (cancellable)."""
    p = S.require()
    if not p.masks.has(req.frame) or (req.forward <= 0 and req.backward <= 0):
        return {"skipped": True}
    jobs.cancel_group(("auto-update",), group=str(p.root))  # a newer edit supersedes the previous update
    job = jobs.submit("auto", "auto-update", "Updating nearby frames",
                      lambda j: ops.auto_update(p, req.frame, req.forward, req.backward, j), group=str(p.root))
    return job.public()


class FrameReq(BaseModel):
    frame: int


@app.post("/mask/clear")
def mask_clear(req: FrameReq) -> dict:
    _cancel_auto(S.require())
    ops.clear_mask(S.require(), req.frame)
    return {"ok": True}


@app.post("/mask/undo")
def mask_undo() -> dict:
    _cancel_auto(S.require())
    r = S.require().masks.undo()
    S.require().touch()
    return {"label": r[0] if r else None, "frames": r[1] if r else []}


@app.post("/mask/redo")
def mask_redo() -> dict:
    _cancel_auto(S.require())
    r = S.require().masks.redo()
    S.require().touch()
    return {"label": r[0] if r else None, "frames": r[1] if r else []}


# ---------------------------------------------------------------- tracking
class PropReq(BaseModel):
    frame: int
    direction: Literal["forward", "backward", "both"] = "both"
    recalc: bool = False


@app.post("/tracking/propagate")
def tracking_propagate(req: PropReq) -> dict:
    p = S.require()
    _cancel_auto(p)
    if not p.masks.has(req.frame):
        raise MediaError("There is no mask on this frame to propagate.",
                         ["Run Auto Roto or paint a mask with Brush or Lasso first"])
    title = "Recalculate from frame " + str(req.frame) if req.recalc else f"Propagate {req.direction}"

    def run(job):
        r = ops.propagate(p, req.frame, req.direction, job, req.recalc)
        bus.publish("tracking_completed", **r)
        return r

    return jobs.submit("process", "propagate", title, run).public()


# ---------------------------------------------------------------- cache / jobs / export
@app.post("/cache/build")
def cache_build() -> dict:
    p = S.require()
    return jobs.submit("cache", "cache", "Caching frames", lambda j: p.build_cache(j)).public()


@app.get("/jobs")
def jobs_list() -> list[dict]:
    return jobs.list()


@app.post("/jobs/{job_id}/cancel")
def job_cancel(job_id: str) -> dict:
    return {"cancelled": jobs.cancel(job_id)}


class ExportReq(BaseModel):
    kind: Literal["mp4", "mov-alpha", "after-effects", "depth"]
    output: str
    settings: dict = {}


@app.post("/export")
def export_start(req: ExportReq) -> dict:
    comp = S.require_comp()
    p = comp.project()
    s = {**comp.data["exportSettings"], **req.settings}
    if req.kind == "mp4":
        exporter.hex_to_bgr(s.get("background", "#FFFFFF"))  # validate BEFORE persisting anything
    if not 0.1 <= float(s.get("scale", 1.0)) <= 4.0:
        raise MediaError("The export scale must be between 10% and 400%.")
    if req.kind == "depth" and depthmod.depth_model_path() is None:
        raise MediaError("The depth model is not installed.", ["Run scripts\\setup.ps1", "Or see README"], stage="depth-model")
    comp.data["exportSettings"] = {**comp.data["exportSettings"], **req.settings, "kind": req.kind}
    comp.write()
    name = Path(req.output).name

    def run(job):
        if req.kind == "after-effects":
            return compose.export_after_effects(comp, s, req.output, job)
        if req.kind == "depth":
            return compose.export_depth(comp, s, req.output, job)
        return compose.export_comp(comp, req.kind, s, req.output, job)

    return jobs.submit("export", req.kind, name, run).public()


class LogReq(BaseModel):
    path: str


@app.post("/system/open-log")
def open_log(req: LogReq) -> dict:
    """Open a log file in the OS default viewer. Only files inside the Zelf-Brush log folder are allowed."""
    target = Path(req.path).resolve()
    if log_dir().resolve() not in target.parents and target != log_dir().resolve():
        raise MediaError("That file is not a Zelf-Brush log.")
    if not target.exists():
        raise MediaError("The log file does not exist.")
    if os.name == "nt":
        os.startfile(str(target))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open" if sys.platform != "darwin" else "open", str(target)])
    return {"opened": str(target)}
