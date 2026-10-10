"""HTTP API + Server-Sent-Events for the desktop UI. Bound to 127.0.0.1 and protected by a per-launch token."""
from __future__ import annotations

import asyncio
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

from . import __version__, export as exporter, ops, video
from .config import CREATE_NO_WINDOW, LOW_CONFIDENCE, app_data_dir, log_dir, workspace_dir
from .engines import get_engine, list_engines
from .engines.base import EngineError
from .events import bus
from .gpu import gpu_info
from .jobs import JobManager
from .logs import setup_logging
from .matte import DEFAULT_MATTE, merge_settings
from .project import DEFAULT_EXPORT, Project
from .video import MediaError

setup_logging()
log = logging.getLogger("rotobrush")
TOKEN = os.environ.get("ROTOBRUSH_TOKEN", "")

@asynccontextmanager
async def lifespan(_: FastAPI):
    bus.bind(asyncio.get_running_loop())
    threading.Thread(target=_autosave_loop, daemon=True, name="autosave").start()
    log.info("RotoBrush backend %s started", __version__)
    yield


app = FastAPI(title="RotoBrush backend", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["tauri://localhost", "http://tauri.localhost", "https://tauri.localhost",
                   "http://localhost:1420", "http://127.0.0.1:1420"],
    allow_methods=["*"], allow_headers=["*"],
)
jobs = JobManager()


class Session:
    project: Project | None = None

    def require(self) -> Project:
        if self.project is None:
            raise MediaError("No project is open.", ["Import a video or open a project first"])
        return self.project


S = Session()


# ---------------------------------------------------------------- plumbing
@app.middleware("http")
async def auth(request: Request, call_next):
    if TOKEN and request.method != "OPTIONS" and request.url.path != "/health":
        if request.headers.get("x-token") != TOKEN and request.query_params.get("token") != TOKEN:
            return JSONResponse({"detail": {"message": "Unauthorized", "hints": []}}, status_code=401)
    return await call_next(request)


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
            p = S.project
            if p and p.autosave():
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
def _open_project(p: Project) -> dict:
    S.project = p
    if not p.cache_complete():
        jobs.submit("cache", "cache", "Caching frames", lambda job: p.build_cache(job))
    bus.publish("project_opened", name=p.meta["name"])
    return p.public_state()


class ImportReq(BaseModel):
    path: str


@app.post("/video/import")
def video_import(req: ImportReq) -> dict:
    info = video.probe(req.path)
    p = Project.create(info, workspace_dir())
    log.info("imported %s (%dx%d @ %.3f fps, %d frames)", info.filename, info.width, info.height, info.fps, info.frame_count)
    return _open_project(p)


@app.post("/video/upload")
async def video_upload(file: UploadFile = File(...)) -> dict:
    """Browser dev mode only (Tauri passes real file paths). Streams the upload to disk in chunks."""
    dest_dir = app_data_dir() / "uploads"
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / Path(file.filename or "upload.mp4").name
    with dest.open("wb") as f:
        while chunk := await file.read(8 * 1024 * 1024):
            f.write(chunk)
    return await asyncio.get_running_loop().run_in_executor(None, lambda: video_import(ImportReq(path=str(dest))))


class PathReq(BaseModel):
    path: str


@app.post("/project/open")
def project_open(req: PathReq) -> dict:
    return _open_project(Project.open(req.path))


@app.get("/project")
def project_get() -> dict:
    return S.require().public_state()


@app.post("/project/save")
def project_save() -> dict:
    p = S.require()
    p.save()
    return {"saved": str(p.root / "project.roto")}


@app.post("/project/save-as")
def project_save_as(req: PathReq) -> dict:
    np_ = S.require().save_as(req.path)
    return _open_project(np_)


@app.post("/project/close")
def project_close() -> dict:
    p = S.project
    if p:
        p.save()
    S.project = None
    return {"closed": True}


class SettingsReq(BaseModel):
    maskSettings: dict | None = None
    exportSettings: dict | None = None
    workArea: dict | None = None
    engine: str | None = None


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
            p.data["workArea"] = {"in": max(0, min(a, b)), "out": min(p.n_frames - 1, max(a, b))}
        if req.engine is not None:
            if req.engine not in ("auto", "opencv", "onnx", "torch"):
                raise MediaError("Unknown engine.")
            p.data["engine"] = req.engine
        p.touch()
    return {"maskSettings": p.data["maskSettings"], "exportSettings": p.data["exportSettings"],
            "workArea": p.data["workArea"], "engine": p.data["engine"]}


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
    ops.select_object(p, req.frame, req.mode, req.points, req.negative, req.box, req.object_id)
    return {"ok": True, "frame": req.frame}


class AutoReq(BaseModel):
    frame: int
    object_ids: list[str] | None = None


@app.post("/rotoscope/auto")
def rotoscope_auto(req: AutoReq) -> dict:
    p = S.require()
    job = jobs.submit("process", "auto-roto", "Auto roto", lambda j: ops.auto_roto(p, req.frame, req.object_ids, j))
    return job.public()


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
    ops.brush(S.require(), req.frame, req.mode, req.points, req.size, req.hardness, req.strength)
    return {"ok": True}


class LassoReq(BaseModel):
    frame: int
    polygon: list[list[float]] = Field(min_length=3)
    mode: Literal["add", "remove"] = "add"
    ai_refine: bool = True


@app.post("/mask/lasso")
def mask_lasso(req: LassoReq) -> dict:
    ops.lasso(S.require(), req.frame, req.polygon, "add" if req.mode == "add" else "subtract", req.ai_refine)
    return {"ok": True}


class FrameReq(BaseModel):
    frame: int


@app.post("/mask/clear")
def mask_clear(req: FrameReq) -> dict:
    ops.clear_mask(S.require(), req.frame)
    return {"ok": True}


@app.post("/mask/undo")
def mask_undo() -> dict:
    r = S.require().masks.undo()
    S.require().touch()
    return {"label": r[0] if r else None, "frames": r[1] if r else []}


@app.post("/mask/redo")
def mask_redo() -> dict:
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
    kind: Literal["mp4", "mov-alpha", "after-effects"]
    output: str
    settings: dict = {}


@app.post("/export")
def export_start(req: ExportReq) -> dict:
    p = S.require()
    s = {**p.data["exportSettings"], **req.settings}
    if req.kind == "mp4":
        exporter.hex_to_bgr(s.get("background", "#FFFFFF"))  # validate BEFORE persisting anything
    if not 0.1 <= float(s.get("scale", 1.0)) <= 4.0:
        raise MediaError("The export scale must be between 10% and 400%.")
    p.data["exportSettings"] = {**p.data["exportSettings"], **req.settings, "kind": req.kind}
    p.touch()
    name = Path(req.output).name

    def run(job):
        if req.kind == "after-effects":
            return exporter.export_after_effects(p, s, req.output, job)
        return exporter.export_video(p, req.kind, s, req.output, job)

    return jobs.submit("export", req.kind, name, run).public()


class LogReq(BaseModel):
    path: str


@app.post("/system/open-log")
def open_log(req: LogReq) -> dict:
    """Open a log file in the OS default viewer. Only files inside the RotoBrush log folder are allowed."""
    target = Path(req.path).resolve()
    if log_dir().resolve() not in target.parents and target != log_dir().resolve():
        raise MediaError("That file is not a RotoBrush log.")
    if not target.exists():
        raise MediaError("The log file does not exist.")
    if os.name == "nt":
        os.startfile(str(target))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open" if sys.platform != "darwin" else "open", str(target)])
    return {"opened": str(target)}
