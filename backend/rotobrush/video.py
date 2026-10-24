"""FFmpeg integration: probing, streaming decode, proxy frame extraction. Never loads a whole video into RAM."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from . import fsutil
from .config import CREATE_NO_WINDOW, binary_candidates, log_dir

log = logging.getLogger("rotobrush.ffmpeg")
SUPPORTED = {".mp4", ".mkv", ".mov"}


class MediaError(Exception):
    """User-facing failure. Carries the real technical context so the UI never has to guess."""

    def __init__(self, message: str, hints: list[str] | None = None, *, stage: str | None = None,
                 exit_code: int | None = None, detail: str | None = None, log: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hints = hints or []
        self.stage, self.exit_code, self.detail, self.log = stage, exit_code, detail, log

    def to_dict(self) -> dict:
        return {"message": self.message, "hints": self.hints, "stage": self.stage,
                "exit_code": self.exit_code, "detail": self.detail, "log": self.log}


_TEXT = {"text": True, "encoding": "utf-8", "errors": "replace"}  # Windows default codepage would mangle ffmpeg output
_resolved: dict[str, str] = {}
_caps: dict[str, set[str]] = {}


def _check_binary(path: str) -> tuple[bool, str]:
    """A tool 'exists' only if it actually runs. Catches broken shims, wrong-arch binaries, missing DLLs."""
    try:
        r = subprocess.run([path, "-version"], capture_output=True, timeout=20, creationflags=CREATE_NO_WINDOW, **_TEXT)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    first = (r.stdout or "").splitlines()[0] if (r.stdout or "").strip() else ""
    if r.returncode != 0 or "version" not in first.lower():
        return False, f"exit code {r.returncode}: {((r.stderr or r.stdout) or '').strip()[:200]}"
    return True, first


def resolve_binary(name: str, candidates: list[str] | None = None) -> str:
    use_cache = candidates is None
    if use_cache and name in _resolved:
        return _resolved[name]
    cands = binary_candidates(name) if candidates is None else candidates
    rejected: list[str] = []
    for c in cands:
        ok, info = _check_binary(c)
        if ok:
            log.info("using %s: %s (%s)", name, c, info)
            if use_cache:
                _resolved[name] = c
            return c
        log.warning("%s candidate rejected: %s -> %s", name, c, info)
        rejected.append(f"{c}: {info}")
    if rejected:
        raise MediaError(f"{name} was found but cannot be run.",
                         ["Reinstall FFmpeg (winget install Gyan.FFmpeg)", "Or put a working ffmpeg.exe and ffprobe.exe in backend\\bin"],
                         stage="ffmpeg-check", detail="; ".join(rejected), log=str(log_dir() / "ffmpeg.log"))
    raise MediaError(f"{name} was not found.",
                     ["Install FFmpeg (winget install Gyan.FFmpeg) and restart Zelf-Brush",
                      "Or put ffmpeg.exe and ffprobe.exe in backend\\bin"], stage="ffmpeg-check")


def binary(name: str) -> str:
    return resolve_binary(name)


def has_encoder(name: str) -> bool:
    if "enc" not in _caps:
        r = subprocess.run([binary("ffmpeg"), "-hide_banner", "-encoders"], capture_output=True, creationflags=CREATE_NO_WINDOW, **_TEXT)
        _caps["enc"] = {ln.split()[1] for ln in (r.stdout or "").splitlines() if ln.startswith(" V") and len(ln.split()) > 1}
    return name in _caps["enc"]


def run_logged(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    log.info("run: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, creationflags=CREATE_NO_WINDOW, **kw)


def first_error_line(stderr: str) -> str:
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if any(k in ln.lower() for k in ("error", "invalid", "unable", "failed", "no such", "not found", "unknown")):
            return ln[:300]
    return (lines[-1] if lines else "no error output")[:300]


@dataclass
class VideoInfo:
    path: str
    filename: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str
    size_bytes: int
    has_audio: bool
    pix_fmt: str
    audio_codec: str = ""
    container: str = ""
    bitrate: int = 0
    streams: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def probe(path: str | Path) -> VideoInfo:
    p = Path(path)
    if p.suffix.lower() not in SUPPORTED:
        raise MediaError(f"{p.suffix or 'This file type'} is not supported.", ["Use an .mp4, .mkv or .mov file"], stage="probe")
    if not p.exists():
        raise MediaError(f"File not found: {p}", stage="probe")
    cmd = [binary("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(p)]
    r = run_logged(cmd, **_TEXT)
    if r.returncode != 0:
        err = first_error_line(r.stderr)
        log.error("ffprobe exit %s for %s: %s", r.returncode, p, (r.stderr or "").strip()[:1000])
        low = (r.stderr or "").lower()
        damaged = any(k in low for k in ("moov atom not found", "invalid data found", "partial file", "end of file"))
        raise MediaError(
            "This file looks incomplete or damaged." if damaged else "The video could not be analysed.",
            ["Check that the file plays in a media player", "Try re-exporting or re-downloading it"] if damaged
            else ["See ffmpeg.log for the full ffprobe output"],
            stage="probe", exit_code=r.returncode, detail=err, log=str(log_dir() / "ffmpeg.log"))
    try:
        data = json.loads(r.stdout or "{}")
    except ValueError:
        raise MediaError("ffprobe returned unreadable output.", stage="probe", log=str(log_dir() / "ffmpeg.log"))
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    if v is None:
        raise MediaError("No video stream was found in this file.", stage="probe")
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = rot or int(float(v.get("tags", {}).get("rotate", 0) or 0))
    if abs(rot) % 180 == 90:
        w, h = h, w
    fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            fr = float(Fraction(v.get(key, "0/0")))
        except (ZeroDivisionError, ValueError):
            fr = 0.0
        if fr > 0:
            fps = fr
            break
    fps = fps or 24.0
    fmt = data.get("format", {})
    dur = float(v.get("duration") or fmt.get("duration") or 0)
    nb = v.get("nb_frames")
    frames = int(nb) if nb and str(nb).isdigit() else int(round(dur * fps))
    if frames <= 0:
        raise MediaError("The video has no readable frames.", stage="probe")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    info = VideoInfo(
        path=str(p.resolve()), filename=p.name, width=w, height=h, fps=fps,
        frame_count=frames, duration=dur or frames / fps, codec=v.get("codec_name", "unknown"),
        size_bytes=p.stat().st_size, has_audio=audio is not None, pix_fmt=v.get("pix_fmt", ""),
        audio_codec=(audio or {}).get("codec_name", ""), container=fmt.get("format_name", ""),
        bitrate=int(fmt.get("bit_rate") or 0), streams=len(streams))
    log.info("[INPUT] %s container=%s video=%s %dx%d %.3ffps %.2fs pix_fmt=%s audio=%s bitrate=%s streams=%d",
             p.name, info.container, info.codec, w, h, fps, info.duration, info.pix_fmt,
             info.audio_codec or "none", info.bitrate or "n/a", info.streams)
    return info


def _seek(info: VideoInfo, frame: int) -> list[str]:
    if frame <= 0:
        return []
    return ["-ss", f"{(frame - 0.5) / info.fps:.6f}"]


def decode_frame(info: VideoInfo, idx: int, w: int, h: int) -> np.ndarray | None:
    """Decode one frame (BGR uint8, HxWx3) at the given size."""
    cmd = [binary("ffmpeg"), "-v", "error", *_seek(info, idx), "-i", info.path, "-frames:v", "1",
           "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    r = subprocess.run(cmd, capture_output=True, creationflags=CREATE_NO_WINDOW)
    if len(r.stdout) < w * h * 3:
        log.warning("decode_frame %d failed: %s", idx, r.stderr.decode(errors="ignore")[:300])
        return None
    return np.frombuffer(r.stdout[: w * h * 3], np.uint8).reshape(h, w, 3).copy()


def read_frames(info: VideoInfo, w: int, h: int, start: int = 0, count: int | None = None,
                pix_fmt: str = "bgr24", fps_out: float | None = None) -> Iterator[np.ndarray]:
    """Stream frames one by one through a pipe (bounded memory: one frame at a time)."""
    channels = {"bgr24": 3, "rgb24": 3, "bgra": 4}[pix_fmt]
    cmd = [binary("ffmpeg"), "-v", "error", *_seek(info, start), "-i", info.path]
    vf = f"fps={fps_out:.6f},scale={w}:{h}" if fps_out else f"scale={w}:{h}"
    cmd += ["-vf", vf, *(["-frames:v", str(count)] if count is not None else []), "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    err = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, creationflags=CREATE_NO_WINDOW)
    size = w * h * channels
    try:
        assert proc.stdout is not None
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, channels).copy()
    finally:
        proc.kill()
        proc.wait()
        err.seek(0)
        msg = err.read().decode(errors="ignore").strip()
        err.close()
        if msg:
            log.warning("read_frames stderr: %s", msg[:500])


def extract_proxy_frames(info: VideoInfo, out_dir: Path, w: int, h: int,
                         on_progress: Callable[[int], None] | None = None,
                         cancelled: Callable[[], bool] | None = None,
                         first: int = 0, count: int | None = None) -> int:
    """Extract proxy JPEGs for source frames first..first+count-1 (all the way to the end when count is None).
    Files are named by SOURCE frame index (000123.jpg). Returns how many frames were written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    pattern = str(out_dir / "%06d.jpg")
    base = [binary("ffmpeg"), "-v", "error", "-nostats", "-progress", "pipe:1", *_seek(info, first), "-i", info.path]
    tail = ["-vf", f"scale={w}:{h}", *(["-frames:v", str(count)] if count else []), "-q:v", "3", "-start_number", str(first), pattern]
    for sync in (["-fps_mode", "passthrough"], ["-vsync", "0"]):
        err = tempfile.TemporaryFile()
        proc = subprocess.Popen(base + sync + tail, stdout=subprocess.PIPE, stderr=err, text=True, encoding="utf-8", errors="replace",
                                creationflags=CREATE_NO_WINDOW)
        assert proc.stdout is not None
        for line in proc.stdout:
            if cancelled and cancelled():
                proc.kill()
                proc.wait()
                err.close()
                raise InterruptedError("cancelled")
            if line.startswith("frame=") and on_progress:
                try:
                    on_progress(int(line.split("=")[1]))
                except ValueError:
                    pass
        proc.wait()
        err.seek(0)
        msg = err.read().decode(errors="ignore")
        err.close()
        if proc.returncode == 0:
            break
        log.error("frame extraction failed (%s): %s", sync, msg[:800])
        if "Unrecognized option" not in msg and "Option not found" not in msg:
            raise MediaError("Frames could not be extracted from this video.",
                             ["The file may be damaged", "See ffmpeg.log for details"])
    return len(list(out_dir.glob("*.jpg")))


_probe_cache: dict[tuple[str, float], VideoInfo] = {}


def probe_cached(path: str | Path) -> VideoInfo:
    """probe() with a tiny cache keyed by path + mtime (the import dialog asks for many preview frames of one file)."""
    p = Path(path)
    key = (str(p), p.stat().st_mtime if p.exists() else 0.0)
    if key not in _probe_cache:
        if len(_probe_cache) > 16:
            _probe_cache.clear()
        _probe_cache[key] = probe(p)
    return _probe_cache[key]


def preview_jpeg(path: str | Path, index: int, width: int = 640) -> bytes:
    """One frame of a not-yet-imported video as JPEG, for the import dialog."""
    import cv2

    info = probe_cached(path)
    index = max(0, min(index, info.frame_count - 1))
    w = max(64, min(width, info.width))
    h = max(2, int(round(w * info.height / info.width / 2)) * 2)
    img = decode_frame(info, index, w, h)
    if img is None:
        raise MediaError("That frame could not be decoded.", ["The video may be damaged near this point"], stage="preview")
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise MediaError("The preview could not be encoded.", stage="preview")
    return buf.tobytes()


def make_intermediate(info: VideoInfo, out: Path, on_progress: Callable[[int], None] | None = None,
                      cancelled: Callable[[], bool] | None = None) -> Path:
    """Re-encode the video stream into a reliable constant-frame-rate ProRes (or FFV1) intermediate."""
    codec = ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"] if has_encoder("prores_ks") \
        else ["-c:v", "ffv1", "-level", "3"]
    tmp = out.with_name(out.stem + ".part.mov")
    tmp.unlink(missing_ok=True)
    cmd = [binary("ffmpeg"), "-y", "-v", "error", "-nostats", "-progress", "pipe:1", "-i", info.path, "-map", "0:v:0", "-an",
           "-vf", f"fps={info.fps:.6f}", *codec, str(tmp)]
    log.info("make_intermediate: %s", " ".join(cmd))
    err = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True, encoding="utf-8", errors="replace",
                            creationflags=CREATE_NO_WINDOW)
    assert proc.stdout is not None
    for line in proc.stdout:
        if cancelled and cancelled():
            proc.kill()
            proc.wait()
            err.close()
            tmp.unlink(missing_ok=True)
            raise InterruptedError("cancelled")
        if line.startswith("frame=") and on_progress:
            try:
                on_progress(int(line.split("=")[1]))
            except ValueError:
                pass
    proc.wait()
    err.seek(0)
    msg = err.read().decode(errors="ignore")
    err.close()
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        log.error("intermediate transcode failed rc=%s: %s", proc.returncode, msg[:1500])
        raise MediaError("The video could not be converted to a stable format.", ["See ffmpeg.log for details"],
                         stage="transcode", exit_code=proc.returncode, detail=first_error_line(msg),
                         log=str(log_dir() / "ffmpeg.log"))
    fsutil.replace(tmp, out)
    return out
