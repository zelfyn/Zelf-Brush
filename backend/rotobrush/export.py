"""Export: MP4 (composited on a background), MOV with real alpha, After Effects package (footage + alpha + JSX).

Frames are streamed one at a time from the ORIGINAL source at export resolution. The proxy-resolution mask is
upsampled and snapped to full-resolution edges with a guided filter before compositing.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import cv2
import numpy as np

from . import matte, video
from .config import CREATE_NO_WINDOW, log_dir
from .gpu import gpu_info
from .jobs import Cancelled, Job
from .project import Project
from .video import MediaError, first_error_line

log = logging.getLogger("rotobrush.export")

ALPHA_PIXFMTS = ("yuva", "argb", "rgba", "bgra", "ya8", "gbrap")
MOV_CODECS = {
    "prores4444": (["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-vendor", "apl0"], "prores"),
    "qtrle": (["-c:v", "qtrle", "-pix_fmt", "argb"], "qtrle"),
    "png": (["-c:v", "png", "-pix_fmt", "rgba"], "png"),
}


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.strip().lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", h):
        raise MediaError("The background colour must look like #RRGGBB.")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def _even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)


def export_size(p: Project, scale: float) -> tuple[int, int]:
    return _even(p.meta["width"] * scale), _even(p.meta["height"] * scale)


def frame_range(p: Project, which: str) -> tuple[int, int]:
    if which == "work":
        wa = p.data["workArea"]
        return int(wa["in"]), int(wa["out"])
    return 0, p.n_frames - 1


def _alpha_full(p: Project, idx: int, frame: np.ndarray, size: tuple[int, int], settings: dict) -> np.ndarray:
    """Upsample the proxy mask to export size and refine it against the full-resolution frame."""
    w, h = size
    raw = p.masks.get(idx)
    if (w, h) != (raw.shape[1], raw.shape[0]):
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_CUBIC)
    scale = w / float(p.proxy[0])
    a = matte.refine_alpha(raw, frame, settings, scale=max(1.0, scale))
    if (w, h) != p.proxy:  # upsampled edges are soft: re-snap them to full-res detail
        guide = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        r = max(2, int(2 * scale))
        a = np.clip(matte.guided_filter(guide, a, r, 1e-4), 0, 1)
    return a


def xlog(label: str, value: object = "") -> None:
    log.info("[EXPORT] %s: %s", label, value)


def _f(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _mp4_plans(s: dict) -> list[tuple[str, list[str]]]:
    """Encoder attempts in order: NVENC (if wanted/available) then software. Software is always last."""
    h265 = s.get("codec", "h264") == "h265"
    hw, crf, bitrate = s.get("hwEncode", "auto"), int(s.get("crf", 18)), s.get("bitrate")
    tag = ["-tag:v", "hvc1"] if h265 else []
    plans: list[tuple[str, list[str]]] = []
    nv = "hevc_nvenc" if h265 else "h264_nvenc"
    if hw == "nvenc" or (hw == "auto" and gpu_info()["detected"] and video.has_encoder(nv)):
        rate = ["-b:v", f"{int(bitrate)}k"] if bitrate else ["-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
        plans.append((nv, ["-c:v", nv, "-preset", "p5", *rate, "-pix_fmt", "yuv420p", *tag]))
    cpu = "libx265" if h265 else "libx264"
    if video.has_encoder(cpu):
        rate = ["-b:v", f"{int(bitrate)}k"] if bitrate else ["-crf", str(crf)]
        plans.append((cpu, ["-c:v", cpu, "-preset", "medium", *rate, "-pix_fmt", "yuv420p", *tag]))
    elif not plans:
        raise MediaError(f"This FFmpeg build has no {cpu} encoder.",
                         ["Install a full FFmpeg build (winget install Gyan.FFmpeg)"], stage="encoder-check")
    return plans


def _save_log(exp_id: str, cmd: list[str], stderr: str) -> str:
    path = log_dir() / f"export_{exp_id}.log"
    path.write_text("COMMAND:\n" + " ".join(cmd) + "\n\nFFMPEG STDERR:\n" + stderr, encoding="utf-8")
    return str(path)


def _encode_once(p: Project, kind: str, label: str, vargs: list[str], audio: bool, a0: int, count: int,
                 size: tuple[int, int], settings: dict, tmp_out: Path, tmpdir: Path, exp_id: str,
                 job: Job | None, span: tuple[float, float]) -> int:
    w, h = size
    fps = p.fps
    stab, decon = float(settings.get("temporal", 0)), bool(settings.get("decontaminate", True))
    sscale = max(1.0, w / float(p.proxy[0]))
    if kind == "mp4":
        bg = hex_to_bgr(settings["_background"])
        pix_in, extra_in = "bgr24", []
        tail = ["-map", "0:v", *vargs]
        if audio:
            extra_in = ["-ss", f"{a0 / fps:.6f}", "-t", f"{count / fps:.6f}", "-i", p.source_path]
            tail += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        tail += ["-movflags", "+faststart"]
    else:
        pix_in, extra_in, tail, bg = "bgra", [], list(vargs), (0, 0, 0)
    cmd = [video.binary("ffmpeg"), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", pix_in, "-s", f"{w}x{h}",
           "-r", f"{fps:.6f}", "-i", "-", *extra_in, *tail, str(tmp_out)]
    xlog("FFmpeg command", " ".join(cmd))
    err_path = tmpdir / f"ffmpeg_{label}.stderr"
    with open(err_path, "wb") as ef:  # child keeps its own handle; ours is closed so Windows can delete the file later
        enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=ef, creationflags=CREATE_NO_WINDOW)
    done, broken = 0, False
    prev_alpha = prev_frame = None
    try:
        assert enc.stdin is not None
        try:
            for n, frame in enumerate(video.read_frames(p.info, w, h, a0, count)):
                if job:
                    job.update(span[0] + (span[1] - span[0]) * n / max(1, count), f"Frame {n + 1}/{count}")
                a = _alpha_full(p, a0 + n, frame, (w, h), settings)
                if prev_alpha is not None and stab > 0:
                    a = matte.stabilize(prev_alpha, prev_frame, frame, a, stab)
                prev_alpha, prev_frame = a, frame
                buf = matte.composite(frame, a, bg, decon, sscale) if kind == "mp4" else matte.foreground_bgra(frame, a, decon, sscale)
                enc.stdin.write(buf.tobytes())
                done += 1
            enc.stdin.close()
        except OSError:  # BrokenPipeError on Linux, often errno 22 on Windows: the encoder died early
            broken = True  # its exit code and stderr tell us why
            try:
                enc.stdin.close()
            except OSError:
                pass
        rc = enc.wait(timeout=600)  # block until FFmpeg has really finished writing and closed the file
    except BaseException:
        enc.kill()
        enc.wait()
        raise
    stderr = err_path.read_text(errors="replace") if err_path.exists() else ""
    xlog("FFmpeg exit code", rc)
    if rc != 0 or broken:
        logp = _save_log(exp_id, cmd, stderr)
        xlog("FFmpeg stderr", first_error_line(stderr))
        raise MediaError(f"Export failed during video encoding: FFmpeg exited with code {rc}.",
                         ["Open the log for the full FFmpeg output", "Try CPU-only encoding in the export dialog"],
                         stage="encoding", exit_code=rc, detail=f"{label}: {first_error_line(stderr)}", log=logp)
    if done < count - 2:
        logp = _save_log(exp_id, cmd, stderr)
        raise MediaError(f"Export failed while reading the source video: only {done} of {count} frames could be decoded.",
                         ["The video may be damaged near that point", "Export the work area that decodes correctly"],
                         stage="decoding", detail=f"{done}/{count} frames", log=logp)
    if stderr.strip():
        xlog("FFmpeg warnings", first_error_line(stderr))
    return done


def export_video(p: Project, kind: str, s: dict, out_path: str | Path, job: Job | None = None,
                 progress_span: tuple[float, float] = (0.0, 1.0)) -> dict:
    if not p.masks.frames():
        raise MediaError("There is no mask to export yet.", ["Run Auto Roto first, or paint a mask on a frame"], stage="preflight")
    final = Path(out_path)
    if kind == "mov-alpha":
        final = final.with_suffix(".mov")
    elif kind != "mp4":
        raise MediaError(f"Unknown export type: {kind}", stage="preflight")
    scale = float(s.get("scale", 1.0))
    w, h = export_size(p, scale)
    a0, a1 = frame_range(p, s.get("range", "full"))
    count = a1 - a0 + 1
    settings = {**p.data["maskSettings"], "_background": s.get("background", "#FFFFFF")}
    exp_id = uuid.uuid4().hex[:8]
    tmpdir = final.parent / f".rotobrush_export_{exp_id}"
    try:
        tmpdir.mkdir(parents=True)
    except OSError as e:
        raise MediaError("RotoBrush cannot write to the output folder.", ["Choose a folder you can write to"],
                         stage="output-path", detail=str(e))
    tmp_out = tmpdir / ("out" + final.suffix)
    info = p.info
    xlog("Input", p.source_path)
    xlog("Input codec", f"{p.meta.get('codec')} / audio {p.meta.get('audio_codec') or 'none'}")
    xlog("Input resolution", f"{p.meta['width']}x{p.meta['height']}")
    xlog("Input FPS", f"{p.fps:.3f}")
    xlog("Input duration", f"{p.meta['duration']:.2f}s")
    xlog("Intermediate path", info.path if info.path != p.source_path else "none (decoding original)")
    xlog("Output", f"{final} ({w}x{h}, frames {a0}-{a1}, temp {tmpdir.name})")
    try:
        has_audio = bool(p.meta.get("has_audio")) and kind == "mp4"
        if kind == "mp4":
            plans = _mp4_plans(s)
            attempts = [(lab, args, has_audio) for lab, args in plans]
            if has_audio:
                attempts.append((plans[-1][0], plans[-1][1], False))  # last resort: video without audio
        else:
            codec = s.get("movCodec", "prores4444")
            if codec not in MOV_CODECS:
                raise MediaError(f"Unknown alpha codec: {codec}", stage="preflight")
            attempts = [(codec, list(MOV_CODECS[codec][0]), False)]
        warnings: list[str] = []
        done, used_audio, errors = 0, False, []
        for i, (label, args, audio) in enumerate(attempts):
            xlog("Encoding started", f"{label}, audio={'yes' if audio else 'no'}")
            try:
                tmp_out.unlink(missing_ok=True)
                done = _encode_once(p, kind, label, args, audio, a0, count, (w, h), settings, tmp_out, tmpdir, exp_id,
                                    job, progress_span)
                used_audio = audio
                break
            except MediaError as e:
                errors.append(e)
                xlog("Encoding attempt failed", f"{label}: {e.detail or e.message}")
                if e.stage != "encoding" or i + 1 == len(attempts):
                    raise (errors[-2] if (len(errors) > 1 and not attempts[i][2] and has_audio) else e)
                nxt = attempts[i + 1]
                if nxt[2] is False and audio:
                    warnings.append(f"Audio could not be encoded and was left out ({e.detail}).")
                xlog("Falling back", f"{label} -> {nxt[0]}{'' if nxt[2] else ' without audio'}")
                if job:
                    job.update(None, f"{label} failed, retrying with {nxt[0]}", force=True)
        xlog("Encoding finished", f"{done} frames")
        if job:
            job.update(None, "Validating output", force=True)
        result = validate_output(tmp_out, kind, w, h, done, s.get("movCodec", "prores4444"),
                                 expect_audio=used_audio, fps=p.fps, exp_id=exp_id)
        xlog("Output validation", "passed")
        try:
            os.replace(tmp_out, final)  # atomic: the final path only ever holds a finished, validated file
        except OSError as e:
            raise MediaError("The finished file could not be moved to the output location.",
                             ["Close any player that has the old file open", "Choose a different file name"],
                             stage="finalizing", detail=str(e))
        result.update(path=str(final), encoder=label, warnings=warnings)
        xlog("Export result", f"success {final} ({result['bytes']} bytes, encoder {label})")
        return result
    except MediaError as e:
        xlog("Export result", f"FAILED stage={e.stage} code={e.exit_code} {e.detail or e.message}")
        raise
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def validate_output(path: Path, kind: str, w: int, h: int, frames: int, mov_codec: str = "prores4444",
                    expect_audio: bool = False, fps: float | None = None, exp_id: str = "validate") -> dict:
    """Never trust an exit code alone: re-open the file and check it is complete, the right size and playable."""
    def fail(msg: str, hints: list[str] | None = None, **kw) -> MediaError:
        return MediaError(msg, hints or ["Open the log for details"], stage="validation", log=str(log_dir() / "export.log"), **kw)

    if not path.exists():
        raise fail("The export produced no file.")
    size = path.stat().st_size
    if size == 0:
        raise fail("The export produced an empty file.")
    r = video.run_logged([video.binary("ffprobe"), "-v", "error", "-count_packets", "-print_format", "json",
                          "-show_format", "-show_streams", str(path)], **video._TEXT)
    if r.returncode != 0:
        raise fail("The exported file cannot be read back.", ["The encoder may have been interrupted"],
                   exit_code=r.returncode, detail=video.first_error_line(r.stderr))
    data = json.loads(r.stdout or "{}")
    streams = data.get("streams", [])
    vs = next((x for x in streams if x.get("codec_type") == "video"), None)
    if vs is None:
        raise fail("The exported file has no video stream.")
    if (vs.get("width"), vs.get("height")) != (w, h):
        raise fail(f"The exported file has the wrong resolution ({vs.get('width')}x{vs.get('height')}, expected {w}x{h}).")
    packets = int(_f(vs.get("nb_read_packets")))
    if packets and abs(packets - frames) > 1:
        raise fail(f"The exported video is truncated: {packets} of {frames} frames were written.")
    dur = _f(vs.get("duration")) or _f(data.get("format", {}).get("duration"))
    if fps and dur and abs(dur - frames / fps) > max(0.3, 3.0 / fps):
        raise fail(f"The exported duration ({dur:.2f}s) does not match the expected {frames / fps:.2f}s.")
    if expect_audio and not any(x.get("codec_type") == "audio" for x in streams):
        raise fail("The audio track is missing from the exported file.")
    if kind == "mp4" and vs.get("codec_name") not in ("h264", "hevc"):
        raise fail(f"The exported video uses an unexpected codec ({vs.get('codec_name')}).")
    info = {"path": str(path), "codec": vs.get("codec_name"), "pix_fmt": vs.get("pix_fmt"), "width": vs.get("width"),
            "height": vs.get("height"), "frames": frames, "packets": packets, "duration": round(dur, 3),
            "audio": any(x.get("codec_type") == "audio" for x in streams), "bytes": size}
    if kind == "mov-alpha":
        if not any(str(vs.get("pix_fmt", "")).startswith(x) for x in ALPHA_PIXFMTS):
            raise fail("The exported movie has no alpha channel.", ["Try the ProRes 4444 codec"])
        if vs.get("codec_name") != MOV_CODECS[mov_codec][1]:
            raise fail("The exported movie uses an unexpected codec.")
        fr = video.run_logged([video.binary("ffmpeg"), "-v", "error", "-i", str(path), "-frames:v", "1", "-vf", "format=rgba",
                               "-f", "rawvideo", "-"])
        if len(fr.stdout) >= w * h * 4:
            alpha = np.frombuffer(fr.stdout[: w * h * 4], np.uint8).reshape(h, w, 4)[..., 3]
            info["alpha_min"], info["alpha_max"] = int(alpha.min()), int(alpha.max())
    # playback check: decode the tail, where a truncated or unfinished file breaks first
    d = video.run_logged([video.binary("ffmpeg"), "-v", "error", "-sseof", "-3", "-i", str(path), "-f", "null", "-"], **video._TEXT)
    bad = any(k in (d.stderr or "").lower() for k in ("error", "invalid", "corrupt", "truncat"))
    if d.returncode != 0 or bad:
        raise fail("The exported video failed a playback check.", exit_code=d.returncode, detail=video.first_error_line(d.stderr))
    info["validated"] = True
    return info


# ------------------------------------------------------------------ After Effects
JSX_TEMPLATE = r'''// RotoBrush -> After Effects import script (generated). Run: File > Scripts > Run Script File...
// Creates a composition that matches the source (size / frame rate / duration) and adds the roto result.
(function () {
    var here = new File($.fileName).parent;
    var CFG = __CFG__;

    function need(f) {
        if (!f.exists) { throw new Error("Missing file: " + f.fsName); }
        return f;
    }

    app.beginUndoGroup("RotoBrush import");
    try {
        var isAbs = /^([A-Za-z]:|\/|\\\\)/.test(CFG.source);
        var sourceFile = need(isAbs ? new File(CFG.source) : new File(here.fsName + "/" + CFG.source));
        var alphaFile = need(new File(here.fsName + "/" + CFG.alpha));

        var sourceItem = app.project.importFile(new ImportOptions(sourceFile));
        var alphaItem = app.project.importFile(new ImportOptions(alphaFile));

        var comp = app.project.items.addComp(CFG.name, CFG.width, CFG.height, 1.0, CFG.duration, CFG.fps);

        var srcLayer = comp.layers.add(sourceItem);
        srcLayer.name = "Source (reference)";
        srcLayer.enabled = false; // hidden: keeps the original footage in the comp for matching / relinking

        var roto = comp.layers.add(alphaItem);
        roto.name = "Roto result (alpha)";
        roto.startTime = CFG.startSeconds;
        srcLayer.moveToEnd();

        comp.openInViewer();
        alert("RotoBrush: composition '" + CFG.name + "' created.\n" +
              CFG.width + "x" + CFG.height + " @ " + CFG.fps + " fps");
    } catch (e) {
        alert("RotoBrush import failed:\n" + e.toString());
    } finally {
        app.endUndoGroup();
    }
})();
'''

README_TEMPLATE = """RotoBrush - After Effects package
=================================

footage/{source}   Original source video{copy_note}
alpha/{alpha}      Roto result with a real alpha channel ({codec})
comp.jsx           ExtendScript that builds the composition

How to use
1. Open Adobe After Effects.
2. File > Scripts > Run Script File...  and choose comp.jsx
   (if scripts are blocked: Preferences > Scripting & Expressions > "Allow Scripts to Write Files and Access Network")
3. A composition "{name}" is created: {width}x{height}, {fps:g} fps, {duration:.3f} s.
   The alpha movie sits on top; the source is added underneath as a hidden reference layer.

Notes
- This is NOT an .aep file. Native .aep projects can only be written by After Effects itself, so RotoBrush
  generates a script that makes After Effects create the project for you.
- The script uses paths relative to itself: keep the folder structure together when you move the package.
- The .jsx was generated automatically and has not been run inside After Effects by the RotoBrush build tests.
"""


def export_after_effects(p: Project, s: dict, out_dir: str | Path, job: Job | None = None) -> dict:
    out = Path(out_dir)
    if out.suffix:
        out = out.with_suffix("")
    (out / "footage").mkdir(parents=True, exist_ok=True)
    (out / "alpha").mkdir(parents=True, exist_ok=True)
    src = Path(p.source_path)
    copy = bool(s.get("copySource", True))
    span_alpha = (0.0, 0.85) if copy else (0.0, 0.97)
    alpha_name = "foreground_alpha.mov"
    info = export_video(p, "mov-alpha", s, out / "alpha" / alpha_name, job, span_alpha)
    if copy:
        dst = out / "footage" / src.name
        size, copied = max(1, src.stat().st_size), 0
        with src.open("rb") as fi, dst.open("wb") as fo:
            while True:
                chunk = fi.read(8 * 1024 * 1024)
                if not chunk:
                    break
                fo.write(chunk)
                copied += len(chunk)
                if job:
                    job.update(0.85 + 0.12 * copied / size, "Copying source footage")
        source_rel = f"footage/{src.name}"
    else:
        source_rel = str(src).replace("\\", "/")
    a0, a1 = frame_range(p, s.get("range", "full"))
    w, h = export_size(p, float(s.get("scale", 1.0)))
    cfg = {
        "name": p.meta["name"], "width": w, "height": h, "fps": round(p.fps, 6),
        "duration": round(p.n_frames / p.fps, 6), "startSeconds": round(a0 / p.fps, 6),
        "source": source_rel, "alpha": f"alpha/{alpha_name}",
    }
    (out / "comp.jsx").write_text(JSX_TEMPLATE.replace("__CFG__", json.dumps(cfg, indent=8)), encoding="utf-8")
    (out / "README.txt").write_text(README_TEMPLATE.format(
        source=src.name, alpha=alpha_name, codec=s.get("movCodec", "prores4444"), name=cfg["name"], width=w, height=h,
        fps=p.fps, duration=cfg["duration"],
        copy_note="" if copy else " (NOT copied: referenced by absolute path)"), encoding="utf-8")
    info.update(package=str(out), files=sorted(f.relative_to(out).as_posix() for f in out.rglob("*") if f.is_file()))
    return info
