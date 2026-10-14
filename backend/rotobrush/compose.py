"""Rendering and export of a composition (stacked layers), plus depth-map export and the After Effects package."""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from . import depth as depthmod
from . import matte, ops, video
from .comp import Comp, fit_size
from .export import (MOV_CODECS, AudioArgs, _alpha_full, _even, _mp4_plans, export_video, hex_to_bgr, run_export, xlog)
from .jobs import Job
from .project import proxy_size
from .video import MediaError

log = logging.getLogger("rotobrush.export")


def _blend(C: np.ndarray, A: np.ndarray, fg: np.ndarray, a: np.ndarray, ox: int, oy: int) -> None:
    """'Over' composite of a straight-alpha layer onto premultiplied accumulators, clipped to the canvas."""
    h, w = a.shape
    H, W = A.shape
    x0, y0, x1, y1 = max(0, ox), max(0, oy), min(W, ox + w), min(H, oy + h)
    if x1 <= x0 or y1 <= y0:
        return
    sa = a[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    sf = fg[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    C[y0:y1, x0:x1] = sf * sa[..., None] + C[y0:y1, x0:x1] * (1 - sa[..., None])
    A[y0:y1, x0:x1] = sa + A[y0:y1, x0:x1] * (1 - sa)


def mask_active(L: dict, p) -> bool:
    return bool(L.get("useMask", True)) and bool(p.masks.frames())


def render_comp(comp: Comp, frame: int, quality: int) -> tuple[bytes, str]:
    """Composite of every visible layer at one comp frame, at proxy resolution, as RGBA (transparent where empty)."""
    pw, ph = proxy_size(comp.width, comp.height)
    sx = pw / comp.width
    C = np.zeros((ph, pw, 3), np.float32)
    A = np.zeros((ph, pw), np.float32)
    for L in reversed(comp.data["layers"]):
        if not L["visible"]:
            continue
        lf = comp.local_frame(L, frame)
        if lf is None:
            continue
        p = comp.projects[L["id"]]
        bgr = p.frame_bgr(lf)
        fw, fh = fit_size(p.meta["width"], p.meta["height"], comp.width, comp.height)
        dw, dh = max(2, int(round(fw * L["scale"] * sx))), max(2, int(round(fh * L["scale"] * sx)))
        alpha = ops.view_alpha(p, lf, bgr) if mask_active(L, p) else np.ones(bgr.shape[:2], np.float32)
        interp = cv2.INTER_AREA if dw < bgr.shape[1] else cv2.INTER_LINEAR
        fg = cv2.resize(bgr, (dw, dh), interpolation=interp).astype(np.float32) / 255.0
        a = cv2.resize(alpha, (dw, dh), interpolation=cv2.INTER_LINEAR) * float(L["opacity"])
        _blend(C, A, fg, a, int(round((pw - dw) / 2 + L["x"] * sx)), int(round((ph - dh) / 2 + L["y"] * sx)))
    out = np.empty((ph, pw, 4), np.uint8)
    out[..., :3] = np.clip(C / np.maximum(A, 1e-4)[..., None] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(A * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if quality < 100:
        out = cv2.resize(out, None, fx=max(0.1, quality / 100.0), fy=max(0.1, quality / 100.0), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise MediaError("The composite could not be rendered.")
    return buf.tobytes(), "image/png"


def _entries(comp: Comp, a0: int, a1: int, scale: float) -> list[dict]:
    """Visible layers that overlap the export range, bottom -> top, with their placement."""
    out = []
    W, H = _even(comp.width * scale), _even(comp.height * scale)
    for L in reversed(comp.data["layers"]):
        if not L["visible"]:
            continue
        ls, le = comp.layer_range(L)
        o0, o1 = max(a0, ls), min(a1, le)
        if o0 > o1:
            continue
        p = comp.projects[L["id"]]
        fw, fh = fit_size(p.meta["width"], p.meta["height"], comp.width, comp.height)
        dw, dh = _even(fw * L["scale"] * scale), _even(fh * L["scale"] * scale)
        out.append({"L": L, "p": p, "o0": o0, "o1": o1, "local0": comp.local_frame(L, o0), "size": (dw, dh),
                    "ox": int(round((W - dw) / 2 + L["x"] * scale)), "oy": int(round((H - dh) / 2 + L["y"] * scale)),
                    "mask": mask_active(L, p)})
    return out


def comp_audio(comp: Comp, entries: list[dict], a0: int, count: int) -> AudioArgs | None:
    aud = [e for e in entries if e["p"].meta.get("has_audio")]
    if not aud:
        return None
    inputs: list[str] = []
    chains: list[str] = []
    for k, e in enumerate(aud):
        inputs += ["-ss", f"{e['local0'] / e['p'].fps:.6f}", "-t", f"{(e['o1'] - e['o0'] + 1) / comp.fps:.6f}", "-i", e["p"].source_path]
        delay = int(round((e["o0"] - a0) / comp.fps * 1000))
        chains.append(f"[{k + 1}:a:0]aresample=48000,adelay={delay}:all=1[a{k}]")  # input 0 is the raw video pipe
    labels = "".join(f"[a{k}]" for k in range(len(aud)))
    # Pad the mix with silence to the exact comp length: a shorter audio track must never cut the video (-shortest would).
    pad = f"apad=whole_dur={count / comp.fps:.6f}"
    chains.append(f"{labels}{pad}[aout]" if len(aud) == 1 else f"{labels}amix=inputs={len(aud)}:duration=longest:normalize=0,{pad}[aout]")
    return inputs, ["-filter_complex", ";".join(chains), "-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]


def export_comp(comp: Comp, kind: str, s: dict, out_path: str | Path, job: Job | None = None,
                span: tuple[float, float] = (0.0, 1.0)) -> dict:
    """Render the whole stack: every visible layer, in order, with its roto matte, opacity, scale and position."""
    final = Path(out_path)
    if kind == "mov-alpha":
        final = final.with_suffix(".mov")
    elif kind != "mp4":
        raise MediaError(f"Unknown export type: {kind}", stage="preflight")
    scale = float(s.get("scale", 1.0))
    W, H = _even(comp.width * scale), _even(comp.height * scale)
    a0, a1 = comp.work_range() if s.get("range") == "work" else (0, comp.duration - 1)
    count = a1 - a0 + 1
    entries = _entries(comp, a0, a1, scale)
    if not entries:
        raise MediaError("There is nothing visible to export in this range.", ["Turn a layer on or widen the work area"], stage="preflight")
    bg = hex_to_bgr(s.get("background", "#FFFFFF")) if kind == "mp4" else (0, 0, 0)

    def source(job_: Job | None, sp: tuple[float, float]) -> Iterator[bytes]:
        gens = {}
        for e in entries:
            n = e["o1"] - e["o0"] + 1
            fo = comp.fps if abs(e["p"].fps - comp.fps) > 0.01 else None
            gens[e["L"]["id"]] = video.read_frames(e["p"].info, *e["size"], e["local0"], n, fps_out=fo)
        prev: dict[str, tuple] = {}
        try:
            for cf in range(a0, a1 + 1):
                if job_:
                    job_.update(sp[0] + (sp[1] - sp[0]) * (cf - a0) / max(1, count), f"Frame {cf - a0 + 1}/{count}")
                if kind == "mp4":
                    C = np.empty((H, W, 3), np.float32)
                    C[:] = np.array(bg, np.float32) / 255.0
                    A = np.ones((H, W), np.float32)
                else:
                    C, A = np.zeros((H, W, 3), np.float32), np.zeros((H, W), np.float32)
                for e in entries:
                    if not e["o0"] <= cf <= e["o1"]:
                        continue
                    L, p = e["L"], e["p"]
                    frame = next(gens[L["id"]], None)
                    if frame is None:
                        continue
                    dw, dh = e["size"]
                    sset = p.data["maskSettings"]
                    if e["mask"]:
                        a = _alpha_full(p, comp.local_frame(L, cf), frame, (dw, dh), sset)
                        pv = prev.get(L["id"])
                        if pv is not None and float(sset.get("temporal", 0)) > 0:
                            a = matte.stabilize(pv[0], pv[1], frame, a, float(sset["temporal"]))
                        prev[L["id"]] = (a, frame)
                        fg = matte.decontaminate(frame, a, max(1.0, dw / float(p.proxy[0]))) if sset.get("decontaminate", True) \
                            else frame.astype(np.float32) / 255.0
                    else:
                        a, fg = np.ones((dh, dw), np.float32), frame.astype(np.float32) / 255.0
                    _blend(C, A, fg, a * float(L["opacity"]), e["ox"], e["oy"])
                if kind == "mp4":
                    yield np.clip(C * 255.0 + 0.5, 0, 255).astype(np.uint8).tobytes()
                else:
                    out = np.empty((H, W, 4), np.uint8)
                    out[..., :3] = np.clip(C / np.maximum(A, 1e-4)[..., None] * 255.0 + 0.5, 0, 255).astype(np.uint8)
                    out[..., 3] = np.clip(A * 255.0 + 0.5, 0, 255).astype(np.uint8)
                    yield out.tobytes()
        finally:
            for g in gens.values():
                g.close()

    if kind == "mp4":
        plans, pix_in, audio = _mp4_plans(s), "bgr24", comp_audio(comp, entries, a0, count)
    else:
        codec = s.get("movCodec", "prores4444")
        if codec not in MOV_CODECS:
            raise MediaError(f"Unknown alpha codec: {codec}", stage="preflight")
        plans, pix_in, audio = [(codec, list(MOV_CODECS[codec][0]))], "bgra", None
    return run_export(final, kind, (W, H), comp.fps, count, pix_in, plans, audio, source, job, span,
                      s.get("movCodec", "prores4444"),
                      describe={"Composition": f"{comp.data['name']} ({len(entries)} layers, {comp.width}x{comp.height} @ {comp.fps:.3f})",
                                "Layers": ", ".join(e["L"]["name"] for e in entries)})


# ------------------------------------------------------------------ depth map export
def export_depth(comp: Comp, s: dict, out_path: str | Path, job: Job | None = None) -> dict:
    """Depth map of the ACTIVE layer's trimmed range: grey MP4 (8-bit) or a 16-bit PNG sequence (best for After Effects)."""
    L = comp.layer()
    p = comp.projects[L["id"]]
    est = depthmod.DepthEstimator.get()
    scale = float(s.get("scale", 1.0))
    w, h = _even(p.meta["width"] * scale), _even(p.meta["height"] * scale)
    a0, a1 = (L["trimIn"], L["trimOut"]) if s.get("range", "work") != "full" else (0, p.n_frames - 1)
    count = a1 - a0 + 1
    invert = bool(s.get("depthInvert", False))
    fmt = s.get("depthFormat", "mp4")
    xlog("Depth", f"{p.meta['name']} frames {a0}-{a1} model on {est.provider}")

    def frames() -> Iterator[np.ndarray]:
        sm = depthmod.DepthSmoother(strength=float(s.get("depthSmooth", 0.5)))
        for n, bgr in enumerate(video.read_frames(p.info, w, h, a0, count)):
            yield sm.step(est.raw(bgr), bgr, invert)

    if fmt == "png16":
        folder = Path(out_path)
        folder = folder.with_suffix("") if folder.suffix else folder
        folder.mkdir(parents=True, exist_ok=True)
        done = 0
        for n, d in enumerate(frames()):
            if job:
                job.update(n / max(1, count), f"Depth frame {n + 1}/{count}")
            if not cv2.imwrite(str(folder / f"depth_{n:05d}.png"), (np.clip(d, 0, 1) * 65535).astype(np.uint16)):
                raise MediaError("A depth frame could not be written.", ["Check the output folder"], stage="output-path")
            done += 1
        if done < count - 2:
            raise MediaError(f"Only {done} of {count} frames could be read from the source.", stage="decoding")
        return {"path": str(folder), "frames": done, "format": "png16", "validated": True, "provider": est.provider,
                "files": done}

    def source(job_: Job | None, sp: tuple[float, float]) -> Iterator[bytes]:
        for n, d in enumerate(frames()):
            if job_:
                job_.update(sp[0] + (sp[1] - sp[0]) * n / max(1, count), f"Depth frame {n + 1}/{count}")
            yield (np.clip(d, 0, 1) * 255.0 + 0.5).astype(np.uint8).tobytes()

    final = Path(out_path).with_suffix(".mp4")
    res = run_export(final, "mp4", (w, h), p.fps, count, "gray", _mp4_plans(s), None, source, job, (0.0, 1.0),
                     describe={"Depth source": p.source_path})
    res.update(format="mp4", provider=est.provider)
    return res


# ------------------------------------------------------------------ After Effects package
JSX_TEMPLATE = r'''// Zelf-Brush -> After Effects import script (generated). Run: File > Scripts > Run Script File...
// Builds a composition (size / frame rate / duration) and stacks every layer with its timing, opacity, scale and position.
(function () {
    var here = new File($.fileName).parent;
    var CFG = __CFG__;

    function resolve(rel) {
        var isAbs = /^([A-Za-z]:|\/|\\\\)/.test(rel);
        var f = isAbs ? new File(rel) : new File(here.fsName + "/" + rel);
        if (!f.exists) { throw new Error("Missing file: " + f.fsName); }
        return f;
    }

    app.beginUndoGroup("Zelf-Brush import");
    try {
        var comp = app.project.items.addComp(CFG.name, CFG.width, CFG.height, 1.0, CFG.duration, CFG.fps);
        for (var i = 0; i < CFG.layers.length; i++) {          // bottom -> top
            var d = CFG.layers[i];
            var item = app.project.importFile(new ImportOptions(resolve(d.file)));
            var lyr = comp.layers.add(item);
            lyr.name = d.name;
            if (d.trimmed) {                                      // alpha movie already cut to the trim range
                lyr.startTime = d.start;
            } else {                                              // original footage: trim inside After Effects
                lyr.startTime = d.start - d.trimInSeconds;
                lyr.inPoint = d.start;
                lyr.outPoint = d.start + d.length;
            }
            lyr.transform.opacity.setValue(d.opacity * 100);
            lyr.transform.scale.setValue([d.scale, d.scale]);
            lyr.transform.position.setValue([CFG.width / 2 + d.x, CFG.height / 2 + d.y]);
            lyr.enabled = d.enabled;
        }
        comp.openInViewer();
        alert("Zelf-Brush: composition '" + CFG.name + "' created.\n" + CFG.width + "x" + CFG.height + " @ " + CFG.fps + " fps, " + CFG.layers.length + " layer(s)");
    } catch (e) {
        alert("Zelf-Brush import failed:\n" + e.toString());
    } finally {
        app.endUndoGroup();
    }
})();
'''

README_TEMPLATE = """Zelf-Brush - After Effects package
==================================

comp.jsx          ExtendScript that builds the composition and all layers
footage/          original videos (used by layers without a roto matte)
alpha/            roto results with a real alpha channel ({codec}), already trimmed

How to use
1. Open Adobe After Effects.
2. File > Scripts > Run Script File...  and choose comp.jsx
   (if scripts are blocked: Preferences > Scripting & Expressions > "Allow Scripts to Write Files and Access Network")
3. Composition "{name}" is created: {width}x{height}, {fps:g} fps, {duration:.3f} s, {n} layer(s).

Notes
- This is NOT an .aep file. Only After Effects can author one, so Zelf-Brush generates a script that makes After
  Effects create the project for you.
- Paths are relative to comp.jsx: keep the folder structure together when you move the package.
- Layers with a roto matte come in as alpha movies (no audio); layers without one use the original footage.
- The .jsx is generated automatically and has not been run inside After Effects by the Zelf-Brush tests.
"""


def export_after_effects(comp: Comp, s: dict, out_dir: str | Path, job: Job | None = None) -> dict:
    out = Path(out_dir)
    if out.suffix:
        out = out.with_suffix("")
    (out / "footage").mkdir(parents=True, exist_ok=True)
    (out / "alpha").mkdir(parents=True, exist_ok=True)
    scale = float(s.get("scale", 1.0))
    copy = bool(s.get("copySource", True))
    layers = [L for L in reversed(comp.data["layers"])]  # bottom -> top
    steps = max(1, len(layers))
    cfg_layers, copied, infos = [], {}, []
    for i, L in enumerate(layers):
        p = comp.projects[L["id"]]
        base = re.sub(r"[^\w\-]+", "_", L["name"]).strip("_") or L["id"]
        fit = min(comp.width / p.meta["width"], comp.height / p.meta["height"])
        d = {"name": L["name"], "opacity": L["opacity"], "scale": round(fit * L["scale"] * 100, 4),
             "x": round(L["x"] * scale, 3), "y": round(L["y"] * scale, 3), "enabled": bool(L["visible"]),
             "start": round(L["start"] / comp.fps, 6), "length": round(comp.layer_frames(L) / comp.fps, 6),
             "trimInSeconds": round(L["trimIn"] / p.fps, 6)}
        span = (i / steps, (i + 0.9) / steps)
        if mask_active(L, p):
            name = f"{base}_alpha.mov"
            infos.append(export_video(p, "mov-alpha", {**s, "range": "work"}, out / "alpha" / name, job, span))
            d.update(file=f"alpha/{name}", trimmed=True)
        else:
            src = Path(p.source_path)
            if copy:
                if src not in copied:
                    shutil.copy2(src, out / "footage" / src.name)
                    copied[src] = True
                d.update(file=f"footage/{src.name}", trimmed=False)
            else:
                d.update(file=str(src).replace("\\", "/"), trimmed=False)
        cfg_layers.append(d)
        if job:
            job.update((i + 1) / steps, f"Prepared layer {i + 1}/{steps}")
    cfg = {"name": comp.data["name"], "width": _even(comp.width * scale), "height": _even(comp.height * scale),
           "fps": round(comp.fps, 6), "duration": round(comp.duration / comp.fps, 6), "layers": cfg_layers}
    (out / "comp.jsx").write_text(JSX_TEMPLATE.replace("__CFG__", json.dumps(cfg, indent=8)), encoding="utf-8")
    (out / "README.txt").write_text(README_TEMPLATE.format(
        codec=s.get("movCodec", "prores4444"), name=cfg["name"], width=cfg["width"], height=cfg["height"], fps=comp.fps,
        duration=cfg["duration"], n=len(cfg_layers)), encoding="utf-8")
    files = sorted(f.relative_to(out).as_posix() for f in out.rglob("*") if f.is_file())
    return {"package": str(out), "files": files, "validated": all(i.get("validated") for i in infos), "layers": len(cfg_layers),
            "alpha_files": len(infos)}
