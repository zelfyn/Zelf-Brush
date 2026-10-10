# RotoBrush

A lightweight desktop RotoBrush editor for Windows: automatic first-pass mask, manual repair tools,
temporal propagation, matte refinement, and export to MP4 / MOV with alpha / After Effects.

```
Import → Detect subject → Auto roto → Brush / Eraser / Lasso / Object → Propagate → Export
```

## Quick start (Windows 10/11)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1        # installs backend venv + npm deps
powershell -ExecutionPolicy Bypass -File scripts\dev.ps1          # desktop window (needs Rust)
powershell -ExecutionPolicy Bypass -File scripts\dev.ps1 -Browser # no Rust needed: http://127.0.0.1:1420
powershell -ExecutionPolicy Bypass -File scripts\build.ps1        # NSIS installer
```

Prerequisites: Python 3.10+, Node 18+, FFmpeg (`winget install Gyan.FFmpeg`, or drop `ffmpeg.exe` and
`ffprobe.exe` into `backend\bin`), Rust (only for the Tauri shell).

## GPU

| Part | Runs on | Notes |
|---|---|---|
| Subject detection, segmentation, per-frame edge decision while tracking | **GPU** (ONNX Runtime + DirectML, or CUDA) | U²-Net model, works on any DX12 GPU incl. GTX 1650; no PyTorch/CUDA toolkit needed |
| MP4 encoding | **GPU (NVENC)** when available, automatic fallback to libx264 | "Encoder: Auto / CPU only" in the export dialog |
| Optical flow, matte refinement (guided filter, feather), compositing, video decoding | CPU | OpenCV/FFmpeg CPU paths |

The status bar shows what really runs the AI: `GPU: <name> · DirectML (AI on GPU)` or `AI running on CPU`.
If the GPU fails mid-run (driver reset, out of memory) the session silently continues on the CPU.
Without a GPU the same engine works on the CPU (detection/segmentation use the network; tracking uses GrabCut because the
network costs ~1 s/frame on CPU). Engine order for "Auto": PyTorch+SAM 2 (if installed) → ONNX → OpenCV.
`scripts\setup.ps1 -Gpu` additionally installs PyTorch (CUDA), Mask R-CNN and SAM 2.

## Getting a Windows .exe

- **On your PC:** double-click `build-exe.bat` (runs setup, then builds). Output: `src-tauri\target\release\bundle\nsis\RotoBrush_0.1.0_x64-setup.exe`.
- **Without installing Rust/Node:** push this folder to GitHub and run *Actions → Build Windows installer*; download the
  `RotoBrush-windows-installer` artifact. The workflow has not been run by the author, so the first run may need small fixes.
- The installer bundles the Python backend and FFmpeg, so end users need nothing else. Users with an NVIDIA GPU can still
  add the deep-learning engine separately (`setup.ps1 -Gpu`).

## Architecture

```
React + TypeScript (Vite)        src/            Editor UI, viewport, timeline, export queue
   │  HTTP + Server-Sent Events  (127.0.0.1, random port + per-launch token in the Tauri build)
Tauri shell (Rust)               src-tauri/      Window, drag & drop, spawns/kills the backend
   │
Python backend (FastAPI)         backend/rotobrush/
   video.py      FFmpeg/ffprobe: probe, streaming decode (1 frame in RAM), proxy cache
   project.py    .roto project, autosave + recovery, frame cache (resumable / reused on reopen)
   masks.py      per-frame alpha PNGs on disk, bounded undo/redo (also restores keyframes + confidence)
   ops.py        Brush, Eraser, Lasso, Object select, analyze, auto roto, propagation, viewport render
   matte.py      shift / smooth / guided-filter edge refine / feather / decontamination / stabilisation
   engines/      SegmentationEngine = detection + interactive segmentation + temporal propagation
                 opencv_engine  always available (HOG + saliency, GrabCut, optical flow + GrabCut tracking)
                 torch_engine   optional (Mask R-CNN, SAM 2), CUDA with VRAM-aware size and OOM fallback
   export.py     MP4 (composite on background), MOV alpha, After Effects package (+ validation)
   gpu.py jobs.py events.py logs.py
```

Project folder: `source/ cache/ masks/ previews/ tracking/ project.roto`. The original video is only
referenced by path, never copied or modified. Logs: `%LOCALAPPDATA%\RotoBrush\logs\{app,ai,ffmpeg,export}.log`.

### How the pipeline works
- **Proxy editing, full-res export.** Masks are edited at proxy size (long side ≤ 960 by default, `ROTOBRUSH_PROXY_SIDE`).
  Export streams the *original* frames, upsamples the mask and re-snaps it to full-resolution edges with a guided filter.
  The Preview 25–100 % control only changes the size of frames sent to the viewport.
- **Temporal consistency.** Propagation does not copy masks. Each step warps the previous mask with dense optical flow
  (DIS), then re-learns colour models from the confident core/background and re-decides only the uncertain band
  (GrabCut; SAM 2 when installed). Confidence = flow forward/backward consistency + agreement with the prediction.
  Frames below 0.55 are flagged ⚠ *Correction recommended* on the timeline.
- **Corrections.** Every Brush/Eraser/Lasso/Object edit creates a *user* keyframe. Propagate forward/backward/both stops
  at the next user keyframe; *Recalculate from here* re-tracks everything after the frame, using each later correction
  as the anchor for the frames after it.
- **Non-destructive.** Raw masks are stored untouched; matte settings are applied at render/export time.

### Shortcuts
`O` original · `M` mask · `T` transparent · `V` overlay · `E` problem areas (when not the eraser tool) ·
`Space` play/pause · `←/→` frame (Shift = 10) · `B/E/L/W/H` tools · `[ ]` brush size ·
`I/P` work-area in/out · `Ctrl+Z` / `Ctrl+Shift+Z` undo/redo · `Ctrl+S` save · `Ctrl+O` open

## Troubleshooting import / export
- **Every failure shows its stage, the FFmpeg exit code and the real FFmpeg message**, with an **Open Log** button.
  Files: `%LOCALAPPDATA%\RotoBrush\logs\` → `app.log`, `ffmpeg.log`, `export.log` (structured `[EXPORT] ...` lines),
  and one `export_<id>.log` (full FFmpeg command + stderr) per failed export.
- "This file looks incomplete or damaged" is only shown when ffprobe itself reports it (e.g. *moov atom not found*).
  If ffmpeg/ffprobe cannot even start you get "ffprobe was found but cannot be run" instead, with the reason.
- FFmpeg lookup order: `ROTOBRUSH_FFMPEG`/`ROTOBRUSH_FFPROBE` → bundled `bin\` → PATH. Each candidate is run with
  `-version` before use; a broken one is skipped and logged.
- If a valid video cannot be decoded in one pass, RotoBrush converts it once into a stable constant-frame-rate
  ProRes intermediate (`cache\intermediate.mov`) and works from that; the original file is never modified.
- Exports are written to a private temp folder next to the output, validated (ffprobe stream/size/frame-count/duration,
  audio present, tail playback decode), and only then moved atomically to the final name. A failed export never
  leaves a partial file or overwrites an existing one.
- Encoder: Auto tries NVENC when an NVIDIA GPU + encoder exist and falls back to libx264 if it fails; "CPU only" skips NVENC.

## Exports
- **MP4:** composited on white / black / custom colour, H.264 or H.265, CRF, resolution, work area, audio kept.
- **MOV alpha:** ProRes 4444 (`yuva444p10le`), QuickTime Animation or PNG-in-MOV. Output is probed after writing
  (codec, size, alpha pixel format, and that the first frame really contains transparent *and* opaque pixels).
- **After Effects:** a folder with `footage/`, `alpha/foreground_alpha.mov`, `comp.jsx`, `README.txt`.
  Run `comp.jsx` in After Effects (File → Scripts → Run Script File) to build the comp. **No `.aep` is written**:
  only After Effects can author one, so the package is an explicit fallback as the spec allows.
- Export queue shows real per-frame progress from the encoder loop; jobs can be cancelled.

## Verification status (be aware)

Tested in a Linux container with real FFmpeg/OpenCV (`scripts\test.ps1` runs the same suite on Windows):
- `backend/tests`: 21 end-to-end tests on a synthetic clip with ground-truth masks: metadata for mp4/mkv/mov,
  detection, auto roto (mean IoU ≈ 0.98 over tracked frames), brush/eraser/lasso/undo/redo, correction propagation,
  matte non-destructiveness, MP4 background pixel check, real alpha in ProRes/QTRLE/PNG MOV, AE package (JSX parses),
  work area, cancel, save/open, cache reuse, autosave recovery, missing-source error, CPU status.
- The React UI type-checks, builds, and was driven in headless Chromium (import, auto roto, view modes, eraser, undo, stepping) with no console errors.

**Not verified:** the Tauri/Rust build, the NSIS installer and PyInstaller bundle, the native Tk file dialogs, and
Tauri drag-and-drop (none can run in that container); `torch_engine.py` (no PyTorch/GPU there); and `comp.jsx`
inside real After Effects (syntax-checked only). Expect to fix small issues on first Windows build.

## Licences to check before you distribute
- FFmpeg "essentials" build bundled by CI is **GPL-3.0**: shipping it in the installer carries GPL obligations. Swap in an LGPL build if needed.
- U²-Net weights (Apache-2.0, from the authors; the ONNX copy is the one published by the rembg project, pinned by SHA-256).
- ONNX Runtime and DirectML: Microsoft licences (MIT / redistributable). SAM 2: Apache-2.0.

## Known limitations
- U²-Net is a *salient object* network: it finds the most prominent subject, it does not know classes (labels are "Subject", or "Person" when the HOG detector agrees) and can miss low-contrast subjects. SAM 2 (optional) is stronger for hard footage.
- The OpenCV engine is classical: HOG detects upright people only, otherwise it proposes the most salient object; GrabCut
  struggles when subject and background share colours; a click with the Object tool is a colour flood-fill seed, not a
  true prompt. Install the GPU engine (SAM 2) for hard footage.
- There is no dedicated hair-matting network. Detail comes from the guided filter; plug a matting model into
  `MatteRefinement` (e.g. MODNet, Apache-2.0) for fine hair. Check licences of any model you add.
- One combined matte per project (selected subjects are unioned); no per-object layers.
- The frame count of files without `nb_frames` (often MKV) is estimated until the cache finishes, then corrected.
- Variable-frame-rate sources are treated as constant frame rate; on-demand frame seeking can be off by a frame until
  the proxy cache completes.
- Stabilisation and decontamination run at export and in the viewport; very large (8K) exports are slow on CPU.
- SAM 2 checkpoint URL in `setup.ps1` points to Meta's public release; verify it is still current.
