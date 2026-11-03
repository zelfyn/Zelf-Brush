"""End-to-end tests: real FFmpeg, real segmentation/tracking, real exports, on a synthetic clip with known masks."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from rotobrush.server import app

W, H, FPS, N = 640, 360, 15, 40


def make_frame(i: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(7)
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    bg = np.stack([90 + 40 * xx / W, 80 + 30 * yy / H, 70 + 0 * xx], -1)
    bg = bg + cv2.GaussianBlur(rng.randn(H, W, 3).astype(np.float32) * 25, (0, 0), 3)
    cx, cy = 200 + 6 * i, 180 + 25 * np.sin(i / 7.0)
    ax, ay = 90 + 0.5 * i, 110
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (int(ax), int(ay)), 15, 0, 360, 255, -1)
    tex = np.stack([30 + 25 * np.sin(xx / 9), 140 + 30 * np.cos(yy / 11), 235 + 0 * xx], -1)  # orange-red (BGR)
    fg = tex + cv2.GaussianBlur(rng.randn(H, W, 3).astype(np.float32) * 10, (0, 0), 2)
    a = cv2.GaussianBlur(m, (0, 0), 0.8).astype(np.float32)[..., None] / 255.0
    return np.clip(fg * a + bg * (1 - a), 0, 255).astype(np.uint8), m


@pytest.fixture(scope="session")
def clips(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("clips")
    mp4 = d / "clip.mp4"
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-shortest",
         "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", "-c:a", "aac", str(mp4)], stdin=subprocess.PIPE)
    for i in range(N):
        p.stdin.write(make_frame(i)[0].tobytes())
    p.stdin.close()
    assert p.wait() == 0
    out = {"mp4": mp4}
    for ext in ("mkv", "mov"):
        f = d / f"clip.{ext}"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp4), "-c", "copy", str(f)], check=True)
        out[ext] = f
    return out


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def wait_job(c: TestClient, job_id: str, timeout: float = 240) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = next(x for x in c.get("/jobs").json() if x["id"] == job_id)
        if j["status"] in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.15)
    raise TimeoutError(job_id)


def wait_cache(c: TestClient) -> None:
    t0 = time.time()
    while time.time() - t0 < 60:
        if c.get("/timeline").json()["cache"]["complete"]:
            return
        time.sleep(0.1)
    raise TimeoutError("cache")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 127, b > 127
    return float((a & b).sum() / max(1, (a | b).sum()))


def decode_mask(c: TestClient, frame: int, mode="mask") -> np.ndarray:
    r = c.get(f"/render/{frame}", params={"mode": mode, "q": 100})
    assert r.status_code == 200
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
    return img if img.ndim == 2 else img[..., 3] if img.shape[2] == 4 else img[..., 0]


@pytest.fixture(scope="session")
def project(client, clips):
    r = client.post("/video/import", json={"path": str(clips["mp4"])})
    assert r.status_code == 200, r.text
    wait_cache(client)
    return r.json()


def test_metadata_all_containers(client, clips):
    for ext in ("mp4", "mkv", "mov"):
        from rotobrush import video
        info = video.probe(clips[ext])
        assert (info.width, info.height) == (W, H)
        assert abs(info.fps - FPS) < 0.01
        assert info.has_audio
        assert abs(info.frame_count - N) <= 1


def test_import_errors_are_actionable(client, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    r = client.post("/video/import", json={"path": str(bad)})
    assert r.status_code == 400 and r.json()["detail"]["hints"]
    r = client.post("/video/import", json={"path": str(tmp_path / "x.avi")})
    assert r.status_code == 400 and "not supported" in r.json()["detail"]["message"]


def test_project_and_cache(client, project):
    assert project["project"]["width"] == W and project["proxy"]["width"] == W
    tl = client.get("/timeline").json()
    assert tl["cache"]["complete"] and tl["cache"]["frames"] == N
    assert client.get("/thumbnail").status_code == 200


def test_analyze_detects_subject(client, project):
    r = client.post("/segmentation/analyze", json={}).json()
    assert r["objects"], "no subject detected"
    gt = make_frame(r["frame"])[1]
    x, y, w, h = r["objects"][0]["bbox"]
    box = np.zeros_like(gt)
    box[y:y + h, x:x + w] = 255
    assert iou(box, gt) > 0.4


def test_auto_roto_temporal_quality(client, project):
    an = client.post("/segmentation/analyze", json={}).json()
    j = wait_job(client, client.post("/rotoscope/auto", json={"frame": an["frame"], "object_ids": [an["objects"][0]["id"]]}).json()["id"])
    assert j["status"] == "completed", j
    ious = []
    for f in range(0, N, 3):
        ious.append(iou(decode_mask(client, f), make_frame(f)[1]))
    print("per-frame IoU:", [round(x, 2) for x in ious])
    assert np.mean(ious) > 0.85 and min(ious) > 0.7
    tl = client.get("/timeline").json()
    assert tl["masked"] == [[0, N - 1]]
    assert any(k["kind"] == "ai" for k in tl["keyframes"])


def test_viewport_modes(client, project):
    for mode in ("original", "mask", "transparent", "overlay", "edge"):
        r = client.get("/render/5", params={"mode": mode, "q": 50})
        assert r.status_code == 200, mode
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
        assert img.shape[1] == W // 2
    assert cv2.imdecode(np.frombuffer(client.get("/render/5", params={"mode": "transparent"}).content, np.uint8),
                        cv2.IMREAD_UNCHANGED).shape[2] == 4


def test_brush_eraser_lasso_undo_redo(client, project):
    f = 10
    before = decode_mask(client, f).copy()
    # eraser through the middle of the object
    gt = make_frame(f)[1]
    ys, xs = np.where(gt > 0)
    cx, cy = int(xs.mean()), int(ys.mean())
    r = client.post("/mask/stroke", json={"frame": f, "mode": "remove", "points": [[cx - 40, cy], [cx + 40, cy]], "size": 40, "hardness": 0.9})
    assert r.status_code == 200
    erased = decode_mask(client, f)
    assert erased[cy, cx] < before[cy, cx] * 0.3
    # brush into empty background corner
    client.post("/mask/stroke", json={"frame": f, "mode": "add", "points": [[30, 30], [60, 30]], "size": 30, "hardness": 1.0})
    assert decode_mask(client, f)[30, 45] > 200
    # lasso subtract with AI refine on / add without
    poly = [[20, 15], [90, 15], [90, 50], [20, 50]]
    client.post("/mask/lasso", json={"frame": f, "polygon": poly, "mode": "remove", "ai_refine": False})
    assert decode_mask(client, f)[30, 45] < 60
    client.post("/mask/lasso", json={"frame": f, "polygon": [[cx - 30, cy - 30], [cx + 30, cy - 30], [cx + 30, cy + 30], [cx - 30, cy + 30]],
                                     "mode": "add", "ai_refine": True, "wait": True})
    # undo everything (4 ops) then redo one
    for _ in range(4):
        assert client.post("/mask/undo").json()["label"]
    restored = decode_mask(client, f)
    assert np.abs(restored.astype(int) - before.astype(int)).max() <= 2
    assert client.post("/mask/redo").json()["label"] == "Eraser"
    assert decode_mask(client, f)[cy, cx] < before[cy, cx] * 0.3
    client.post("/mask/undo")
    tl = client.get("/timeline").json()
    assert tl["can_redo"]


def test_correction_propagation_and_markers(client, project):
    f = 20
    # user corrects frame 20 with the true mask silhouette (lasso, exact polygon)
    gt = make_frame(f)[1]
    cnt = max(cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)[:, 0, :]
    client.post("/mask/clear", json={"frame": f})
    assert client.post("/mask/lasso", json={"frame": f, "polygon": cnt.tolist(), "mode": "add", "ai_refine": False}).status_code == 200
    tl = client.get("/timeline").json()
    assert {"frame": f, "kind": "user"} in tl["keyframes"]
    j = wait_job(client, client.post("/tracking/propagate", json={"frame": f, "direction": "both"}).json()["id"])
    assert j["status"] == "completed", j
    assert j["result"]["frames"] > 0
    assert iou(decode_mask(client, f + 8), make_frame(f + 8)[1]) > 0.8
    j = wait_job(client, client.post("/tracking/propagate", json={"frame": f, "direction": "forward", "recalc": True}).json()["id"])
    assert j["status"] == "completed"
    # undo of the whole propagation restores previous masks in one step
    assert client.post("/mask/undo").json()["label"] == "Recalculate"


def test_propagate_without_mask_is_actionable(client, project):
    client.post("/mask/clear", json={"frame": 38})
    client.post("/mask/undo")  # back to state before clear -> mask exists again
    r = client.post("/tracking/propagate", json={"frame": 0, "direction": "forward"})
    assert r.status_code in (200, 400)


def test_matte_settings_non_destructive(client, project):
    raw_before = decode_mask(client, 3, "mask").copy()
    r = client.patch("/project/settings", json={"maskSettings": {"shift": 6, "feather": 3}})
    assert r.status_code == 200
    grown = decode_mask(client, 3, "mask")
    assert (grown > 127).sum() > (raw_before > 127).sum()
    client.patch("/project/settings", json={"maskSettings": {"shift": 0, "feather": 0}})
    assert abs(int((decode_mask(client, 3, "mask") > 127).sum()) - int((raw_before > 127).sum())) < 50


def _export(client, kind, out, **settings):
    j = wait_job(client, client.post("/export", json={"kind": kind, "output": str(out), "settings": settings}).json()["id"], 300)
    assert j["status"] == "completed", j["error"]
    return j


def test_export_mp4_custom_background(client, project, tmp_path):
    out = tmp_path / "final.mp4"
    j = _export(client, "mp4", out, background="#7B5CFF", scale=1.0, crf=18)
    assert j["result"]["validated"] and j["result"]["codec"] == "h264"
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert abs(n - N) <= 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    ok, fr = cap.read()
    assert ok and fr.shape[:2] == (H, W)
    corner = fr[5:15, 5:15].reshape(-1, 3).mean(0)  # BGR of #7B5CFF = (255, 92, 123)
    assert np.abs(corner - np.array([255, 92, 123])).max() < 12, corner
    gt = make_frame(5)[1]
    cy, cx = np.array(np.where(gt > 0)).mean(1).astype(int)
    assert np.abs(fr[cy, cx].astype(int) - np.array([255, 92, 123])).max() > 40  # object kept its own colour
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True).stdout.strip()
    assert probe == "aac"


def test_export_mp4_half_resolution_and_bad_colour(client, project, tmp_path):
    j = _export(client, "mp4", tmp_path / "half.mp4", background="#000000", scale=0.5)
    assert (j["result"]["width"], j["result"]["height"]) == (W // 2, H // 2)
    r = client.post("/export", json={"kind": "mp4", "output": str(tmp_path / "x.mp4"), "settings": {"background": "purple"}})
    assert r.status_code == 400


@pytest.mark.parametrize("codec,name", [("prores4444", "prores"), ("qtrle", "qtrle"), ("png", "png")])
def test_export_mov_alpha_is_real(client, project, tmp_path, codec, name):
    out = tmp_path / f"fg_{codec}.mov"
    j = _export(client, "mov-alpha", out, movCodec=codec)
    r = j["result"]
    assert r["validated"] and r["codec"] == name
    assert r["alpha_min"] == 0 and r["alpha_max"] == 255, r  # genuine transparency, not an opaque file
    st = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt", "-of", "json", str(out)],
                                   capture_output=True, text=True).stdout)["streams"][0]
    assert any(st["pix_fmt"].startswith(x) for x in ("yuva", "argb", "rgba"))


def test_export_after_effects_package(client, project, tmp_path):
    out = tmp_path / "RotoExport"
    j = _export(client, "after-effects", out, scale=1.0, range="full", movCodec="prores4444")
    files = j["result"]["files"]
    alphas = [f for f in files if f.startswith("alpha/") and f.endswith("_alpha.mov")]
    assert {"comp.jsx", "README.txt"} <= set(files) and len(alphas) == 1, files
    assert not any(f.startswith("footage/") for f in files), "a layer with a roto matte uses the alpha movie, not the raw footage"
    jsx = (out / "comp.jsx").read_text()
    assert f'"width": {W}' in jsx and f'"height": {H}' in jsx and "addComp" in jsx and "importFile" in jsx
    assert "aep" not in [p.suffix.strip(".") for p in out.rglob("*")]
    if shutil.which("node"):
        chk = tmp_path / "comp_check.js"
        chk.write_text(jsx)
        r = subprocess.run(["node", "--check", str(chk)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_export_work_area_and_cancel(client, project, tmp_path):
    client.patch("/project/settings", json={"workArea": {"in": 10, "out": 19}})
    j = _export(client, "mov-alpha", tmp_path / "wa.mov", range="work", movCodec="prores4444")
    assert j["result"]["frames"] == 10
    client.patch("/project/settings", json={"workArea": {"in": 0, "out": N - 1}})
    job = client.post("/export", json={"kind": "mp4", "output": str(tmp_path / "c.mp4"), "settings": {"range": "full"}}).json()
    client.post(f"/jobs/{job['id']}/cancel")
    assert wait_job(client, job["id"])["status"] in ("cancelled", "completed")


def test_save_open_roundtrip_and_cache_reuse(client, project, tmp_path):
    root = Path(project["root"])
    assert client.post("/project/save").status_code == 200
    saved = json.loads((root / "project.roto").read_text())
    assert saved["project"]["width"] == W and saved["keyframes"] and saved["tracking"]["confidence"]
    n_masks = len(list((root / "masks").glob("*.png")))
    mtime = (root / "cache" / "frames" / "000000.jpg").stat().st_mtime
    r = client.post("/project/open", json={"path": str(root / "project.roto")})
    assert r.status_code == 200 and not r.json()["recovered"]
    time.sleep(0.5)
    assert (root / "cache" / "frames" / "000000.jpg").stat().st_mtime == mtime, "cache was rebuilt instead of reused"
    tl = client.get("/timeline").json()
    assert tl["masked"] == [[0, N - 1]] and len(list((root / "masks").glob("*.png"))) == n_masks
    assert iou(decode_mask(client, 12), make_frame(12)[1]) > 0.8
    # save-as copies masks + tracking
    dest = tmp_path / "copy"
    r = client.post("/project/save-as", json={"path": str(dest)})
    assert r.status_code == 200 and (dest / "project.roto").exists() and len(list((dest / "masks").glob("*.png"))) == n_masks


def test_autosave_recovery(client, project):
    from rotobrush.project import Project
    from rotobrush.server import S
    p = S.project
    p.data["maskSettings"]["feather"] = 2.5
    p.touch()
    assert p.autosave()
    q = Project.open(p.root)
    assert q.recovered and q.data["maskSettings"]["feather"] == 2.5


def test_gpu_status_and_cpu_fallback(client, project):
    st = client.get("/system/status").json()
    assert st["gpu"]["mode"] in ("cpu", "cuda", "directml") and st["gpu"]["message"]
    assert st["active_engine"]["name"] in ("opencv", "onnx", "torch")
    if not st["gpu"]["detected"] and st["gpu"]["mode"] == "cpu":
        assert "CPU mode" in st["gpu"]["message"]
    if st["gpu"]["mode"] == "directml" and not st["gpu"]["detected"]:
        assert "DirectX 12" in st["gpu"]["message"]  # must not claim acceleration it cannot confirm


def test_missing_source_is_reported(client, project, tmp_path):
    root = Path(project["root"])
    data = json.loads((root / "project.roto").read_text())
    data["project"]["source"] = str(tmp_path / "gone.mp4")
    d2 = tmp_path / "broken"
    d2.mkdir()
    (d2 / "project.roto").write_text(json.dumps(data))
    r = client.post("/project/open", json={"path": str(d2)})
    assert r.status_code == 400 and "missing" in r.json()["detail"]["message"]
    client.post("/project/open", json={"path": str(root)})
    assert Path(project["project"]["source"]).exists()  # original footage untouched


# =====================================================================================================
# Export / import robustness (revision: "This file could not be read" and silent export failures)
# =====================================================================================================
from rotobrush import export as exporter, video  # noqa: E402
from rotobrush.config import log_dir  # noqa: E402
from rotobrush.server import S  # noqa: E402


def _fill_masks(p) -> None:
    for i in range(p.n_frames):
        p.masks.set(i, make_frame(min(i, N - 1))[1])


@pytest.fixture(scope="module")
def rb(client, clips):
    """A fresh project with ground-truth masks on every frame (fast, deterministic)."""
    r = client.post("/video/import", json={"path": str(clips["mp4"])})
    assert r.status_code == 200, r.text
    wait_cache(client)
    _fill_masks(S.project)
    return S.project


def _job(client, kind, out, **settings):
    return wait_job(client, client.post("/export", json={"kind": kind, "output": str(out), "settings": settings}).json()["id"], 300)


def test_broken_ffprobe_is_not_reported_as_corrupt_file(client, clips, tmp_path, monkeypatch):
    # Simulates the real bug: a bundled "ffprobe.exe" that exists but cannot run (e.g. a Chocolatey shim copied away).
    fake = tmp_path / "ffprobe.exe"
    fake.write_text("this is not an executable")
    monkeypatch.setenv("ROTOBRUSH_FFPROBE", str(fake))
    monkeypatch.setenv("ROTOBRUSH_NO_PATH_FALLBACK", "1")
    video._resolved.clear()
    try:
        r = client.post("/video/import", json={"path": str(clips["mp4"])})
        d = r.json()["detail"]
        assert r.status_code == 400
        assert d["stage"] == "ffmpeg-check" and "cannot be run" in d["message"]
        assert "corrupt" not in d["message"].lower() and "damaged" not in d["message"].lower()
        assert str(fake) in d["detail"]
    finally:
        video._resolved.clear()


def test_broken_bundled_binary_falls_back_to_a_working_one(client, clips, tmp_path, monkeypatch):
    fake = tmp_path / "ffprobe.exe"
    fake.write_text("nope")
    monkeypatch.setenv("ROTOBRUSH_FFPROBE", str(fake))  # PATH fallback stays enabled
    video._resolved.clear()
    try:
        info = video.probe(clips["mp4"])
        assert info.width == W and video.resolve_binary("ffprobe") != str(fake)
    finally:
        video._resolved.clear()


def test_all_candidates_broken_gives_clear_error(tmp_path):
    fake = tmp_path / "ffmpeg.exe"
    fake.write_text("x")
    with pytest.raises(video.MediaError) as e:
        video.resolve_binary("ffmpeg", candidates=[str(fake)])
    assert e.value.stage == "ffmpeg-check" and e.value.hints
    with pytest.raises(video.MediaError) as e:
        video.resolve_binary("ffmpeg", candidates=[])
    assert "not found" in e.value.message


def test_truncated_file_is_classified_with_evidence(client, clips, tmp_path):
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(clips["mp4"].read_bytes()[:3000])
    r = client.post("/video/import", json={"path": str(cut)})
    d = r.json()["detail"]
    assert r.status_code == 400 and d["stage"] == "probe" and d["detail"] and d["exit_code"] not in (None, 0)
    assert "incomplete or damaged" in d["message"]


@pytest.mark.parametrize("ext", ["mp4", "mkv", "mov"])
def test_valid_files_import_without_error(client, clips, ext):
    r = client.post("/video/import", json={"path": str(clips[ext])})
    assert r.status_code == 200, r.text
    assert r.json()["project"]["audio_codec"] == "aac" and r.json()["project"]["container"]
    wait_cache(client)


def test_export_success_is_validated_playable_and_logged(client, rb, tmp_path):
    out = tmp_path / "ok.mp4"
    j = _job(client, "mp4", out, background="#FFFFFF", hwEncode="cpu")
    assert j["status"] == "completed", j["error"]
    r = j["result"]
    assert r["validated"] and r["codec"] == "h264" and r["audio"] and r["warnings"] == []
    assert abs(r["duration"] - N / FPS) < 0.3 and r["packets"] in (N - 1, N, N + 1)
    # independent re-check: the whole file decodes with zero errors
    d = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-f", "null", "-"], capture_output=True, text=True)
    assert d.returncode == 0 and d.stderr.strip() == ""
    assert not [x for x in tmp_path.iterdir() if x.name.startswith(".rotobrush_export_")], "temp folder was not cleaned"
    text = (log_dir() / "export.log").read_text(encoding="utf-8")
    for key in ("[EXPORT] Input:", "[EXPORT] Input codec:", "[EXPORT] Input resolution:", "[EXPORT] Input FPS:",
                "[EXPORT] Input duration:", "[EXPORT] Intermediate path:", "[EXPORT] FFmpeg command:",
                "[EXPORT] Encoding started:", "[EXPORT] Encoding finished:", "[EXPORT] FFmpeg exit code: 0",
                "[EXPORT] Output validation: passed", "[EXPORT] Export result: success"):
        assert key in text, key


def test_ffmpeg_failure_reports_real_error_and_leaves_no_partial_file(client, rb, tmp_path):
    out = tmp_path / "bad.xyz"  # FFmpeg cannot choose a muxer for this extension: a genuine encoder failure
    out.write_bytes(b"PREVIOUS GOOD FILE")
    j = _job(client, "mp4", out, hwEncode="cpu")
    assert j["status"] == "failed"
    e = j["error"]
    assert e["stage"] == "encoding" and e["exit_code"] not in (None, 0)
    assert "FFmpeg exited with code" in e["message"] and e["detail"]
    assert Path(e["log"]).exists()
    body = Path(e["log"]).read_text(encoding="utf-8")
    assert "COMMAND:" in body and "libx264" in body and "FFMPEG STDERR:" in body
    assert out.read_bytes() == b"PREVIOUS GOOD FILE", "a failed export must not touch the existing output"
    assert not [x for x in tmp_path.iterdir() if x.name.startswith(".rotobrush_export_")]
    assert "[EXPORT] Export result: FAILED stage=encoding" in (log_dir() / "export.log").read_text(encoding="utf-8")


def test_unwritable_output_folder_is_reported(client, rb, tmp_path):
    blocker = tmp_path / "afile"
    blocker.write_text("x")
    j = _job(client, "mp4", blocker / "out.mp4")
    assert j["status"] == "failed" and j["error"]["stage"] == "output-path"


def test_hardware_encoder_failure_falls_back_to_software(client, rb, tmp_path):
    # Forcing NVENC fails on machines without an NVIDIA encoder; the export must still finish with libx264.
    j = _job(client, "mp4", tmp_path / "hw.mp4", hwEncode="nvenc")
    assert j["status"] == "completed", j["error"]
    assert j["result"]["encoder"] in ("libx264", "h264_nvenc") and j["result"]["validated"]
    text = (log_dir() / "export.log").read_text(encoding="utf-8")
    if j["result"]["encoder"] == "libx264":
        assert "Falling back: h264_nvenc -> libx264" in text


def test_two_exports_do_not_collide(client, rb, tmp_path):
    ids = [client.post("/export", json={"kind": "mp4", "output": str(tmp_path / f"c{i}.mp4"), "settings": {"hwEncode": "cpu"}}).json()["id"]
           for i in range(2)]
    assert all(wait_job(client, i, 300)["status"] == "completed" for i in ids)
    assert not [x for x in tmp_path.iterdir() if x.name.startswith(".rotobrush_export_")]


def test_validate_output_rejects_bad_files(client, rb, tmp_path):
    good = tmp_path / "good.mp4"
    assert _job(client, "mp4", good, hwEncode="cpu")["status"] == "completed"
    V = lambda f, **kw: exporter.validate_output(f, "mp4", W, H, kw.pop("frames", N), fps=FPS, **kw)  # noqa: E731
    V(good, expect_audio=True)
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(video.MediaError, match="empty"):
        V(empty)
    with pytest.raises(video.MediaError, match="no file"):
        V(tmp_path / "missing.mp4")
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(good.read_bytes()[: good.stat().st_size // 2])
    with pytest.raises(video.MediaError):
        V(cut)
    with pytest.raises(video.MediaError, match="truncated"):
        V(good, frames=N + 10)
    with pytest.raises(video.MediaError, match="resolution"):
        exporter.validate_output(good, "mp4", W // 2, H // 2, N, fps=FPS)
    noaudio = tmp_path / "noaudio.mov"
    assert _job(client, "mov-alpha", noaudio)["status"] == "completed"
    with pytest.raises(video.MediaError, match="audio"):
        exporter.validate_output(noaudio, "mov-alpha", W, H, N, fps=FPS, expect_audio=True)


def test_undecodable_source_falls_back_to_stable_intermediate(client, clips, monkeypatch):
    real = video.extract_proxy_frames
    calls = {"n": 0}

    def flaky(info, *a, **k):
        calls["n"] += 1
        if not info.path.endswith("intermediate.mov"):
            raise video.MediaError("Frames could not be extracted from this video.", stage="frame-extraction", detail="simulated decoder failure")
        return real(info, *a, **k)

    monkeypatch.setattr(video, "extract_proxy_frames", flaky)
    r = client.post("/video/import", json={"path": str(clips["mkv"])})
    assert r.status_code == 200
    wait_cache(client)
    p = S.project
    assert p.intermediate_path.exists() and p.cache_complete() and p.cached_frame_count() >= N - 1
    assert p.info.path.endswith("intermediate.mov") and p.source_path.endswith("clip.mkv")
    # the whole pipeline works from the intermediate, original footage untouched
    _fill_masks(p)
    monkeypatch.setattr(video, "extract_proxy_frames", real)
    j = _job(client, "mp4", p.root / "from_intermediate.mp4", hwEncode="cpu")
    assert j["status"] == "completed" and j["result"]["audio"] and clips["mkv"].exists()


def test_open_log_endpoint_only_serves_log_files(client, tmp_path):
    other = tmp_path / "secret.txt"
    other.write_text("x")
    assert client.post("/system/open-log", json={"path": str(other)}).status_code == 400


# =====================================================================================================
# GPU-capable ONNX engine (DirectML / CUDA via ONNX Runtime). On machines without a GPU these run on the CPU provider.
# =====================================================================================================
from rotobrush import engines as engine_registry  # noqa: E402
from rotobrush.engines import onnx_engine  # noqa: E402

needs_model = pytest.mark.skipif(onnx_engine.find_model() is None or not onnx_engine.onnx_status()[0],
                                 reason="u2net.onnx / onnxruntime not available")


@pytest.fixture
def onnx_project(client, clips, monkeypatch):
    monkeypatch.delenv("ROTOBRUSH_ONNX_TRACK", raising=False)
    engine_registry.reset_engines()
    r = client.post("/video/import", json={"path": str(clips["mp4"])})
    assert r.status_code == 200, r.text
    wait_cache(client)
    assert client.patch("/project/settings", json={"engine": "onnx", "aiModel": "fast"}).status_code == 200
    yield S.project
    engine_registry.reset_engines()


@needs_model
def test_onnx_engine_is_listed_and_reports_its_device(client, onnx_project):
    st = client.get("/system/status").json()
    assert next(e for e in st["engines"] if e["name"] == "onnx")["available"]
    eng = st["active_engine"]
    assert eng["name"] == "onnx"
    # before any model has loaded nothing is claimed: no guessing a GPU from the installed providers
    assert eng["ready"] is False and eng["provider"] == "pending" and eng["accelerated"] is False
    engine_registry.get_engine("onnx").runtime()                     # load a model (a GPU-less machine falls back to the CPU)
    eng = client.get("/system/status").json()["active_engine"]
    assert eng["ready"] is True and eng["provider"] in ("CPU", "CUDA", "DirectML")
    assert eng["accelerated"] == (eng["provider"] != "CPU"), "provider and accelerated must always agree"


@needs_model
def test_onnx_detection_segmentation_and_auto_roto(client, onnx_project):
    an = client.post("/segmentation/analyze", json={}).json()
    assert an["engine"] == "onnx" and an["objects"], an
    gt = make_frame(an["frame"])[1]
    x, y, w, h = an["objects"][0]["bbox"]
    box = np.zeros_like(gt)
    box[y:y + h, x:x + w] = 255
    gy, gx = np.where(gt > 0)
    gtbox = np.zeros_like(gt)
    gtbox[gy.min():gy.max() + 1, gx.min():gx.max() + 1] = 255
    assert iou(box, gtbox) > 0.9  # box vs the object's own bounding box (an ellipse fills only ~78% of its box)
    j = wait_job(client, client.post("/rotoscope/auto", json={"frame": an["frame"], "object_ids": [an["objects"][0]["id"]]}).json()["id"], 600)
    assert j["status"] == "completed", j["error"]
    ious = [iou(decode_mask(client, f), make_frame(f)[1]) for f in range(0, N, 4)]
    assert np.mean(ious) > 0.9 and min(ious) > 0.8, [round(v, 2) for v in ious]


@needs_model
def test_onnx_network_tracking_path(client, onnx_project, monkeypatch):
    """Forces the GPU-style tracking path (network re-decides the edge band each frame), even when running on CPU."""
    monkeypatch.setenv("ROTOBRUSH_ONNX_TRACK", "1")
    engine_registry.reset_engines()
    client.patch("/project/settings", json={"workArea": {"in": 10, "out": 25}})
    f = 16
    S.project.masks.set(f, make_frame(f)[1])
    S.project.add_keyframe(f, "user")
    j = wait_job(client, client.post("/tracking/propagate", json={"frame": f, "direction": "both"}).json()["id"], 600)
    assert j["status"] == "completed", j["error"] and j["result"]["engine"] == "onnx"
    ious = [iou(decode_mask(client, i), make_frame(i)[1]) for i in range(10, 26)]
    assert np.mean(ious) > 0.95 and min(ious) > 0.85, [round(v, 2) for v in ious]
    assert min(client.get("/timeline").json()["confidence"].values()) > 0.7


@needs_model
def test_onnx_ai_refine_snaps_a_rough_lasso_to_edges(client, onnx_project):
    f = 12
    gt = make_frame(f)[1]
    ys, xs = np.where(gt > 0)
    cnt = max(cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    M = cv2.moments(cnt)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    rough = ((cnt[:, 0, :] - [cx, cy]) * 1.06 + [cx, cy] + 3).astype(int)  # ~6% too big and shifted: a sloppy lasso
    plain = np.zeros_like(gt)
    cv2.fillPoly(plain, [rough.astype(np.int32)], 255)
    client.post("/mask/clear", json={"frame": f})
    assert client.post("/mask/lasso", json={"frame": f, "polygon": rough.tolist(), "mode": "add", "ai_refine": True, "wait": True}).status_code == 200
    refined = decode_mask(client, f)
    assert iou(refined, gt) > iou(plain, gt) and iou(refined, gt) > 0.93, (iou(refined, gt), iou(plain, gt))


@needs_model
def test_gpu_runtime_failure_falls_back_to_cpu_and_keeps_working():
    class DeviceLost:
        def run(self, *a, **k):
            raise RuntimeError("DXGI_ERROR_DEVICE_REMOVED")

        def get_providers(self):
            return ["DmlExecutionProvider", "CPUExecutionProvider"]

    rt = onnx_engine.OrtRuntime(onnx_engine.find_model(), force_cpu=True)
    assert rt.provider == "CPUExecutionProvider" and not rt.accelerated
    rt.session = DeviceLost()
    assert rt.accelerated  # pretends to be a GPU session that then dies mid-run
    out = rt.infer(np.zeros((1, 3, 320, 320), np.float32))
    assert out.shape == (1, 1, 320, 320) and rt.fell_back and rt.provider == "CPUExecutionProvider"
    del rt
    engine_registry.reset_engines()


@needs_model
def test_unusable_gpu_provider_at_startup_falls_back(monkeypatch):
    import onnxruntime as ort
    real = ort.InferenceSession
    seen = []

    def picky(path, sess_options=None, providers=None, **kw):
        seen.append(providers[0])
        if providers[0] == "DmlExecutionProvider":
            raise RuntimeError("no DX12 adapter")
        return real(path, sess_options=sess_options, providers=["CPUExecutionProvider"], **kw)

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", picky)
    rt = onnx_engine.OrtRuntime(onnx_engine.find_model())
    assert seen[0] == "DmlExecutionProvider" and rt.provider == "CPUExecutionProvider" and "no DX12 adapter" in rt.error
    del rt
    monkeypatch.undo()
    engine_registry.reset_engines()


def test_gpu_info_recognises_directml(monkeypatch):
    from rotobrush import gpu
    monkeypatch.setattr(gpu, "_ort_info", {"installed": True, "providers": ["DmlExecutionProvider", "CPUExecutionProvider"]})
    monkeypatch.setattr(gpu, "_nvidia_smi", lambda: {"name": "NVIDIA GeForce GTX 1650", "vram_total_mb": 4096, "vram_used_mb": 300, "driver": "551"})
    monkeypatch.setattr(gpu, "_cache", None)
    info = gpu.gpu_info(force=True)
    assert info["mode"] in ("directml", "cuda") and info["directml_available"] and info["name"].endswith("GTX 1650")
    monkeypatch.setattr(gpu, "_ort_info", {"installed": True, "providers": ["CPUExecutionProvider"]})
    monkeypatch.setattr(gpu, "_torch_probe", lambda: {"installed": False, "cuda": False, "version": None})
    monkeypatch.setattr(gpu, "_cache", None)
    assert "no GPU-enabled AI runtime" in gpu.gpu_info(force=True)["message"]
    monkeypatch.setattr(gpu, "_cache", None)


def test_auto_engine_falls_back_to_opencv_when_model_missing(monkeypatch):
    monkeypatch.setattr(onnx_engine, "find_model", lambda *a, **k: None)
    monkeypatch.setattr(engine_registry, "_instances", {})
    eng = engine_registry.get_engine("auto")
    assert eng.name in ("opencv", "torch")
    assert engine_registry.get_engine("onnx").name in ("opencv", "torch")


@pytest.mark.parametrize("engine", ["opencv", "onnx"])
def test_backward_propagation_really_walks_backwards(client, clips, monkeypatch, engine):
    """Regression: backward tracking used to process zero frames because the loop bound only worked forwards."""
    if engine == "onnx" and (onnx_engine.find_model() is None or not onnx_engine.onnx_status()[0]):
        pytest.skip("u2net.onnx not available")
    r = client.post("/video/import", json={"path": str(clips["mp4"])})
    assert r.status_code == 200
    wait_cache(client)
    client.patch("/project/settings", json={"engine": engine, "aiModel": "fast", "workArea": {"in": 12, "out": N - 1}})
    f = 30
    S.project.masks.set(f, make_frame(f)[1])
    S.project.add_keyframe(f, "user")
    j = wait_job(client, client.post("/tracking/propagate", json={"frame": f, "direction": "backward"}).json()["id"], 600)
    assert j["status"] == "completed", j["error"]
    assert j["result"]["frames"] == f - 12, j["result"]          # frames 29..12 inclusive
    assert S.project.masks.has(12) and not S.project.masks.has(11), "must stop at the work-area start"
    ious = [iou(decode_mask(client, i), make_frame(i)[1]) for i in range(12, 30)]
    assert np.mean(ious) > 0.9 and min(ious) > 0.75, [round(v, 2) for v in ious]
    client.patch("/project/settings", json={"engine": "auto"})
    engine_registry.reset_engines()


# =====================================================================================================
# Concurrency: autosave thread vs. tracking job writing the same files (WinError 32 on Windows CI)
# =====================================================================================================
def test_atomic_write_survives_concurrent_writers(tmp_path):
    import threading
    from rotobrush.project import _atomic_write
    target = tmp_path / "tracking.json"
    errors: list[BaseException] = []

    def hammer(k: int) -> None:
        try:
            for i in range(150):
                _atomic_write(target, json.dumps({"writer": k, "i": i, "pad": "x" * 2000}))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=hammer, args=(k,)) for k in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, errors[:2]
    assert json.loads(target.read_text())["pad"] == "x" * 2000          # last write is complete, never half-written
    assert [p.name for p in tmp_path.iterdir()] == ["tracking.json"], "temporary files were left behind"


def test_replace_retries_when_windows_has_the_file_briefly_locked(tmp_path, monkeypatch):
    from rotobrush import fsutil
    src, dst = tmp_path / "a.tmp", tmp_path / "a.json"
    src.write_text("new")
    dst.write_text("old")
    real, calls = os.replace, {"n": 0}

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(32, "The process cannot access the file because it is being used by another process")
        return real(a, b)

    monkeypatch.setattr(fsutil.os, "replace", flaky)
    fsutil.replace(src, dst)
    assert dst.read_text() == "new" and calls["n"] == 3


def test_replace_gives_up_with_a_clear_error_if_the_file_stays_locked(tmp_path, monkeypatch):
    from rotobrush import fsutil
    src, dst = tmp_path / "b.tmp", tmp_path / "b.json"
    src.write_text("x")
    monkeypatch.setattr(fsutil.os, "replace", lambda a, b: (_ for _ in ()).throw(PermissionError(32, "locked")))
    monkeypatch.setattr(fsutil.time, "sleep", lambda s: None)
    with pytest.raises(PermissionError):
        fsutil.replace(src, dst, attempts=3)


def test_tracking_job_survives_autosave_hammering_the_same_files(client, clips):
    """The real-world race: the 20 s autosave thread fires while a tracking job is saving masks and tracking.json."""
    import threading
    r = client.post("/video/import", json={"path": str(clips["mp4"])})
    assert r.status_code == 200
    wait_cache(client)
    client.patch("/project/settings", json={"engine": "opencv", "workArea": {"in": 0, "out": N - 1}})
    p = S.project
    p.masks.set(0, make_frame(0)[1])
    p.add_keyframe(0, "user")
    stop, errors = threading.Event(), []

    def autosaver() -> None:
        while not stop.is_set():
            try:
                p.touch()
                p.autosave()
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                return

    th = threading.Thread(target=autosaver)
    th.start()
    try:
        j = wait_job(client, client.post("/tracking/propagate", json={"frame": 0, "direction": "forward"}).json()["id"], 300)
    finally:
        stop.set()
        th.join()
    assert not errors, errors[:1]
    assert j["status"] == "completed", j["error"]
    assert p.masks.has(N - 1)
    client.patch("/project/settings", json={"engine": "auto"})


# =====================================================================================================
# Composition: stacked layers, trim, move, composite preview, multi-layer export, depth, models
# =====================================================================================================
from rotobrush import compose, depth as depthmod  # noqa: E402

W2, H2, FPS2, N2 = 320, 180, 10, 20


@pytest.fixture(scope="session")
def clip2(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("clip2") / "second.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={W2}x{H2}:rate={FPS2}:duration={N2 / FPS2}",
                    "-f", "lavfi", "-i", f"sine=frequency=880:duration={N2 / FPS2}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(out)], check=True)
    return out


def lid_headers(lid: str) -> dict:
    return {"x-layer": lid}


@pytest.fixture
def stack(client, clips, clip2):
    """Comp = clip.mp4 (640x360 @15, with ground-truth masks) on TOP of second.mp4 (320x180 @10, no roto)."""
    r = client.post("/video/import", json={"path": str(clips["mp4"])})
    assert r.status_code == 200, r.text
    wait_cache(client)
    first = r.json()["comp"]["layers"][0]["id"]
    r = client.post("/layers/add", json={"path": str(clip2), "start": 0})
    assert r.status_code == 200, r.text
    wait_cache(client)
    second = r.json()["comp"]["layers"][0]["id"]
    r = client.post("/layers/reorder", json={"order": [first, second]})
    comp = r.json()["comp"]
    assert comp["layers"][0]["filename"] == "clip.mp4" and comp["layers"][1]["filename"] == "second.mp4"
    client.post(f"/layers/{first}/activate")
    _fill_masks(S.comp.project(first))
    return S.comp, first, second


def test_new_comp_keeps_the_single_clip_response_shape(client, clips):
    r = client.post("/video/import", json={"path": str(clips["mp4"])})
    d = r.json()
    assert d["project"]["width"] == W and d["proxy"]["width"] == W and "root" in d      # what the editor panels read
    c = d["comp"]
    assert len(c["layers"]) == 1 and c["active"] == c["layers"][0]["id"] and c["duration"] == N
    assert (c["width"], c["height"]) == (W, H) and abs(c["fps"] - FPS) < 0.01
    wait_cache(client)


def test_layers_are_independent_stacked_and_timed(client, stack):
    comp, top, bottom = stack
    assert [L["id"] for L in comp.data["layers"]] == [top, bottom] and comp.data["active"] == top
    assert comp.project(top).masks.frames() and not comp.project(bottom).masks.frames()
    # the roto tools address a layer through the x-layer header: strokes land on that layer only
    r = client.post("/mask/stroke", headers=lid_headers(bottom), json={"frame": 3, "mode": "add", "points": [[40, 40], [80, 40]], "size": 20, "hardness": 1})
    assert r.status_code == 200
    assert comp.project(bottom).masks.has(3) and (comp.project(bottom).masks.get(3) > 127).any()
    tl_top = client.get("/timeline", headers=lid_headers(top)).json()
    tl_bot = client.get("/timeline", headers=lid_headers(bottom)).json()
    assert tl_top["masked"] == [[0, N - 1]] and tl_bot["masked"] == [[3, 3]]
    # move + duration: second clip is 20 frames @10 fps = 30 comp frames @15 fps
    assert comp.layer_frames(comp.layer(bottom)) == 30
    st = client.patch(f"/layers/{bottom}", json={"start": 25}).json()["comp"]
    assert next(L for L in st["layers"] if L["id"] == bottom)["end"] == 25 + 30 - 1 and st["duration"] == 55
    assert client.post(f"/layers/{bottom}/activate").json()["comp"]["active"] == bottom
    client.patch(f"/layers/{bottom}", json={"start": 0})
    client.post(f"/layers/{top}/activate")


def test_trim_limits_the_layer_and_the_roto_range(client, stack):
    comp, top, _ = stack
    st = client.patch(f"/layers/{top}", json={"trimIn": 10, "trimOut": 29}).json()["comp"]
    L = next(x for x in st["layers"] if x["id"] == top)
    assert (L["trimIn"], L["trimOut"], L["length"]) == (10, 29, 20)
    assert comp.project(top).data["workArea"] == {"in": 10, "out": 29}, "tracking must stay inside the trim"
    assert comp.local_frame(comp.layer(top), L["start"]) == 10 and comp.local_frame(comp.layer(top), L["start"] + 19) == 29
    assert comp.local_frame(comp.layer(top), L["start"] + 20) is None
    # propagation never leaves the trimmed range
    p = comp.project(top)
    for i in range(N):
        p.masks.delete(i)
    p.masks.set(20, make_frame(20)[1])
    p.add_keyframe(20, "user")
    client.patch("/project/settings", headers=lid_headers(top), json={"engine": "opencv"})
    j = wait_job(client, client.post("/tracking/propagate", headers=lid_headers(top), json={"frame": 20, "direction": "both"}).json()["id"])
    assert j["status"] == "completed", j["error"]
    assert p.masks.frames()[0] == 10 and p.masks.frames()[-1] == 29
    r = client.patch(f"/layers/{top}", json={"trimIn": 25, "trimOut": 12})
    assert r.status_code == 400 and r.json()["detail"]["stage"] == "trim"
    client.patch(f"/layers/{top}", json={"trimIn": 0, "trimOut": N - 1})
    client.patch("/project/settings", headers=lid_headers(top), json={"engine": "auto"})
    _fill_masks(p)


def test_composite_preview_stacks_layers_with_matte_and_opacity(client, stack):
    comp, top, bottom = stack
    f = 6
    gt = make_frame(f)[1]
    cy, cx = np.array(np.where(gt > 0)).mean(1).astype(int)

    def png(frame=f):
        r = client.get(f"/comp/render/{frame}")
        assert r.status_code == 200 and r.headers["content-type"] in ("image/png", "image/jpeg")
        png.mime = r.headers["content-type"]
        return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)

    img = png()
    assert img.shape == (H, W, 3) and png.mime == "image/jpeg"                       # bottom layer fills the frame -> opaque JPEG
    obj = np.array([30, 140, 235])                                                    # BGR of the orange subject
    assert np.abs(img[cy, cx, :3].astype(int) - obj).max() < 60, img[cy, cx]          # subject comes from the TOP layer
    bot = cv2.resize(S.comp.project(bottom).frame_bgr(int(round(f * FPS2 / FPS))), (W, H))
    assert np.abs(img[5, 5, :3].astype(int) - bot[5, 5].astype(int)).max() < 40       # outside the matte: the layer below
    client.patch(f"/layers/{top}", json={"opacity": 0.5})
    half = png()
    assert np.abs(half[cy, cx, :3].astype(int) - img[cy, cx, :3].astype(int)).max() > 10  # blended with what is below
    client.patch(f"/layers/{top}", json={"opacity": 1.0, "visible": False})
    assert np.abs(png()[cy, cx, :3].astype(int) - bot[cy, cx].astype(int)).max() < 40      # hidden layer is gone
    client.patch(f"/layers/{top}", json={"visible": True})
    client.patch(f"/layers/{top}", json={"useMask": False})                           # matte off -> whole top frame is opaque
    assert np.abs(png()[5, 5, :3].astype(int) - S.comp.project(top).frame_bgr(f)[5, 5].astype(int)).max() < 40
    client.patch(f"/layers/{top}", json={"useMask": True})
    assert client.get("/comp/render/9999").status_code == 200                         # past the end is clamped, not an error


def test_export_stack_mp4_has_both_layers_and_mixed_audio(client, stack, tmp_path):
    comp, top, bottom = stack
    out = tmp_path / "stack.mp4"
    j = _job(client, "mp4", out, hwEncode="cpu", background="#000000")
    assert j["status"] == "completed", j["error"]
    r = j["result"]
    assert r["validated"] and r["audio"] and abs(r["duration"] - comp.duration / comp.fps) < 0.35 and r["warnings"] == []
    n_audio = int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(out)],
                                 capture_output=True, text=True).stdout.count("\n"))
    assert n_audio == 1                                                              # both layers mixed into ONE track
    cap = cv2.VideoCapture(str(out))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 6)
    ok, fr = cap.read()
    gt = make_frame(6)[1]
    cy, cx = np.array(np.where(gt > 0)).mean(1).astype(int)
    assert ok and np.abs(fr[cy, cx].astype(int) - np.array([30, 140, 235])).max() < 70
    assert fr[5, 5].sum() > 40, "the bottom layer must show where the top matte is empty (background is black)"
    # a hidden top layer: only the bottom clip remains, so the object colour is gone
    client.patch(f"/layers/{top}", json={"visible": False})
    out2 = tmp_path / "bottom_only.mp4"
    assert _job(client, "mp4", out2, hwEncode="cpu")["status"] == "completed"
    cap = cv2.VideoCapture(str(out2))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 6)
    ok, fr2 = cap.read()
    assert ok and np.abs(fr2[cy, cx].astype(int) - np.array([30, 140, 235])).max() > 40
    client.patch(f"/layers/{top}", json={"visible": True})


def test_export_stack_alpha_only_where_layers_exist(client, stack, tmp_path):
    comp, top, bottom = stack
    client.patch(f"/layers/{bottom}", json={"visible": False})
    j = _job(client, "mov-alpha", tmp_path / "top_alpha.mov")
    assert j["status"] == "completed", j["error"]
    assert j["result"]["alpha_min"] == 0 and j["result"]["alpha_max"] == 255 and j["result"]["frames"] == comp.duration
    client.patch(f"/layers/{bottom}", json={"visible": True})
    for L in comp.data["layers"]:                                                    # nothing visible -> clear message
        client.patch(f"/layers/{L['id']}", json={"visible": False})
    j = _job(client, "mp4", tmp_path / "none.mp4")
    assert j["status"] == "failed" and j["error"]["stage"] == "preflight" and "nothing visible" in j["error"]["message"]
    for L in comp.data["layers"]:
        client.patch(f"/layers/{L['id']}", json={"visible": True})


def test_layer_with_another_frame_rate_stays_in_sync(client, stack, tmp_path):
    comp, top, bottom = stack
    client.patch(f"/layers/{bottom}", json={"start": 20})        # 20 + 30 comp frames -> duration 50
    assert comp.duration == 50
    j = _job(client, "mp4", tmp_path / "long.mp4", hwEncode="cpu")
    assert j["status"] == "completed", j["error"]
    assert abs(j["result"]["duration"] - 50 / FPS) < 0.35 and j["result"]["packets"] in (49, 50, 51)
    client.patch(f"/layers/{bottom}", json={"start": 0})


def test_reorder_delete_and_persistence(client, stack, tmp_path):
    comp, top, bottom = stack
    assert client.post("/layers/reorder", json={"order": [bottom, top]}).json()["comp"]["layers"][0]["id"] == bottom
    assert client.post("/layers/reorder", json={"order": [bottom]}).status_code == 400
    client.post("/layers/reorder", json={"order": [top, bottom]})
    client.patch(f"/layers/{top}", json={"trimIn": 5, "trimOut": 30, "opacity": 0.8, "x": 12, "name": "Hero"})
    assert client.post("/project/save").status_code == 200
    root = comp.root
    assert (root / "comp.roto").exists() and (root / "layers" / top / "project.roto").exists()
    r = client.post("/project/open", json={"path": str(root)})                          # reopen the whole composition
    assert r.status_code == 200
    lay = r.json()["comp"]["layers"]
    h = next(x for x in lay if x["id"] == top)
    assert [x["id"] for x in lay] == [top, bottom] and (h["name"], h["trimIn"], h["trimOut"], h["opacity"], h["x"]) == ("Hero", 5, 30, 0.8, 12)
    assert h["hasMask"] and r.json()["comp"]["active"] == top
    wait_cache(client)
    d = tmp_path / "copy_comp"
    r = client.post("/project/save-as", json={"path": str(d)})
    assert r.status_code == 200 and (d / "comp.roto").exists() and len(r.json()["comp"]["layers"]) == 2
    # removing a layer never deletes its files
    client.post("/project/open", json={"path": str(root)})
    assert client.delete(f"/layers/{bottom}").status_code == 200 and (root / "layers" / bottom / "project.roto").exists()
    r = client.delete(f"/layers/{top}")
    assert r.status_code == 400 and "at least one layer" in r.json()["detail"]["message"]


def test_object_selection_by_box_and_background_click(client, clips):
    client.post("/video/import", json={"path": str(clips["mp4"])})
    wait_cache(client)
    client.patch("/project/settings", json={"engine": "onnx" if onnx_engine.onnx_status()[0] else "opencv", "aiModel": "fast"})
    f = 12
    gt = make_frame(f)[1]
    ys, xs = np.where(gt > 0)
    box = [int(xs.min()) - 6, int(ys.min()) - 6, int(xs.max() - xs.min()) + 12, int(ys.max() - ys.min()) + 12]
    r = client.post("/segmentation/select", json={"frame": f, "mode": "replace", "box": box})
    assert r.status_code == 200, r.text
    assert iou(decode_mask(client, f), gt) > 0.9
    r = client.post("/segmentation/select", json={"frame": 3, "mode": "add", "points": [[4, 4]]})   # empty corner
    d = r.json()["detail"]
    assert r.status_code == 400 and d["stage"] == "object-selection" and any("box" in h.lower() for h in d["hints"])
    engine_registry.reset_engines()


def test_ai_model_setting_and_auto_choice_logic():
    from rotobrush.engines.onnx_engine import decide_model
    photo = np.zeros((100, 100), np.float32)
    photo[20:80, 20:80] = 1.0
    assert decide_model(photo, photo) == "general"                                   # same area -> general
    face_only = np.zeros((100, 100), np.float32)
    face_only[30:60, 30:60] = 1.0
    full = np.zeros((100, 100), np.float32)
    full[10:90, 20:80] = 1.0
    assert decide_model(face_only, full) == "anime"                                  # anime finds far more subject
    blurry = np.full((100, 100), 0.5, np.float32)
    assert decide_model(face_only, blurry * 0 + np.where(full > 0, 0.5, 0.0)) == "general"   # ...but not if it is unsure


def test_ai_model_endpoint_validates(client, clips):
    client.post("/video/import", json={"path": str(clips["mp4"])})
    wait_cache(client)
    assert client.patch("/project/settings", json={"aiModel": "anime"}).json()["aiModel"] == "anime"
    assert client.patch("/project/settings", json={"aiModel": "bogus"}).status_code == 400
    client.patch("/project/settings", json={"aiModel": "auto"})


big_models = pytest.mark.skipif(not {"general", "anime"} <= set(onnx_engine.available_models()), reason="isnet models not installed")


@big_models
def test_general_and_anime_models_run_and_auto_picks_one(client, clips):
    engine_registry.reset_engines()
    client.post("/video/import", json={"path": str(clips["mp4"])})
    wait_cache(client)
    client.patch("/project/settings", json={"engine": "onnx", "aiModel": "auto"})
    an = client.post("/segmentation/analyze", json={}).json()
    assert an["objects"] and an["model"] in ("general", "anime"), an
    assert S.project.data["aiModelResolved"] == an["model"]
    gt = make_frame(an["frame"])[1]
    gy, gx = np.where(gt > 0)
    x, y, w, h = an["objects"][0]["bbox"]
    assert abs(x - gx.min()) < 25 and abs(y - gy.min()) < 25
    client.patch("/project/settings", json={"engine": "auto"})
    engine_registry.reset_engines()


needs_depth = pytest.mark.skipif(depthmod.depth_model_path() is None or not onnx_engine.onnx_status()[0], reason="depth model not installed")


@needs_depth
def test_depth_preview_and_exports(client, clips, tmp_path):
    client.post("/video/import", json={"path": str(clips["mp4"])})
    wait_cache(client)
    client.patch("/layers/" + S.comp.data["active"], json={"trimIn": 4, "trimOut": 9})
    r = client.get("/render/5", params={"mode": "depth", "q": 100})
    assert r.status_code == 200
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (H, W) and img.std() > 5, "the depth view must show real structure, not a flat image"
    assert (S.project.root / "previews" / "depth" / "000005.png").exists()                  # cached on disk
    folder = tmp_path / "depth_seq"
    j = _job(client, "depth", folder, depthFormat="png16", range="work")
    assert j["status"] == "completed", j["error"]
    files = sorted(Path(j["result"]["path"]).glob("depth_*.png"))
    assert len(files) == 6                                                                  # trim 4..9
    d0 = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
    assert d0.dtype == np.uint16 and d0.shape == (H, W) and d0.max() > 40000 and d0.min() < 20000
    j2 = _job(client, "depth", tmp_path / "inv", depthFormat="png16", range="work", depthInvert=True)
    d1 = cv2.imread(str(sorted(Path(j2["result"]["path"]).glob("depth_*.png"))[0]), cv2.IMREAD_UNCHANGED)
    assert abs(float(d0.mean() + d1.mean()) / 65535.0 - 1.0) < 0.15, "invert must flip near and far"
    j3 = _job(client, "depth", tmp_path / "depth.mp4", depthFormat="mp4", range="work", hwEncode="cpu")
    assert j3["status"] == "completed", j3["error"]
    assert j3["result"]["validated"] and j3["result"]["packets"] in (5, 6, 7)
    cap = cv2.VideoCapture(str(tmp_path / "depth.mp4"))
    ok, fr = cap.read()
    assert ok and fr.std() > 5
    client.patch("/layers/" + S.comp.data["active"], json={"trimIn": 0, "trimOut": N - 1})
    depthmod.DepthEstimator.reset()


def test_depth_without_the_model_is_a_clear_error(client, clips, tmp_path, monkeypatch):
    monkeypatch.setattr(depthmod, "depth_model_path", lambda: None)
    client.post("/video/import", json={"path": str(clips["mp4"])})
    wait_cache(client)
    r = client.post("/export", json={"kind": "depth", "output": str(tmp_path / "d.mp4"), "settings": {}})
    assert r.status_code == 400 and r.json()["detail"]["stage"] == "depth-model" and r.json()["detail"]["hints"]


# =====================================================================================================
# Revision 2: crisp edges, always-composite preview, import dialog (trim), no-lag background AI
# =====================================================================================================
import threading  # noqa: E402


def _soft_per_contour_px(a_u8: np.ndarray) -> float:
    soft = int(((a_u8 > 12) & (a_u8 < 243)).sum())
    contours = cv2.findContours((a_u8 > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    assert contours, "rendered mask is empty: no foreground contour was found"
    cnt = max(contours, key=cv2.contourArea)
    return soft / cv2.arcLength(cnt, True)


def test_edges_are_crisp_by_default_and_feather_is_only_what_the_user_adds(client, clips):
    client.post("/video/import", json={"path": str(clips["mp4"])})          # its own project: never a stale shared fixture
    wait_cache(client)
    _fill_masks(S.project)
    f = 10
    mask_png = lambda: cv2.imdecode(np.frombuffer(client.get(f"/render/{f}", params={"mode": "mask"}).content, np.uint8), cv2.IMREAD_GRAYSCALE)  # noqa: E731
    assert client.get("/project").json()["maskSettings"]["hardEdge"] is True
    crisp = _soft_per_contour_px(mask_png())
    assert crisp < 0.3, f"default edge must have no feather, got {crisp:.2f} soft pixels per contour pixel"
    client.patch("/project/settings", json={"maskSettings": {"feather": 4}})
    feathered = _soft_per_contour_px(mask_png())
    assert feathered > 5 and feathered > 10 * max(crisp, 0.05), (crisp, feathered)
    client.patch("/project/settings", json={"maskSettings": {"feather": 0, "hardEdge": False}})
    assert _soft_per_contour_px(mask_png()) > 1.0, "the old soft behaviour is still available as an opt-out"
    client.patch("/project/settings", json={"maskSettings": {"hardEdge": True}})
    assert _soft_per_contour_px(mask_png()) < 0.3


def test_render_cache_is_scoped_to_project(tmp_path, monkeypatch):
    """Two projects can have identical frame/mask versions but different mask pixels."""
    from collections import OrderedDict
    from rotobrush import matte, ops

    monkeypatch.setattr(ops, "_render_cache", OrderedDict())

    class Masks:
        def __init__(self, mask):
            self.mask = mask

        def version(self, frame):
            return 1

        def has(self, frame):
            return True

        def get(self, frame):
            return self.mask.copy()

    class ProjectLike:
        def __init__(self, root, mask):
            self.root = root
            self.data = {"maskSettings": dict(matte.DEFAULT_MATTE)}
            self.masks = Masks(mask)

        def frame_bgr(self, frame):
            return np.full((*self.masks.mask.shape, 3), 120, np.uint8)

    empty = np.zeros((120, 160), np.uint8)
    filled = empty.copy()
    cv2.ellipse(filled, (80, 60), (35, 45), 0, 0, 360, 255, -1)
    a = ProjectLike(tmp_path / "project_a", empty)
    b = ProjectLike(tmp_path / "project_b", filled)

    img_a = cv2.imdecode(np.frombuffer(ops.render(a, 10, "mask", 100)[0], np.uint8), cv2.IMREAD_COLOR)
    img_b = cv2.imdecode(np.frombuffer(ops.render(b, 10, "mask", 100)[0], np.uint8), cv2.IMREAD_COLOR)
    assert not (img_a[..., 0] > 127).any()
    assert (img_b[..., 0] > 127).any(), "project B must not receive project A's cached empty mask"


def test_onnx_status_uses_one_provider_snapshot():
    """A GPU-to-CPU fallback must not produce a mismatched provider/accelerated pair."""
    class ChangingRuntime:
        def __init__(self):
            self.lock = threading.Lock()
            self.reads = 0

        @property
        def provider(self):
            self.reads += 1
            return "DmlExecutionProvider" if self.reads == 1 else "CPUExecutionProvider"

    eng = object.__new__(onnx_engine.OnnxEngine)
    eng._lock = threading.Lock()
    runtime = ChangingRuntime()
    eng._rts = {"fast": runtime}

    info = eng.info()
    assert info["provider"] == "DirectML"
    assert info["accelerated"] is True
    assert runtime.reads == 1, "provider must be read only once for one status response"


def test_detected_masks_are_stored_hard(client, clips):
    from rotobrush.engines import cvseg
    fr, gt = make_frame(12)
    ys, xs = np.where(gt > 0)
    m = cvseg.grabcut_box(fr, (int(xs.min()) - 8, int(ys.min()) - 8, int(xs.max() - xs.min()) + 16, int(ys.max() - ys.min()) + 16))
    assert set(np.unique(m).tolist()) <= {0, 255}, "segmentation output must be 0/255, never blurred"


def test_composite_is_transparent_png_where_nothing_covers_and_jpeg_when_opaque(client, stack):
    comp, top, bottom = stack
    client.patch(f"/layers/{bottom}", json={"visible": False})
    r = client.get("/comp/render/6")
    assert r.headers["content-type"] == "image/png"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img.shape == (H, W, 4) and img[..., 3].min() == 0 and img[..., 3].max() == 255      # matte -> real transparency
    client.patch(f"/layers/{bottom}", json={"visible": True})
    assert client.get("/comp/render/6").headers["content-type"] == "image/jpeg"
    # cached: a repeat is identical; a mask edit invalidates it
    a1 = client.get("/comp/render/6").content
    assert client.get("/comp/render/6").content == a1
    comp.project(top).masks.set(6, np.zeros((comp.project(top).proxy[1], comp.project(top).proxy[0]), np.uint8))
    assert client.get("/comp/render/6").content != a1
    _fill_masks(comp.project(top))


def test_import_dialog_endpoints_probe_preview_and_stage(client, clips):
    r = client.post("/video/probe", json={"path": str(clips["mp4"])})
    assert r.status_code == 200 and (r.json()["width"], r.json()["height"], r.json()["frame_count"]) == (W, H, N)
    r = client.get("/video/frame", params={"path": str(clips["mp4"]), "index": 5, "w": 320})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (180, 320, 3)
    assert client.get("/video/frame", params={"path": str(clips["mp4"]), "index": 9999}).status_code == 200   # clamped
    assert client.get("/video/frame", params={"path": str(clips["mp4"]) + ".nope", "index": 0}).status_code == 400
    with open(clips["mp4"], "rb") as fh:
        r = client.post("/video/stage", files={"file": ("staged.mp4", fh, "video/mp4")})
    assert r.status_code == 200 and Path(r.json()["path"]).exists() and r.json()["info"]["width"] == W
    assert client.get("/system/status").status_code == 200       # staging imported nothing
    r = client.post("/video/probe", json={"path": str(clips["mp4"].with_suffix(".txt"))})
    assert r.status_code == 400


def test_import_with_trim_caches_only_the_kept_frames(client, clips):
    r = client.post("/video/import", json={"path": str(clips["mp4"]), "trimIn": 10, "trimOut": 29, "name": "My cut"})
    assert r.status_code == 200, r.text
    L = r.json()["comp"]["layers"][0]
    assert (L["name"], L["trimIn"], L["trimOut"], L["length"]) == ("My cut", 10, 29, 20)
    assert r.json()["workArea"] == {"in": 10, "out": 29} and r.json()["comp"]["duration"] == 20
    wait_cache(client)
    files = sorted(int(f.stem) for f in (S.project.frames_dir).glob("*.jpg"))
    assert files == list(range(10, 30)), "only the trimmed range may be extracted, named by source frame"
    tl = client.get("/timeline").json()
    assert tl["cache"] == {"frames": 20, "complete": True, "needed": 20}
    assert client.get("/render/3", params={"mode": "original"}).status_code == 200      # outside the cut still decodes on demand
    assert client.get("/render/15", params={"mode": "original"}).status_code == 200
    # widening the trim later re-caches the larger range in the background
    lid = r.json()["comp"]["active"]
    client.patch(f"/layers/{lid}", json={"trimIn": 0})
    wait_cache(client)
    assert sorted(int(f.stem) for f in S.project.frames_dir.glob("*.jpg"))[0] == 0
    assert client.get("/timeline").json()["cache"]["needed"] == 30
    # invalid cut
    assert client.post("/video/import", json={"path": str(clips["mp4"]), "trimIn": 30, "trimOut": 5}).status_code == 400


def test_add_layer_with_cut_name_and_start(client, clips, clip2):
    client.post("/video/import", json={"path": str(clips["mp4"])})
    wait_cache(client)
    r = client.post("/layers/add", json={"path": str(clip2), "start": 12, "trimIn": 4, "trimOut": 13, "name": "Intro"})
    assert r.status_code == 200, r.text
    L = r.json()["comp"]["layers"][0]
    assert (L["name"], L["start"], L["trimIn"], L["trimOut"]) == ("Intro", 12, 4, 13)
    wait_cache(client)
    assert sorted(int(f.stem) for f in S.project.frames_dir.glob("*.jpg")) == list(range(4, 14))


def test_previews_and_timeline_stay_responsive_while_tracking_runs(client, clips):
    """Regression: a tracking job used to hold the project and mask locks for its whole run, freezing every preview."""
    client.post("/video/import", json={"path": str(clips["mp4"])})
    wait_cache(client)
    client.patch("/project/settings", json={"engine": "opencv"})
    S.project.masks.set(0, make_frame(0)[1])
    S.project.add_keyframe(0, "user")
    job = client.post("/tracking/propagate", json={"frame": 0, "direction": "forward"}).json()
    lat = []
    t_end = time.time() + 8
    while time.time() < t_end:
        j = next(x for x in client.get("/jobs").json() if x["id"] == job["id"])
        if j["status"] != "running" and lat:
            break
        if j["status"] == "running":
            for path, params in (("/render/3", {"mode": "mask"}), ("/timeline", {}), ("/comp/render/3", {}), ("/project", {})):
                t0 = time.time()
                assert client.get(path, params=params).status_code == 200
                lat.append(time.time() - t0)
        time.sleep(0.05)
    assert len(lat) >= 8, "the job finished before it could be measured"
    assert max(lat) < 1.0, f"a preview waited {max(lat):.2f}s behind the tracking job"
    assert wait_job(client, job["id"])["status"] == "completed"
    client.patch("/project/settings", json={"engine": "auto"})


def _auto_project(client, clips, engine="opencv"):
    client.post("/video/import", json={"path": str(clips["mp4"])})
    wait_cache(client)
    client.patch("/project/settings", json={"engine": engine, "aiModel": "fast"})
    return S.project


def test_auto_update_carries_an_edit_to_neighbours_and_undoes_in_one_step(client, clips):
    p = _auto_project(client, clips)
    f = 10
    p.masks.set(f, make_frame(f)[1])
    original = p.masks.get(f).copy()
    # the user edits frame 10 ...
    assert client.post("/mask/stroke", json={"frame": f, "mode": "add", "points": [[20, 20], [60, 20]], "size": 12, "hardness": 1}).status_code == 200
    edited = p.masks.get(f).copy()
    assert not np.array_equal(edited, original)
    # ... then the app updates the neighbours in the background
    job = client.post("/mask/auto-update", json={"frame": f, "forward": 5, "backward": 2}).json()
    j = wait_job(client, job["id"])
    assert j["status"] == "completed" and j["result"]["frames"] == 7, j
    assert [i for i in p.masks.frames() if i != f] == [8, 9, 11, 12, 13, 14, 15], "limited to the requested range"
    assert iou(p.masks.get(12), make_frame(12)[1]) > 0.8
    # ONE undo reverts the edit AND everything the AI derived from it
    assert client.post("/mask/undo").json()["label"] == "Brush"
    assert p.masks.frames() == [f] and np.array_equal(p.masks.get(f), original)
    assert client.post("/mask/redo").status_code == 200
    # nothing to do without a mask / with zero range
    assert client.post("/mask/auto-update", json={"frame": 33, "forward": 5}).json() == {"skipped": True}
    assert client.post("/mask/auto-update", json={"frame": f, "forward": 0, "backward": 0}).json() == {"skipped": True}
    client.patch("/project/settings", json={"engine": "auto"})


def test_a_new_edit_cancels_the_background_update_and_is_never_overwritten(client, clips):
    p = _auto_project(client, clips)
    p.masks.set(0, make_frame(0)[1])
    job = client.post("/mask/auto-update", json={"frame": 0, "forward": 39, "backward": 0}).json()
    time.sleep(0.5)
    n_before = len(p.masks.frames())
    t0 = time.time()
    r = client.post("/mask/stroke", json={"frame": 20, "mode": "add", "points": [[100, 100], [140, 100]], "size": 30, "hardness": 1})
    took = time.time() - t0
    assert r.status_code == 200 and took < 2.5, "the edit must not wait for the whole background run"
    j = next(x for x in client.get("/jobs").json() if x["id"] == job["id"])
    assert j["status"] == "cancelled", j["status"]
    mine = p.masks.get(20).copy()
    time.sleep(0.6)
    assert np.array_equal(p.masks.get(20), mine), "the cancelled job must not overwrite the user's frame"
    assert len(p.masks.frames()) < N, "it was stopped part-way, not allowed to run to the end"
    assert len(p.masks.frames()) >= n_before
    client.patch("/project/settings", json={"engine": "auto"})


def test_lasso_applies_instantly_and_ai_snaps_it_to_edges_in_the_background(client, clips):
    p = _auto_project(client, clips)
    f = 12
    gt = make_frame(f)[1]
    cnt = max(cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)[:, 0, :].astype(np.float32)
    c = cnt.mean(0)
    rough = ((cnt - c) * 1.08 + c + 4).astype(np.int32)
    raw = np.zeros_like(gt)
    cv2.fillPoly(raw, [rough], 255)
    client.post("/mask/clear", json={"frame": f})
    t0 = time.time()
    r = client.post("/mask/lasso", json={"frame": f, "polygon": rough.tolist(), "mode": "add", "ai_refine": True}).json()
    assert time.time() - t0 < 1.0 and "refining" in r, r
    assert iou(p.masks.get(f), raw) > 0.95, "the rough polygon is applied immediately, before any AI work"
    j = wait_job(client, r["refining"])
    assert j["status"] == "completed" and j["result"]["applied"] is True, j
    assert iou(p.masks.get(f), gt) > iou(raw, gt), "after the background step the edge follows the object better"
    assert client.post("/mask/undo").json()["label"] == "Lasso"       # lasso + refinement are ONE undo step
    assert not (p.masks.get(f) > 127).any()
    client.patch("/project/settings", json={"engine": "auto"})


def test_ai_refine_is_dropped_if_the_frame_is_edited_again(client, clips):
    p = _auto_project(client, clips)
    f = 14
    poly = [[100, 80], [260, 80], [260, 260], [100, 260]]
    client.post("/mask/clear", json={"frame": f})
    r = client.post("/mask/lasso", json={"frame": f, "polygon": poly, "mode": "add", "ai_refine": True}).json()
    client.post("/mask/stroke", json={"frame": f, "mode": "remove", "points": [[180, 170]], "size": 40, "hardness": 1})  # newer edit
    j = wait_job(client, r["refining"])
    assert j["status"] in ("cancelled", "completed")
    if j["status"] == "completed":
        assert j["result"]["applied"] is False
    assert p.masks.get(f)[170, 180] < 60, "the user's later erase must survive"
    client.patch("/project/settings", json={"engine": "auto"})


@needs_model
def test_interactive_ai_calls_jump_ahead_of_background_ones():
    class SlowSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, *a, **k):
            time.sleep(0.05)
            return [np.zeros((1, 1, 4, 4), np.float32)]

    rt = onnx_engine.OrtRuntime(onnx_engine.find_model(), force_cpu=True)
    rt.session = SlowSession()
    stop = threading.Event()

    def background():
        while not stop.is_set():
            rt.infer(np.zeros((1, 3, 4, 4), np.float32))

    th = threading.Thread(target=background)
    th.start()
    time.sleep(0.1)
    waits = []
    with onnx_engine.interactive():
        for _ in range(6):
            t0 = time.time()
            rt.infer(np.zeros((1, 3, 4, 4), np.float32))
            waits.append(time.time() - t0)
            time.sleep(0.03)
    stop.set()
    th.join()
    assert max(waits) < 0.3, f"an interactive call waited {max(waits):.2f}s behind background inference"
    del rt
    engine_registry.reset_engines()


@needs_model
def test_status_never_claims_a_gpu_when_directml_exists_but_no_device_does(client, onnx_project, monkeypatch):
    """CI runners have the DirectML runtime but no GPU: ORT lists the provider, then fails to create a device."""
    import onnxruntime as ort
    real = ort.InferenceSession

    def no_device(path, sess_options=None, providers=None, **kw):
        if providers and providers[0] == "DmlExecutionProvider":
            raise RuntimeError("No devices detected that match the filter criteria")
        return real(path, sess_options=sess_options, providers=["CPUExecutionProvider"], **kw)

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", no_device)
    engine_registry.reset_engines()
    eng = client.get("/system/status").json()["active_engine"]
    assert (eng["ready"], eng["provider"], eng["accelerated"]) == (False, "pending", False)   # not guessing "DirectML"
    engine_registry.get_engine("onnx").runtime()
    eng = client.get("/system/status").json()["active_engine"]
    assert (eng["ready"], eng["provider"], eng["accelerated"]) == (True, "CPU", False)
    engine_registry.reset_engines()
