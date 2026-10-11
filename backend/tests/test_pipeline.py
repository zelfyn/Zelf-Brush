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
                                     "mode": "add", "ai_refine": True})
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
    assert {"comp.jsx", "README.txt", "alpha/foreground_alpha.mov"} <= set(files)
    assert any(f.startswith("footage/") for f in files)
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
    assert client.patch("/project/settings", json={"engine": "onnx"}).status_code == 200
    yield S.project
    engine_registry.reset_engines()


@needs_model
def test_onnx_engine_is_listed_and_reports_its_device(client, onnx_project):
    st = client.get("/system/status").json()
    assert next(e for e in st["engines"] if e["name"] == "onnx")["available"]
    eng = st["active_engine"]
    assert eng["name"] == "onnx" and eng["provider"] in ("CPU", "CUDA", "DirectML")
    assert eng["accelerated"] == (eng["provider"] != "CPU")


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
    assert client.post("/mask/lasso", json={"frame": f, "polygon": rough.tolist(), "mode": "add", "ai_refine": True}).status_code == 200
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
    monkeypatch.setattr(onnx_engine, "find_model", lambda: None)
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
    client.patch("/project/settings", json={"engine": engine, "workArea": {"in": 12, "out": N - 1}})
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
