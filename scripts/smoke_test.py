"""Black-box check of a packaged backend: start it like the app does, import a real video, export an MP4.

  python scripts/smoke_test.py --backend backend/dist/rotobrush-backend/rotobrush-backend.exe \
                               --ffmpeg  backend/dist/rotobrush-backend/bin/ffmpeg.exe --expect-engine onnx
  python scripts/smoke_test.py --dev          # run `python -m rotobrush` from ./backend instead

The system PATH is NOT used for FFmpeg (ROTOBRUSH_NO_PATH_FALLBACK=1), so a broken bundled ffmpeg/ffprobe fails here
instead of being silently rescued by another install on the build machine.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

TOKEN = "smoke"


def call(base: str, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(f"{base}{path}", method=method, headers={"x-token": TOKEN, "Content-Type": "application/json"},
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def wait_job(base: str, job_id: str, timeout: float = 180) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, jobs = call(base, "GET", "/jobs")
        j = next(x for x in jobs if x["id"] == job_id)  # type: ignore[union-attr]
        if j["status"] in ("completed", "failed", "cancelled"):
            return j
        time.sleep(0.3)
    raise TimeoutError(job_id)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--expect-engine", help="fail unless /system/status reports this active engine (e.g. onnx)")
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="roto_smoke_"))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    env = {**os.environ, "ROTOBRUSH_DATA": str(tmp / "data"), "ROTOBRUSH_WORKSPACE": str(tmp / "ws"),
           "ROTOBRUSH_TOKEN": TOKEN, "ROTOBRUSH_NO_PATH_FALLBACK": "1"}
    repo = Path(__file__).resolve().parents[1]
    if a.dev:
        cmd, cwd = [sys.executable, "-m", "rotobrush", "--port", str(port)], repo / "backend"
        env["ROTOBRUSH_FFMPEG"], env["ROTOBRUSH_FFPROBE"] = a.ffmpeg, str(Path(a.ffmpeg).with_name(Path(a.ffmpeg).name.replace("ffmpeg", "ffprobe")))
    else:
        if not a.backend:
            ap.error("--backend is required unless --dev")
        cmd, cwd = [a.backend, "--port", str(port)], Path(a.backend).parent
    sample = tmp / "sample.mp4"
    g = subprocess.run([a.ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15:duration=2",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", str(sample)], capture_output=True, text=True)
    if g.returncode != 0:
        print("SMOKE FAIL: the bundled ffmpeg cannot run:", g.stderr.strip()[:400])
        return 1
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"{base}/health", timeout=2)
                break
            except OSError:
                time.sleep(0.3)
        else:
            print("SMOKE FAIL: backend did not start")
            return 1
        st, d = call(base, "POST", "/video/import", {"path": str(sample)})
        if st != 200:
            print("SMOKE FAIL: import rejected a valid video:", json.dumps(d.get("detail", d)))
            return 1
        assert d["project"]["width"] == 320 and d["project"]["audio_codec"] == "aac", d["project"]
        st, sysinfo = call(base, "GET", "/system/status")
        eng = sysinfo["active_engine"]
        print(f"engine: {eng['name']} on {eng['provider']} (GPU accelerated: {eng['accelerated']}); gpu: {sysinfo['gpu']['message']}")
        if a.expect_engine and eng["name"] != a.expect_engine:
            reasons = [f"{e['name']}: {e['reason']}" for e in sysinfo["engines"] if e["reason"]]
            print(f"SMOKE FAIL: expected engine '{a.expect_engine}' but '{eng['name']}' is active.", "; ".join(reasons))
            return 1
        for _ in range(100):
            if call(base, "GET", "/timeline")[1].get("cache", {}).get("complete"):
                break
            time.sleep(0.3)
        else:
            print("SMOKE FAIL: frame cache never completed")
            return 1
        poly = [[60, 30], [260, 30], [260, 150], [60, 150]]
        assert call(base, "POST", "/mask/lasso", {"frame": 0, "polygon": poly, "mode": "add", "ai_refine": False})[0] == 200
        j = wait_job(base, call(base, "POST", "/tracking/propagate", {"frame": 0, "direction": "forward"})[1]["id"])
        if j["status"] != "completed":
            print("SMOKE FAIL: tracking:", j["error"])
            return 1
        out = tmp / "out.mp4"
        j = wait_job(base, call(base, "POST", "/export", {"kind": "mp4", "output": str(out), "settings": {"hwEncode": "cpu"}})[1]["id"])
        if j["status"] != "completed" or not out.exists() or out.stat().st_size == 0:
            print("SMOKE FAIL: export:", json.dumps(j["error"]))
            return 1
        r = j["result"]
        assert r["validated"] and r["codec"] == "h264" and r["audio"], r
        print(f"SMOKE OK: import + cache + tracking + export validated ({r['bytes']} bytes, {r['frames']} frames, encoder {r['encoder']})")
        return 0
    finally:
        proc.kill()
        proc.wait()


if __name__ == "__main__":
    sys.exit(main())
