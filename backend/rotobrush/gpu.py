"""GPU / inference capability detection (nvidia-smi + optional PyTorch)."""
from __future__ import annotations

import subprocess
import time
from typing import Any

from .config import CREATE_NO_WINDOW, find_binary

_cache: tuple[float, dict] | None = None
_torch_info: dict | None = None
_ort_info: dict | None = None


def _ort_probe() -> dict:
    global _ort_info
    if _ort_info is None:
        _ort_info = {"installed": False, "providers": []}
        try:
            import onnxruntime as ort

            _ort_info = {"installed": True, "providers": list(ort.get_available_providers())}
        except Exception:  # noqa: BLE001
            pass
    return _ort_info


def _torch_probe() -> dict:
    global _torch_info
    if _torch_info is not None:
        return _torch_info
    info: dict[str, Any] = {"installed": False, "cuda": False, "version": None}
    try:
        import torch  # noqa: WPS433 (lazy import: slow)

        info.update(installed=True, version=torch.__version__, cuda=bool(torch.cuda.is_available()))
    except Exception:  # noqa: BLE001
        pass
    _torch_info = info
    return info


def _nvidia_smi() -> dict | None:
    exe = find_binary("nvidia-smi") or "nvidia-smi"
    try:
        r = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=4, creationflags=CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    name, total, used, driver = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
    return {"name": name, "vram_total_mb": int(float(total)), "vram_used_mb": int(float(used)), "driver": driver}


def gpu_info(force: bool = False) -> dict:
    global _cache
    if _cache and not force and time.time() - _cache[0] < 4:
        return _cache[1]
    smi = _nvidia_smi()
    t = _torch_probe()
    o = _ort_probe()
    ort_cuda = "CUDAExecutionProvider" in o["providers"]
    ort_dml = "DmlExecutionProvider" in o["providers"]
    cuda = bool(smi and (t["cuda"] or ort_cuda))
    info: dict[str, Any] = {
        "detected": smi is not None,
        "name": smi["name"] if smi else None,
        "vram_total_mb": smi["vram_total_mb"] if smi else 0,
        "vram_used_mb": smi["vram_used_mb"] if smi else 0,
        "driver": smi["driver"] if smi else None,
        "torch_installed": t["installed"],
        "ort_providers": o["providers"],
        "cuda_available": cuda,
        "directml_available": ort_dml,
    }
    if cuda:
        info["mode"], info["message"] = "cuda", "CUDA available"
    elif ort_dml:
        info["mode"] = "directml"
        info["message"] = ("GPU acceleration available through DirectML" if smi else
                           "DirectML runtime installed; it needs a DirectX 12 GPU, otherwise AI runs on the CPU")
    elif smi:
        info["mode"] = "cpu"
        info["message"] = "NVIDIA GPU detected, but no GPU-enabled AI runtime is installed. Using CPU mode."
    else:
        info["mode"] = "cpu"
        info["message"] = "NVIDIA GPU not detected. Using CPU mode. Processing may be slower."
    _cache = (time.time(), info)
    return info
