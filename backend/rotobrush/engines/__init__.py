"""Engine registry. Add a new engine by subclassing SegmentationEngine and registering it here."""
from __future__ import annotations

import logging

from .base import EngineError, SegmentationEngine
from .opencv_engine import OpenCVEngine

log = logging.getLogger("rotobrush.ai")
_instances: dict[str, SegmentationEngine] = {}


def list_engines() -> list[dict]:
    from .onnx_engine import onnx_status
    from .torch_engine import sam2_paths, torch_available

    out = [{"name": "opencv", "available": True, "description": OpenCVEngine.description, "reason": ""}]
    ok, why = onnx_status()
    out.append({"name": "onnx", "available": ok, "reason": why,
                "description": "ONNX Runtime U²-Net (GPU via DirectML/CUDA when available)"})
    ok, why = torch_available()
    out.append({"name": "torch", "available": ok, "reason": why,
                "description": "PyTorch: Mask R-CNN detection" + (" + SAM 2" if ok and sam2_paths() else "")})
    return out


def _build(name: str) -> SegmentationEngine:
    if name == "torch":
        from .torch_engine import TorchEngine
        return TorchEngine()
    if name == "onnx":
        from .onnx_engine import OnnxEngine
        return OnnxEngine()
    return OpenCVEngine()


def get_engine(name: str = "auto") -> SegmentationEngine:
    """auto = best available: torch (+SAM 2) > onnx (GPU via DirectML/CUDA) > opencv. Never raises for a missing engine."""
    if name in _instances:
        return _instances[name]
    order = ["torch", "onnx", "opencv"] if name == "auto" else [name, "opencv"]
    for cand in order:
        if cand in _instances:
            return _instances[cand]
        if cand != "opencv" and not next(e for e in list_engines() if e["name"] == cand)["available"]:
            continue
        try:
            eng = _build(cand)
        except EngineError as e:
            log.warning("engine %s unavailable (%s)", cand, e.message)
            continue
        _instances[cand] = eng
        if name == "auto":
            _instances["auto"] = eng
        return eng
    raise EngineError("No segmentation engine could be started.")


def reset_engines() -> None:
    """Drop all cached engines and free their models (an ONNX session holds ~0.7 GB)."""
    import gc

    _instances.clear()
    gc.collect()
