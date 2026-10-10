"""Replaceable module interfaces. The UI and API only talk to these, never to a specific model."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

GetFrame = Callable[[int], np.ndarray]


class EngineError(Exception):
    def __init__(self, message: str, hints: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hints = hints or []


@dataclass
class Detection:
    id: str
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h in proxy pixels
    mask: np.ndarray | None = None   # optional instance mask (uint8 0/255)


class ObjectDetection(ABC):
    @abstractmethod
    def detect(self, bgr: np.ndarray) -> list[Detection]: ...


class InteractiveSegmentation(ABC):
    @abstractmethod
    def segment(self, bgr: np.ndarray, *, box: tuple[int, int, int, int] | None = None,
                fg_points: list[tuple[float, float]] | None = None,
                bg_points: list[tuple[float, float]] | None = None) -> np.ndarray:
        """Return uint8 mask (0..255) at frame resolution."""

    @abstractmethod
    def refine_prior(self, bgr: np.ndarray, prior: np.ndarray, band: int) -> np.ndarray:
        """Snap a rough mask (e.g. a lasso) to real edges."""


class TemporalPropagation(ABC):
    @abstractmethod
    def propagate(self, get_frame: GetFrame, start: int, mask: np.ndarray, step: int, last: int,
                  stop_at: int | None, cancelled: Callable[[], bool],
                  on_frame: Callable[[int, np.ndarray, float], None]) -> None:
        """Walk from `start` in direction `step` (+1/-1) until `last` or `stop_at` (exclusive)."""


class SegmentationEngine:
    name = "base"
    description = ""
    provider = "CPU"          # what actually runs the neural network: CPU / CUDA / DirectML ...
    accelerated = False       # True only when a GPU runs the inference
    detection: ObjectDetection
    interactive: InteractiveSegmentation
    temporal: TemporalPropagation

    def info(self) -> dict:
        return {"name": self.name, "description": self.description, "provider": self.provider,
                "accelerated": self.accelerated}
