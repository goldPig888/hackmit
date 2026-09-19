from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detector/tracker observation. `bbox` is x, y, width, height in pixels."""

    object_id: str
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    timestamp_s: float

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2, y + h / 2

    @property
    def area(self) -> float:
        return self.bbox[2] * self.bbox[3]


@dataclass
class TrackState:
    object_id: str
    label: str
    confidence: float
    position_m: np.ndarray
    velocity_mps: np.ndarray
    position_sigma_m: float
    ttc_s: float | None
    closing: bool
    timestamp_s: float


@dataclass(frozen=True)
class RiskAssessment:
    object_id: str
    label: str
    risk: float
    direction: Literal["left", "right", "center"]
    ttc_s: float | None
    closest_distance_m: float
    time_to_conflict_s: float
    conflict_probability: float


@dataclass(frozen=True)
class HapticEvent:
    direction: Literal["left", "right", "center"]
    risk: float
    ttc_s: float | None
    conflict_s: float
    object_id: str
    intensity: Literal["weak", "medium", "strong"]
    metadata: dict[str, float | str | None] = field(default_factory=dict)
