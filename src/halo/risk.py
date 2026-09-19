from __future__ import annotations

import math

from .models import RiskAssessment, TrackState
from .trajectory import Conflict


CLASS_MULTIPLIER = {"car": 1.0, "truck": 1.0, "bus": 1.0, "motorcycle": 0.9, "bicycle": 0.75, "person": 0.7}


def assess(track: TrackState, conflict: Conflict, direction: str, tau0_s: float = 2.5, d0_m: float = 1.2) -> RiskAssessment:
    temporal = math.exp(-conflict.time_s / tau0_s)
    spatial = math.exp(-conflict.closest_distance_m / d0_m)
    cls = CLASS_MULTIPLIER.get(track.label.lower(), 0.65)
    # Require closure or a meaningful predicted geometric overlap.
    gate = 1.0 if track.closing else 0.35
    risk = max(0.0, min(1.0, conflict.probability * temporal * spatial * cls * gate * track.confidence))
    return RiskAssessment(track.object_id, track.label, risk, direction, track.ttc_s, conflict.closest_distance_m, conflict.time_s, conflict.probability)


def intensity(risk: float) -> str | None:
    if risk < 0.30:
        return None
    if risk < 0.55:
        return "weak"
    if risk < 0.75:
        return "medium"
    return "strong"
