from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import ImageGroundProjector, bearing_direction
from .kalman import ConstantVelocityKalman
from .models import Detection, RiskAssessment, TrackState
from .risk import assess
from .trajectory import closest_conflict, predict_path, rider_path
from .ttc import ExpansionTTC


@dataclass
class _Track:
    filter: ConstantVelocityKalman
    ttc: ExpansionTTC
    last_timestamp_s: float
    label: str


class HaloPipeline:
    """Detector-agnostic state, conflict, and interpretable-risk pipeline."""

    def __init__(self, projector: ImageGroundProjector | None = None, ttc_gate_s: float = 8.0) -> None:
        self.projector = projector or ImageGroundProjector()
        self.ttc_gate_s = ttc_gate_s
        self.tracks: dict[str, _Track] = {}

    def update(self, detection: Detection, rider_speed_mps: float = 4.0, yaw_rate_rps: float = 0.0) -> RiskAssessment | None:
        x, y, w, h = detection.bbox
        measured = self.projector.project(detection.center, h)
        track = self.tracks.get(detection.object_id)
        if track is None:
            track = _Track(ConstantVelocityKalman(measured), ExpansionTTC(), detection.timestamp_s, detection.label)
            self.tracks[detection.object_id] = track
        else:
            track.filter.predict(detection.timestamp_s - track.last_timestamp_s)
            track.filter.update(measured)
            track.last_timestamp_s = detection.timestamp_s
            track.label = detection.label
        ttc_s = track.ttc.update(detection.timestamp_s, detection.area)
        velocity = track.filter.velocity
        # Negative radial velocity means the object is coming toward the rider.
        radial_rate = float(np.dot(track.filter.position, velocity) / max(np.linalg.norm(track.filter.position), 0.1))
        closing = radial_rate < -0.10 or (ttc_s is not None and ttc_s <= self.ttc_gate_s)
        state = TrackState(detection.object_id, detection.label, detection.confidence, track.filter.position,
                           velocity, track.filter.sigma_m, ttc_s, closing, detection.timestamp_s)
        if not closing:
            return None
        _, object_path = predict_path(state.position_m, state.velocity_mps)
        _, ego_path = rider_path(rider_speed_mps, yaw_rate_rps)
        conflict = closest_conflict(object_path, ego_path, state.position_sigma_m)
        return assess(state, conflict, bearing_direction(state.position_m))
