from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from .models import HapticEvent, RiskAssessment
from .risk import intensity


class HapticPublisher:
    """Records events locally and optionally POSTs them to an ESP32 endpoint."""

    def __init__(self, output_dir: str | Path, endpoint: str | None = None) -> None:
        self.path = Path(output_dir) / "haptics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.endpoint = endpoint or os.getenv("HALO_HAPTIC_URL")

    def publish(self, assessment: RiskAssessment) -> HapticEvent | None:
        level = intensity(assessment.risk)
        if level is None:
            return None
        event = HapticEvent(assessment.direction, assessment.risk, assessment.ttc_s, assessment.time_to_conflict_s,
                            assessment.object_id, level, {"closest_distance_m": assessment.closest_distance_m})
        payload = {"direction": event.direction, "risk": round(event.risk, 3), "ttc_s": event.ttc_s,
                   "conflict_s": round(event.conflict_s, 2), "object_id": event.object_id, "intensity": event.intensity}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload) + "\n")
        if self.endpoint:
            try:
                request = Request(self.endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
                with urlopen(request, timeout=0.25):
                    pass
            except OSError:
                # Safety output remains observable locally when hardware drops out.
                pass
        return event

    def publish_reflex(self, direction: str, object_id: str) -> HapticEvent | None:
        """Immediate-motion threat event: always strong, bypasses risk gating."""
        if direction not in ("left", "right", "center"):
            direction = "center"
        event = HapticEvent(direction, 1.0, 0.0, 0.0, object_id, "strong", {"reflex": True})
        payload = {"direction": event.direction, "risk": 1.0, "ttc_s": 0.0,
                   "conflict_s": 0.0, "object_id": event.object_id,
                   "intensity": "strong", "reflex": True}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload) + "\n")
        if self.endpoint:
            try:
                request = Request(self.endpoint, data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")
                with urlopen(request, timeout=0.25):
                    pass
            except OSError:
                pass
        return event
