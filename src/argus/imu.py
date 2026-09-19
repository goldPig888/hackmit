from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiderMotion:
    speed_mps: float = 4.0
    yaw_rate_rps: float = 0.0

    def update(self, yaw_rate_rps: float, acceleration_forward_mps2: float, dt_s: float) -> None:
        self.yaw_rate_rps = yaw_rate_rps
        self.speed_mps = max(0.0, min(15.0, self.speed_mps + acceleration_forward_mps2 * dt_s))
