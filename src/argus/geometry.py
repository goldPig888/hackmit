from __future__ import annotations

import numpy as np


class ImageGroundProjector:
    """Explicitly approximate image-to-local-ground mapping for an MVP.

    Tune pixels_per_meter and horizon_y for the mounted camera. A calibrated
    homography should replace this before treating predictions as metric truth.
    """

    def __init__(self, frame_width: int = 1280, horizon_y: float = 280, pixels_per_meter: float = 55) -> None:
        self.frame_width = frame_width
        self.horizon_y = horizon_y
        self.pixels_per_meter = pixels_per_meter

    def project(self, center_px: tuple[float, float], bbox_height_px: float) -> np.ndarray:
        cx, cy = center_px
        # bottom-of-box rises toward horizon with distance; blend it with a
        # fixed scale term to avoid an unstable singularity at the horizon.
        ground_y = cy + bbox_height_px / 2
        forward = max(1.0, (720 - ground_y) / self.pixels_per_meter)
        lateral = (cx - self.frame_width / 2) / self.pixels_per_meter
        return np.array([lateral, forward], dtype=float)


def bearing_direction(position_m: np.ndarray, deadband_m: float = 0.35) -> str:
    if abs(float(position_m[0])) <= deadband_m:
        return "center"
    return "right" if position_m[0] > 0 else "left"
