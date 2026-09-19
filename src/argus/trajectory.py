from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Conflict:
    closest_distance_m: float
    time_s: float
    probability: float


def predict_path(position_m: np.ndarray, velocity_mps: np.ndarray, horizon_s: float = 4.0, step_s: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(0.0, horizon_s + step_s / 2, step_s)
    return times, position_m[None, :] + times[:, None] * velocity_mps[None, :]


def rider_path(speed_mps: float, yaw_rate_rps: float, horizon_s: float = 4.0, step_s: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(0.0, horizon_s + step_s / 2, step_s)
    if abs(yaw_rate_rps) < 1e-4:
        path = np.column_stack((np.zeros_like(times), speed_mps * times))
    else:
        radius = speed_mps / yaw_rate_rps
        path = np.column_stack((radius * (1 - np.cos(yaw_rate_rps * times)), radius * np.sin(yaw_rate_rps * times)))
    return times, path


def closest_conflict(object_path: np.ndarray, rider_path_m: np.ndarray, base_sigma_m: float) -> Conflict:
    distances = np.linalg.norm(object_path - rider_path_m, axis=1)
    index = int(np.argmin(distances))
    distance = float(distances[index])
    sigma = base_sigma_m + 0.15 * index
    # A soft overlap estimate explicitly degrades with uncertainty.
    probability = float(np.exp(-0.5 * (distance / max(sigma, 0.05)) ** 2))
    return Conflict(distance, index * 0.2, probability)
