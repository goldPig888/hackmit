from __future__ import annotations

import numpy as np


class ConstantVelocityKalman:
    """Small 2-D constant-velocity Kalman filter with position-only measurements."""

    def __init__(self, position: np.ndarray, measurement_sigma_m: float = 0.6) -> None:
        self.x = np.array([position[0], position[1], 0.0, 0.0], dtype=float)
        self.P = np.diag([measurement_sigma_m**2, measurement_sigma_m**2, 16.0, 16.0])
        self.measurement_sigma_m = measurement_sigma_m

    def predict(self, dt_s: float) -> None:
        dt_s = max(0.001, min(dt_s, 1.0))
        F = np.array([[1, 0, dt_s, 0], [0, 1, 0, dt_s], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        q = 2.0
        Q = q * np.array(
            [[dt_s**4 / 4, 0, dt_s**3 / 2, 0], [0, dt_s**4 / 4, 0, dt_s**3 / 2],
             [dt_s**3 / 2, 0, dt_s**2, 0], [0, dt_s**3 / 2, 0, dt_s**2]], dtype=float
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, position: np.ndarray) -> None:
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        R = np.eye(2) * self.measurement_sigma_m**2
        innovation = np.asarray(position, dtype=float) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ H) @ self.P

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:].copy()

    @property
    def sigma_m(self) -> float:
        return float(np.sqrt(max(np.trace(self.P[:2, :2]) / 2, 0.0)))
