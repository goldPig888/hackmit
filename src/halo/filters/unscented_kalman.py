"""Simplified Unscented Kalman Filter for nonlinear state estimation."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class UKFState:
    """UKF state vector for object tracking."""
    x: np.ndarray  # State vector
    P: np.ndarray  # Covariance matrix
    timestamp_s: float


class UnscentedKalmanFilter:
    """Simplified Unscented Kalman Filter for tracking.
    
    For Hackathon, a simplified UKF that handles the nonlinear
    measurement model without complex Jacobian calculations.
    Uses sigma points to approximate the nonlinear transformation.
    """

    def __init__(self, state_dim: int = 6, meas_dim: int = 3, 
                 alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        """Initialize UKF.
        
        Args:
            state_dim: State dimension
            meas_dim: Measurement dimension
            alpha: UKF alpha parameter
            beta: UKF beta parameter
            kappa: UKF kappa parameter
        """
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        
        # Calculate lambda parameter
        self.lambda_ = alpha**2 * (state_dim + kappa) - state_dim
        
        # Weights for sigma points
        self.Wm, self.Wc = self._calculate_weights()
        
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.timestamp_s: Optional[float] = None

    def _calculate_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate UKF weights for sigma points.
        
        Returns:
            Tuple of (mean weights, covariance weights)
        """
        n = self.state_dim
        
        # Mean weights
        Wm = np.zeros(2 * n + 1)
        Wm[0] = self.lambda_ / (n + self.lambda_)
        Wm[1:] = 1 / (2 * (n + self.lambda_))
        
        # Covariance weights
        Wc = np.zeros(2 * n + 1)
        Wc[0] = self.lambda_ / (n + self.lambda_) + (1 - self.alpha**2 + self.beta)
        Wc[1:] = 1 / (2 * (n + self.lambda_))
        
        return Wm, Wc

    def initialize(self, initial_state: np.ndarray, initial_covariance: Optional[np.ndarray] = None) -> None:
        """Initialize UKF with initial state.
        
        Args:
            initial_state: Initial state vector
            initial_covariance: Initial covariance matrix
        """
        self.x = initial_state.copy()
        
        if initial_covariance is None:
            self.P = np.eye(self.state_dim)
        else:
            self.P = initial_covariance.copy()
        
        self.timestamp_s = 0.0

    def generate_sigma_points(self) -> np.ndarray:
        """Generate sigma points around current state.
        
        Returns:
            Sigma points matrix (state_dim x 2*state_dim+1)
        """
        n = self.state_dim
        
        # Calculate sigma point scaling
        sqrt_P = np.linalg.cholesky((n + self.lambda_) * self.P).T
        
        # Generate sigma points
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = self.x
        
        for i in range(n):
            sigma_points[i + 1] = self.x + sqrt_P[:, i]
            sigma_points[i + 1 + n] = self.x - sqrt_P[:, i]
        
        return sigma_points.T

    def predict(self, dt: float, process_func: Optional[Callable] = None, 
                process_noise: Optional[np.ndarray] = None) -> None:
        """Predict state forward using process model.
        
        Args:
            dt: Time step
            process_func: Process function f(x, dt) (default: constant velocity)
            process_noise: Process noise covariance
        """
        if self.x is None:
            return
        
        # Default process model: constant velocity
        if process_func is None:
            def process_func(x, dt):
                # x_new = x + v * dt for position, v unchanged
                x_new = x.copy()
                x_new[:3] += x[3:] * dt  # Position += velocity * dt
                return x_new
        
        if process_noise is None:
            process_noise = 0.1 * np.eye(self.state_dim)
        
        # Generate sigma points
        sigma_points = self.generate_sigma_points()
        
        # Transform sigma points through process model
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(sigma_points.shape[1]):
            sigma_points_pred[:, i] = process_func(sigma_points[:, i], dt)
        
        # Calculate predicted mean
        x_pred = np.zeros(self.state_dim)
        for i in range(len(self.Wm)):
            x_pred += self.Wm[i] * sigma_points_pred[:, i]
        
        # Calculate predicted covariance
        P_pred = np.zeros((self.state_dim, self.state_dim))
        for i in range(len(self.Wc)):
            diff = sigma_points_pred[:, i] - x_pred
            P_pred += self.Wc[i] * np.outer(diff, diff)
        
        P_pred += process_noise
        
        self.x = x_pred
        self.P = P_pred
        self.timestamp_s += dt

    def update(self, measurement: np.ndarray, meas_func: Callable, 
              meas_noise: Optional[np.ndarray] = None) -> None:
        """Update state with measurement.
        
        Args:
            measurement: Measurement vector
            meas_func: Measurement function h(x)
            meas_noise: Measurement noise covariance
        """
        if self.x is None:
            return
        
        if meas_noise is None:
            meas_noise = 0.5 * np.eye(self.meas_dim)
        
        # Generate sigma points
        sigma_points = self.generate_sigma_points()
        
        # Transform sigma points through measurement model
        sigma_meas = np.zeros((self.meas_dim, 2 * self.state_dim + 1))
        for i in range(sigma_points.shape[1]):
            sigma_meas[:, i] = meas_func(sigma_points[:, i])
        
        # Calculate predicted measurement mean
        z_pred = np.zeros(self.meas_dim)
        for i in range(len(self.Wm)):
            z_pred += self.Wm[i] * sigma_meas[:, i]
        
        # Calculate predicted measurement covariance
        P_zz = np.zeros((self.meas_dim, self.meas_dim))
        for i in range(len(self.Wc)):
            diff = sigma_meas[:, i] - z_pred
            P_zz += self.Wc[i] * np.outer(diff, diff)
        
        P_zz += meas_noise
        
        # Calculate cross-covariance
        P_xz = np.zeros((self.state_dim, self.meas_dim))
        for i in range(len(self.Wc)):
            x_diff = sigma_points[:, i] - self.x
            z_diff = sigma_meas[:, i] - z_pred
            P_xz += self.Wc[i] * np.outer(x_diff, z_diff)
        
        # Calculate Kalman gain
        K = P_xz @ np.linalg.inv(P_zz)
        
        # Update state
        self.x = self.x + K @ (measurement - z_pred)
        
        # Update covariance
        self.P = self.P - K @ P_zz @ K.T

    @property
    def position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.x[:3].copy() if self.x is not None else np.zeros(3)

    @property
    def velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.x[3:].copy() if self.x is not None else np.zeros(3)

    @property
    def covariance(self) -> np.ndarray:
        """Get current covariance matrix."""
        return self.P.copy() if self.P is not None else np.eye(self.state_dim)