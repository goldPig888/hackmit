"""Extended Kalman Filter for nonlinear state estimation."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable
import math


@dataclass
class EKFState:
    """EKF state vector for object tracking in HALO frame."""
    # State: [X, Y, Z, Vx, Vy, Vz] (position and velocity)
    x: np.ndarray
    
    # Covariance matrix (6x6)
    P: np.ndarray
    
    timestamp_s: float


class ExtendedKalmanFilter:
    """Extended Kalman Filter for nonlinear object state estimation.
    
    The EKF is necessary because our measurements (bearing, elevation, distance)
    are nonlinear functions of the state. The standard Kalman filter only works
    for linear systems, but EKF can handle nonlinear measurement models through
    linearization using Jacobians.
    
    State: [X, Y, Z, Vx, Vy, Vz] - 3D position and velocity in HALO frame
    Measurement: [bearing, elevation, distance] - Spherical coordinates
    """

    def __init__(self, initial_position: np.ndarray, initial_velocity: Optional[np.ndarray] = None,
                 process_noise: float = 0.1, measurement_noise: float = 0.5):
        """Initialize EKF for object tracking.
        
        Args:
            initial_position: Initial 3D position [X, Y, Z] (meters)
            initial_velocity: Initial 3D velocity [Vx, Vy, Vz] (m/s)
            process_noise: Process noise covariance scalar
            measurement_noise: Measurement noise covariance scalar
        """
        # State vector: [X, Y, Z, Vx, Vy, Vz]
        if initial_velocity is None:
            initial_velocity = np.zeros(3)
        
        self.x = np.concatenate([initial_position, initial_velocity])
        
        # Initial covariance (high uncertainty for new objects)
        self.P = np.diag([
            1.0, 1.0, 1.0,  # Position uncertainty
            2.0, 2.0, 2.0   # Velocity uncertainty
        ])
        
        # Process noise covariance (Q)
        self.Q = process_noise * np.eye(6)
        
        # Measurement noise covariance (R)
        self.R = measurement_noise * np.eye(3)
        
        self.timestamp_s: Optional[float] = None

    def predict(self, dt: float) -> None:
        """Predict state forward by time step.
        
        State transition model (constant velocity):
        x(t+dt) = x(t) + v(t) * dt
        v(t+dt) = v(t)
        
        Args:
            dt: Time step in seconds
        """
        dt = max(0.001, min(dt, 1.0))  # Clamp dt to reasonable range
        
        # State transition matrix (F)
        F = np.array([
            [1, 0, 0, dt, 0, 0],   # X = X + Vx*dt
            [0, 1, 0, 0, dt, 0],   # Y = Y + Vy*dt
            [0, 0, 1, 0, 0, dt],   # Z = Z + Vz*dt
            [0, 0, 0, 1, 0, 0],   # Vx = Vx
            [0, 0, 0, 0, 1, 0],   # Vy = Vy
            [0, 0, 0, 0, 0, 1]    # Vz = Vz
        ], dtype=float)
        
        # Predict state: x_pred = F * x
        self.x = F @ self.x
        
        # Predict covariance: P_pred = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement: np.ndarray, dt: float, timestamp_s: Optional[float] = None) -> None:
        """Update state with new measurement.
        
        Measurement: [bearing, elevation, distance] (spherical coordinates)
        These are nonlinear functions of the state, requiring EKF linearization.
        
        Args:
            measurement: Measurement vector [bearing, elevation, distance]
            dt: Time step since last update
            timestamp_s: Current timestamp (optional)
        """
        # First predict to current time
        self.predict(dt)
        
        # Calculate expected measurement from state
        h = self._measurement_function(self.x)
        
        # Calculate measurement residual
        y = measurement - h
        
        # Calculate measurement Jacobian (H)
        H = self._measurement_jacobian(self.x)
        
        # Calculate innovation covariance: S = H * P * H^T + R
        S = H @ self.P @ H.T + self.R
        
        # Calculate Kalman gain: K = P * H^T * S^-1
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state: x = x + K * y
        self.x = self.x + K @ y
        
        # Update covariance: P = (I - K * H) * P
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P
        
        if timestamp_s is not None:
            if self.timestamp_s is None:
                self.timestamp_s = timestamp_s - dt
            self.timestamp_s += dt

    def _measurement_function(self, x: np.ndarray) -> np.ndarray:
        """Convert state to measurement space (nonlinear).
        
        State: [X, Y, Z, Vx, Vy, Vz]
        Measurement: [bearing, elevation, distance]
        
        Args:
            x: State vector
            
        Returns:
            Measurement vector [bearing, elevation, distance]
        """
        X, Y, Z = x[0], x[1], x[2]
        
        # Bearing: angle from forward (Y) axis
        bearing = np.arctan2(X, Y)
        
        # Elevation: angle from horizontal plane
        horizontal_distance = np.sqrt(X**2 + Y**2)
        elevation = np.arctan2(Z, horizontal_distance)
        
        # Distance: Euclidean distance
        distance = np.sqrt(X**2 + Y**2 + Z**2)
        
        return np.array([bearing, elevation, distance])

    def _measurement_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Calculate Jacobian of measurement function.
        
        Jacobian H = ∂h/∂x where h is the measurement function.
        
        Args:
            x: State vector
            
        Returns:
            3x6 Jacobian matrix
        """
        X, Y, Z = x[0], x[1], x[2]
        
        # Avoid division by zero
        X2_Y2 = X**2 + Y**2
        X2_Y2_Z2 = X2_Y2 + Z**2
        distance = np.sqrt(X2_Y2_Z2)
        horizontal_distance = np.sqrt(X2_Y2)
        
        # Derivatives for bearing = atan2(X, Y)
        # ∂bearing/∂X = Y / (X^2 + Y^2)
        # ∂bearing/∂Y = -X / (X^2 + Y^2)
        d_bearing_dX = Y / max(X2_Y2, 1e-6)
        d_bearing_dY = -X / max(X2_Y2, 1e-6)
        
        # Derivatives for elevation = atan2(Z, sqrt(X^2 + Y^2))
        # ∂elevation/∂X = -X*Z / (sqrt(X^2+Y^2) * (X^2+Y^2+Z^2))
        # ∂elevation/∂Y = -Y*Z / (sqrt(X^2+Y^2) * (X^2+Y^2+Z^2))
        # ∂elevation/∂Z = sqrt(X^2+Y^2) / (X^2+Y^2+Z^2)
        if horizontal_distance > 1e-6:
            d_elevation_dX = -X * Z / (horizontal_distance * X2_Y2_Z2)
            d_elevation_dY = -Y * Z / (horizontal_distance * X2_Y2_Z2)
            d_elevation_dZ = horizontal_distance / X2_Y2_Z2
        else:
            d_elevation_dX = 0.0
            d_elevation_dY = 0.0
            d_elevation_dZ = 1.0
        
        # Derivatives for distance = sqrt(X^2 + Y^2 + Z^2)
        # ∂distance/∂X = X / distance
        # ∂distance/∂Y = Y / distance
        # ∂distance/∂Z = Z / distance
        if distance > 1e-6:
            d_distance_dX = X / distance
            d_distance_dY = Y / distance
            d_distance_dZ = Z / distance
        else:
            d_distance_dX = 1.0
            d_distance_dY = 0.0
            d_distance_dZ = 0.0
        
        # Jacobian matrix (3x6)
        H = np.array([
            [d_bearing_dX, d_bearing_dY, 0.0, 0.0, 0.0, 0.0],      # ∂bearing/∂state
            [d_elevation_dX, d_elevation_dY, d_elevation_dZ, 0.0, 0.0, 0.0],  # ∂elevation/∂state
            [d_distance_dX, d_distance_dY, d_distance_dZ, 0.0, 0.0, 0.0]   # ∂distance/∂state
        ], dtype=float)
        
        return H

    @property
    def position(self) -> np.ndarray:
        """Get current position estimate."""
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        """Get current velocity estimate."""
        return self.x[3:].copy()

    @property
    def covariance(self) -> np.ndarray:
        """Get current covariance matrix."""
        return self.P.copy()

    @property
    def position_uncertainty(self) -> float:
        """Get position uncertainty (trace of position covariance)."""
        return float(np.sqrt(np.trace(self.P[:3, :3]) / 3))

    def get_state_with_uncertainty(self) -> tuple[np.ndarray, np.ndarray]:
        """Get state and covariance for uncertainty propagation.
        
        Returns:
            Tuple of (state vector, covariance matrix)
        """
        return self.x.copy(), self.P.copy()

    def reset(self, initial_position: np.ndarray, initial_velocity: Optional[np.ndarray] = None) -> None:
        """Reset filter to initial state.
        
        Args:
            initial_position: New initial position
            initial_velocity: New initial velocity
        """
        if initial_velocity is None:
            initial_velocity = np.zeros(3)
        
        self.x = np.concatenate([initial_position, initial_velocity])
        self.P = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        self.timestamp_s = None