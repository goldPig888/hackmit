"""Ego (rider/wearer) motion tracking in stabilized frame."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from collections import deque


@dataclass
class EgoState:
    """Ego (rider/wearer) state in HALO coordinate frame."""
    # Position [X, Y, Z] (meters) - rider is at origin by definition
    position: np.ndarray
    
    # Velocity [Vx, Vy, Vz] (m/s)
    velocity: np.ndarray
    
    # Orientation (yaw angle in radians from forward)
    yaw: float
    
    # Yaw rate (rad/s)
    yaw_rate: float
    
    # Speed (m/s)
    speed: float
    
    # Metadata
    timestamp_s: float
    confidence: float

    def predict_future_position(self, dt: float) -> np.ndarray:
        """Predict future ego position using bicycle model.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Predicted position [X, Y, Z]
        """
        # Bicycle model for trajectory prediction
        # If yaw rate is small, approximate as straight line
        if abs(self.yaw_rate) < 1e-4:
            # Straight line motion
            dx = self.speed * np.sin(self.yaw) * dt
            dy = self.speed * np.cos(self.yaw) * dt
            dz = 0.0
        else:
            # Circular motion (turning)
            radius = self.speed / self.yaw_rate
            angle_change = self.yaw_rate * dt
            
            # Change in position due to turning
            dx = radius * (1 - np.cos(angle_change))
            dy = radius * np.sin(angle_change)
            dz = 0.0
        
        return self.position + np.array([dx, dy, dz])

    def predict_future_path(self, horizon_s: float = 4.0, step_s: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
        """Predict ego trajectory for future time horizon.
        
        Args:
            horizon_s: Prediction horizon in seconds
            step_s: Time step for prediction
            
        Returns:
            Tuple of (times, path) where path is (N, 3) array of positions
        """
        times = np.arange(0.0, horizon_s + step_s / 2, step_s)
        path = np.zeros((len(times), 3))
        
        current_pos = self.position.copy()
        current_yaw = self.yaw
        
        for i, dt in enumerate(times):
            if i == 0:
                path[i] = current_pos
                continue
            
            # Integrate motion
            if abs(self.yaw_rate) < 1e-4:
                # Straight line
                dx = self.speed * np.sin(current_yaw) * dt
                dy = self.speed * np.cos(current_yaw) * dt
            else:
                # Circular motion
                radius = self.speed / self.yaw_rate
                angle_change = self.yaw_rate * dt
                dx = radius * (1 - np.cos(angle_change))
                dy = radius * np.sin(angle_change)
                current_yaw += angle_change
            
            current_pos = current_pos + np.array([dx, dy, 0.0])
            path[i] = current_pos
        
        return times, path


class EgoTracker:
    """Track ego (rider/wearer) motion using IMU data."""

    def __init__(self, initial_speed: float = 1.5, history_size: int = 100):
        """Initialize ego tracker.
        
        Args:
            initial_speed: Initial walking speed assumption (m/s)
            history_size: Size of state history
        """
        self.initial_speed = initial_speed
        self.history_size = history_size
        self.state_history: deque[EgoState] = deque(maxlen=history_size)
        
        # Current state
        self.current_state: Optional[EgoState] = None
        
        # Default: rider at origin, facing forward
        self.default_state = EgoState(
            position=np.array([0.0, 0.0, 0.0]),  # At origin
            velocity=np.array([0.0, initial_speed, 0.0]),  # Moving forward
            yaw=0.0,  # Facing forward
            yaw_rate=0.0,
            speed=initial_speed,
            timestamp_s=0.0,
            confidence=0.5
        )

    def update_from_imu(self, yaw: float, yaw_rate: float, timestamp_s: float,
                      speed: Optional[float] = None, confidence: float = 0.8) -> EgoState:
        """Update ego state from IMU data.
        
        Args:
            yaw: Current yaw angle (radians)
            yaw_rate: Current yaw rate (rad/s)
            speed: Current speed (m/s), uses default if None
            timestamp_s: Current timestamp
            confidence: Confidence in IMU measurement
            
        Returns:
            Updated ego state
        """
        if speed is None:
            speed = self.initial_speed
        
        # Calculate velocity components from yaw and speed
        vx = speed * np.sin(yaw)
        vy = speed * np.cos(yaw)
        vz = 0.0  # Assume no vertical motion for walking
        
        velocity = np.array([vx, vy, vz])
        
        # Position stays at origin (we track ego relative to itself)
        position = np.array([0.0, 0.0, 0.0])
        
        self.current_state = EgoState(
            position=position,
            velocity=velocity,
            yaw=yaw,
            yaw_rate=yaw_rate,
            speed=speed,
            timestamp_s=timestamp_s,
            confidence=confidence
        )
        
        self.state_history.append(self.current_state)
        
        return self.current_state

    def update_from_gps(self, gps_position: np.ndarray, gps_velocity: np.ndarray,
                       timestamp_s: float, confidence: float = 0.9) -> EgoState:
        """Update ego state from GPS data (if available).
        
        Args:
            gps_position: GPS position [latitude, longitude, altitude]
            gps_velocity: GPS velocity [north, east, down]
            timestamp_s: Current timestamp
            confidence: Confidence in GPS measurement
            
        Returns:
            Updated ego state
        """
        # Convert GPS to local HALO frame (simplified)
        # In full implementation, would use proper coordinate transformation
        local_x = gps_velocity[1]  # East velocity
        local_y = gps_velocity[0]  # North velocity
        local_z = -gps_velocity[2]  # Down to up conversion
        
        velocity = np.array([local_x, local_y, local_z])
        speed = float(np.linalg.norm(velocity))
        
        # Calculate yaw from velocity direction
        yaw = np.arctan2(local_x, local_y)
        
        # Estimate yaw rate from recent history
        yaw_rate = 0.0
        if len(self.state_history) >= 2:
            prev_yaw = self.state_history[-1].yaw
            dt = timestamp_s - self.state_history[-1].timestamp_s
            if dt > 0:
                yaw_rate = (yaw - prev_yaw) / dt
        
        self.current_state = EgoState(
            position=np.array([0.0, 0.0, 0.0]),  # Keep at origin
            velocity=velocity,
            yaw=yaw,
            yaw_rate=yaw_rate,
            speed=speed,
            timestamp_s=timestamp_s,
            confidence=confidence
        )
        
        self.state_history.append(self.current_state)
        
        return self.current_state

    def get_current_state(self) -> EgoState:
        """Get current ego state.
        
        Returns:
            Current ego state or default if unavailable
        """
        return self.current_state if self.current_state else self.default_state

    def get_predicted_path(self, horizon_s: float = 4.0, step_s: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
        """Get predicted ego trajectory.
        
        Args:
            horizon_s: Prediction horizon in seconds
            step_s: Time step for prediction
            
        Returns:
            Tuple of (times, path)
        """
        state = self.get_current_state()
        return state.predict_future_path(horizon_s, step_s)

    def is_turning(self, threshold: float = 0.1) -> bool:
        """Check if ego is currently turning.
        
        Args:
            threshold: Yaw rate threshold for turning [rad/s]
            
        Returns:
            True if ego is turning
        """
        state = self.get_current_state()
        return abs(state.yaw_rate) > threshold

    def get_turn_direction(self) -> str:
        """Get current turn direction.
        
        Returns:
            "left", "right", or "straight"
        """
        state = self.get_current_state()
        if state.yaw_rate > 0.1:
            return "left"
        elif state.yaw_rate < -0.1:
            return "right"
        else:
            return "straight"

    def reset(self) -> None:
        """Reset ego tracker state."""
        self.state_history.clear()
        self.current_state = None