"""Object state representation in stabilized world frame."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class CoordinateFrame(Enum):
    """Coordinate frame definitions."""
    CAMERA = "camera"  # Camera coordinates (moving)
    HALO = "halo"    # HALO world frame (stabilized)
    WORLD = "world"    # Global world frame


@dataclass
class ObjectState:
    """Object state in stabilized HALO coordinate frame.
    
    HALO coordinate frame:
    - +Y: Forward direction
    - +X: Right direction  
    - +Z: Up direction
    """
    object_id: str
    label: str
    
    # Position in HALO frame [X, Y, Z] (meters)
    position: np.ndarray
    
    # Velocity in HALO frame [Vx, Vy, Vz] (m/s)
    velocity: np.ndarray
    
    # Bearing and elevation (radians)
    bearing: float
    elevation: float
    
    # Relative depth estimate (meters)
    relative_depth: float
    
    # Uncertainty covariance matrix
    covariance: np.ndarray
    
    # Metadata
    confidence: float
    timestamp_s: float
    duration_tracked: float = 0.0
    is_valid: bool = True

    @property
    def speed(self) -> float:
        """Get object speed magnitude."""
        return float(np.linalg.norm(self.velocity))

    @property
    def heading(self) -> float:
        """Get object heading direction (radians from forward)."""
        return np.arctan2(self.velocity[0], self.velocity[1])

    @property
    def distance(self) -> float:
        """Get distance from origin (rider position)."""
        return float(np.linalg.norm(self.position))

    def predict_future_position(self, dt: float) -> np.ndarray:
        """Predict future position assuming constant velocity.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Predicted position [X, Y, Z]
        """
        return self.position + self.velocity * dt

    def get_bearing_rate(self) -> float:
        """Calculate rate of bearing change.
        
        Returns:
            Bearing rate in rad/s
        """
        # Calculate bearing from velocity direction
        vx, vy = self.velocity[0], self.velocity[1]
        if abs(vx) < 1e-6 and abs(vy) < 1e-6:
            return 0.0
        
        current_bearing = np.arctan2(vx, vy)
        return current_bearing  # Simplified - would need history for rate

    def is_approaching(self, approach_threshold: float = 0.1) -> bool:
        """Check if object is approaching the rider.
        
        Args:
            approach_threshold: Bearing rate threshold for approaching
            
        Returns:
            True if object appears to be approaching
        """
        # Object is approaching if it's moving toward origin
        # Check if velocity points toward origin
        to_object = self.position  # Vector from origin to object
        velocity_toward_object = -self.velocity  # Velocity toward origin
        
        # Check if velocity has component toward object
        approach_component = np.dot(to_object, velocity_toward_object)
        
        return approach_component > 0 and self.speed > 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "object_id": self.object_id,
            "label": self.label,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "bearing": float(self.bearing),
            "elevation": float(self.elevation),
            "relative_depth": float(self.relative_depth),
            "speed": float(self.speed),
            "distance": float(self.distance),
            "confidence": float(self.confidence),
            "timestamp_s": float(self.timestamp_s),
            "duration_tracked": float(self.duration_tracked),
            "is_valid": self.is_valid
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ObjectState":
        """Create ObjectState from dictionary."""
        return cls(
            object_id=data["object_id"],
            label=data["label"],
            position=np.array(data["position"]),
            velocity=np.array(data["velocity"]),
            bearing=data["bearing"],
            elevation=data["elevation"],
            relative_depth=data["relative_depth"],
            covariance=np.eye(6),  # Default covariance
            confidence=data["confidence"],
            timestamp_s=data["timestamp_s"],
            duration_tracked=data.get("duration_tracked", 0.0),
            is_valid=data.get("is_valid", True)
        )