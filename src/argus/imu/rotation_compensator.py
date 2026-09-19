"""Rotation compensation for stabilizing object observations."""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import deque

from .camera_pose import CameraPose, CameraPoseEstimator


@dataclass
class StabilizedObservation:
    """Object observation in stabilized world frame."""
    object_id: str
    world_ray: np.ndarray  # 3D ray in world coordinates
    bearing: float  # Bearing angle in radians
    elevation: float  # Elevation angle in radians
    timestamp_s: float
    confidence: float
    original_pixel: Tuple[float, float]  # Original pixel coordinates


class RotationCompensator:
    """Compensate for camera rotation to stabilize object observations.
    
    This is the key component that addresses the fundamental issue:
    when the camera rotates, objects appear to move in pixel coordinates
    even if they're stationary in the world. This component transforms
    observations into a stable world frame to distinguish camera motion
    from actual object motion.
    """

    def __init__(self, camera_pose_estimator: CameraPoseEstimator, history_size: int = 50):
        """Initialize rotation compensator.
        
        Args:
            camera_pose_estimator: Camera pose estimator for ego-motion
            history_size: Size of observation history for each object
        """
        self.camera_pose_estimator = camera_pose_estimator
        self.history_size = history_size
        self.object_histories: dict[str, deque[StabilizedObservation]] = {}
        
        # ARGUS local coordinate frame definition
        # +Y = forward, +X = right, +Z = up
        self.argus_frame_basis = np.eye(3)

    def compensate_observation(self, object_id: str, pixel: Tuple[float, float], 
                             confidence: float, timestamp_s: float) -> Optional[StabilizedObservation]:
        """Compensate observation for camera rotation.
        
        This implements the key operation: r_t^w = R_t * K^-1 * p_t
        transforming pixel observations into the stable world frame.
        
        Args:
            object_id: Object identifier
            pixel: (u, v) pixel coordinates
            confidence: Detection confidence
            timestamp_s: Timestamp
            
        Returns:
            Stabilized observation or None if camera pose unavailable
        """
        camera_pose = self.camera_pose_estimator.get_current_pose()
        if camera_pose is None:
            return None
        
        # Transform pixel to world ray (the key operation)
        world_ray = self.camera_pose_estimator.pixel_to_world_ray(pixel, camera_pose)
        
        # Calculate bearing and elevation in ARGUS frame
        bearing, elevation = self._ray_to_bearing_elevation(world_ray)
        
        # Create stabilized observation
        stabilized = StabilizedObservation(
            object_id=object_id,
            world_ray=world_ray,
            bearing=bearing,
            elevation=elevation,
            timestamp_s=timestamp_s,
            confidence=confidence,
            original_pixel=pixel
        )
        
        # Store in object history
        if object_id not in self.object_histories:
            self.object_histories[object_id] = deque(maxlen=self.history_size)
        
        self.object_histories[object_id].append(stabilized)
        
        return stabilized

    def _ray_to_bearing_elevation(self, world_ray: np.ndarray) -> Tuple[float, float]:
        """Convert world ray to bearing and elevation angles.
        
        Args:
            world_ray: 3D ray in world coordinates
            
        Returns:
            Tuple of (bearing, elevation) in radians
        """
        x, y, z = world_ray
        
        # Bearing: angle in horizontal plane (measured from forward)
        bearing = np.arctan2(x, y)  # Returns angle from Y axis (forward)
        
        # Elevation: angle from horizontal plane
        horizontal_distance = np.sqrt(x**2 + y**2)
        elevation = np.arctan2(z, horizontal_distance)
        
        return bearing, elevation

    def bearing_to_pixel(self, bearing: float, elevation: float, 
                       camera_pose: CameraPose) -> Tuple[float, float]:
        """Convert bearing/elevation back to pixel coordinates.
        
        Args:
            bearing: Bearing angle in radians
            elevation: Elevation angle in radians
            camera_pose: Current camera pose
            
        Returns:
            (u, v) pixel coordinates
        """
        # Convert bearing/elevation to world ray
        y = np.cos(elevation) * np.cos(bearing)
        x = np.cos(elevation) * np.sin(bearing)
        z = np.sin(elevation)
        
        world_ray = np.array([x, y, z])
        
        # Transform world ray to camera ray: r_c = R^T * r_w
        camera_ray = camera_pose.rotation_matrix.T @ world_ray
        
        # Convert camera ray to pixel: p = K * r_c
        # Need to scale ray to unit depth first
        camera_ray = camera_ray / camera_ray[2]  # Normalize by Z
        pixel_homogeneous = self.camera_pose_estimator.intrinsics.K @ camera_ray
        
        u, v = pixel_homogeneous[0], pixel_homogeneous[1]
        
        return (u, v)

    def get_object_velocity(self, object_id: str) -> Optional[np.ndarray]:
        """Estimate object velocity from stabilized observations.
        
        This calculates v_object = v_observed - v_ego by analyzing
        how the stabilized bearing changes over time.
        
        Args:
            object_id: Object identifier
            
        Returns:
            Velocity vector [bearing_rate, elevation_rate] or None
        """
        if object_id not in self.object_histories or len(self.object_histories[object_id]) < 2:
            return None
        
        history = list(self.object_histories[object_id])
        
        # Get most recent observations
        recent = history[-2:]
        
        dt = recent[1].timestamp_s - recent[0].timestamp_s
        if dt <= 0:
            return None
        
        # Calculate bearing and elevation rates
        bearing_rate = (recent[1].bearing - recent[0].bearing) / dt
        elevation_rate = (recent[1].elevation - recent[0].elevation) / dt
        
        return np.array([bearing_rate, elevation_rate])

    def is_object_moving(self, object_id: str, threshold: float = 0.1) -> bool:
        """Determine if object is actually moving vs camera motion.
        
        Args:
            object_id: Object identifier
            threshold: Velocity threshold for considering object as moving [rad/s]
            
        Returns:
            True if object appears to be moving independently
        """
        velocity = self.get_object_velocity(object_id)
        if velocity is None:
            return False
        
        # Check if velocity exceeds threshold
        speed = np.linalg.norm(velocity)
        return speed > threshold

    def get_stabilized_trajectory(self, object_id: str, duration_s: float = 2.0) -> list[Tuple[float, float, float]]:
        """Get stabilized trajectory for an object.
        
        Args:
            object_id: Object identifier
            duration_s: Duration of trajectory to retrieve
            
        Returns:
            List of (timestamp, bearing, elevation) tuples
        """
        if object_id not in self.object_histories:
            return []
        
        current_time = time.time()
        cutoff_time = current_time - duration_s
        
        trajectory = []
        for obs in self.object_histories[object_id]:
            if obs.timestamp_s >= cutoff_time:
                trajectory.append((obs.timestamp_s, obs.bearing, obs.elevation))
        
        return trajectory

    def get_relative_motion(self, object_id: str) -> dict:
        """Analyze relative motion characteristics.
        
        Args:
            object_id: Object identifier
            
        Returns:
            Dictionary with motion analysis results
        """
        if object_id not in self.object_histories or len(self.object_histories[object_id]) < 3:
            return {
                "has_sufficient_data": False,
                "is_approaching": False,
                "bearing_change_rate": 0.0,
                "confidence": 0.0
            }
        
        history = list(self.object_histories[object_id])
        
        # Calculate bearing change over recent history
        recent = history[-5:] if len(history) >= 5 else history
        bearing_changes = []
        
        for i in range(1, len(recent)):
            dt = recent[i].timestamp_s - recent[i-1].timestamp_s
            if dt > 0:
                bearing_change = abs(recent[i].bearing - recent[i-1].bearing)
                bearing_changes.append(bearing_change / dt)
        
        avg_bearing_rate = np.mean(bearing_changes) if bearing_changes else 0.0
        
        # Determine if approaching (bearing stays relatively constant while size might increase)
        is_approaching = avg_bearing_rate < 0.1  # Low bearing change rate
        
        return {
            "has_sufficient_data": True,
            "is_approaching": is_approaching,
            "bearing_change_rate": avg_bearing_rate,
            "confidence": len(history) / self.history_size
        }

    def cleanup_object(self, object_id: str) -> None:
        """Remove object from tracking.
        
        Args:
            object_id: Object identifier to remove
        """
        if object_id in self.object_histories:
            del self.object_histories[object_id]

    def reset(self) -> None:
        """Reset all tracking state."""
        self.object_histories.clear()