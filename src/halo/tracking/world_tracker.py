"""World frame tracker for stabilized object tracking."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, List
from collections import deque
import time

from ..imu.rotation_compensator import RotationCompensator, StabilizedObservation
from ..imu.camera_pose import CameraPoseEstimator
from .object_state import ObjectState
from .ego_tracker import EgoTracker


class WorldFrameTracker:
    """Track objects in stabilized world frame using camera rotation compensation.
    
    This is the core component that implements the architecture:
    Perception → Ego-motion compensation → World-space tracking → Prediction
    
    It distinguishes camera motion from object motion by:
    1. Using IMU data to estimate camera rotation
    2. Transforming pixel observations to world rays
    3. Tracking objects in the stabilized HALO frame
    4. Separating ego motion from object motion
    """

    def __init__(self, camera_pose_estimator: CameraPoseEstimator, 
                 ego_tracker: Optional[EgoTracker] = None):
        """Initialize world frame tracker.
        
        Args:
            camera_pose_estimator: Camera pose estimator for ego-motion
            ego_tracker: Optional ego tracker for rider motion
        """
        self.camera_pose_estimator = camera_pose_estimator
        self.rotation_compensator = RotationCompensator(camera_pose_estimator)
        self.ego_tracker = ego_tracker or EgoTracker()
        
        # Object tracking in world frame
        self.world_objects: Dict[str, ObjectState] = {}
        self.object_observation_history: Dict[str, deque[StabilizedObservation]] = {}
        
        # Depth estimation parameters (monocular)
        self.default_depth = 10.0  # Default depth assumption (meters)
        self.depth_scale_factor = 1.0  # Scale factor for depth estimation
        
        # Tracking parameters
        self.max_object_age = 30.0  # Maximum time to track object without detection
        self.min_observations = 3  # Minimum observations before reliable tracking

    def update_object(self, object_id: str, label: str, pixel: tuple[float, float],
                    confidence: float, bbox_size: float, timestamp_s: float) -> Optional[ObjectState]:
        """Update object tracking with new observation.
        
        This implements the key pipeline:
        1. Compensate observation for camera rotation
        2. Estimate depth from bbox size
        3. Update object state in world frame
        4. Calculate velocity from stabilized observations
        
        Args:
            object_id: Object identifier
            label: Object class label
            pixel: (u, v) pixel coordinates
            confidence: Detection confidence
            bbox_size: Bounding box size (height in pixels)
            timestamp_s: Current timestamp
            
        Returns:
            Updated object state or None if compensation failed
        """
        # Step 1: Compensate for camera rotation
        stabilized_obs = self.rotation_compensator.compensate_observation(
            object_id, pixel, confidence, timestamp_s
        )
        
        if stabilized_obs is None:
            return None
        
        # Step 2: Estimate relative depth from bbox size
        relative_depth = self._estimate_depth_from_bbox(bbox_size, label)
        
        # Step 3: Convert world ray to 3D position in HALO frame
        position = self._world_ray_to_position(
            stabilized_obs.world_ray, 
            stabilized_obs.bearing, 
            relative_depth
        )
        
        # Step 4: Update or create object state
        if object_id in self.world_objects:
            # Update existing object
            object_state = self._update_existing_object(
                object_id, position, stabilized_obs, timestamp_s
            )
        else:
            # Create new object
            object_state = self._create_new_object(
                object_id, label, position, stabilized_obs, relative_depth, timestamp_s
            )
        
        self.world_objects[object_id] = object_state
        
        return object_state

    def _estimate_depth_from_bbox(self, bbox_size: float, label: str) -> float:
        """Estimate relative depth from bounding box size.
        
        Args:
            bbox_size: Bounding box height in pixels
            label: Object class label
            
        Returns:
            Estimated depth in meters
        """
        # Simple depth estimation: depth ∝ 1/size
        # Calibrated for typical object sizes
        typical_sizes = {
            "person": 1.7,    # Average person height (m)
            "car": 1.5,       # Average car height (m)
            "truck": 2.5,     # Average truck height (m)
            "bus": 3.0,       # Average bus height (m)
            "bicycle": 1.0,  # Average bicycle height (m)
            "motorcycle": 1.2
        }
        
        typical_size = typical_sizes.get(label, 1.5)
        
        # Depth estimation using pinhole camera model approximation
        # depth = (typical_size * focal_length) / bbox_size
        focal_length = self.camera_pose_estimator.intrinsics.fy
        estimated_depth = (typical_size * focal_length) / bbox_size
        
        # Apply scaling factor and clamp to reasonable range
        estimated_depth *= self.depth_scale_factor
        estimated_depth = max(1.0, min(estimated_depth, 50.0))  # Clamp between 1m and 50m
        
        return estimated_depth

    def _world_ray_to_position(self, world_ray: np.ndarray, bearing: float, 
                             depth: float) -> np.ndarray:
        """Convert world ray to 3D position in HALO frame.
        
        Args:
            world_ray: 3D ray in world coordinates
            bearing: Bearing angle in radians
            depth: Depth estimate in meters
            
        Returns:
            3D position [X, Y, Z] in HALO frame
        """
        # Convert spherical coordinates to Cartesian
        # X = depth * sin(bearing) * cos(elevation)
        # Y = depth * cos(bearing) * cos(elevation)  
        # Z = depth * sin(elevation)
        
        x = depth * np.sin(bearing)
        y = depth * np.cos(bearing)
        z = 0.0  # Assume ground plane for now
        
        return np.array([x, y, z])

    def _create_new_object(self, object_id: str, label: str, position: np.ndarray,
                        stabilized_obs: StabilizedObservation, depth: float,
                        timestamp_s: float) -> ObjectState:
        """Create new object state.
        
        Args:
            object_id: Object identifier
            label: Object class label
            position: 3D position in HALO frame
            stabilized_obs: Stabilized observation
            depth: Estimated depth
            timestamp_s: Current timestamp
            
        Returns:
            New object state
        """
        # Initial velocity estimate (assume stationary)
        velocity = np.zeros(3)
        
        # Initial covariance (high uncertainty for new objects)
        covariance = np.diag([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])  # [X,Y,Z,Vx,Vy,Vz]
        
        return ObjectState(
            object_id=object_id,
            label=label,
            position=position,
            velocity=velocity,
            bearing=stabilized_obs.bearing,
            elevation=stabilized_obs.elevation,
            relative_depth=depth,
            covariance=covariance,
            confidence=stabilized_obs.confidence,
            timestamp_s=timestamp_s,
            duration_tracked=0.0,
            is_valid=True
        )

    def _update_existing_object(self, object_id: str, new_position: np.ndarray,
                              stabilized_obs: StabilizedObservation,
                              timestamp_s: float) -> ObjectState:
        """Update existing object state with new observation.
        
        Args:
            object_id: Object identifier
            new_position: New 3D position
            stabilized_obs: New stabilized observation
            timestamp_s: Current timestamp
            
        Returns:
            Updated object state
        """
        old_state = self.world_objects[object_id]
        
        # Calculate time difference
        dt = timestamp_s - old_state.timestamp_s
        if dt <= 0:
            dt = 0.01  # Minimum time step
        
        # Estimate velocity from position change
        velocity = (new_position - old_state.position) / dt
        
        # Smooth velocity with previous estimate (simple exponential smoothing)
        alpha = 0.3
        smoothed_velocity = alpha * velocity + (1 - alpha) * old_state.velocity
        
        # Update duration tracked
        duration_tracked = old_state.duration_tracked + dt
        
        # Update covariance (simple uncertainty growth model)
        new_covariance = old_state.covariance * (1 + dt * 0.1)
        
        return ObjectState(
            object_id=object_id,
            label=old_state.label,
            position=new_position,
            velocity=smoothed_velocity,
            bearing=stabilized_obs.bearing,
            elevation=stabilized_obs.elevation,
            relative_depth=old_state.relative_depth,  # Keep previous depth estimate
            covariance=new_covariance,
            confidence=min(1.0, old_state.confidence + 0.05),  # Increase confidence with more observations
            timestamp_s=timestamp_s,
            duration_tracked=duration_tracked,
            is_valid=True
        )

    def get_object_state(self, object_id: str) -> Optional[ObjectState]:
        """Get current state of tracked object.
        
        Args:
            object_id: Object identifier
            
        Returns:
            Object state or None if not tracked
        """
        return self.world_objects.get(object_id)

    def get_all_objects(self) -> List[ObjectState]:
        """Get all currently tracked objects.
        
        Returns:
            List of all object states
        """
        return list(self.world_objects.values())

    def cleanup_old_objects(self, current_time: float) -> None:
        """Remove objects that haven't been updated recently.
        
        Args:
            current_time: Current timestamp
        """
        to_remove = []
        
        for object_id, state in self.world_objects.items():
            age = current_time - state.timestamp_s
            if age > self.max_object_age:
                to_remove.append(object_id)
        
        for object_id in to_remove:
            del self.world_objects[object_id]
            self.rotation_compensator.cleanup_object(object_id)

    def get_object_predictions(self, object_id: str, horizon_s: float = 4.0, 
                            step_s: float = 0.2) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Get predicted trajectory for an object.
        
        Args:
            object_id: Object identifier
            horizon_s: Prediction horizon in seconds
            step_s: Time step for prediction
            
        Returns:
            Tuple of (times, positions) or None if object not found
        """
        state = self.get_object_state(object_id)
        if state is None:
            return None
        
        times = np.arange(0.0, horizon_s + step_s / 2, step_s)
        positions = np.zeros((len(times), 3))
        
        for i, dt in enumerate(times):
            positions[i] = state.predict_future_position(dt)
        
        return times, positions

    def get_relative_motion(self, object_id: str) -> Dict:
        """Analyze relative motion between object and ego.
        
        Args:
            object_id: Object identifier
            
        Returns:
            Dictionary with relative motion analysis
        """
        object_state = self.get_object_state(object_id)
        if object_state is None:
            return {"error": "Object not found"}
        
        ego_state = self.ego_tracker.get_current_state()
        
        # Calculate relative velocity
        relative_velocity = object_state.velocity - ego_state.velocity
        
        # Calculate relative position
        relative_position = object_state.position - ego_state.position
        
        # Calculate closing rate (negative if approaching)
        closing_rate = np.dot(relative_position, relative_velocity) / max(np.linalg.norm(relative_position), 0.1)
        
        return {
            "relative_velocity": relative_velocity.tolist(),
            "relative_position": relative_position.tolist(),
            "closing_rate": float(closing_rate),
            "object_speed": float(object_state.speed),
            "ego_speed": float(ego_state.speed),
            "is_approaching": closing_rate < -0.1,
            "bearing_difference": float(object_state.bearing - ego_state.yaw)
        }

    def reset(self) -> None:
        """Reset all tracking state."""
        self.world_objects.clear()
        self.rotation_compensator.reset()
        self.ego_tracker.reset()