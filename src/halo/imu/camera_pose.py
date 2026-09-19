"""Camera pose estimation for ego-motion compensation."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int  # Image width
    height: int  # Image height

    @property
    def K(self) -> np.ndarray:
        """Camera intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=float)

    @property
    def K_inv(self) -> np.ndarray:
        """Inverse camera intrinsic matrix."""
        return np.linalg.inv(self.K)


@dataclass
class CameraPose:
    """Camera pose in world coordinates."""
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3x1 translation vector
    timestamp_s: float
    confidence: float = 1.0

    @property
    def quaternion(self) -> np.ndarray:
        """Convert rotation matrix to quaternion."""
        return self._rotation_matrix_to_quaternion(self.rotation_matrix)

    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Quaternion [w, x, y, z]
        """
        # Use trace method for robust conversion
        trace = np.trace(R)
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                w = (R[2, 1] - R[1, 2]) / S
                x = 0.25 * S
                y = (R[0, 1] + R[1, 0]) / S
                z = (R[0, 2] + R[2, 0]) / S
            elif R[1, 1] > R[2, 2]:
                S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                w = (R[0, 2] - R[2, 0]) / S
                x = (R[0, 1] + R[1, 0]) / S
                y = 0.25 * S
                z = (R[1, 2] + R[2, 1]) / S
            else:
                S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                w = (R[1, 0] - R[0, 1]) / S
                x = (R[0, 2] + R[2, 0]) / S
                y = (R[1, 2] + R[2, 1]) / S
                z = 0.25 * S
        
        return np.array([w, x, y, z])


class CameraPoseEstimator:
    """Estimate camera pose from IMU data for ego-motion compensation."""

    def __init__(self, intrinsics: Optional[CameraIntrinsics] = None):
        """Initialize camera pose estimator.
        
        Args:
            intrinsics: Camera intrinsic parameters (defaults to typical phone camera)
        """
        if intrinsics is None:
            # Default phone camera parameters (typical values)
            self.intrinsics = CameraIntrinsics(
                fx=800.0,  # Focal length (pixels)
                fy=800.0,
                cx=640.0,  # Principal point (image center)
                cy=360.0,
                width=1280,
                height=720
            )
        else:
            self.intrinsics = intrinsics
        
        self.current_pose: Optional[CameraPose] = None
        self.pose_history = []

    def update_from_imu(self, quaternion: np.ndarray, timestamp_s: float, 
                       angular_velocity: Optional[np.ndarray] = None) -> CameraPose:
        """Update camera pose from IMU quaternion.
        
        Args:
            quaternion: Camera orientation quaternion [w, x, y, z]
            timestamp_s: Current timestamp
            angular_velocity: Optional angular velocity for confidence estimation
            
        Returns:
            Current camera pose
        """
        # Convert quaternion to rotation matrix
        rotation_matrix = self._quaternion_to_rotation_matrix(quaternion)
        
        # For now, assume camera is at origin (translation = [0, 0, 0])
        # In full implementation, this would come from visual-inertial odometry
        translation = np.zeros(3)
        
        # Estimate confidence based on angular velocity (lower confidence during rapid motion)
        confidence = 1.0
        if angular_velocity is not None:
            angular_speed = np.linalg.norm(angular_velocity)
            confidence = max(0.5, 1.0 - angular_speed / 5.0)  # Reduce confidence during fast rotation
        
        self.current_pose = CameraPose(
            rotation_matrix=rotation_matrix,
            translation=translation,
            timestamp_s=timestamp_s,
            confidence=confidence
        )
        
        self.pose_history.append(self.current_pose)
        
        # Keep limited history
        if len(self.pose_history) > 100:
            self.pose_history.pop(0)
        
        return self.current_pose

    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix.
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = q
        
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ])

    def get_relative_rotation(self, earlier_pose: CameraPose, later_pose: CameraPose) -> np.ndarray:
        """Get relative rotation between two poses.
        
        Args:
            earlier_pose: Earlier camera pose
            later_pose: Later camera pose
            
        Returns:
            Relative rotation matrix (later_pose * earlier_pose^T)
        """
        return later_pose.rotation_matrix @ earlier_pose.rotation_matrix.T

    def pixel_to_camera_ray(self, pixel: tuple[float, float]) -> np.ndarray:
        """Convert pixel coordinates to camera ray.
        
        Args:
            pixel: (u, v) pixel coordinates
            
        Returns:
            3D camera ray (normalized)
        """
        u, v = pixel
        pixel_homogeneous = np.array([u, v, 1.0])
        
        # Convert to camera ray: r_c = K^-1 * p
        camera_ray = self.intrinsics.K_inv @ pixel_homogeneous
        
        # Normalize
        camera_ray = camera_ray / np.linalg.norm(camera_ray)
        
        return camera_ray

    def camera_ray_to_world_ray(self, camera_ray: np.ndarray, camera_pose: CameraPose) -> np.ndarray:
        """Transform camera ray to world coordinates.
        
        Args:
            camera_ray: 3D ray in camera coordinates
            camera_pose: Current camera pose
            
        Returns:
            3D ray in world coordinates
        """
        # r_w = R * r_c (for rotation only, no translation for rays)
        world_ray = camera_pose.rotation_matrix @ camera_ray
        
        return world_ray

    def pixel_to_world_ray(self, pixel: tuple[float, float], camera_pose: CameraPose) -> np.ndarray:
        """Convert pixel directly to world ray using current camera pose.
        
        Args:
            pixel: (u, v) pixel coordinates
            camera_pose: Current camera pose
            
        Returns:
            3D ray in world coordinates
        """
        camera_ray = self.pixel_to_camera_ray(pixel)
        world_ray = self.camera_ray_to_world_ray(camera_ray, camera_pose)
        
        return world_ray

    def get_current_pose(self) -> Optional[CameraPose]:
        """Get current camera pose.
        
        Returns:
            Current camera pose or None if not available
        """
        return self.current_pose

    def reset(self) -> None:
        """Reset pose estimator state."""
        self.current_pose = None
        self.pose_history = []