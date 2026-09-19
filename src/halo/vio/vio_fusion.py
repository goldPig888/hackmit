"""Visual-Inertial Odometry fusion for robust ego-motion estimation."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from collections import deque

from .visual_odometry import VisualMotionEstimate


@dataclass
class FusedMotionEstimate:
    """Fused ego-motion estimate from IMU and vision."""
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    angular_velocity: np.ndarray  # 3x1 angular velocity vector
    translation: np.ndarray  # 3x1 translation vector
    confidence: float  # Overall confidence
    imu_confidence: float  # Confidence in IMU estimate
    vision_confidence: float  # Confidence in visual estimate
    timestamp_s: float


class VIOFusion:
    """Fuse IMU and visual odometry for robust ego-motion estimation.
    
    Implements the complementary filter approach:
    R_camera = α * R_IMU + (1-α) * R_vision
    
    For proper implementation, quaternions should be used instead of
    direct matrix averaging, but for Hackathon this simplified
    approach provides the conceptual foundation.
    """

    def __init__(self, fusion_alpha: float = 0.7, history_size: int = 10):
        """Initialize VIO fusion.
        
        Args:
            fusion_alpha: Weight for IMU (0-1), higher = trust IMU more
            history_size: Size of motion history for smoothing
        """
        self.fusion_alpha = fusion_alpha
        self.history_size = history_size
        
        self.imu_history: deque = deque(maxlen=history_size)
        self.vision_history: deque = deque(maxlen=history_size)
        self.fused_history: deque = deque(maxlen=history_size)
        
        self.current_estimate: Optional[FusedMotionEstimate] = None

    def update_from_imu(self, imu_rotation: np.ndarray, angular_velocity: np.ndarray,
                       timestamp_s: float, confidence: float = 0.8) -> None:
        """Update with IMU-based motion estimate.
        
        Args:
            imu_rotation: Rotation matrix from IMU
            angular_velocity: Angular velocity vector
            timestamp_s: Current timestamp
            confidence: Confidence in IMU measurement
        """
        self.imu_history.append({
            'rotation': imu_rotation,
            'angular_velocity': angular_velocity,
            'timestamp': timestamp_s,
            'confidence': confidence
        })
        
        self._update_fused_estimate()

    def update_from_vision(self, visual_estimate: VisualMotionEstimate) -> None:
        """Update with visual odometry estimate.
        
        Args:
            visual_estimate: Visual motion estimate
        """
        self.vision_history.append({
            'rotation': visual_estimate.rotation_matrix,
            'angular_velocity': np.zeros(3),  # Vision doesn't directly give angular velocity
            'timestamp': visual_estimate.timestamp_s,
            'confidence': visual_estimate.confidence
        })
        
        self._update_fused_estimate()

    def _update_fused_estimate(self) -> None:
        """Update fused estimate using complementary filter.
        
        Implements: R_camera = α * R_IMU + (1-α) * R_vision
        """
        if not self.imu_history or not self.vision_history:
            return
        
        # Get most recent estimates
        imu_data = self.imu_history[-1]
        vision_data = self.vision_history[-1]
        
        # Complementary filter for rotation
        # For proper implementation, would use quaternion interpolation
        R_imu = imu_data['rotation']
        R_vision = vision_data['rotation']
        
        # Weighted average of rotation matrices (simplified)
        R_fused = self.fusion_alpha * R_imu + (1 - self.fusion_alpha) * R_vision
        
        # Orthogonalize the fused rotation matrix
        U, _, Vt = np.linalg.svd(R_fused)
        R_fused = U @ Vt
        
        # Fuse angular velocities (vision doesn't provide this, so use IMU)
        omega_fused = imu_data['angular_velocity'].copy()
        
        # Fuse confidences
        conf_imu = imu_data['confidence']
        conf_vision = vision_data['confidence']
        conf_fused = self.fusion_alpha * conf_imu + (1 - self.fusion_alpha) * conf_vision
        
        # Translation (simplified - assume minimal for rotation-only case)
        translation = np.zeros(3)
        
        self.current_estimate = FusedMotionEstimate(
            rotation_matrix=R_fused,
            angular_velocity=omega_fused,
            translation=translation,
            confidence=conf_fused,
            imu_confidence=conf_imu,
            vision_confidence=conf_vision,
            timestamp_s=imu_data['timestamp']
        )
        
        self.fused_history.append(self.current_estimate)

    def get_current_estimate(self) -> Optional[FusedMotionEstimate]:
        """Get current fused motion estimate.
        
        Returns:
            Current fused estimate or None if unavailable
        """
        return self.current_estimate

    def get_smoothed_estimate(self, window_size: int = 3) -> Optional[FusedMotionEstimate]:
        """Get temporally smoothed fused estimate.
        
        Args:
            window_size: Number of recent estimates to average
            
        Returns:
            Smoothed estimate or None if insufficient history
        """
        if len(self.fused_history) < window_size:
            return None
        
        recent = list(self.fused_history)[-window_size:]
        
        # Average rotation matrices
        avg_rotation = np.mean([est.rotation_matrix for est in recent], axis=0)
        
        # Orthogonalize
        U, _, Vt = np.linalg.svd(avg_rotation)
        avg_rotation = U @ Vt
        
        # Average angular velocities
        avg_angular_velocity = np.mean([est.angular_velocity for est in recent], axis=0)
        
        # Average translation
        avg_translation = np.mean([est.translation for est in recent], axis=0)
        
        # Average confidence
        avg_confidence = np.mean([est.confidence for est in recent])
        
        return FusedMotionEstimate(
            rotation_matrix=avg_rotation,
            angular_velocity=avg_angular_velocity,
            translation=avg_translation,
            confidence=avg_confidence,
            imu_confidence=np.mean([est.imu_confidence for est in recent]),
            vision_confidence=np.mean([est.vision_confidence for est in recent]),
            timestamp_s=recent[-1].timestamp_s
        )

    def reset(self) -> None:
        """Reset fusion state."""
        self.imu_history.clear()
        self.vision_history.clear()
        self.fused_history.clear()
        self.current_estimate = None