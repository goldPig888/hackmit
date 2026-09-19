"""Visual odometry for ego-motion estimation from background features."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
import cv2


@dataclass
class VisualMotionEstimate:
    """Visual ego-motion estimate."""
    rotation_matrix: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3x1 translation vector
    confidence: float  # Confidence in estimate
    num_features: int  # Number of features used
    timestamp_s: float


class VisualOdometryEstimator:
    """Estimate camera ego-motion from visual features.
    
    This implements the visual component of visual-inertial odometry:
    - Track background features (masking out YOLO detections)
    - Use Shi-Tomasi/ORB features + Lucas-Kanade optical flow
    - Estimate camera rotation using RANSAC
    - Fuse with IMU data for robust ego-motion estimation
    """

    def __init__(self, max_features: int = 500, quality_level: float = 0.01,
                 min_distance: float = 7.0, lk_params: Optional[dict] = None):
        """Initialize visual odometry estimator.
        
        Args:
            max_features: Maximum number of features to track
            quality_level: Shi-Tomasi quality level
            min_distance: Minimum distance between features
            lk_params: Lucas-Kanade optical flow parameters
        """
        self.max_features = max_features
        self.quality_level = quality_level
        self.min_distance = min_distance
        
        # Lucas-Kanade optical flow parameters
        if lk_params is None:
            self.lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                flags=cv2.OPTFLOW_LK_GET_PYRAMIDAL
            )
        else:
            self.lk_params = lk_params
        
        # Feature detection parameters for ORB
        self.orb = cv2.ORB_create(nfeatures=max_features)
        
        # Previous frame and features
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_keypoints: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None
        
        # Motion history
        self.motion_history: List[VisualMotionEstimate] = []
        self.max_history = 20

    def detect_features(self, frame: np.ndarray, 
                       mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Detect ORB features in frame.
        
        Args:
            frame: Input frame
            mask: Optional mask to exclude regions (e.g., YOLO detections)
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        keypoints, descriptors = self.orb.detectAndCompute(gray, mask)
        
        if keypoints is None:
            return np.array([]), None
        
        # Convert to numpy array
        keypoints_array = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        
        return keypoints_array, descriptors

    def track_features(self, frame: np.ndarray, 
                       prev_keypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Track features using Lucas-Kanade optical flow.
        
        Args:
            frame: Current frame
            prev_keypoints: Previous frame keypoints
            
        Returns:
            Tuple of (new_keypoints, status, error)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        if prev_keypoints is None or len(prev_keypoints) < 4:
            return np.array([]), np.array([]), np.array([])
        
        # Reshape keypoints for LK optical flow
        prev_keypoints_reshaped = prev_keypoints.reshape(-1, 1, 2)
        
        # Track features
        new_keypoints, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_keypoints_reshaped,
            **self.lk_params
        )
        
        # Filter good tracks
        good_new = new_keypoints[status == 1]
        good_old = prev_keypoints[status == 1]
        
        return good_new.reshape(-1, 2), good_old.reshape(-1, 2), error[status == 1]

    def estimate_camera_motion(self, frame: np.ndarray, 
                             detections: Optional[List[Tuple]] = None) -> Optional[VisualMotionEstimate]:
        """Estimate camera motion from frame.
        
        Args:
            frame: Current frame
            detections: Optional list of detection bounding boxes to mask
            
        Returns:
            Visual motion estimate or None if estimation failed
        """
        import time
        current_time = time.time()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Create mask to exclude detection regions
        mask = self._create_detection_mask(frame, detections) if detections else None
        
        # Detect features in current frame
        keypoints, descriptors = self.detect_features(gray, mask)
        
        if len(keypoints) < 10:
            # Not enough features, try without mask
            keypoints, descriptors = self.detect_features(gray, None)
        
        if len(keypoints) < 8:
            return None  # Not enough features for reliable estimation
        
        # If we have previous frame, track features
        if self.prev_frame is not None and self.prev_keypoints is not None:
            # Track features using Lucas-Kanade
            new_keypoints, old_keypoints, error = self.track_features(frame, self.prev_keypoints)
            
            if len(new_keypoints) >= 8:
                # Estimate essential matrix or homography
                motion_estimate = self._estimate_motion_from_correspondences(
                    old_keypoints, new_keypoints, current_time
                )
                
                # Update state
                self.prev_frame = frame.copy()
                self.prev_gray = gray.copy()
                self.prev_keypoints = keypoints
                
                return motion_estimate
        
        # Update state for next iteration
        self.prev_frame = frame.copy()
        self.prev_gray = gray.copy()
        self.prev_keypoints = keypoints
        
        return None

    def _create_detection_mask(self, frame: np.ndarray, 
                            detections: List[Tuple]) -> np.ndarray:
        """Create mask to exclude detection regions.
        
        Args:
            frame: Input frame
            detections: List of (x, y, w, h) bounding boxes
            
        Returns:
            Binary mask (1 = background, 0 = detection regions)
        """
        mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        
        for (x, y, w, h) in detections:
            # Add padding to ensure detections are fully masked
            padding = 10
            x1 = max(0, int(x) - padding)
            y1 = max(0, int(y) - padding)
            x2 = min(frame.shape[1], int(x + w) + padding)
            y2 = min(frame.shape[0], int(y + h) + padding)
            
            mask[y1:y2, x1:x2] = 0
        
        return mask

    def _estimate_motion_from_correspondences(self, old_points: np.ndarray, 
                                           new_points: np.ndarray,
                                           timestamp_s: float) -> VisualMotionEstimate:
        """Estimate camera motion from point correspondences.
        
        Args:
            old_points: Previous frame points
            new_points: Current frame points
            timestamp_s: Current timestamp
            
        Returns:
            Visual motion estimate
        """
        # Estimate fundamental matrix using RANSAC
        F, mask = cv2.findFundamentalMat(old_points, new_points, 
                                       cv2.FM_RANSAC, 0.1, 0.99)
        
        # Estimate homography for planar scenes (more robust for camera rotation)
        H, mask = cv2.findHomography(old_points, new_points, 
                                 cv2.RANSAC, 5.0)
        
        if H is not None:
            # Extract rotation from homography
            # For pure rotation, H ≈ R (since translation is negligible)
            # Decompose H to get rotation matrix
            try:
                # Use SVD to extract rotation
                U, S, Vt = np.linalg.svd(H)
                R = U @ Vt
                
                # Ensure proper rotation matrix (det(R) = 1)
                if np.linalg.det(R) < 0:
                    R = -R
                
                confidence = min(1.0, mask.mean() if mask is not None else 0.8)
                
                return VisualMotionEstimate(
                    rotation_matrix=R,
                    translation=np.zeros(3),
                    confidence=confidence,
                    num_features=len(old_points),
                    timestamp_s=timestamp_s
                )
            except Exception:
                pass
        
        # Fallback: estimate simple rotation from point correspondences
        # Calculate centroid motion
        old_centroid = np.mean(old_points, axis=0)
        new_centroid = np.mean(new_points, axis=0)
        
        # Simple rotation estimate (not very accurate but better than nothing)
        if np.linalg.norm(old_centroid - new_centroid) > 1.0:
            # Assuming pure rotation around center
            # Calculate angle between centroids (simplified)
            angle = np.arctan2(new_centroid[1] - old_centroid[1], 
                               new_centroid[0] - old_centroid[0])
            
            # Create 2D rotation matrix and extend to 3D
            c, s = np.cos(angle), np.sin(angle)
            R_2d = np.array([[c, -s], [s, c]])
            
            # Extend to 3D (assuming rotation around Z axis)
            R = np.eye(3)
            R[:2, :2] = R_2d
            
            return VisualMotionEstimate(
                rotation_matrix=R,
                translation=np.zeros(3),
                confidence=0.5,
                num_features=len(old_points),
                timestamp_s=timestamp_s
            )
        
        # No reliable motion estimate
        return VisualMotionEstimate(
            rotation_matrix=np.eye(3),
            translation=np.zeros(3),
            confidence=0.1,
            num_features=len(old_points),
            timestamp_s=timestamp_s
        )

    def get_smoothed_motion(self, window_size: int = 5) -> Optional[VisualMotionEstimate]:
        """Get smoothed motion estimate from recent history.
        
        Args:
            window_size: Number of recent estimates to average
            
        Returns:
            Smoothed motion estimate or None if insufficient history
        """
        if len(self.motion_history) < window_size:
            return None
        
        recent_estimates = self.motion_history[-window_size:]
        
        # Average rotation matrices (simplified - using quaternion averaging would be better)
        avg_rotation = np.mean([est.rotation_matrix for est in recent_estimates], axis=0)
        
        # Orthogonalize the averaged rotation matrix
        U, _, Vt = np.linalg.svd(avg_rotation)
        avg_rotation = U @ Vt
        
        avg_confidence = np.mean([est.confidence for est in recent_estimates])
        num_features = int(np.mean([est.num_features for est in recent_estimates]))
        
        return VisualMotionEstimate(
            rotation_matrix=avg_rotation,
            translation=np.zeros(3),
            confidence=avg_confidence,
            num_features=num_features,
            timestamp_s=recent_estimates[-1].timestamp_s
        )

    def reset(self) -> None:
        """Reset visual odometry estimator."""
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_gray = None
        self.motion_history.clear()