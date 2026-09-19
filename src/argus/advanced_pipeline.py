"""Advanced ARGUS pipeline with camera-frame realignment and CPA-based collision detection."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
import time

from .models import Detection, RiskAssessment, TrackState
from .imu import IMUProcessor, CameraPoseEstimator, RotationCompensator
from .tracking import WorldFrameTracker, ObjectState, EgoTracker
from .collision import CPADetector, CollisionRiskAssessor, RiskLevel, CollisionRiskAssessment
from .filters import ExtendedKalmanFilter
from .vio import VisualOdometryEstimator, VIOFusion


@dataclass
class AdvancedPipelineConfig:
    """Configuration for advanced ARGUS pipeline."""
    enable_imu_compensation: bool = True
    enable_visual_odometry: bool = False  # Disabled by default for Hackathon
    enable_ekf: bool = True
    enable_cpa_collision: bool = True
    fusion_alpha: float = 0.7  # IMU weight in fusion
    safe_distance: float = 2.0  # meters
    time_horizon: float = 4.0  # seconds
    max_object_age: float = 30.0  # seconds


class AdvancedArgusPipeline:
    """Advanced ARGUS pipeline with camera-frame realignment.
    
    This implements the architecture you described:
    Perception → Ego-motion compensation → World-space tracking → Prediction
    
    Key improvements over original pipeline:
    1. IMU-based camera rotation compensation
    2. Stabilized world frame tracking
    3. EKF for nonlinear state estimation
    4. CPA-based collision detection (not just TTC)
    5. Optional visual-inertial odometry fusion
    """

    def __init__(self, config: Optional[AdvancedPipelineConfig] = None):
        """Initialize advanced ARGUS pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or AdvancedPipelineConfig()
        
        # Initialize IMU processing
        self.imu_processor = IMUProcessor()
        self.camera_pose_estimator = CameraPoseEstimator()
        self.rotation_compensator = RotationCompensator(self.camera_pose_estimator)
        
        # Initialize tracking in world frame
        self.world_tracker = WorldFrameTracker(self.camera_pose_estimator)
        self.ego_tracker = EgoTracker()
        
        # Initialize collision detection
        self.cpa_detector = CPADetector(
            safe_distance=self.config.safe_distance,
            time_horizon=self.config.time_horizon
        )
        self.risk_assessor = CollisionRiskAssessor(self.cpa_detector)
        
        # Initialize visual-inertial odometry (optional)
        if self.config.enable_visual_odometry:
            self.visual_odometry = VisualOdometryEstimator()
            self.vio_fusion = VIOFusion(fusion_alpha=self.config.fusion_alpha)
        else:
            self.visual_odometry = None
            self.vio_fusion = None
        
        # EKF filters for each object
        self.ekf_filters: Dict[str, ExtendedKalmanFilter] = {}
        
        # Internal state
        self.last_timestamp: Optional[float] = None
        self.current_risk_assessments: List[CollisionRiskAssessment] = []

    def update(self, detection: Detection, 
             imu_data: Optional[tuple] = None,
             frame: Optional[np.ndarray] = None) -> Optional[RiskAssessment]:
        """Update pipeline with new detection.
        
        This implements the full advanced pipeline:
        1. Compensate observation for camera rotation
        2. Update object in stabilized world frame
        3. Apply EKF for state estimation
        4. Update ego state from IMU
        5. Calculate CPA-based collision risk
        6. Return comprehensive risk assessment
        
        Args:
            detection: Detection from detector
            imu_data: Optional (gyro, accel, quaternion) from IMU
            frame: Optional frame for visual odometry
            
        Returns:
            Risk assessment or None if risk is below threshold
        """
        timestamp = detection.timestamp_s
        dt = timestamp - self.last_timestamp if self.last_timestamp else 0.01
        self.last_timestamp = timestamp
        
        # Step 1: Process IMU data and update camera pose
        if imu_data is not None:
            gyro, accel, quaternion = imu_data
            imu_reading = self.imu_processor.add_reading(gyro, accel, timestamp)
            
            # Update camera pose from IMU
            camera_pose = self.camera_pose_estimator.update_from_imu(
                quaternion, timestamp, angular_velocity=gyro
            )
            
            # Update ego tracker from IMU
            yaw = np.arctan2(camera_pose.rotation_matrix[0, 1], camera_pose.rotation_matrix[1, 1])
            self.ego_tracker.update_from_imu(yaw, gyro[2], timestamp, speed=None)
        
        # Step 2: Optional visual odometry (if enabled)
        if self.config.enable_visual_odometry and frame is not None:
            detections = [(detection.bbox[0], detection.bbox[1], 
                           detection.bbox[2], detection.bbox[3])]
            visual_estimate = self.visual_odometry.estimate_camera_motion(frame, detections)
            
            if visual_estimate is not None:
                self.vio_fusion.update_from_vision(visual_estimate)
                
                # Use fused estimate for camera pose
                fused_estimate = self.vio_fusion.get_current_estimate()
                if fused_estimate is not None:
                    self.camera_pose_estimator.current_pose.rotation_matrix = fused_estimate.rotation_matrix
        
        # Step 3: Compensate detection for camera rotation
        pixel_center = (detection.bbox[0] + detection.bbox[2]/2, 
                         detection.bbox[1] + detection.bbox[3]/2)
        
        stabilized_obs = self.rotation_compensator.compensate_observation(
            detection.object_id, pixel_center, detection.confidence, timestamp
        )
        
        if stabilized_obs is None:
            return None
        
        # Step 4: Update object in world frame
        bbox_size = detection.bbox[3]  # Use height as size proxy
        object_state = self.world_tracker.update_object(
            detection.object_id, detection.label, pixel_center,
            detection.confidence, bbox_size, timestamp
        )
        
        if object_state is None:
            return None
        
        # Step 5: Apply EKF for improved state estimation
        if self.config.enable_ekf:
            if detection.object_id not in self.ekf_filters:
                # Initialize EKF for new object
                self.ekf_filters[detection.object_id] = ExtendedKalmanFilter(
                    object_state.position
                )
            
            # Convert measurement to EKF format (bearing, elevation, distance)
            measurement = np.array([
                stabilized_obs.bearing,
                stabilized_obs.elevation,
                object_state.relative_depth
            ])
            
            # Update EKF
            self.ekf_filters[detection.object_id].update(measurement, dt, timestamp)
            
            # Update object state with EKF estimate
            object_state.position = self.ekf_filters[detection.object_id].position
            object_state.velocity = self.ekf_filters[detection.object_id].velocity
            object_state.covariance = self.ekf_filters[detection.object_id].covariance
        
        # Step 6: Get ego state
        ego_state = self.ego_tracker.get_current_state()
        
        # Step 7: CPA-based collision detection
        if self.config.enable_cpa_collision:
            # Get predictions for both trajectories
            times, object_trajectory = self.world_tracker.get_object_predictions(
                detection.object_id, self.config.time_horizon
            )
            times_ego, ego_trajectory = self.ego_tracker.get_predicted_path(
                self.config.time_horizon
            )
            
            # Comprehensive risk assessment
            risk_assessment = self.risk_assessor.assess_risk(
                object_state, ego_state, object_trajectory, ego_trajectory, times
            )
            
            # Convert to legacy RiskAssessment format for compatibility
            legacy_assessment = self._convert_to_legacy_risk(risk_assessment)
            
            # Store current risk assessments
            self.current_risk_assessments.append(risk_assessment)
            
            # Keep only recent assessments
            if len(self.current_risk_assessments) > 50:
                self.current_risk_assessments.pop(0)
            
            # Only return assessment if risk is significant
            if risk_assessment.probability > 0.2:
                return legacy_assessment
        
        return None

    def _convert_to_legacy_risk(self, risk_assessment: CollisionRiskAssessment) -> RiskAssessment:
        """Convert new risk assessment to legacy format for compatibility.
        
        Args:
            risk_assessment: New collision risk assessment
            
        Returns:
            Legacy risk assessment format
        """
        direction_map = {
            "left": "left",
            "right": "right", 
            "center": "center"
        }
        
        return RiskAssessment(
            object_id=risk_assessment.object_id,
            label="",  # Would need to track object label separately
            risk=risk_assessment.probability,
            direction=direction_map.get(risk_assessment.direction, "center"),
            ttc_s=risk_assessment.time_to_collision if risk_assessment.time_to_collision != float('inf') else None,
            closest_distance_m=risk_assessment.miss_distance,
            time_to_conflict_s=risk_assessment.cpa_result.time_to_cpa,
            conflict_probability=risk_assessment.probability
        )

    def get_all_object_states(self) -> List[ObjectState]:
        """Get all currently tracked object states.
        
        Returns:
            List of object states in world frame
        """
        return self.world_tracker.get_all_objects()

    def get_ego_state(self) -> object:
        """Get current ego state.
        
        Returns:
            Current ego state
        """
        return self.ego_tracker.get_current_state()

    def get_risk_assessments(self) -> List[CollisionRiskAssessment]:
        """Get recent risk assessments.
        
        Returns:
            List of recent risk assessments
        """
        return self.current_risk_assessments

    def cleanup_old_objects(self) -> None:
        """Remove objects that haven't been updated recently."""
        self.world_tracker.cleanup_old_objects(time.time())
        
        # Clean up EKF filters for removed objects
        active_ids = set(obj.object_id for obj in self.world_tracker.get_all_objects())
        for object_id in list(self.ekf_filters.keys()):
            if object_id not in active_ids:
                del self.ekf_filters[object_id]

    def get_system_status(self) -> Dict:
        """Get comprehensive system status.
        
        Returns:
            Dictionary with system status information
        """
        ego_state = self.ego_tracker.get_current_state()
        object_states = self.world_tracker.get_all_objects()
        
        return {
            "num_tracked_objects": len(object_states),
            "ego_speed": float(ego_state.speed),
            "ego_yaw": float(ego_state.yaw),
            "ego_yaw_rate": float(ego_state.yaw_rate),
            "is_turning": self.ego_tracker.is_turning(),
            "turn_direction": self.ego_tracker.get_turn_direction(),
            "camera_confidence": float(self.camera_pose_estimator.get_current_pose().confidence) if self.camera_pose_estimator.get_current_pose() else 0.0,
            "fusion_enabled": self.config.enable_visual_odometry,
            "ekf_enabled": self.config.enable_ekf,
            "cpa_enabled": self.config.enable_cpa_collision,
            "timestamp": time.time()
        }

    def reset(self) -> None:
        """Reset pipeline state."""
        self.imu_processor.reset()
        self.camera_pose_estimator.reset()
        self.rotation_compensator.reset()
        self.world_tracker.reset()
        self.ego_tracker.reset()
        self.ekf_filters.clear()
        self.current_risk_assessments.clear()
        self.last_timestamp = None
        
        if self.visual_odometry:
            self.visual_odometry.reset()
        if self.vio_fusion:
            self.vio_fusion.reset()