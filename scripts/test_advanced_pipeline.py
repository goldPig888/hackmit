#!/usr/bin/env python3
"""Test the advanced HALO pipeline with camera-frame realignment."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import time
from halo import (
    AdvancedHaloPipeline, AdvancedPipelineConfig,
    IMUProcessor, CameraPoseEstimator, RotationCompensator,
    WorldFrameTracker, ObjectState, EgoTracker,
    CPADetector, CollisionRiskAssessor, RiskLevel, CollisionRiskAssessment,
    ExtendedKalmanFilter
)
from halo.models import Detection


def test_imu_integration():
    """Test IMU processing and camera pose estimation."""
    print("=" * 60)
    print("Testing IMU Integration")
    print("=" * 60)
    
    # Initialize IMU processor
    imu_processor = IMUProcessor(sample_rate=50.0)
    
    # Simulate IMU data
    gyro = np.array([0.01, 0.02, 0.03])  # rad/s
    accel = np.array([0.1, 0.2, 9.8])    # m/s^2
    
    # Add readings
    timestamp = 0.0
    for i in range(10):
        reading = imu_processor.add_reading(gyro, accel, timestamp)
        timestamp += 0.02
    
    # Get rotation matrix
    rotation_matrix = imu_processor.get_current_rotation_matrix()
    
    print(f"✓ IMU processor created")
    print(f"✓ Added {len(imu_processor.reading_buffer)} readings")
    print(f"✓ Rotation matrix shape: {rotation_matrix.shape}")
    print(f"✓ Rotation matrix valid: {np.allclose(rotation_matrix @ rotation_matrix.T, np.eye(3), atol=0.1)}")
    
    # Test camera pose estimator
    camera_pose_estimator = CameraPoseEstimator()
    quaternion = imu_processor.current_quaternion
    camera_pose = camera_pose_estimator.update_from_imu(quaternion, timestamp)
    
    print(f"✓ Camera pose estimated")
    print(f"✓ Camera pose confidence: {camera_pose.confidence:.2f}")
    
    return True


def test_rotation_compensation():
    """Test camera rotation compensation."""
    print("\n" + "=" * 60)
    print("Testing Rotation Compensation")
    print("=" * 60)
    
    # Initialize components
    camera_pose_estimator = CameraPoseEstimator()
    rotation_compensator = RotationCompensator(camera_pose_estimator)
    
    # Simulate camera pose
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    camera_pose = camera_pose_estimator.update_from_imu(quaternion, 0.0)
    
    # Test compensation
    pixel = (640, 360)  # Center of image
    timestamp = 0.0
    
    for i in range(5):
        stabilized = rotation_compensator.compensate_observation(
            "test-obj", pixel, 0.9, timestamp
        )
        timestamp += 0.1
    
    print(f"✓ Rotation compensator created")
    print(f"✓ Compensated {len(rotation_compensator.object_histories['test-obj'])} observations")
    
    if stabilized:
        print(f"✓ Stabilized bearing: {stabilized.bearing:.3f} rad")
        print(f"✓ Stabilized elevation: {stabilized.elevation:.3f} rad")
    
    return True


def test_world_frame_tracking():
    """Test world frame tracking."""
    print("\n" + "=" * 60)
    print("Testing World Frame Tracking")
    print("=" * 60)
    
    # Initialize components
    camera_pose_estimator = CameraPoseEstimator()
    world_tracker = WorldFrameTracker(camera_pose_estimator)
    
    # Set up camera pose (required for compensation)
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    camera_pose = camera_pose_estimator.update_from_imu(quaternion, 0.0)
    
    # Simulate object detection
    pixel = (500, 300)
    bbox_size = 100
    timestamp = 0.0
    
    # Update object multiple times
    for i in range(10):
        # Simulate object approaching
        pixel = (500 + i*5, 300 + i*3)
        bbox_size = 100 + i*2
        
        object_state = world_tracker.update_object(
            "car-1", "car", pixel, 0.9, bbox_size, timestamp
        )
        timestamp += 0.1
    
    print(f"✓ World tracker created")
    
    if object_state:
        print(f"✓ Tracked object: {object_state.object_id}")
        print(f"✓ Object position: {object_state.position}")
        print(f"✓ Object velocity: {object_state.velocity}")
        print(f"✓ Object speed: {object_state.speed:.2f} m/s")
        print(f"✓ Relative depth: {object_state.relative_depth:.2f} m")
    else:
        print("✗ No object state returned")
        return False
    
    return True


def test_ego_tracking():
    """Test ego (rider) tracking."""
    print("\n" + "=" * 60)
    print("Testing Ego Tracking")
    print("=" * 60)
    
    # Initialize ego tracker
    ego_tracker = EgoTracker(initial_speed=1.5)
    
    # Simulate IMU updates
    timestamp = 0.0
    for i in range(10):
        yaw = 0.1 * i  # Turning left
        yaw_rate = 0.1
        ego_state = ego_tracker.update_from_imu(yaw, yaw_rate, timestamp)
        timestamp += 0.1
    
    print(f"✓ Ego tracker created")
    print(f"✓ Ego speed: {ego_state.speed:.2f} m/s")
    print(f"✓ Ego yaw: {ego_state.yaw:.3f} rad")
    print(f"✓ Ego yaw rate: {ego_state.yaw_rate:.3f} rad/s")
    print(f"✓ Is turning: {ego_tracker.is_turning()}")
    print(f"✓ Turn direction: {ego_tracker.get_turn_direction()}")
    
    # Test trajectory prediction
    times, path = ego_tracker.get_predicted_path(horizon_s=2.0, step_s=0.2)
    print(f"✓ Predicted {len(times)} trajectory points")
    
    return True


def test_cpa_collision_detection():
    """Test CPA-based collision detection."""
    print("\n" + "=" * 60)
    print("Testing CPA Collision Detection")
    print("=" * 60)
    
    # Initialize CPA detector
    cpa_detector = CPADetector(safe_distance=2.0, time_horizon=4.0)
    
    # Test case 1: Approaching collision
    object_position = np.array([0.0, 10.0, 0.0])  # 10m ahead
    object_velocity = np.array([0.0, -5.0, 0.0])  # Moving toward ego at 5 m/s
    ego_position = np.array([0.0, 0.0, 0.0])
    ego_velocity = np.array([0.0, 1.5, 0.0])  # Ego moving forward at 1.5 m/s
    
    cpa_result = cpa_detector.calculate_cpa(
        object_position, object_velocity,
        ego_position, ego_velocity
    )
    
    print(f"✓ CPA detector created")
    print(f"✓ Time to CPA: {cpa_result.time_to_cpa:.2f} s")
    print(f"✓ Distance at CPA: {cpa_result.distance_at_cpa:.2f} m")
    print(f"✓ Will collide: {cpa_result.will_collide}")
    print(f"✓ Confidence: {cpa_result.confidence:.2f}")
    
    # Test case 2: Safe passage
    object_position = np.array([5.0, 10.0, 0.0])  # 5m to the right, 10m ahead
    object_velocity = np.array([0.0, -5.0, 0.0])
    
    cpa_result_safe = cpa_detector.calculate_cpa(
        object_position, object_velocity,
        ego_position, ego_velocity
    )
    
    print(f"✓ Safe passage test:")
    print(f"  - Time to CPA: {cpa_result_safe.time_to_cpa:.2f} s")
    print(f"  - Distance at CPA: {cpa_result_safe.distance_at_cpa:.2f} m")
    print(f"  - Will collide: {cpa_result_safe.will_collide}")
    
    return True


def test_extended_kalman_filter():
    """Test Extended Kalman Filter."""
    print("\n" + "=" * 60)
    print("Testing Extended Kalman Filter")
    print("=" * 60)
    
    # Initialize EKF
    initial_position = np.array([5.0, 10.0, 0.0])
    ekf = ExtendedKalmanFilter(initial_position)
    
    print(f"✓ EKF initialized")
    print(f"✓ Initial position: {ekf.position}")
    print(f"✓ Initial velocity: {ekf.velocity}")
    
    # Simulate measurements and updates
    dt = 0.1
    timestamp = 0.0
    for i in range(10):
        # Simulate measurement (bearing, elevation, distance)
        bearing = np.arctan2(5.0, 10.0 - i*0.5)
        elevation = 0.0
        distance = 10.0 - i*0.5
        
        measurement = np.array([bearing, elevation, distance])
        ekf.update(measurement, dt)
        timestamp += dt
    
    print(f"✓ Updated EKF with {10} measurements")
    print(f"✓ Final position: {ekf.position}")
    print(f"✓ Final velocity: {ekf.velocity}")
    print(f"✓ Position uncertainty: {ekf.position_uncertainty:.3f} m")
    
    return True


def test_full_advanced_pipeline():
    """Test the full advanced pipeline."""
    print("\n" + "=" * 60)
    print("Testing Full Advanced Pipeline")
    print("=" * 60)
    
    # Create configuration
    config = AdvancedPipelineConfig(
        enable_imu_compensation=True,
        enable_visual_odometry=False,  # Disabled for testing
        enable_ekf=True,
        enable_cpa_collision=True
    )
    
    # Initialize pipeline
    pipeline = AdvancedHaloPipeline(config)
    
    print(f"✓ Advanced pipeline created")
    print(f"✓ IMU compensation: {config.enable_imu_compensation}")
    print(f"✓ EKF enabled: {config.enable_ekf}")
    print(f"✓ CPA collision: {config.enable_cpa_collision}")
    
    # Simulate detection with IMU data
    timestamp = 0.0
    for i in range(20):
        # Create detection
        detection = Detection(
            object_id="car-1",
            label="car",
            confidence=0.9,
            bbox=(450 + i*5, 280 + i*3, 100 + i*2, 80 + i*2),
            timestamp_s=timestamp
        )
        
        # Create IMU data
        gyro = np.array([0.01, 0.02, 0.0])
        accel = np.array([0.1, 0.2, 9.8])
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        imu_data = (gyro, accel, quaternion)
        
        # Update pipeline
        risk_assessment = pipeline.update(detection, imu_data=imu_data)
        
        timestamp += 0.1
    
    print(f"✓ Processed {20} detections")
    
    # Get system status
    status = pipeline.get_system_status()
    print(f"✓ System status:")
    print(f"  - Tracked objects: {status['num_tracked_objects']}")
    print(f"  - Ego speed: {status['ego_speed']:.2f} m/s")
    print(f"  - Ego yaw: {status['ego_yaw']:.3f} rad")
    print(f"  - Is turning: {status['is_turning']}")
    
    # Get object states
    object_states = pipeline.get_all_object_states()
    if object_states:
        obj = object_states[0]
        print(f"✓ Tracked object: {obj.object_id}")
        print(f"  - Position: {obj.position}")
        print(f"  - Velocity: {obj.velocity}")
        print(f"  - Speed: {obj.speed:.2f} m/s")
    
    # Get risk assessments
    risk_assessments = pipeline.get_risk_assessments()
    print(f"✓ Risk assessments: {len(risk_assessments)}")
    
    if risk_assessments:
        risk = risk_assessments[-1]
        print(f"  - Risk level: {risk.risk_level.value}")
        print(f"  - Probability: {risk.probability:.2f}")
        print(f"  - Direction: {risk.direction}")
    
    return True


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "HALO Advanced Pipeline Tests" + " " * 18 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    tests = [
        ("IMU Integration", test_imu_integration),
        ("Rotation Compensation", test_rotation_compensation),
        ("World Frame Tracking", test_world_frame_tracking),
        ("Ego Tracking", test_ego_tracking),
        ("CPA Collision Detection", test_cpa_collision_detection),
        ("Extended Kalman Filter", test_extended_kalman_filter),
        ("Full Advanced Pipeline", test_full_advanced_pipeline),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)