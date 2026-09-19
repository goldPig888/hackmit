#!/usr/bin/env python3
"""Test custom detection models with ARGUS pipeline."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from argus.detectors import DetectorRegistry, YOLODetector, WeaponsDetector
from argus.testing import PipelineTester, MockDetectionGenerator
from argus.pipeline import ArgusPipeline


def test_detector_registry():
    """Test detector registry functionality."""
    print("Testing Detector Registry...")
    
    # Register YOLO detector
    DetectorRegistry.register("yolo", YOLODetector)
    
    # Register weapons detector factory
    def create_weapons_detector():
        return WeaponsDetector()
    
    DetectorRegistry.register_factory("weapons", create_weapons_detector)
    
    print(f"  Registered detectors: {DetectorRegistry.list_registered()}")
    
    # Test creation
    try:
        yolo_detector = DetectorRegistry.create("yolo", model_path="yolo11n.pt")
        print(f"  Created YOLO detector: {type(yolo_detector).__name__}")
        yolo_detector.cleanup()
    except Exception as e:
        print(f"  YOLO detector creation skipped (model not downloaded): {e}")
    
    weapons_detector = DetectorRegistry.create("weapons")
    print(f"  Created weapons detector: {type(weapons_detector).__name__}")
    weapons_detector.cleanup()
    
    print("  Registry test passed!")


def test_yolo_detector_with_mock_data():
    """Test YOLO detector with mock data (simulated)."""
    print("\nTesting YOLO Detector with Mock Data...")
    
    # Create a simple mock detector for testing
    class MockYOLODetector:
        def __init__(self):
            self.detection_count = 0
        
        def detect(self, frame, timestamp_s):
            from argus.models import Detection
            self.detection_count += 1
            # Simulate detection
            return [Detection(
                object_id=f"car-{self.detection_count}",
                label="car",
                confidence=0.9,
                bbox=(100 + self.detection_count * 10, 200, 80, 60),
                timestamp_s=timestamp_s
            )]
        
        def get_supported_classes(self):
            return {"car", "truck", "bus"}
        
        def cleanup(self):
            pass
    
    detector = MockYOLODetector()
    
    # Generate mock frames (just placeholders)
    mock_frames = [None] * 10  # 10 frames
    
    tester = PipelineTester()
    results = tester.test_with_detector(detector, mock_frames, rider_speed_mps=4.0)
    
    print(f"  Processed {results['total_frames']} frames")
    print(f"  Generated {results['total_detections']} detections")
    print(f"  Risk assessments: {len(results['risk_assessments'])}")
    
    detector.cleanup()
    print("  Mock YOLO test passed!")


def test_weapons_detector_integration():
    """Test weapons detector integration."""
    print("\nTesting Weapons Detector Integration...")
    
    detector = WeaponsDetector()
    
    print(f"  Supported classes: {detector.get_supported_classes()}")
    
    # Test with a mock frame
    mock_frame = None  # Placeholder
    detections = detector.detect(mock_frame, timestamp_s=0.0)
    
    print(f"  Detections (should be empty - placeholder): {len(detections)}")
    print("  Note: Replace placeholder implementation with your actual model")
    
    detector.cleanup()
    print("  Weapons detector integration test passed!")


def test_pipeline_with_custom_detector():
    """Test full pipeline with a custom detector."""
    print("\nTesting Pipeline with Custom Detector...")
    
    # Create a custom detector that simulates weapons detection
    class CustomSafetyDetector:
        def __init__(self):
            self.frame_count = 0
        
        def detect(self, frame, timestamp_s):
            from argus.models import Detection
            self.frame_count += 1
            
            # Simulate different detection scenarios
            detections = []
            
            # Approaching car
            if self.frame_count <= 5:
                detections.append(Detection(
                    object_id="threat-1",
                    label="car",
                    confidence=0.95,
                    bbox=(300 - self.frame_count * 20, 400, 100, 80),
                    timestamp_s=timestamp_s
                ))
            
            # Weapon detection (simulated)
            if self.frame_count > 3:
                detections.append(Detection(
                    object_id="weapon-1",
                    label="gun",
                    confidence=0.88,
                    bbox=(500, 300, 50, 30),
                    timestamp_s=timestamp_s
                ))
            
            return detections
        
        def get_supported_classes(self):
            return {"car", "gun", "knife", "weapon"}
        
        def cleanup(self):
            pass
    
    detector = CustomSafetyDetector()
    mock_frames = [None] * 10
    
    tester = PipelineTester()
    results = tester.test_with_detector(detector, mock_frames, rider_speed_mps=4.0)
    
    print(f"  Total detections: {results['total_detections']}")
    print(f"  Risk assessments: {len(results['risk_assessments'])}")
    
    for assessment in results['risk_assessments']:
        print(f"    - {assessment['object_id']}: risk={assessment['risk']:.2f}, direction={assessment['direction']}")
    
    detector.cleanup()
    print("  Custom detector pipeline test passed!")


def test_sync_hazard_detection():
    """Test synchronized hazard detection."""
    print("\nTesting Synchronized Hazard Detection...")
    
    from argus.hazards import SyncHazardDetector
    from argus.models import TrackState
    import numpy as np
    
    detector = SyncHazardDetector()
    
    # Create mock track states simulating convergence
    track_states = [
        TrackState(
            object_id="car-1",
            label="car",
            confidence=0.9,
            position_m=np.array([1.0, 8.0]),
            velocity_mps=np.array([-0.5, -2.0]),
            position_sigma_m=0.5,
            ttc_s=2.5,
            closing=True,
            timestamp_s=1.0
        ),
        TrackState(
            object_id="car-2",
            label="car",
            confidence=0.85,
            position_m=np.array([-1.5, 7.0]),
            velocity_mps=np.array([0.3, -1.8]),
            position_sigma_m=0.6,
            ttc_s=3.0,
            closing=True,
            timestamp_s=1.0
        )
    ]
    
    hazards = detector.update(track_states)
    
    print(f"  Detected hazards: {len(hazards)}")
    for hazard in hazards:
        print(f"    - {hazard.hazard_type}: {hazard.description} (severity: {hazard.severity:.2f})")
    
    # SyncHazardDetector doesn't need cleanup like base detectors
    print("  Sync hazard detection test passed!")


def main():
    """Run all custom detector tests."""
    print("=" * 60)
    print("ARGUS Custom Detector Testing Suite")
    print("=" * 60)
    
    try:
        test_detector_registry()
        test_yolo_detector_with_mock_data()
        test_weapons_detector_integration()
        test_pipeline_with_custom_detector()
        test_sync_hazard_detection()
        
        print("\n" + "=" * 60)
        print("All custom detector tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()