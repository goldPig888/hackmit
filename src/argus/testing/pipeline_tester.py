"""Pipeline-level testing utilities for ARGUS."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..detectors.base import BaseDetector

from ..pipeline import ArgusPipeline
from ..models import Detection


class PipelineTester:
    """Test the full ARGUS pipeline with custom detectors."""

    def __init__(self, output_dir: str | Path = "runs/pipeline_tests"):
        """Initialize pipeline tester.
        
        Args:
            output_dir: Directory to save test results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_with_detector(self, detector: BaseDetector, frames: list, 
                          rider_speed_mps: float = 4.0, yaw_rate_rps: float = 0.0) -> dict:
        """Test pipeline with a custom detector.
        
        Args:
            detector: Custom detector instance
            frames: List of frames to process
            rider_speed_mps: Rider speed in m/s
            yaw_rate_rps: Rider yaw rate in rad/s
            
        Returns:
            Dictionary with pipeline test results
        """
        pipeline = ArgusPipeline()
        results = {
            "detector_type": type(detector).__name__,
            "total_frames": len(frames),
            "total_detections": 0,
            "risk_assessments": [],
            "haptic_events": []
        }
        
        try:
            for i, frame in enumerate(frames):
                timestamp = i / 10.0  # Assume 10 FPS for testing
                detections = detector.detect(frame, timestamp)
                results["total_detections"] += len(detections)
                
                for detection in detections:
                    assessment = pipeline.update(detection, rider_speed_mps, yaw_rate_rps)
                    if assessment:
                        from ..risk import intensity
                        event_intensity = intensity(assessment.risk)
                        if event_intensity:
                            results["risk_assessments"].append({
                                "timestamp": timestamp,
                                "object_id": assessment.object_id,
                                "risk": assessment.risk,
                                "direction": assessment.direction,
                                "ttc": assessment.ttc_s,
                                "intensity": event_intensity
                            })
        finally:
            detector.cleanup()
        
        return results

    def test_with_detections(self, detections_list: list[list[Detection]],
                           rider_speed_mps: float = 4.0, yaw_rate_rps: float = 0.0) -> dict:
        """Test pipeline with pre-generated detections.
        
        Args:
            detections_list: List of detection lists (one per frame)
            rider_speed_mps: Rider speed in m/s
            yaw_rate_rps: Rider yaw rate in rad/s
            
        Returns:
            Dictionary with pipeline test results
        """
        pipeline = ArgusPipeline()
        results = {
            "total_frames": len(detections_list),
            "total_detections": sum(len(d) for d in detections_list),
            "risk_assessments": [],
            "track_states": []
        }
        
        for frame_detections in detections_list:
            for detection in frame_detections:
                assessment = pipeline.update(detection, rider_speed_mps, yaw_rate_rps)
                if assessment:
                    from ..risk import intensity
                    event_intensity = intensity(assessment.risk)
                    if event_intensity:
                        results["risk_assessments"].append({
                            "timestamp": detection.timestamp_s,
                            "object_id": assessment.object_id,
                            "risk": assessment.risk,
                            "direction": assessment.direction,
                            "ttc": assessment.ttc_s,
                            "intensity": event_intensity
                        })
        
        return results

    def test_rider_scenarios(self) -> dict:
        """Test pipeline with different rider scenarios.
        
        Returns:
            Dictionary with scenario test results
        """
        from .mock_data import MockDetectionGenerator
        
        generator = MockDetectionGenerator()
        scenarios = {
            "straight_approach": {
                "detections": list(generator.approaching_car()),
                "rider_speed": 4.0,
                "yaw_rate": 0.0
            },
            "left_turn_approach": {
                "detections": list(generator.approaching_car()),
                "rider_speed": 4.0,
                "yaw_rate": 0.3
            },
            "right_turn_approach": {
                "detections": list(generator.approaching_car()),
                "rider_speed": 4.0,
                "yaw_rate": -0.3
            }
        }
        
        results = {}
        for scenario_name, scenario_data in scenarios.items():
            detections_per_frame = [[d] for d in scenario_data["detections"]]
            results[scenario_name] = self.test_with_detections(
                detections_per_frame,
                scenario_data["rider_speed"],
                scenario_data["yaw_rate"]
            )
        
        return results

    def save_results(self, results: dict, name: str) -> Path:
        """Save pipeline test results to file.
        
        Args:
            results: Test results dictionary
            name: Name for the test file
            
        Returns:
            Path to saved results file
        """
        import json
        
        output_path = self.output_dir / f"{name}.json"
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        
        return output_path