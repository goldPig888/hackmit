"""Component-level testing utilities for HALO."""

from __future__ import annotations

import numpy as np
from typing import Callable
from pathlib import Path

from ..geometry import ImageGroundProjector
from ..kalman import ConstantVelocityKalman
from ..ttc import ExpansionTTC
from ..trajectory import predict_path, rider_path, closest_conflict
from ..risk import assess, intensity
from ..models import Detection, TrackState


class ComponentTester:
    """Test individual HALO pipeline components in isolation."""

    def __init__(self, output_dir: str | Path = "runs/component_tests"):
        """Initialize component tester.
        
        Args:
            output_dir: Directory to save test results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_geometry_projection(self, detections: list[Detection]) -> dict[str, np.ndarray]:
        """Test image-to-ground projection.
        
        Args:
            detections: List of detections to project
            
        Returns:
            Dictionary mapping object_id to ground positions
        """
        projector = ImageGroundProjector()
        results = {}
        
        for detection in detections:
            ground_pos = projector.project(detection.center, detection.bbox[3])
            results[detection.object_id] = ground_pos
            
        return results

    def test_kalman_filter(self, detections: list[Detection]) -> dict[str, dict]:
        """Test Kalman filter tracking.
        
        Args:
            detections: List of detections for a single object
            
        Returns:
            Dictionary with filter state evolution
        """
        if not detections:
            return {}
            
        projector = ImageGroundProjector()
        first_det = detections[0]
        measured = projector.project(first_det.center, first_det.bbox[3])
        filter = ConstantVelocityKalman(measured)
        
        results = {
            "positions": [],
            "velocities": [],
            "sigmas": [],
            "timestamps": []
        }
        
        for i, detection in enumerate(detections):
            if i > 0:
                dt = detection.timestamp_s - detections[i-1].timestamp_s
                filter.predict(dt)
            
            measured = projector.project(detection.center, detection.bbox[3])
            filter.update(measured)
            
            results["positions"].append(filter.position.copy())
            results["velocities"].append(filter.velocity.copy())
            results["sigmas"].append(filter.sigma_m)
            results["timestamps"].append(detection.timestamp_s)
        
        return results

    def test_ttc_estimation(self, detections: list[Detection]) -> dict:
        """Test time-to-collision estimation.
        
        Args:
            detections: List of detections for a single object
            
        Returns:
            Dictionary with TTC evolution
        """
        ttc = ExpansionTTC()
        results = {
            "ttc_values": [],
            "timestamps": []
        }
        
        for detection in detections:
            ttc_value = ttc.update(detection.timestamp_s, detection.area)
            results["ttc_values"].append(ttc_value)
            results["timestamps"].append(detection.timestamp_s)
        
        return results

    def test_trajectory_prediction(self, position: np.ndarray, velocity: np.ndarray,
                                   rider_speed: float = 4.0, yaw_rate: float = 0.0) -> dict:
        """Test trajectory prediction for both object and rider.
        
        Args:
            position: Object position in meters
            velocity: Object velocity in m/s
            rider_speed: Rider speed in m/s
            yaw_rate: Rider yaw rate in rad/s
            
        Returns:
            Dictionary with predicted paths
        """
        times, object_path = predict_path(position, velocity)
        times_rider, rider_path_arr = rider_path(rider_speed, yaw_rate)
        
        conflict = closest_conflict(object_path, rider_path_arr, base_sigma_m=0.6)
        
        return {
            "object_path": object_path,
            "rider_path": rider_path_arr,
            "times": times,
            "conflict": conflict
        }

    def test_risk_assessment(self, track_state: TrackState, conflict) -> dict:
        """Test risk assessment component.
        
        Args:
            track_state: Track state object
            conflict: Conflict object from trajectory analysis
            
        Returns:
            Dictionary with risk assessment results
        """
        from ..geometry import bearing_direction
        direction = bearing_direction(track_state.position_m)
        assessment = assess(track_state, conflict, direction)
        
        return {
            "risk_score": assessment.risk,
            "direction": assessment.direction,
            "ttc": assessment.ttc_s,
            "closest_distance": assessment.closest_distance_m,
            "time_to_conflict": assessment.time_to_conflict_s,
            "conflict_probability": assessment.conflict_probability,
            "intensity": intensity(assessment.risk)
        }

    def run_component_suite(self, detections: list[Detection]) -> dict:
        """Run full component test suite on detection sequence.
        
        Args:
            detections: List of detections for a single object
            
        Returns:
            Dictionary with all component test results
        """
        results = {
            "geometry": self.test_geometry_projection(detections),
            "kalman": self.test_kalman_filter(detections),
            "ttc": self.test_ttc_estimation(detections)
        }
        
        # Test trajectory prediction with final state
        if results["kalman"]["positions"]:
            final_pos = results["kalman"]["positions"][-1]
            final_vel = results["kalman"]["velocities"][-1]
            results["trajectory"] = self.test_trajectory_prediction(final_pos, final_vel)
        
        return results

    def save_results(self, results: dict, name: str) -> Path:
        """Save test results to file.
        
        Args:
            results: Test results dictionary
            name: Name for the test file
            
        Returns:
            Path to saved results file
        """
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if hasattr(obj, '__dataclass_fields__'):  # Handle dataclasses
                return {k: convert(getattr(obj, k)) for k in obj.__dataclass_fields__}
            if isinstance(obj, list):
                return [convert(item) for item in obj]
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        
        serializable = convert(results)
        
        output_path = self.output_dir / f"{name}.json"
        with output_path.open("w") as f:
            json.dump(serializable, f, indent=2)
        
        return output_path