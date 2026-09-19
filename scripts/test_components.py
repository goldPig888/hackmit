#!/usr/bin/env python3
"""Test individual ARGUS pipeline components."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from argus.testing import ComponentTester, MockDetectionGenerator
from argus.models import Detection


def test_geometry():
    """Test geometry projection component."""
    print("Testing Geometry Projection...")
    tester = ComponentTester()
    
    generator = MockDetectionGenerator()
    detections = list(generator.approaching_car(num_frames=10))
    
    results = tester.test_geometry_projection(detections)
    
    for obj_id, position in results.items():
        print(f"  {obj_id}: position = ({position[0]:.2f}, {position[1]:.2f}) meters")
    
    return results


def test_kalman_filter():
    """Test Kalman filter component."""
    print("\nTesting Kalman Filter...")
    tester = ComponentTester()
    
    generator = MockDetectionGenerator()
    detections = list(generator.approaching_car(num_frames=20))
    
    results = tester.test_kalman_filter(detections)
    
    print(f"  Tracked {len(results['positions'])} frames")
    print(f"  Final position: ({results['positions'][-1][0]:.2f}, {results['positions'][-1][1]:.2f})")
    print(f"  Final velocity: ({results['velocities'][-1][0]:.2f}, {results['velocities'][-1][1]:.2f}) m/s")
    print(f"  Final uncertainty: {results['sigmas'][-1]:.3f} meters")
    
    return results


def test_ttc_estimation():
    """Test TTC estimation component."""
    print("\nTesting TTC Estimation...")
    tester = ComponentTester()
    
    generator = MockDetectionGenerator()
    detections = list(generator.approaching_car(num_frames=30))
    
    results = tester.test_ttc_estimation(detections)
    
    ttc_strings = [f'{ttc:.1f}s' if ttc else 'None' for ttc in results['ttc_values'][-5:]]
    print(f"  TTC values: {ttc_strings}")
    
    return results


def test_trajectory_prediction():
    """Test trajectory prediction component."""
    print("\nTesting Trajectory Prediction...")
    tester = ComponentTester()
    
    import numpy as np
    position = np.array([2.0, 10.0])  # 2m right, 10m ahead
    velocity = np.array([-1.5, -3.0])  # Moving left and toward rider
    
    results = tester.test_trajectory_prediction(position, velocity, rider_speed=4.0, yaw_rate=0.0)
    
    print(f"  Object path points: {len(results['object_path'])}")
    print(f"  Rider path points: {len(results['rider_path'])}")
    print(f"  Closest conflict: {results['conflict'].closest_distance_m:.2f}m at {results['conflict'].time_s:.1f}s")
    print(f"  Conflict probability: {results['conflict'].probability:.2f}")
    
    return results


def test_full_component_suite():
    """Test full component suite."""
    print("\nTesting Full Component Suite...")
    tester = ComponentTester()
    
    generator = MockDetectionGenerator()
    detections = list(generator.approaching_car(num_frames=25))
    
    results = tester.run_component_suite(detections)
    
    print(f"  Geometry: {len(results['geometry'])} objects projected")
    print(f"  Kalman: {len(results['kalman']['positions'])} frames tracked")
    print(f"  TTC: {len([t for t in results['ttc']['ttc_values'] if t is not None])} valid estimates")
    
    if 'trajectory' in results:
        print(f"  Trajectory: conflict at {results['trajectory']['conflict'].closest_distance_m:.2f}m")
    
    # Save results
    output_path = tester.save_results(results, "component_suite_test")
    print(f"  Results saved to: {output_path}")
    
    return results


def main():
    """Run all component tests."""
    print("=" * 60)
    print("ARGUS Component Testing Suite")
    print("=" * 60)
    
    try:
        test_geometry()
        test_kalman_filter()
        test_ttc_estimation()
        test_trajectory_prediction()
        test_full_component_suite()
        
        print("\n" + "=" * 60)
        print("All component tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()