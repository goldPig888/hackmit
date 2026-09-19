"""CPA (Closest Point of Approach) collision detection."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class CPAResult:
    """Result of Closest Point of Approach calculation."""
    time_to_cpa: float  # Time to closest approach (seconds)
    distance_at_cpa: float  # Distance at closest approach (meters)
    relative_position_at_cpa: np.ndarray  # Relative position at CPA [X, Y, Z]
    relative_velocity: np.ndarray  # Relative velocity [Vx, Vy, Vz]
    will_collide: bool  # Whether collision is predicted
    confidence: float  # Confidence in prediction


class CPADetector:
    """Calculate Closest Point of Approach between object and ego trajectories.
    
    This implements the mathematical collision reasoning described:
    - t_CPA = -(r_0 · v_rel) / ||v_rel||^2
    - d_CPA = ||r_0 + v_rel * t_CPA||
    
    This is much better than simple TTC because it handles vehicles that
    are approaching but will safely pass beside you.
    """

    def __init__(self, safe_distance: float = 2.0, time_horizon: float = 4.0):
        """Initialize CPA detector.
        
        Args:
            safe_distance: Minimum safe distance for collision (meters)
            time_horizon: Maximum time to consider for CPA (seconds)
        """
        self.safe_distance = safe_distance
        self.time_horizon = time_horizon

    def calculate_cpa(self, object_position: np.ndarray, object_velocity: np.ndarray,
                     ego_position: np.ndarray, ego_velocity: np.ndarray) -> CPAResult:
        """Calculate Closest Point of Approach.
        
        This implements the core CPA math:
        r(t) = r_0 + v_rel * t
        t_CPA = -(r_0 · v_rel) / ||v_rel||^2
        d_CPA = ||r_0 + v_rel * t_CPA||
        
        Args:
            object_position: Object position [X, Y, Z] (meters)
            object_velocity: Object velocity [Vx, Vy, Vz] (m/s)
            ego_position: Ego position [X, Y, Z] (meters)
            ego_velocity: Ego velocity [Vx, Vy, Vz] (m/s)
            
        Returns:
            CPA result with collision assessment
        """
        # Calculate relative position and velocity
        relative_position = object_position - ego_position  # r_0
        relative_velocity = object_velocity - ego_velocity  # v_rel
        
        # Calculate relative speed
        relative_speed = np.linalg.norm(relative_velocity)
        
        # If relative speed is very small, objects are nearly stationary relative to each other
        if relative_speed < 0.01:
            return CPAResult(
                time_to_cpa=float('inf'),
                distance_at_cpa=float(np.linalg.norm(relative_position)),
                relative_position_at_cpa=relative_position,
                relative_velocity=relative_velocity,
                will_collide=False,
                confidence=0.5
            )
        
        # Calculate time to CPA: t_CPA = -(r_0 · v_rel) / ||v_rel||^2
        t_cpa = -np.dot(relative_position, relative_velocity) / (relative_speed ** 2)
        
        # Calculate position at CPA: r_CPA = r_0 + v_rel * t_CPA
        position_at_cpa = relative_position + relative_velocity * t_cpa
        
        # Calculate distance at CPA: d_CPA = ||r_CPA||
        distance_at_cpa = np.linalg.norm(position_at_cpa)
        
        # Determine if collision will occur
        # Collision if: 0 < t_CPA < time_horizon AND d_CPA < safe_distance
        will_collide = (0 < t_cpa < self.time_horizon) and (distance_at_cpa < self.safe_distance)
        
        # Calculate confidence based on how close we are to the conditions
        if will_collide:
            # Higher confidence if CPA is well within time horizon and distance threshold
            time_confidence = 1.0 - (t_cpa / self.time_horizon)
            distance_confidence = 1.0 - (distance_at_cpa / self.safe_distance)
            confidence = max(0.5, min(1.0, (time_confidence + distance_confidence) / 2))
        else:
            # Lower confidence for non-collision cases
            if t_cpa < 0:
                # Objects are moving apart
                confidence = 0.8
            elif t_cpa > self.time_horizon:
                # CPA is beyond our time horizon
                confidence = 0.6
            else:
                # CPA time is reasonable but distance is safe
                confidence = 0.9
        
        return CPAResult(
            time_to_cpa=t_cpa,
            distance_at_cpa=distance_at_cpa,
            relative_position_at_cpa=position_at_cpa,
            relative_velocity=relative_velocity,
            will_collide=will_collide,
            confidence=confidence
        )

    def calculate_trajectory_cpa(self, object_trajectory: np.ndarray, 
                                 ego_trajectory: np.ndarray,
                                 times: np.ndarray) -> CPAResult:
        """Calculate CPA from full trajectory predictions.
        
        This is more accurate than constant-velocity assumption as it
        uses the actual predicted trajectories.
        
        Args:
            object_trajectory: Object trajectory positions (N, 3)
            ego_trajectory: Ego trajectory positions (N, 3)
            times: Time points for trajectories
            
        Returns:
            CPA result with collision assessment
        """
        if len(object_trajectory) != len(ego_trajectory):
            raise ValueError("Trajectories must have same length")
        
        # Calculate relative positions at each time step
        relative_positions = object_trajectory - ego_trajectory
        distances = np.linalg.norm(relative_positions, axis=1)
        
        # Find minimum distance and corresponding time
        min_distance_idx = int(np.argmin(distances))
        min_distance = float(distances[min_distance_idx])
        t_cpa = float(times[min_distance_idx])
        
        # Estimate relative velocity at CPA (from surrounding points)
        if min_distance_idx > 0 and min_distance_idx < len(times) - 1:
            dt = times[min_distance_idx + 1] - times[min_distance_idx - 1]
            if dt > 0:
                rel_pos_before = relative_positions[min_distance_idx - 1]
                rel_pos_after = relative_positions[min_distance_idx + 1]
                relative_velocity = (rel_pos_after - rel_pos_before) / (2 * dt)
            else:
                relative_velocity = np.zeros(3)
        else:
            relative_velocity = np.zeros(3)
        
        # Determine collision
        will_collide = (0 < t_cpa < self.time_horizon) and (min_distance < self.safe_distance)
        
        return CPAResult(
            time_to_cpa=t_cpa,
            distance_at_cpa=min_distance,
            relative_position_at_cpa=relative_positions[min_distance_idx],
            relative_velocity=relative_velocity,
            will_collide=will_collide,
            confidence=0.9 if will_collide else 0.8
        )

    def assess_miss_distance(self, cpa_result: CPAResult) -> Tuple[str, float]:
        """Assess the nature of a near-miss or safe passage.
        
        Args:
            cpa_result: CPA calculation result
            
        Returns:
            Tuple of (assessment_type, miss_distance)
            assessment_type: "collision", "near_miss", "safe_pass", "moving_away"
        """
        if cpa_result.will_collide:
            return "collision", cpa_result.distance_at_cpa
        
        if cpa_result.time_to_cpa < 0:
            # Objects are moving apart
            return "moving_away", cpa_result.distance_at_cpa
        
        if cpa_result.time_to_cpa > self.time_horizon:
            # CPA is beyond consideration time
            return "safe_pass", cpa_result.distance_at_cpa
        
        # CPA is within time horizon but distance is safe
        if cpa_result.distance_at_cpa < self.safe_distance * 2:
            return "near_miss", cpa_result.distance_at_cpa
        else:
            return "safe_pass", cpa_result.distance_at_cpa

    def calculate_bearing_rate_conflict(self, object_state, ego_state) -> bool:
        """Check for constant-bearing/decreasing-range collision pattern.
        
        This implements the concerning pattern you described:
        |θ̇| ≈ 0 while Ȧ > 0
        (bearing stays constant while object expands)
        
        Args:
            object_state: Object state
            ego_state: Ego state
            
        Returns:
            True if showing concerning collision pattern
        """
        # Calculate relative bearing
        relative_bearing = object_state.bearing - ego_state.yaw
        
        # Estimate bearing rate (simplified - would need history)
        bearing_rate = 0.0  # Would be calculated from bearing history
        
        # Check if bearing rate is small (constant bearing)
        constant_bearing = abs(bearing_rate) < 0.1  # rad/s threshold
        
        # Check if object is approaching (decreasing range)
        is_approaching = object_state.is_approaching()
        
        # Constant bearing + approaching = concerning pattern
        return constant_bearing and is_approaching

    def get_collision_probability(self, cpa_result: CPAResult, 
                               object_speed: float, object_class: str) -> float:
        """Calculate comprehensive collision probability.
        
        Combines CPA results with other factors for a more nuanced risk assessment.
        
        Args:
            cpa_result: CPA calculation result
            object_speed: Object speed in m/s
            object_class: Object class label
            
        Returns:
            Collision probability between 0 and 1
        """
        # Base probability from CPA result
        if cpa_result.will_collide:
            base_prob = 0.9
        else:
            base_prob = 0.1
        
        # Adjust based on distance at CPA
        distance_factor = max(0.0, 1.0 - (cpa_result.distance_at_cpa / (self.safe_distance * 2)))
        
        # Adjust based on time to CPA (sooner = more concerning)
        if 0 < cpa_result.time_to_cpa < self.time_horizon:
            time_factor = 1.0 - (cpa_result.time_to_cpa / self.time_horizon)
        else:
            time_factor = 0.3
        
        # Adjust based on object speed (faster = more concerning)
        speed_factor = min(1.0, object_speed / 10.0)  # Normalize to max 10 m/s
        
        # Adjust based on object class
        class_multipliers = {
            "car": 1.0,
            "truck": 1.2,
            "bus": 1.1,
            "motorcycle": 0.9,
            "bicycle": 0.7,
            "person": 0.5
        }
        class_factor = class_multipliers.get(object_class.lower(), 0.8)
        
        # Combine factors
        probability = base_prob * distance_factor * time_factor * speed_factor * class_factor
        probability = max(0.0, min(1.0, probability))
        
        return probability