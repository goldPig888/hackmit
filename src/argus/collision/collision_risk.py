"""Comprehensive collision risk assessment."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum

from .cpa_detector import CPADetector, CPAResult
from ..tracking.object_state import ObjectState
from ..tracking.ego_tracker import EgoState


class RiskLevel(Enum):
    """Risk level classification."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CollisionRiskAssessment:
    """Comprehensive collision risk assessment."""
    object_id: str
    risk_level: RiskLevel
    probability: float  # Overall collision probability (0-1)
    cpa_result: CPAResult
    time_to_collision: float  # Time to potential collision (seconds)
    miss_distance: float  # Distance at closest approach (meters)
    conflict_type: str  # "collision", "near_miss", "safe_pass", "moving_away"
    direction: str  # "left", "right", "center"
    contributing_factors: Dict[str, float]
    timestamp_s: float
    confidence: float


class CollisionRiskAssessor:
    """Assess collision risk using CPA and comprehensive factors."""

    def __init__(self, cpa_detector: Optional[CPADetector] = None):
        """Initialize collision risk assessor.
        
        Args:
            cpa_detector: CPA detector instance (creates default if None)
        """
        self.cpa_detector = cpa_detector or CPADetector()
        
        # Risk thresholds
        self.probability_thresholds = {
            RiskLevel.SAFE: 0.0,
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.4,
            RiskLevel.HIGH: 0.7,
            RiskLevel.CRITICAL: 0.9
        }

    def assess_risk(self, object_state: ObjectState, ego_state: EgoState,
                    object_predictions: Optional[np.ndarray] = None,
                    ego_predictions: Optional[np.ndarray] = None,
                    times: Optional[np.ndarray] = None) -> CollisionRiskAssessment:
        """Comprehensive risk assessment for object-ego pair.
        
        This implements the full collision reasoning pipeline:
        1. Calculate CPA between trajectories
        2. Assess collision probability
        3. Determine conflict type
        4. Calculate risk level
        5. Identify contributing factors
        
        Args:
            object_state: Object state in world frame
            ego_state: Ego state in world frame
            object_predictions: Optional predicted object trajectory
            ego_predictions: Optional predicted ego trajectory
            times: Time points for trajectory predictions
            
        Returns:
            Comprehensive collision risk assessment
        """
        # Calculate CPA
        if object_predictions is not None and ego_predictions is not None and times is not None:
            cpa_result = self.cpa_detector.calculate_trajectory_cpa(
                object_predictions, ego_predictions, times
            )
        else:
            cpa_result = self.cpa_detector.calculate_cpa(
                object_state.position, object_state.velocity,
                ego_state.position, ego_state.velocity
            )
        
        # Get conflict type and miss distance
        conflict_type, miss_distance = self.cpa_detector.assess_miss_distance(cpa_result)
        
        # Calculate collision probability
        collision_probability = self.cpa_detector.get_collision_probability(
            cpa_result, object_state.speed, object_state.label
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(collision_probability, cpa_result, conflict_type)
        
        # Determine threat direction
        direction = self._determine_threat_direction(object_state, ego_state)
        
        # Identify contributing factors
        contributing_factors = self._analyze_contributing_factors(
            object_state, ego_state, cpa_result, collision_probability
        )
        
        # Estimate time to collision
        if cpa_result.will_collide:
            time_to_collision = cpa_result.time_to_cpa
        else:
            time_to_collision = float('inf')
        
        return CollisionRiskAssessment(
            object_id=object_state.object_id,
            risk_level=risk_level,
            probability=collision_probability,
            cpa_result=cpa_result,
            time_to_collision=time_to_collision,
            miss_distance=miss_distance,
            conflict_type=conflict_type,
            direction=direction,
            contributing_factors=contributing_factors,
            timestamp_s=object_state.timestamp_s,
            confidence=cpa_result.confidence
        )

    def _determine_risk_level(self, probability: float, cpa_result: CPAResult, 
                            conflict_type: str) -> RiskLevel:
        """Determine risk level from probability and CPA results.
        
        Args:
            probability: Collision probability
            cpa_result: CPA calculation result
            conflict_type: Type of conflict
            
        Returns:
            Risk level classification
        """
        # Start with probability-based classification
        if probability >= self.probability_thresholds[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif probability >= self.probability_thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif probability >= self.probability_thresholds[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        elif probability >= self.probability_thresholds[RiskLevel.LOW]:
            return RiskLevel.LOW
        else:
            return RiskLevel.SAFE

    def _determine_threat_direction(self, object_state: ObjectState, 
                                 ego_state: EgoState) -> str:
        """Determine which side the threat is coming from.
        
        Args:
            object_state: Object state
            ego_state: Ego state
            
        Returns:
            Direction: "left", "right", or "center"
        """
        # Calculate relative bearing
        relative_bearing = object_state.bearing - ego_state.yaw
        
        # Normalize to [-π, π]
        relative_bearing = (relative_bearing + np.pi) % (2 * np.pi) - np.pi
        
        # Determine direction
        if abs(relative_bearing) < 0.3:  # ~17 degrees
            return "center"
        elif relative_bearing > 0:
            return "right"
        else:
            return "left"

    def _analyze_contributing_factors(self, object_state: ObjectState, ego_state: EgoState,
                                   cpa_result: CPAResult, probability: float) -> Dict[str, float]:
        """Analyze factors contributing to risk assessment.
        
        Args:
            object_state: Object state
            ego_state: Ego state
            cpa_result: CPA result
            probability: Collision probability
            
        Returns:
            Dictionary of contributing factors with their influence (0-1)
        """
        factors = {}
        
        # Distance factor (closer = higher risk)
        distance = np.linalg.norm(object_state.position)
        distance_factor = max(0.0, 1.0 - (distance / 20.0))  # Normalize to 20m range
        factors["distance"] = distance_factor
        
        # Speed factor (faster = higher risk)
        speed_factor = min(1.0, object_state.speed / 15.0)  # Normalize to 15 m/s
        factors["speed"] = speed_factor
        
        # Heading alignment factor (heading toward rider = higher risk)
        relative_bearing = abs(object_state.bearing - ego_state.yaw)
        heading_factor = max(0.0, 1.0 - (relative_bearing / np.pi))  # Normalize to π range
        factors["heading_alignment"] = heading_factor
        
        # CPA distance factor
        if cpa_result.distance_at_cpa < float('inf'):
            cpa_factor = max(0.0, 1.0 - (cpa_result.distance_at_cpa / (self.cpa_detector.safe_distance * 3)))
            factors["cpa_distance"] = cpa_factor
        else:
            factors["cpa_distance"] = 0.0
        
        # Time factor (sooner CPA = higher risk)
        if 0 < cpa_result.time_to_cpa < self.cpa_detector.time_horizon:
            time_factor = 1.0 - (cpa_result.time_to_cpa / self.cpa_detector.time_horizon)
            factors["time_to_cpa"] = time_factor
        else:
            factors["time_to_cpa"] = 0.0
        
        # Object class factor
        class_risk = {
            "truck": 0.9,
            "bus": 0.85,
            "car": 0.8,
            "motorcycle": 0.7,
            "bicycle": 0.5,
            "person": 0.4
        }
        factors["object_class"] = class_risk.get(object_state.label.lower(), 0.6)
        
        # Ego motion factor (turning = higher risk)
        ego_turning = abs(ego_state.yaw_rate) > 0.1
        factors["ego_turning"] = 1.0 if ego_turning else 0.3
        
        return factors

    def assess_multiple_objects(self, object_states: List[ObjectState], 
                              ego_state: EgoState) -> List[CollisionRiskAssessment]:
        """Assess risk for multiple objects and return sorted by severity.
        
        Args:
            object_states: List of object states
            ego_state: Current ego state
            
        Returns:
            List of collision risks sorted by probability (highest first)
        """
        risks = []
        
        for object_state in object_states:
            risk = self.assess_risk(object_state, ego_state)
            risks.append(risk)
        
        # Sort by probability (highest risk first)
        risks.sort(key=lambda r: r.probability, reverse=True)
        
        return risks

    def get_highest_risk(self, object_states: List[ObjectState], 
                       ego_state: EgoState) -> Optional[CollisionRiskAssessment]:
        """Get the highest risk object among multiple objects.
        
        Args:
            object_states: List of object states
            ego_state: Current ego state
            
        Returns:
            Highest risk assessment or None if no objects
        """
        risks = self.assess_multiple_objects(object_states, ego_state)
        return risks[0] if risks else None

    def filter_by_risk_level(self, risks: List[CollisionRiskAssessment], 
                           min_level: RiskLevel) -> List[CollisionRiskAssessment]:
        """Filter risks by minimum risk level.
        
        Args:
            risks: List of collision risks
            min_level: Minimum risk level to include
            
        Returns:
            Filtered list of risks
        """
        level_order = [RiskLevel.SAFE, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        min_index = level_order.index(min_level)
        
        return [risk for risk in risks if level_order.index(risk.risk_level) >= min_index]