"""Pattern analysis for hazard detection."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
from enum import Enum

if TYPE_CHECKING:
    from ..models import TrackState


class HazardPattern(Enum):
    """Types of hazard patterns."""
    CONVERGENCE = "convergence"
    COORDINATED = "coordinated"
    ENCIRCLEMENT = "encirclement"
    AMBUSH = "ambush"
    TRAP = "trap"
    FLANKING = "flanking"
    UNKNOWN = "unknown"


@dataclass
class PatternMatch:
    """Result of pattern matching."""
    pattern: HazardPattern
    confidence: float
    involved_objects: list[str]
    metadata: dict


class PatternAnalyzer:
    """Analyze movement patterns for sophisticated hazard detection.
    
    Identifies complex threat patterns like:
    - Ambush scenarios (one object distracts, others approach)
    - Trap scenarios (blocking escape routes)
    - Coordinated flanking maneuvers
    """

    def __init__(self, min_objects: int = 2, confidence_threshold: float = 0.6):
        """Initialize pattern analyzer.
        
        Args:
            min_objects: Minimum number of objects for pattern detection
            confidence_threshold: Minimum confidence to report patterns
        """
        self.min_objects = min_objects
        self.confidence_threshold = confidence_threshold
        self.pattern_rules: dict[HazardPattern, Callable] = {
            HazardPattern.AMBUSH: self._detect_ambush,
            HazardPattern.TRAP: self._detect_trap,
            HazardPattern.FLANKING: self._detect_flanking
        }

    def analyze(self, track_states: list[TrackState]) -> list[PatternMatch]:
        """Analyze current track states for hazard patterns.
        
        Args:
            track_states: Current track states
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        if len(track_states) < self.min_objects:
            return patterns
        
        for pattern_type, rule_func in self.pattern_rules.items():
            try:
                match = rule_func(track_states)
                if match and match.confidence >= self.confidence_threshold:
                    patterns.append(match)
            except Exception:
                # Pattern detection errors shouldn't crash the system
                continue
        
        return patterns

    def _detect_ambush(self, track_states: list[TrackState]) -> PatternMatch | None:
        """Detect ambush pattern: one visible threat while others approach from blind spots.
        
        Args:
            track_states: Current track states
            
        Returns:
            Pattern match if ambush detected
        """
        if len(track_states) < 2:
            return None
        
        # Look for one prominent object (distraction) and others approaching
        # from sides/rear while rider is focused on the distraction
        
        # Sort by distance - closest might be the distraction
        sorted_by_distance = sorted(track_states, key=lambda ts: np.linalg.norm(ts.position_m))
        
        if len(sorted_by_distance) < 2:
            return None
        
        closest = sorted_by_distance[0]
        others = sorted_by_distance[1:]
        
        # Check if others are approaching from sides while closest is in front
        front_objects = [ts for ts in others if ts.position_m[1] > 0]  # In front
        side_objects = [ts for ts in others if abs(ts.position_m[0]) > 2.0]  # On sides
        
        if len(side_objects) >= 1 and len(front_objects) >= 1:
            closing_sides = sum(1 for ts in side_objects if ts.closing)
            if closing_sides >= 1:
                confidence = min(1.0, closing_sides / len(side_objects) + 0.3)
                return PatternMatch(
                    pattern=HazardPattern.AMBUSH,
                    confidence=confidence,
                    involved_objects=[ts.object_id for ts in track_states],
                    metadata={
                        "distraction_object": closest.object_id,
                        "flanking_objects": [ts.object_id for ts in side_objects if ts.closing]
                    }
                )
        
        return None

    def _detect_trap(self, track_states: list[TrackState]) -> PatternMatch | None:
        """Detect trap pattern: objects blocking potential escape routes.
        
        Args:
            track_states: Current track states
            
        Returns:
            Pattern match if trap detected
        """
        if len(track_states) < 2:
            return None
        
        # Check if objects are positioned to block different directions
        # Divide space into quadrants relative to rider
        quadrants = {
            "front_left": [],
            "front_right": [],
            "rear_left": [],
            "rear_right": []
        }
        
        for ts in track_states:
            x, y = ts.position_m
            if y > 0:  # Front
                if x < 0:
                    quadrants["front_left"].append(ts.object_id)
                else:
                    quadrants["front_right"].append(ts.object_id)
            else:  # Rear
                if x < 0:
                    quadrants["rear_left"].append(ts.object_id)
                else:
                    quadrants["rear_right"].append(ts.object_id)
        
        # Count occupied quadrants
        occupied_quadrants = sum(1 for q in quadrants.values() if len(q) > 0)
        
        if occupied_quadrants >= 3:
            confidence = min(1.0, occupied_quadrants / 4.0)
            return PatternMatch(
                pattern=HazardPattern.TRAP,
                confidence=confidence,
                involved_objects=[ts.object_id for ts in track_states],
                metadata={
                    "occupied_quadrants": occupied_quadrants,
                    "quadrant_distribution": {k: len(v) for k, v in quadrants.items()}
                }
            )
        
        return None

    def _detect_flanking(self, track_states: list[TrackState]) -> PatternMatch | None:
        """Detect flanking pattern: objects approaching from both sides simultaneously.
        
        Args:
            track_states: Current track states
            
        Returns:
            Pattern match if flanking detected
        """
        if len(track_states) < 2:
            return None
        
        # Separate objects by left/right side
        left_objects = [ts for ts in track_states if ts.position_m[0] < -1.0 and ts.closing]
        right_objects = [ts for ts in track_states if ts.position_m[0] > 1.0 and ts.closing]
        
        if len(left_objects) >= 1 and len(right_objects) >= 1:
            # Calculate symmetry confidence
            left_count = len(left_objects)
            right_count = len(right_objects)
            symmetry = 1.0 - abs(left_count - right_count) / max(left_count + right_count, 1)
            
            confidence = symmetry * 0.8  # Base confidence from symmetry
            
            return PatternMatch(
                pattern=HazardPattern.FLANKING,
                confidence=confidence,
                involved_objects=[ts.object_id for ts in left_objects + right_objects],
                metadata={
                    "left_count": left_count,
                    "right_count": right_count,
                    "symmetry": symmetry
                }
            )
        
        return None

    def set_custom_rule(self, pattern: HazardPattern, rule_func: Callable) -> None:
        """Set a custom pattern detection rule.
        
        Args:
            pattern: Pattern type
            rule_func: Function that takes track_states and returns PatternMatch or None
        """
        self.pattern_rules[pattern] = rule_func