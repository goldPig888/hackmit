"""Synchronized hazard detection for coordinated threats."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from ..models import Detection, TrackState


@dataclass
class SyncHazard:
    """Represents a synchronized hazard event."""
    hazard_type: str
    involved_objects: list[str]
    severity: float
    description: str
    timestamp_s: float


class SyncHazardDetector:
    """Detect synchronized/coordinated hazards from multiple objects.
    
    Identifies patterns like:
    - Multiple vehicles converging on rider
    - Coordinated movement patterns
    - Ambush scenarios
    - Correlated acceleration/velocity changes
    """

    def __init__(self, convergence_threshold_m: float = 5.0, 
                 correlation_threshold: float = 0.7,
                 time_window_s: float = 2.0):
        """Initialize sync hazard detector.
        
        Args:
            convergence_threshold_m: Distance threshold for convergence detection
            correlation_threshold: Correlation threshold for coordinated movement
            time_window_s: Time window for pattern analysis
        """
        self.convergence_threshold_m = convergence_threshold_m
        self.correlation_threshold = correlation_threshold
        self.time_window_s = time_window_s
        self.object_history: dict[str, list[TrackState]] = defaultdict(list)
        self.detected_hazards: list[SyncHazard] = []

    def update(self, track_states: list[TrackState]) -> list[SyncHazard]:
        """Update detector with new track states and detect hazards.
        
        Args:
            track_states: List of current track states
            
        Returns:
            List of newly detected sync hazards
        """
        current_time = max((ts.timestamp_s for ts in track_states), default=0.0)
        
        # Update object history
        for track_state in track_states:
            self.object_history[track_state.object_id].append(track_state)
            # Remove old data outside time window
            self.object_history[track_state.object_id] = [
                ts for ts in self.object_history[track_state.object_id]
                if current_time - ts.timestamp_s <= self.time_window_s
            ]
        
        # Detect various hazard patterns
        new_hazards = []
        new_hazards.extend(self._detect_convergence(track_states, current_time))
        new_hazards.extend(self._detect_coordinated_movement(track_states, current_time))
        new_hazards.extend(self._detect_encirclement(track_states, current_time))
        
        self.detected_hazards.extend(new_hazards)
        return new_hazards

    def _detect_convergence(self, track_states: list[TrackState], current_time: float) -> list[SyncHazard]:
        """Detect multiple objects converging on the rider.
        
        Args:
            track_states: Current track states
            current_time: Current timestamp
            
        Returns:
            List of convergence hazards
        """
        hazards = []
        if len(track_states) < 2:
            return hazards
        
        # Check if multiple objects are approaching rider (position near origin)
        approaching_objects = [
            ts for ts in track_states
            if np.linalg.norm(ts.position_m) < self.convergence_threshold_m
            and ts.closing
        ]
        
        if len(approaching_objects) >= 2:
            severity = min(1.0, len(approaching_objects) / 3.0)  # More objects = higher severity
            hazard = SyncHazard(
                hazard_type="convergence",
                involved_objects=[ts.object_id for ts in approaching_objects],
                severity=severity,
                description=f"{len(approaching_objects)} objects converging on rider position",
                timestamp_s=current_time
            )
            hazards.append(hazard)
        
        return hazards

    def _detect_coordinated_movement(self, track_states: list[TrackState], 
                                    current_time: float) -> list[SyncHazard]:
        """Detect coordinated movement patterns between objects.
        
        Args:
            track_states: Current track states
            current_time: Current timestamp
            
        Returns:
            List of coordinated movement hazards
        """
        hazards = []
        if len(track_states) < 2:
            return hazards
        
        # Calculate velocity correlations between object pairs
        for i, ts1 in enumerate(track_states):
            for ts2 in track_states[i+1:]:
                # Get velocity histories
                hist1 = self.object_history[ts1.object_id]
                hist2 = self.object_history[ts2.object_id]
                
                if len(hist1) < 3 or len(hist2) < 3:
                    continue
                
                # Calculate velocity correlation
                vel1 = np.array([ts.velocity_mps for ts in hist1])
                vel2 = np.array([ts.velocity_mps for ts in hist2])
                
                # Normalize and correlate
                if len(vel1) == len(vel2):
                    correlation = self._calculate_correlation(vel1, vel2)
                    
                    if correlation > self.correlation_threshold:
                        # Check if they're moving toward rider
                        both_closing = ts1.closing and ts2.closing
                        if both_closing:
                            severity = correlation
                            hazard = SyncHazard(
                                hazard_type="coordinated_movement",
                                involved_objects=[ts1.object_id, ts2.object_id],
                                severity=severity,
                                description=f"Coordinated movement detected (correlation: {correlation:.2f})",
                                timestamp_s=current_time
                            )
                            hazards.append(hazard)
        
        return hazards

    def _detect_encirclement(self, track_states: list[TrackState], 
                           current_time: float) -> list[SyncHazard]:
        """Detect potential encirclement patterns.
        
        Args:
            track_states: Current track states
            current_time: Current timestamp
            
        Returns:
            List of encirclement hazards
        """
        hazards = []
        if len(track_states) < 3:
            return hazards
        
        # Check if objects surround the rider from different angles
        angles = []
        for ts in track_states:
            if np.linalg.norm(ts.position_m) < self.convergence_threshold_m * 2:
                angle = np.arctan2(ts.position_m[1], ts.position_m[0])
                angles.append(angle)
        
        if len(angles) >= 3:
            # Check angular spread
            angles_sorted = sorted(angles)
            max_gap = max(
                (angles_sorted[(i+1) % len(angles_sorted)] - angles_sorted[i]) % (2 * np.pi)
                for i in range(len(angles_sorted))
            )
            
            # If objects cover most angles around rider
            if max_gap < np.pi / 2:  # Less than 90 degree gap
                severity = 1.0 - (max_gap / np.pi)
                hazard = SyncHazard(
                    hazard_type="encirclement",
                    involved_objects=[ts.object_id for ts in track_states if ts.object_id in [ts.object_id for ts in track_states]],
                    severity=severity,
                    description=f"Potential encirclement detected ({len(angles)} objects)",
                    timestamp_s=current_time
                )
                hazards.append(hazard)
        
        return hazards

    def _calculate_correlation(self, arr1: np.ndarray, arr2: np.ndarray) -> float:
        """Calculate correlation between two arrays.
        
        Args:
            arr1: First array
            arr2: Second array
            
        Returns:
            Correlation coefficient
        """
        # Flatten if needed
        arr1 = arr1.flatten()
        arr2 = arr2.flatten()
        
        # Calculate correlation for each dimension
        correlations = []
        for dim in range(min(arr1.shape[0], arr2.shape[0])):
            if len(arr1) > 1 and len(arr2) > 1:
                corr_matrix = np.corrcoef(arr1[dim:dim+1], arr2[dim:dim+1])
                if not np.isnan(corr_matrix[0, 1]):
                    correlations.append(abs(corr_matrix[0, 1]))
        
        return float(np.mean(correlations)) if correlations else 0.0

    def get_recent_hazards(self, time_window_s: float = 5.0) -> list[SyncHazard]:
        """Get hazards detected within the time window.
        
        Args:
            time_window_s: Time window in seconds
            
        Returns:
            List of recent hazards
        """
        current_time = max((h.timestamp_s for h in self.detected_hazards), default=0.0)
        return [
            h for h in self.detected_hazards
            if current_time - h.timestamp_s <= time_window_s
        ]

    def clear_history(self) -> None:
        """Clear all detection history."""
        self.object_history.clear()
        self.detected_hazards.clear()