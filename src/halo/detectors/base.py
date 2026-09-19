"""Base detector interface for custom detection models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Detection


class BaseDetector(ABC):
    """Abstract base class for custom detection models.
    
    All custom detectors should inherit from this class to ensure
    compatibility with the HALO pipeline.
    """

    @abstractmethod
    def detect(self, frame, timestamp_s: float) -> list[Detection]:
        """Process a frame and return detections.
        
        Args:
            frame: Input frame (type depends on detector - could be numpy array, etc.)
            timestamp_s: Current timestamp in seconds
            
        Returns:
            List of Detection objects
        """
        pass

    @abstractmethod
    def get_supported_classes(self) -> set[str]:
        """Return the set of object classes this detector can identify."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources and cleanup detector."""
        pass