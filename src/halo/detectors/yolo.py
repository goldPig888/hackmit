"""YOLO detector implementation for HALO."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Detection

from .base import BaseDetector


class YOLODetector(BaseDetector):
    """YOLO-based detector using ultralytics package.
    
    Wraps YOLO for compatibility with HALO detector plugin system.
    """

    def __init__(self, model_path: str = "yolo11n.pt", tracker_config: str = "bytetrack.yaml", 
                 allowed_classes: set[str] | None = None):
        """Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights
            tracker_config: Path to tracker configuration
            allowed_classes: Set of class names to detect (None = all classes)
        """
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError("YOLO detector requires ultralytics package. Install with: pip install ultralytics") from error
        
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.allowed_classes = allowed_classes or {
            "car", "truck", "bus", "motorcycle", "bicycle", "person"
        }
        self._supported_classes = set(self.model.names.values())
        # Compact per-epoch display IDs: ByteTrack's internal counter keeps
        # climbing forever, so map raw IDs onto 0..N and restart on reset().
        self._id_map: dict[int, int] = {}

    def reset(self) -> None:
        """Restart tracking epoch — next detection becomes ID 0."""
        self._id_map.clear()
        try:
            from ultralytics.trackers.basetrack import BaseTrack
            BaseTrack.reset_id()
            if getattr(self.model, "predictor", None) is not None:
                self.model.predictor = None
        except Exception:
            pass

    def detect(self, frame, timestamp_s: float) -> list[Detection]:
        """Process frame with YOLO detection and tracking.
        
        Args:
            frame: Input frame (numpy array)
            timestamp_s: Current timestamp in seconds
            
        Returns:
            List of Detection objects
        """
        import numpy as np
        
        from ..models import Detection
        
        result = self.model.track(frame, persist=True, tracker=self.tracker_config, verbose=False)[0]
        detections = []
        
        if result.boxes.id is not None:
            names = result.names
            for box, track_id, cls, confidence in zip(
                result.boxes.xywh.cpu().tolist(),
                result.boxes.id.int().cpu().tolist(),
                result.boxes.cls.int().cpu().tolist(),
                result.boxes.conf.cpu().tolist()
            ):
                label = names[cls]
                if label not in self.allowed_classes:
                    continue
                
                x, y, w, h = box
                compact_id = self._id_map.setdefault(track_id, len(self._id_map))
                object_id = f"{label}-{compact_id}"
                detections.append(Detection(
                    object_id=object_id,
                    label=label,
                    confidence=float(confidence),
                    bbox=(float(x - w/2), float(y - h/2), float(w), float(h)),
                    timestamp_s=timestamp_s
                ))
        
        return detections

    def get_supported_classes(self) -> set[str]:
        """Return classes this YOLO model can detect."""
        return self._supported_classes

    def cleanup(self) -> None:
        """Release YOLO model resources."""
        # YOLO models are automatically cleaned up by Python garbage collection
        pass