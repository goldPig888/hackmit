"""ARGUS reusable safety-estimation library."""

from .models import Detection, HapticEvent, RiskAssessment, TrackState
from .pipeline import ArgusPipeline

# New modules for custom detection and testing
from .detectors import BaseDetector, DetectorRegistry, register_detector
from .testing import ComponentTester, MockDetectionGenerator, MockIMUGenerator, PipelineTester
from .hazards import SyncHazardDetector, PatternAnalyzer
from .fine_tuning import ModelTrainer, ModelEvaluator, DataLoader

__all__ = [
    "ArgusPipeline", "Detection", "HapticEvent", "RiskAssessment", "TrackState",
    "BaseDetector", "DetectorRegistry", "register_detector",
    "ComponentTester", "MockDetectionGenerator", "MockIMUGenerator", "PipelineTester",
    "SyncHazardDetector", "PatternAnalyzer",
    "ModelTrainer", "ModelEvaluator", "DataLoader"
]
