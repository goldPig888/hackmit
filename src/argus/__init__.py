"""ARGUS reusable safety-estimation library."""

from .models import Detection, HapticEvent, RiskAssessment, TrackState
from .pipeline import ArgusPipeline

# New modules for custom detection and testing
from .detectors import BaseDetector, DetectorRegistry, register_detector
from .testing import ComponentTester, MockDetectionGenerator, MockIMUGenerator, PipelineTester
from .hazards import SyncHazardDetector, PatternAnalyzer
from .fine_tuning import ModelTrainer, ModelEvaluator, DataLoader
from .dashboard import DashboardServer, DataStreamer

# Advanced camera-frame realignment modules
from .imu import IMUProcessor, CameraPoseEstimator, RotationCompensator
from .tracking import WorldFrameTracker, ObjectState, EgoTracker
from .collision import CPADetector, CollisionRiskAssessor, RiskLevel, CollisionRiskAssessment
from .filters import ExtendedKalmanFilter, UnscentedKalmanFilter
from .vio import VisualOdometryEstimator, VIOFusion
from .advanced_pipeline import AdvancedArgusPipeline, AdvancedPipelineConfig

__all__ = [
    "ArgusPipeline", "Detection", "HapticEvent", "RiskAssessment", "TrackState",
    "BaseDetector", "DetectorRegistry", "register_detector",
    "ComponentTester", "MockDetectionGenerator", "MockIMUGenerator", "PipelineTester",
    "SyncHazardDetector", "PatternAnalyzer",
    "ModelTrainer", "ModelEvaluator", "DataLoader",
    "DashboardServer", "DataStreamer",
    "IMUProcessor", "CameraPoseEstimator", "RotationCompensator",
    "WorldFrameTracker", "ObjectState", "EgoTracker",
    "CPADetector", "CollisionRiskAssessor", "RiskLevel", "CollisionRiskAssessment",
    "ExtendedKalmanFilter", "UnscentedKalmanFilter",
    "VisualOdometryEstimator", "VIOFusion",
    "AdvancedArgusPipeline", "AdvancedPipelineConfig"
]
