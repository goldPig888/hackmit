"""Custom detection models plugin system."""

from .base import BaseDetector
from .registry import DetectorRegistry, register_detector
from .yolo import YOLODetector
from .weapons import WeaponsDetector, FineTunedWeaponsDetector

__all__ = [
    "BaseDetector", 
    "DetectorRegistry", 
    "register_detector",
    "YOLODetector",
    "WeaponsDetector",
    "FineTunedWeaponsDetector"
]