"""Detector registry for managing custom detection models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .base import BaseDetector


class DetectorRegistry:
    """Registry for custom detection models.
    
    Allows registration and retrieval of detector implementations.
    """

    _detectors: dict[str, type[BaseDetector]] = {}
    _factories: dict[str, Callable[..., BaseDetector]] = {}

    @classmethod
    def register(cls, name: str, detector_class: type[BaseDetector]) -> None:
        """Register a detector class.
        
        Args:
            name: Unique identifier for this detector
            detector_class: Detector class inheriting from BaseDetector
        """
        cls._detectors[name] = detector_class

    @classmethod
    def register_factory(cls, name: str, factory: Callable[..., BaseDetector]) -> None:
        """Register a detector factory function.
        
        Args:
            name: Unique identifier for this detector
            factory: Function that returns a configured BaseDetector instance
        """
        cls._factories[name] = factory

    @classmethod
    def get(cls, name: str) -> type[BaseDetector] | Callable[..., BaseDetector] | None:
        """Get a detector by name.
        
        Args:
            name: Detector identifier
            
        Returns:
            Detector class or factory function, or None if not found
        """
        if name in cls._detectors:
            return cls._detectors[name]
        if name in cls._factories:
            return cls._factories[name]
        return None

    @classmethod
    def list_registered(cls) -> list[str]:
        """List all registered detector names."""
        return sorted(set(cls._detectors.keys()) | set(cls._factories.keys()))

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> BaseDetector:
        """Create a detector instance by name.
        
        Args:
            name: Detector identifier
            *args: Positional arguments to pass to detector constructor/factory
            **kwargs: Keyword arguments to pass to detector constructor/factory
            
        Returns:
            Configured detector instance
            
        Raises:
            ValueError: If detector name is not registered
        """
        detector_cls_or_factory = cls.get(name)
        if detector_cls_or_factory is None:
            raise ValueError(f"Detector '{name}' not registered. Available: {cls.list_registered()}")
        return detector_cls_or_factory(*args, **kwargs)


def register_detector(name: str):
    """Decorator for registering detector classes.
    
    Usage:
        @register_detector("my_custom_detector")
        class MyDetector(BaseDetector):
            ...
    """
    def decorator(detector_class: type[BaseDetector]) -> type[BaseDetector]:
        DetectorRegistry.register(name, detector_class)
        return detector_class
    return decorator