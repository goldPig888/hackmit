"""Mock data generators for testing ARGUS components."""

from __future__ import annotations

import numpy as np
from typing import Generator

from ..models import Detection


class MockDetectionGenerator:
    """Generate synthetic detection data for testing."""

    def __init__(self, frame_width: int = 1280, frame_height: int = 720):
        """Initialize mock detection generator.
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        self.frame_width = frame_width
        self.frame_height = frame_height

    def approaching_car(self, num_frames: int = 45, start_pos: tuple[float, float] = (260, 360),
                       growth_rate: float = 2.7, lateral_speed: float = 8.0) -> Generator[Detection, None, None]:
        """Generate detections for an approaching car.
        
        Args:
            num_frames: Number of frames to generate
            start_pos: Starting (x, y) position
            growth_rate: How fast the bounding box grows (pixels per frame)
            lateral_speed: Lateral movement speed (pixels per frame)
            
        Yields:
            Detection objects simulating an approaching car
        """
        for frame in range(num_frames):
            timestamp = frame / 10.0
            size = 30 + frame * growth_rate
            x = start_pos[0] + frame * lateral_speed
            y = start_pos[1] - frame * 2  # Slight upward movement
            
            yield Detection(
                object_id="car-12",
                label="car",
                confidence=0.94,
                bbox=(x, y, size, size * 0.72),
                timestamp_s=timestamp
            )

    def crossing_vehicle(self, num_frames: int = 60, start_x: float = 100, 
                        y_position: float = 400, speed: float = 15.0) -> Generator[Detection, None, None]:
        """Generate detections for a vehicle crossing from left to right.
        
        Args:
            num_frames: Number of frames to generate
            start_x: Starting x position
            y_position: Y position (constant)
            speed: Movement speed in pixels per frame
            
        Yields:
            Detection objects simulating a crossing vehicle
        """
        for frame in range(num_frames):
            timestamp = frame / 10.0
            x = start_x + frame * speed
            size = 80
            
            yield Detection(
                object_id="car-31",
                label="car",
                confidence=0.89,
                bbox=(x, y_position, size, size * 0.6),
                timestamp_s=timestamp
            )

    def static_object(self, num_frames: int = 30, position: tuple[float, float] = (640, 360),
                     size: float = 100) -> Generator[Detection, None, None]:
        """Generate detections for a static object.
        
        Args:
            num_frames: Number of frames to generate
            position: (x, y) position
            size: Bounding box size
            
        Yields:
            Detection objects for a static object
        """
        for frame in range(num_frames):
            timestamp = frame / 10.0
            
            yield Detection(
                object_id="static-1",
                label="person",
                confidence=0.92,
                bbox=(position[0], position[1], size, size * 1.8),
                timestamp_s=timestamp
            )

    def multiple_objects(self, num_frames: int = 45) -> Generator[list[Detection], None, None]:
        """Generate multiple objects in each frame.
        
        Args:
            num_frames: Number of frames to generate
            
        Yields:
            Lists of Detection objects with multiple objects per frame
        """
        for frame in range(num_frames):
            timestamp = frame / 10.0
            detections = []
            
            # Approaching car
            size = 30 + frame * 2.0
            detections.append(Detection(
                object_id="car-12",
                label="car",
                confidence=0.94,
                bbox=(260 + frame * 6, 360 - frame * 2, size, size * 0.72),
                timestamp_s=timestamp
            ))
            
            # Crossing vehicle
            x = 100 + frame * 12
            if x < 1200:  # Only if still in frame
                detections.append(Detection(
                    object_id="car-31",
                    label="car",
                    confidence=0.87,
                    bbox=(x, 400, 80, 48),
                    timestamp_s=timestamp
                ))
            
            yield detections


class MockIMUGenerator:
    """Generate synthetic IMU data for testing."""

    def straight_motion(self, num_samples: int = 100, sample_rate: float = 10.0) -> Generator[tuple[float, np.ndarray, np.ndarray], None, None]:
        """Generate IMU data for straight motion.
        
        Args:
            num_samples: Number of samples to generate
            sample_rate: Sample rate in Hz
            
        Yields:
            Tuple of (timestamp, acceleration, angular_velocity)
        """
        for i in range(num_samples):
            timestamp = i / sample_rate
            # Slight forward acceleration, minimal rotation
            acceleration = np.array([0.1, 0.0, 9.8])  # x, y, z (m/s^2)
            angular_velocity = np.array([0.0, 0.0, 0.01])  # roll, pitch, yaw (rad/s)
            yield timestamp, acceleration, angular_velocity

    def left_turn(self, num_samples: int = 100, sample_rate: float = 10.0,
                 yaw_rate: float = 0.3) -> Generator[tuple[float, np.ndarray, np.ndarray], None, None]:
        """Generate IMU data for a left turn.
        
        Args:
            num_samples: Number of samples to generate
            sample_rate: Sample rate in Hz
            yaw_rate: Yaw rate in rad/s
            
        Yields:
            Tuple of (timestamp, acceleration, angular_velocity)
        """
        for i in range(num_samples):
            timestamp = i / sample_rate
            # Centripetal acceleration during turn
            acceleration = np.array([-0.5, 0.0, 9.8])
            angular_velocity = np.array([0.0, 0.0, yaw_rate])
            yield timestamp, acceleration, angular_velocity

    def right_turn(self, num_samples: int = 100, sample_rate: float = 10.0,
                  yaw_rate: float = -0.3) -> Generator[tuple[float, np.ndarray, np.ndarray], None, None]:
        """Generate IMU data for a right turn.
        
        Args:
            num_samples: Number of samples to generate
            sample_rate: Sample rate in Hz
            yaw_rate: Yaw rate in rad/s (negative for right turn)
            
        Yields:
            Tuple of (timestamp, acceleration, angular_velocity)
        """
        for i in range(num_samples):
            timestamp = i / sample_rate
            acceleration = np.array([0.5, 0.0, 9.8])
            angular_velocity = np.array([0.0, 0.0, yaw_rate])
            yield timestamp, acceleration, angular_velocity