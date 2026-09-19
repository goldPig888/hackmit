"""IMU data processing for camera motion estimation."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import deque
import time


@dataclass
class IMUReading:
    """Single IMU reading."""
    timestamp_s: float
    gyro: np.ndarray  # Angular velocity [rad/s] (roll, pitch, yaw)
    accel: np.ndarray  # Linear acceleration [m/s^2] (x, y, z)
    quaternion: Optional[np.ndarray] = None  # Orientation quaternion [w, x, y, z]


class IMUProcessor:
    """Process IMU data for camera motion estimation.
    
    Handles gyroscope and accelerometer data to estimate camera orientation
    and angular velocity for ego-motion compensation.
    """

    def __init__(self, sample_rate: float = 50.0, buffer_size: int = 100):
        """Initialize IMU processor.
        
        Args:
            sample_rate: IMU sample rate in Hz
            buffer_size: Size of internal buffer for smoothing
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.reading_buffer: deque[IMUReading] = deque(maxlen=buffer_size)
        
        # Current orientation estimate (simple integration)
        self.current_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self.last_timestamp: Optional[float] = None
        
        # Calibration parameters
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.is_calibrated = False

    def add_reading(self, gyro: np.ndarray, accel: np.ndarray, timestamp_s: float) -> IMUReading:
        """Add a new IMU reading.
        
        Args:
            gyro: Angular velocity [rad/s] (roll, pitch, yaw)
            accel: Linear acceleration [m/s^2] (x, y, z)
            timestamp_s: Timestamp in seconds
            
        Returns:
            IMUReading object with processed data
        """
        # Apply calibration if available
        if self.is_calibrated:
            gyro = gyro - self.gyro_bias
            accel = accel - self.accel_bias
        
        # Update orientation estimate (simple quaternion integration)
        if self.last_timestamp is not None:
            dt = timestamp_s - self.last_timestamp
            if dt > 0:
                self._integrate_gyroscope(gyro, dt)
        
        self.last_timestamp = timestamp_s
        
        reading = IMUReading(
            timestamp_s=timestamp_s,
            gyro=gyro,
            accel=accel,
            quaternion=self.current_quaternion.copy()
        )
        
        self.reading_buffer.append(reading)
        return reading

    def _integrate_gyroscope(self, gyro: np.ndarray, dt: float) -> None:
        """Integrate gyroscope data to update orientation.
        
        Args:
            gyro: Angular velocity [rad/s]
            dt: Time step in seconds
        """
        # Convert gyro to quaternion rate of change
        # q_dot = 0.5 * q * omega (quaternion multiplication)
        omega = np.array([0, *gyro])  # [0, wx, wy, wz]
        
        # Quaternion multiplication: q * omega
        q = self.current_quaternion
        q_dot = 0.5 * self._quaternion_multiply(q, omega)
        
        # Integrate: q_new = q + q_dot * dt
        self.current_quaternion = q + q_dot * dt
        
        # Normalize quaternion
        self.current_quaternion = self.current_quaternion / np.linalg.norm(self.current_quaternion)

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions.
        
        Args:
            q1: First quaternion [w, x, y, z]
            q2: Second quaternion [w, x, y, z]
            
        Returns:
            Result quaternion
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def get_current_rotation_matrix(self) -> np.ndarray:
        """Get current rotation matrix from quaternion.
        
        Returns:
            3x3 rotation matrix
        """
        return self._quaternion_to_rotation_matrix(self.current_quaternion)

    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix.
        
        Args:
            q: Quaternion [w, x, y, z]
            
        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = q
        
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ])

    def get_angular_velocity(self) -> np.ndarray:
        """Get current angular velocity from latest reading.
        
        Returns:
            Angular velocity [rad/s] (roll, pitch, yaw)
        """
        if not self.reading_buffer:
            return np.zeros(3)
        
        return self.reading_buffer[-1].gyro

    def get_linear_acceleration(self) -> np.ndarray:
        """Get current linear acceleration from latest reading.
        
        Returns:
            Linear acceleration [m/s^2] (x, y, z)
        """
        if not self.reading_buffer:
            return np.zeros(3)
        
        return self.reading_buffer[-1].accel

    def calibrate(self, duration_s: float = 5.0) -> None:
        """Calibrate IMU by measuring bias while stationary.
        
        Args:
            duration_s: Calibration duration in seconds
        """
        print(f"Calibrating IMU for {duration_s} seconds... Keep device stationary.")
        
        start_time = time.time()
        gyro_samples = []
        accel_samples = []
        
        while time.time() - start_time < duration_s:
            if self.reading_buffer:
                latest = self.reading_buffer[-1]
                gyro_samples.append(latest.gyro)
                accel_samples.append(latest.accel)
            time.sleep(0.01)
        
        if gyro_samples and accel_samples:
            self.gyro_bias = np.mean(gyro_samples, axis=0)
            self.accel_bias = np.mean(accel_samples, axis=0)
            self.is_calibrated = True
            
            print(f"Calibration complete. Gyro bias: {self.gyro_bias}, Accel bias: {self.accel_bias}")

    def get_smoothed_reading(self, window_size: int = 5) -> Optional[IMUReading]:
        """Get smoothed IMU reading over recent window.
        
        Args:
            window_size: Number of samples to average
            
        Returns:
            Smoothed IMU reading or None if insufficient data
        """
        if len(self.reading_buffer) < window_size:
            return None
        
        recent_readings = list(self.reading_buffer)[-window_size:]
        
        # Average gyro and accel
        avg_gyro = np.mean([r.gyro for r in recent_readings], axis=0)
        avg_accel = np.mean([r.accel for r in recent_readings], axis=0)
        avg_timestamp = np.mean([r.timestamp_s for r in recent_readings])
        
        # Use most recent quaternion (averaging quaternions is complex)
        avg_quaternion = recent_readings[-1].quaternion
        
        return IMUReading(
            timestamp_s=avg_timestamp,
            gyro=avg_gyro,
            accel=avg_accel,
            quaternion=avg_quaternion
        )

    def reset(self) -> None:
        """Reset IMU processor state."""
        self.reading_buffer.clear()
        self.current_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        self.last_timestamp = None
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.is_calibrated = False