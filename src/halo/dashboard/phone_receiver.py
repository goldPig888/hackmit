"""Receive camera frames and IMU data streamed from a phone browser page.

The phone page (dashboard_static/phone_stream.html) pushes:
- Binary WebSocket messages: 8-byte big-endian double timestamp (ms) + JPEG bytes
- JSON WebSocket messages: {"type": "imu", "ts": ms, "rot": [deg/s x3],
  "accel": [m/s^2 x3], "ori": [alpha, beta, gamma deg], "screen": angle deg}

This receiver converts the raw device-orientation data into the exact
(gyro, accel, quaternion) tuple that AdvancedHaloPipeline.update expects:
- gyro is rotated into the earth frame so gyro[2] is the vertical yaw rate
  regardless of how the phone is held
- the quaternion encodes R_earth<-camera = R_earth<-device @ R_device<-camera,
  where R_device<-camera depends on the phone's screen orientation because the
  streamed image is normalized to display orientation
"""

from __future__ import annotations

import json
import math
import struct
import threading
from collections import deque
from typing import Optional

import numpy as np

from ..imu.camera_pose import CameraPose


# R_device<-camera keyed by screen orientation angle.
# Normalized-image convention: +x image right, +y image down, +z optical axis.
# iOS device convention: +x right, +y toward top of screen, +z out of screen.
# If bearings look mirrored in field testing, the phone page's ROT_FIX knob is
# the thing to check first — these matrices assume the streamed image is
# upright relative to the display.
R_DEV_TO_CAM = {
    0: np.diag([1.0, -1.0, -1.0]),                                        # portrait
    180: np.diag([-1.0, 1.0, -1.0]),                                      # portrait upside-down
    90: np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),  # landscape, top of phone to the left
    270: np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),   # landscape, top of phone to the right
}


def _device_orientation_to_matrix(alpha_deg: float, beta_deg: float, gamma_deg: float) -> np.ndarray:
    """Convert W3C deviceorientation angles to a device->earth rotation matrix.

    Intrinsic Z-X'-Y'' composition: R = Rz(alpha) @ Rx(beta) @ Ry(gamma).
    """
    a, b, g = math.radians(alpha_deg), math.radians(beta_deg), math.radians(gamma_deg)
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cg, sg = math.cos(g), math.sin(g)
    rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cb, -sb], [0.0, sb, cb]])
    ry = np.array([[cg, 0.0, sg], [0.0, 1.0, 0.0], [-sg, 0.0, cg]])
    return rz @ rx @ ry


def _matrix_to_quaternion(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to [w, x, y, z] via CameraPose's conversion."""
    return CameraPose(
        rotation_matrix=rotation_matrix,
        translation=np.zeros(3),
        timestamp_s=0.0,
    ).quaternion


class PhoneStreamReceiver:
    """Thread-safe buffer for phone-streamed frames and IMU samples.

    Written by the aiohttp ingest handler (server thread), read by the CV
    processing loop (main thread). Deques are bounded so a slow consumer
    never accumulates backlog; callers should take the newest data.
    """

    def __init__(self, frame_buffer: int = 30, imu_buffer: int = 600) -> None:
        self._frames: deque[tuple[float, bytes]] = deque(maxlen=frame_buffer)
        self._imu: deque[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=imu_buffer)
        self._lock = threading.Lock()
        self._processed_frame: Optional[bytes] = None
        self.connected = False
        self.stats = {"frames": 0, "imu": 0, "last_frame_ts": None, "last_imu_ts": None}

    # ---- ingest side (called from the WebSocket handler) ----

    def push_frame_message(self, data: bytes) -> None:
        """Parse a binary frame message: 8-byte BE double ms + JPEG payload."""
        if len(data) < 9:
            return
        ts_ms = struct.unpack(">d", data[:8])[0]
        self.push_frame(ts_ms / 1000.0, data[8:])

    def push_frame(self, ts_s: float, jpeg: bytes) -> None:
        with self._lock:
            self._frames.append((ts_s, jpeg))
            self.stats["frames"] += 1
            self.stats["last_frame_ts"] = ts_s

    def push_imu_message(self, text: str) -> None:
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            return
        if msg.get("type") != "imu":
            return
        ts_s = float(msg.get("ts", 0.0)) / 1000.0

        # Device attitude: alpha/beta/gamma -> R_earth<-device
        ori = msg.get("ori") or [0.0, 0.0, 0.0]
        r_earth_dev = _device_orientation_to_matrix(ori[0], ori[1], ori[2])

        # Device->camera extrinsic for the current screen orientation
        screen_angle = int(msg.get("screen", 0)) % 360
        r_dev_cam = R_DEV_TO_CAM.get(screen_angle, R_DEV_TO_CAM[0])
        quat_cam = _matrix_to_quaternion(r_earth_dev @ r_dev_cam)

        # Gyro: device-frame deg/s -> earth-frame rad/s so [2] is yaw rate
        rot = msg.get("rot") or [0.0, 0.0, 0.0]
        gyro_dev = np.deg2rad(np.asarray(rot, dtype=float))
        gyro_earth = r_earth_dev @ gyro_dev

        accel = np.asarray(msg.get("accel") or [0.0, 0.0, 0.0], dtype=float)

        with self._lock:
            self._imu.append((ts_s, gyro_earth, accel, quat_cam))
            self.stats["imu"] += 1
            self.stats["last_imu_ts"] = ts_s

    # ---- consumer side (called from the CV loop) ----

    def take_latest_frame(self) -> Optional[tuple[float, bytes]]:
        """Return the newest frame and drop anything older."""
        with self._lock:
            if not self._frames:
                return None
            newest = self._frames.pop()
            self._frames.clear()
            return newest

    def imu_tuple_at(self, ts_s: float, tolerance_s: float = 0.25) -> Optional[tuple]:
        """Return (gyro, accel, quaternion) nearest to the given timestamp.

        Falls back to the most recent sample if nothing is within tolerance.
        """
        with self._lock:
            if not self._imu:
                return None
            best = min(self._imu, key=lambda s: abs(s[0] - ts_s))
            if abs(best[0] - ts_s) > tolerance_s:
                best = self._imu[-1]
            return best[1], best[2], best[3]

    def latest_imu(self) -> Optional[tuple]:
        with self._lock:
            if not self._imu:
                return None
            s = self._imu[-1]
            return s[1], s[2], s[3]

    def set_processed_frame(self, jpeg: bytes) -> None:
        """Store an annotated JPEG for the /video MJPEG endpoint."""
        with self._lock:
            self._processed_frame = jpeg

    def processed_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._processed_frame

    def latest_raw_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._frames[-1][1] if self._frames else None
