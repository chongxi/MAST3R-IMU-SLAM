"""IMU tracking utilities for MASt3R-SLAM experiments.

This module introduces an ``IMUTracker`` helper that integrates planar velocity
measurements (``vx``, ``vy``) together with yaw observations into a running pose
estimate.  It is intentionally lightweight so it can be used by stand-alone
prototypes without touching the existing visual pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class ImuSample:
    """Container for a single IMU sample.

    Attributes
    ----------
    timestamp : float
        Arrival time in seconds (monotonic clock recommended).
    vx : float
        Linear velocity along the IMU's x-axis (m/s), expressed in the IMU/body
        frame.
    vy : float
        Linear velocity along the IMU's y-axis (m/s), expressed in the IMU/body
        frame.
    yaw : float
        Heading angle measurement.  Interpreted as degrees by default but can be
        treated as radians if ``ImuTracker`` is instantiated with
        ``yaw_units="radians"``.
    dt : float | None
        Time delta relative to the previous sample.  This field is optional;
        ``ImuTracker`` will compute its own delta when absent.
    """

    timestamp: float
    vx: float
    vy: float
    yaw: float
    dt: float | None = None


class IMUTracker:
    """Integrate planar velocity+yaw observations into pose estimates."""

    def __init__(
        self,
        yaw_units: str = "degrees",
        drift_correction: float | None = None,
    ) -> None:
        if yaw_units not in {"degrees", "radians"}:
            raise ValueError("yaw_units must be 'degrees' or 'radians'")
        self.yaw_units = yaw_units
        self.drift_correction = drift_correction
        self.reset()

    def reset(
        self,
        timestamp: float | None = None,
        position: Sequence[float] | None = None,
        yaw: float | None = None,
    ) -> None:
        self.position = np.zeros(3, dtype=np.float64)
        if position is not None:
            self.position[: len(position)] = position
        self.yaw = 0.0
        if yaw is not None:
            self.yaw = self._to_radians(yaw)
        self.last_timestamp = timestamp
        self.history: List[Tuple[float, np.ndarray, float]] = []
        if timestamp is not None:
            self.history.append((timestamp, self.position.copy(), self.yaw))

    def _to_radians(self, yaw: float) -> float:
        return math.radians(yaw) if self.yaw_units == "degrees" else yaw

    def update(self, sample: ImuSample) -> Tuple[np.ndarray, float]:
        dt = sample.dt
        if dt is None and self.last_timestamp is not None:
            dt = sample.timestamp - self.last_timestamp
        if dt is None or dt <= 0.0:
            dt = 0.0

        yaw_rad = self._to_radians(sample.yaw)
        if self.drift_correction is not None and self.history:
            yaw_rad = self._apply_yaw_correction(yaw_rad)

        # Rotate body-frame velocities into world frame using current yaw.
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)
        vx_body, vy_body = sample.vx, sample.vy
        vx_world = vx_body * cos_yaw - vy_body * sin_yaw
        vy_world = vx_body * sin_yaw + vy_body * cos_yaw

        self.position[0] += vx_world * dt
        self.position[1] += vy_world * dt
        self.yaw = yaw_rad
        self.last_timestamp = sample.timestamp
        self.history.append((sample.timestamp, self.position.copy(), self.yaw))
        return self.position.copy(), self.yaw

    def _apply_yaw_correction(self, yaw_rad: float) -> float:
        """Apply optional drift correction relative to the previous yaw."""
        if not self.history:
            return yaw_rad
        prev_yaw = self.history[-1][2]
        delta = yaw_rad - prev_yaw
        # Wrap delta to [-pi, pi]
        delta = (delta + math.pi) % (2 * math.pi) - math.pi
        corrected = prev_yaw + delta * (1.0 - self.drift_correction)
        return corrected

    def get_pose(self) -> Tuple[np.ndarray, float]:
        return self.position.copy(), self.yaw

    def path(self) -> np.ndarray:
        if not self.history:
            return np.zeros((0, 3))
        return np.array([record[1] for record in self.history])

    def path_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        points = self.path()
        if points.size == 0:
            return np.array([]), np.array([])
        return points[:, 0], points[:, 1]

    def as_homogeneous(self) -> np.ndarray:
        """Return the current pose as a 4x4 homogeneous matrix."""
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 0] = cos_yaw
        matrix[0, 1] = -sin_yaw
        matrix[1, 0] = sin_yaw
        matrix[1, 1] = cos_yaw
        matrix[:3, 3] = self.position
        return matrix
