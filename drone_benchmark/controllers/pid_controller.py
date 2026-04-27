"""
Simple 3-axis position PID controller.

Given a position error, outputs a velocity command (or optionally acceleration).
Designed to wrap around PX4's inner-loop attitude/rate controllers — we only
run the outer position loop here and let PX4 handle attitude.

Architecture:
  position error → PID → velocity setpoint → send_velocity_setpoint()
"""

import time
import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PIDAxis:
    """Single-axis PID with anti-windup and derivative filtering."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limit: float = 3.0,
        integral_limit: float = 1.0,
        derivative_filter_alpha: float = 0.8,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.alpha = derivative_filter_alpha  # low-pass on derivative

        self._integral = 0.0
        self._prev_error = 0.0
        self._filtered_derivative = 0.0
        self._last_time: float = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._filtered_derivative = 0.0
        self._last_time = 0.0

    def compute(self, error: float, now: float) -> float:
        if self._last_time == 0.0:
            self._last_time = now
            self._prev_error = error
            return self.kp * error

        dt = now - self._last_time
        if dt <= 0.0:
            return self.kp * error

        # Proportional
        p = self.kp * error

        # Integral with anti-windup (clamp)
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)
        i = self.ki * self._integral

        # Derivative with low-pass filter to reduce noise
        raw_derivative = (error - self._prev_error) / dt
        self._filtered_derivative = (
            self.alpha * self._filtered_derivative + (1.0 - self.alpha) * raw_derivative
        )
        d = self.kd * self._filtered_derivative

        output = np.clip(p + i + d, -self.output_limit, self.output_limit)

        self._prev_error = error
        self._last_time = now
        return float(output)


class PIDController:
    """
    3-axis position PID controller for quadrotor outer-loop control.

    Inputs:  current position (x,y,z NED) + target position (x,y,z NED)
    Outputs: velocity setpoint (vx,vy,vz NED)

    Example:
        ctrl = PIDController(kp=[0.8,0.8,1.0], ki=[0.05,0.05,0.1], kd=[0.2,0.2,0.3])
        vx, vy, vz = ctrl.compute(current=(0.1,0.2,-1.4), target=(0,0,-1.5))
    """

    def __init__(
        self,
        kp: List[float],
        ki: List[float],
        kd: List[float],
        max_velocity: float = 3.0,
    ):
        assert len(kp) == len(ki) == len(kd) == 3, "Need 3-element gain arrays"
        self.max_velocity = max_velocity
        self.axes = [
            PIDAxis(kp[i], ki[i], kd[i], output_limit=max_velocity)
            for i in range(3)
        ]

    def reset(self) -> None:
        for ax in self.axes:
            ax.reset()

    def compute(
        self,
        current: Tuple[float, float, float],
        target: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """
        Compute velocity command from position error.

        Args:
            current: (x, y, z) actual position in NED
            target:  (x, y, z) desired position in NED

        Returns:
            (vx, vy, vz) velocity command in NED
        """
        now = time.time()
        vx = self.axes[0].compute(target[0] - current[0], now)
        vy = self.axes[1].compute(target[1] - current[1], now)
        vz = self.axes[2].compute(target[2] - current[2], now)
        return vx, vy, vz

    def compute_from_state(self, state: dict, target: Tuple[float, float, float]):
        """
        Convenience: extract position from a telemetry state dict.

        state keys expected: pos_x, pos_y, pos_z
        """
        current = (state["pos_x"], state["pos_y"], state["pos_z"])
        return self.compute(current, target)
