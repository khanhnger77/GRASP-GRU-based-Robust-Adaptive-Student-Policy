"""
Figure-8 (Lemniscate of Bernoulli) trajectory in the horizontal plane.

Parametric equations (NED):
    x(t) = cx + A * sin(ω*t)
    y(t) = cy + B * sin(ω*t) * cos(ω*t)   = cy + (B/2) * sin(2ω*t)
    z(t) = cz  (constant altitude)
"""

import math
from typing import Tuple


class Figure8Trajectory:
    """
    Smooth figure-8 path around a centre point.

    Args:
        center:    (x, y, z) NED centre of the figure-8
        scale_x:   half-width in the x direction (metres)
        scale_y:   half-width in the y direction (metres)
        period:    time for one full loop (seconds)
        duration:  total trajectory duration (seconds)
    """

    def __init__(
        self,
        center: Tuple[float, float, float],
        scale_x: float = 2.0,
        scale_y: float = 1.0,
        period: float = 10.0,
        duration: float = 60.0,
    ):
        self.cx, self.cy, self.cz = center
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.omega = 2.0 * math.pi / period
        self.duration = duration

    def get_setpoint(self, t: float) -> Tuple[float, float, float]:
        x = self.cx + self.scale_x * math.sin(self.omega * t)
        y = self.cy + self.scale_y * math.sin(self.omega * t) * math.cos(self.omega * t)
        return x, y, self.cz

    def is_done(self, t: float) -> bool:
        return t >= self.duration

    @classmethod
    def from_speed(
        cls,
        center: Tuple[float, float, float],
        scale_x: float,
        scale_y: float,
        speed: float,
        duration: float,
    ) -> "Figure8Trajectory":
        """
        Build trajectory from a desired average linear speed (m/s).

        Approximates period from perimeter ≈ 2π * sqrt((A²+B²)/2).
        """
        A, B = scale_x, scale_y
        perimeter = 2.0 * math.pi * math.sqrt((A**2 + B**2) / 2.0)
        period = perimeter / max(speed, 0.01)
        return cls(center, scale_x, scale_y, period, duration)
