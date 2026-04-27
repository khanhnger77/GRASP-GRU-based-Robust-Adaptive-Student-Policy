"""Circular trajectory in the horizontal plane."""

import math
from typing import Tuple


class CircleTrajectory:
    def __init__(
        self,
        center: Tuple[float, float, float],
        radius: float = 2.0,
        period: float = 10.0,
        duration: float = 30.0,
    ):
        self.cx, self.cy, self.cz = center
        self.radius = radius
        self.omega = 2.0 * math.pi / period
        self.duration = duration

    def get_setpoint(self, t: float) -> Tuple[float, float, float]:
        x = self.cx + self.radius * math.cos(self.omega * t)
        y = self.cy + self.radius * math.sin(self.omega * t)
        return x, y, self.cz

    def is_done(self, t: float) -> bool:
        return t >= self.duration
