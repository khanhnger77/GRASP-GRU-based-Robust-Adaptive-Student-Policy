"""Hover trajectory — constant position setpoint."""

from typing import Tuple


class HoverTrajectory:
    """Returns the same position for the entire duration."""

    def __init__(self, position: Tuple[float, float, float], duration: float):
        self.position = position
        self.duration = duration

    def get_setpoint(self, t: float) -> Tuple[float, float, float]:
        return self.position

    def is_done(self, t: float) -> bool:
        return t >= self.duration
