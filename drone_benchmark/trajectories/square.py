"""Square waypoint trajectory."""

import math
from typing import List, Tuple


class SquareTrajectory:
    """
    Visits the four corners of a square at constant altitude.
    Dwells at each corner for `dwell` seconds before moving to the next.
    """

    def __init__(
        self,
        center: Tuple[float, float, float],
        side: float = 2.0,
        dwell: float = 5.0,
    ):
        cx, cy, cz = center
        h = side / 2.0
        self.waypoints: List[Tuple[float, float, float]] = [
            (cx + h, cy + h, cz),
            (cx - h, cy + h, cz),
            (cx - h, cy - h, cz),
            (cx + h, cy - h, cz),
        ]
        self.dwell = dwell
        self.duration = dwell * len(self.waypoints)

    def get_setpoint(self, t: float) -> Tuple[float, float, float]:
        idx = int(t / self.dwell) % len(self.waypoints)
        return self.waypoints[idx]

    def is_done(self, t: float) -> bool:
        return t >= self.duration
