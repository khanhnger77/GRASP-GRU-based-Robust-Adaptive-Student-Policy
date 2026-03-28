"""Common math utilities."""

import math
from typing import Tuple
import numpy as np


def ned_to_enu(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert NED (North-East-Down) coordinates to ENU (East-North-Up)."""
    return y, x, -z


def enu_to_ned(e: float, n: float, u: float) -> Tuple[float, float, float]:
    return n, e, -u


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler → quaternion [w, x, y, z]."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ])


def wrap_pi(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def distance_3d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
