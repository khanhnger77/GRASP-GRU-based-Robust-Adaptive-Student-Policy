"""
Control effort / energy metric.

Approximates total energy expenditure as the sum of squared actuator
outputs (motor commands) over the test duration.  Lower is more efficient.
"""

from typing import List
import numpy as np
from mavlink_interface.telemetry_logger import TelemetrySnapshot


def compute_control_effort(snapshots: List[TelemetrySnapshot]) -> float:
    """
    Sum of squared motor outputs across all snapshots and all 4 motors.
    Normalised by number of snapshots so tests of different lengths compare fairly.
    """
    if not snapshots:
        return float("nan")

    efforts = []
    for s in snapshots:
        total = s.motor0**2 + s.motor1**2 + s.motor2**2 + s.motor3**2
        efforts.append(total)

    return float(np.mean(efforts))


def compute_total_thrust(snapshots: List[TelemetrySnapshot], dt: float = 0.02) -> float:
    """
    Approximate total thrust integral: sum(motor_avg) * dt.

    Motor values are assumed to be normalised in [0, 1].
    """
    if not snapshots:
        return float("nan")

    thrust = sum(
        (s.motor0 + s.motor1 + s.motor2 + s.motor3) / 4.0
        for s in snapshots
    ) * dt
    return float(thrust)
