"""
Settling time metric.

Settling time: the first time at which the 3D position error drops below
`threshold` metres AND stays there for `window` seconds.
"""

from typing import List, Optional
import numpy as np
from mavlink_interface.telemetry_logger import TelemetrySnapshot


def compute_settling_time(
    snapshots: List[TelemetrySnapshot],
    threshold: float = 0.1,
    window: float = 2.0,
) -> Optional[float]:
    """
    Returns the elapsed time (seconds from first snapshot) at which the
    drone first settles within `threshold` metres of the setpoint and
    stays there for `window` seconds.

    Returns None if it never settles.
    """
    if not snapshots:
        return None

    t0 = snapshots[0].timestamp
    times = np.array([s.timestamp - t0 for s in snapshots])
    errors = np.array([
        np.sqrt((s.pos_x - s.sp_x) ** 2 + (s.pos_y - s.sp_y) ** 2 + (s.pos_z - s.sp_z) ** 2)
        for s in snapshots
    ])

    for i, t in enumerate(times):
        # Find the window end
        end_t = t + window
        window_mask = (times >= t) & (times <= end_t)
        if window_mask.sum() < 2:
            continue
        if np.all(errors[window_mask] < threshold):
            return float(t)

    return None
