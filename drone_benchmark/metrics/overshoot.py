"""Overshoot metric for step-response tests."""

from typing import List, Optional
import numpy as np
from mavlink_interface.telemetry_logger import TelemetrySnapshot


def compute_overshoot(
    snapshots: List[TelemetrySnapshot],
    axis: str = "x",
) -> Optional[float]:
    """
    Compute percentage overshoot along a single axis for a step input.

    Overshoot = (peak_error - final_value) / step_size * 100

    Args:
        snapshots: telemetry log
        axis:      "x", "y", or "z"

    Returns:
        Overshoot percentage, or None if insufficient data.
    """
    if len(snapshots) < 10:
        return None

    pos_attr = f"pos_{axis}"
    sp_attr  = f"sp_{axis}"

    positions = np.array([getattr(s, pos_attr) for s in snapshots])
    setpoints = np.array([getattr(s, sp_attr)  for s in snapshots])

    step_size = setpoints[-1] - setpoints[0]
    if abs(step_size) < 1e-6:
        return 0.0

    final_value = positions[-1]
    # Find peak excursion beyond the target
    if step_size > 0:
        peak = np.max(positions)
        overshoot = (peak - final_value) / abs(step_size) * 100.0
    else:
        peak = np.min(positions)
        overshoot = (final_value - peak) / abs(step_size) * 100.0

    return float(max(overshoot, 0.0))
