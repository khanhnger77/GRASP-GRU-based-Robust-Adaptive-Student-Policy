"""Position RMSE and per-axis MAE metrics."""

from typing import List, Tuple
import numpy as np
from mavlink_interface.telemetry_logger import TelemetrySnapshot


def compute_rmse(snapshots: List[TelemetrySnapshot]) -> float:
    """3D position RMSE over all logged snapshots."""
    if not snapshots:
        return float("nan")
    errors = [
        (s.pos_x - s.sp_x) ** 2 + (s.pos_y - s.sp_y) ** 2 + (s.pos_z - s.sp_z) ** 2
        for s in snapshots
    ]
    return float(np.sqrt(np.mean(errors)))


def compute_per_axis_rmse(snapshots: List[TelemetrySnapshot]) -> Tuple[float, float, float]:
    """RMSE per axis (x, y, z)."""
    if not snapshots:
        return (float("nan"),) * 3
    ex = np.array([s.pos_x - s.sp_x for s in snapshots])
    ey = np.array([s.pos_y - s.sp_y for s in snapshots])
    ez = np.array([s.pos_z - s.sp_z for s in snapshots])
    return float(np.sqrt(np.mean(ex**2))), float(np.sqrt(np.mean(ey**2))), float(np.sqrt(np.mean(ez**2)))


def compute_max_error(snapshots: List[TelemetrySnapshot]) -> float:
    """Maximum 3D Euclidean position error."""
    if not snapshots:
        return float("nan")
    errors = [
        np.sqrt((s.pos_x - s.sp_x) ** 2 + (s.pos_y - s.sp_y) ** 2 + (s.pos_z - s.sp_z) ** 2)
        for s in snapshots
    ]
    return float(np.max(errors))
