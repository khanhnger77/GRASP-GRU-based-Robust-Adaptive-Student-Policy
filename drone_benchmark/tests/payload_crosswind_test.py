"""
Payload Crosswind Test
======================
Drone flies A→B with payload attached under constant crosswind.
Highlights RL vs PID difference: PID drifts laterally, RL compensates.

Phases
------
  Phase 1 — apply_wind:   constant crosswind enabled before flight
  Phase 2 — leg_ab:       fly from origin A to waypoint B (10 m in X)
  Phase 3 — hold_b:       hover at B briefly
  Phase 4 — leg_ba:       return B→A
  Phase 5 — wind_off:     crosswind removed

Metrics
-------
  cross_track_rmse_ab:  lateral (Y) RMS deviation during A→B leg
  cross_track_rmse_ba:  lateral (Y) RMS deviation during B→A leg
  altitude_rmse:        Z RMS deviation (full flight)
  path_rmse:            3D RMSE vs setpoint (full flight)
  max_roll_deg:         peak payload pendulum roll (deg)
  max_pitch_deg:        peak payload pendulum pitch (deg)
  lateral_drift_ab:     mean absolute Y error during A→B (m)

Model: gz_x500_payload_drop
"""

import logging
import subprocess
import time
from typing import Any, Dict, List

import numpy as np

from tests.base_test import BaseTest
from mavlink_interface.telemetry_logger import TelemetrySnapshot
from metrics import compute_rmse, compute_max_error
from utils.plotting import plot_trajectory, plot_error

logger = logging.getLogger(__name__)


def _set_wind(world: str, x: float, y: float, z: float) -> bool:
    cmd = [
        "gz", "topic",
        "-t", f"/world/{world}/wind",
        "-m", "gz.msgs.Wind",
        "-p", f"linear_velocity: {{x: {x} y: {y} z: {z}}}",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3.0)
        if r.returncode == 0:
            return True
        logger.error("gz wind failed (rc=%d): %s", r.returncode, r.stderr.strip())
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.error("gz wind error: %s", e)
        return False


def _cross_track_rmse(snaps: List[TelemetrySnapshot], axis: str = "y") -> float:
    """RMS deviation on given axis (actual vs setpoint)."""
    if not snaps:
        return float("nan")
    if axis == "y":
        errs = [s.pos_y - s.sp_y for s in snaps]
    elif axis == "x":
        errs = [s.pos_x - s.sp_x for s in snaps]
    else:
        errs = [s.pos_z - s.sp_z for s in snaps]
    return float(np.sqrt(np.mean(np.array(errs) ** 2)))


def _euler_peak(snaps: List[TelemetrySnapshot]) -> Dict[str, float]:
    if not snaps:
        return {"max_roll_deg": float("nan"), "max_pitch_deg": float("nan")}
    return {
        "max_roll_deg":  float(np.degrees(np.max(np.abs([s.roll  for s in snaps])))),
        "max_pitch_deg": float(np.degrees(np.max(np.abs([s.pitch for s in snaps])))),
    }


class PayloadCrosswindTest(BaseTest):
    """
    Fly A→B→A with payload in constant crosswind.

    Timeline (configurable in config.yaml tests.payload_crosswind):
        t=0               wind applied
        t=wind_settle     start A→B leg
        t=+leg_duration   hold at B
        t=+hold_b         start B→A
        t=+leg_duration   wind removed, test ends
    """

    def run_test(self) -> None:
        cfg       = self.config["tests"]["payload_crosswind"]
        world     = cfg.get("gz_world", "default")
        wind_x    = cfg.get("wind_x", 0.0)
        wind_y    = cfg.get("wind_y", 4.0)
        B         = cfg["waypoint_b"]          # [x, y, z_rel]
        leg_dur   = cfg["leg_duration"]        # seconds per leg
        hold_b    = cfg["hold_b"]              # seconds hovering at B
        wind_settle = cfg.get("wind_settle", 2.0)
        rate_hz   = self.config["mavlink"]["setpoint_rate_hz"]

        A = [0.0, 0.0, B[2]]  # origin at same altitude

        logger.info(
            "PayloadCrosswind | wind=(%.1f, %.1f) m/s | A→B=%s | leg=%.0fs hold=%.0fs",
            wind_x, wind_y, B, leg_dur, hold_b,
        )

        if not self._using_rl:
            self.controller.reset()

        # Apply wind first, wait for it to settle
        logger.info("Applying crosswind (%.1f, %.1f, 0) m/s...", wind_x, wind_y)
        _set_wind(world, wind_x, wind_y, 0.0)
        t_wind_end = time.time() + wind_settle
        while time.time() < t_wind_end:
            self.send_setpoint(*A)
            self._sleep_rate(rate_hz)

        # Leg A→B
        logger.info("=== A→B leg start ===")
        t_ab_start = self.elapsed
        t_ab_end   = t_ab_start + leg_dur
        while self.elapsed < t_ab_end:
            alpha = min(1.0, (self.elapsed - t_ab_start) / leg_dur)
            sp = [A[i] + alpha * (B[i] - A[i]) for i in range(3)]
            self.send_setpoint(*sp)
            self._sleep_rate(rate_hz)
        # Hold at B
        t_hold_end = self.elapsed + hold_b
        while self.elapsed < t_hold_end:
            self.send_setpoint(*B)
            self._sleep_rate(rate_hz)

        # Leg B→A
        logger.info("=== B→A leg start ===")
        t_ba_start = self.elapsed
        t_ba_end   = t_ba_start + leg_dur
        while self.elapsed < t_ba_end:
            alpha = min(1.0, (self.elapsed - t_ba_start) / leg_dur)
            sp = [B[i] + alpha * (A[i] - B[i]) for i in range(3)]
            self.send_setpoint(*sp)
            self._sleep_rate(rate_hz)

        # Remove wind
        logger.info("=== Wind OFF ===")
        _set_wind(world, 0.0, 0.0, 0.0)

        self._compute_metrics(t_ab_start, t_ab_end, t_ba_start, t_ba_end, wind_settle)

    def _compute_metrics(
        self,
        t_ab_start: float,
        t_ab_end: float,
        t_ba_start: float,
        t_ba_end: float,
        wind_settle: float,
    ) -> None:
        all_data = self.tlog.get_data()
        if not all_data:
            self.results = {"error": "no telemetry data"}
            return

        t0 = all_data[0].timestamp
        ab_data = [s for s in all_data
                   if t_ab_start <= (s.timestamp - t0) < t_ab_end]
        ba_data = [s for s in all_data
                   if t_ba_start <= (s.timestamp - t0) < t_ba_end]
        flight_data = [s for s in all_data if (s.timestamp - t0) >= wind_settle]

        euler = _euler_peak(flight_data)

        ab_mean_drift = (float(np.mean(np.abs([s.pos_y - s.sp_y for s in ab_data])))
                         if ab_data else float("nan"))

        self.results = {
            "test":                   "payload_crosswind",
            "controller":             type(self.controller).__name__,
            "cross_track_rmse_ab":    _cross_track_rmse(ab_data, "y"),
            "cross_track_rmse_ba":    _cross_track_rmse(ba_data, "y"),
            "lateral_drift_ab_mean":  ab_mean_drift,
            "altitude_rmse":          _cross_track_rmse(flight_data, "z"),
            "path_rmse":              compute_rmse(flight_data),
            "max_position_error":     compute_max_error(flight_data),
            "max_roll_deg":           euler["max_roll_deg"],
            "max_pitch_deg":          euler["max_pitch_deg"],
            "n_ab":                   len(ab_data),
            "n_ba":                   len(ba_data),
        }

        logger.info("=== PayloadCrosswind Results ===")
        for k, v in self.results.items():
            logger.info("  %-34s %s", k + ":",
                        f"{v:.4f}" if isinstance(v, float) else v)

        plot_cfg  = self.config["plotting"]
        ctrl_name = type(self.controller).__name__.lower()
        label     = f"payload_crosswind_{ctrl_name}"
        out       = f"{plot_cfg['output_dir']}/payload_crosswind"

        plot_trajectory(all_data,
                        title=f"Payload Crosswind — {type(self.controller).__name__}",
                        save_path=f"{out}/{label}_trajectory.png",
                        show=plot_cfg["show"], dpi=plot_cfg["dpi"])
        plot_error(all_data,
                   title=f"Payload Crosswind Error — {type(self.controller).__name__}",
                   save_path=f"{out}/{label}_error.png",
                   show=plot_cfg["show"], dpi=plot_cfg["dpi"])
