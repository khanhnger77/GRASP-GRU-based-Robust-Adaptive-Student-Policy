"""
Waypoint Drop Test
==================
Drone flies A→B with payload, drops payload at B, continues to C.
Evaluates navigation accuracy, drop precision, and post-drop recovery.

Phases
------
  Phase 1 — leg_ab:   fly origin A to drop point B (payload attached)
  Phase 2 — hold_b:   hover at B briefly to stabilise before drop
  Phase 3 — drop:     gz topic payload detach command
  Phase 4 — recover:  hold B position post-drop (measure recovery)
  Phase 5 — leg_bc:   fly from B to return point C

Metrics
-------
  path_ab_rmse:         3D RMSE during A→B leg
  drop_position_error:  3D distance from B setpoint at moment of drop
  euler_spike_roll/pitch/yaw_deg: attitude jerk at drop event
  recovery_time_s:      seconds until error < settle_threshold for 2s
  path_bc_rmse:         3D RMSE during B→C leg
  drop_success:         bool — gz topic returned OK

Model: gz_x500_payload_drop
"""

import logging
import subprocess
import time
from typing import Any, Dict, List, Optional

import numpy as np

from tests.base_test import BaseTest
from mavlink_interface.telemetry_logger import TelemetrySnapshot
from metrics import compute_rmse, compute_max_error, compute_settling_time
from utils.plotting import plot_trajectory, plot_error

logger = logging.getLogger(__name__)

_GZ_DETACH_CMD = [
    "gz", "topic",
    "-t", "/payload/detach",
    "-m", "gz.msgs.Empty",
    "-p", "unused: true",
]


def _drop_payload() -> bool:
    try:
        r = subprocess.run(_GZ_DETACH_CMD, capture_output=True, text=True, timeout=3.0)
        if r.returncode == 0:
            logger.info("Payload detached (gz topic OK).")
            return True
        logger.error("gz detach failed (rc=%d): %s", r.returncode, r.stderr.strip())
        return False
    except FileNotFoundError:
        logger.error("gz CLI not found.")
        return False
    except subprocess.TimeoutExpired:
        logger.error("gz detach command timed out.")
        return False


def _attitude_spike(
    snapshots: List[TelemetrySnapshot],
    t_drop: float,
    window: float = 3.0,
) -> Dict[str, float]:
    pre  = [s for s in snapshots if s.timestamp < t_drop]
    post = [s for s in snapshots if t_drop <= s.timestamp <= t_drop + window]
    if not pre or not post:
        return {"roll": float("nan"), "pitch": float("nan"), "yaw": float("nan")}
    base_roll  = float(np.mean([s.roll  for s in pre[-10:]]))
    base_pitch = float(np.mean([s.pitch for s in pre[-10:]]))
    base_yaw   = float(np.mean([s.yaw   for s in pre[-10:]]))
    return {
        "roll":  float(np.degrees(np.max(np.abs([s.roll  - base_roll  for s in post])))),
        "pitch": float(np.degrees(np.max(np.abs([s.pitch - base_pitch for s in post])))),
        "yaw":   float(np.degrees(np.max(np.abs([s.yaw   - base_yaw   for s in post])))),
    }


class WaypointDropTest(BaseTest):
    """
    A→B (with payload) → drop → recover → B→C.

    Configurable in config.yaml under tests.waypoint_drop.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drop_timestamp: Optional[float] = None
        self._drop_success: bool = False

    def run_test(self) -> None:
        cfg      = self.config["tests"]["waypoint_drop"]
        B        = cfg["drop_point"]       # [x, y, z_rel] — where payload is dropped
        C        = cfg["return_point"]     # [x, y, z_rel] — final destination
        A        = [0.0, 0.0, B[2]]       # start at origin, same altitude
        leg_dur  = cfg["leg_duration"]     # seconds per leg A→B and B→C
        hold_b   = cfg["hold_b"]           # seconds hovering at B before drop
        recover  = cfg["recover_hold"]     # seconds at B after drop before moving to C
        rate_hz  = self.config["mavlink"]["setpoint_rate_hz"]

        logger.info(
            "WaypointDrop | A=%s B=%s C=%s | leg=%.0fs hold_b=%.0fs recover=%.0fs",
            A, B, C, leg_dur, hold_b, recover,
        )

        if not self._using_rl:
            self.controller.reset()

        # ── Leg A→B ──────────────────────────────────────────────────────
        logger.info("=== A→B leg ===")
        t_ab_start = self.elapsed
        t_ab_end   = t_ab_start + leg_dur
        while self.elapsed < t_ab_end:
            alpha = min(1.0, (self.elapsed - t_ab_start) / leg_dur)
            sp = [A[i] + alpha * (B[i] - A[i]) for i in range(3)]
            self.send_setpoint(*sp)
            self._sleep_rate(rate_hz)

        # ── Hold at B (pendulum settle) ───────────────────────────────────
        logger.info("=== Hold at B (%.0fs) ===", hold_b)
        t_hold_end = self.elapsed + hold_b
        while self.elapsed < t_hold_end:
            self.send_setpoint(*B)
            self._sleep_rate(rate_hz)

        # ── Drop payload ──────────────────────────────────────────────────
        self._drop_timestamp = time.time()
        logger.info("=== PAYLOAD DROP at t=%.2fs ===", self.elapsed)
        self._drop_success = _drop_payload()

        # ── Recover at B ─────────────────────────────────────────────────
        logger.info("=== Recovery hold at B (%.0fs) ===", recover)
        t_recover_start = self.elapsed
        t_recover_end   = t_recover_start + recover
        while self.elapsed < t_recover_end:
            self.send_setpoint(*B)
            self._sleep_rate(rate_hz)

        # ── Leg B→C ──────────────────────────────────────────────────────
        logger.info("=== B→C leg ===")
        t_bc_start = self.elapsed
        t_bc_end   = t_bc_start + leg_dur
        while self.elapsed < t_bc_end:
            alpha = min(1.0, (self.elapsed - t_bc_start) / leg_dur)
            sp = [B[i] + alpha * (C[i] - B[i]) for i in range(3)]
            self.send_setpoint(*sp)
            self._sleep_rate(rate_hz)

        self._compute_metrics(
            t_ab_start, t_ab_end,
            t_recover_start, t_recover_end,
            t_bc_start, t_bc_end,
            B,
        )

    def _compute_metrics(
        self,
        t_ab_start: float, t_ab_end: float,
        t_recover_start: float, t_recover_end: float,
        t_bc_start: float, t_bc_end: float,
        B: list,
    ) -> None:
        cfg      = self.config["tests"]["waypoint_drop"]
        thresh   = cfg.get("settle_threshold", 0.10)
        all_data = self.tlog.get_data()
        if not all_data:
            self.results = {"error": "no telemetry data"}
            return

        t0 = all_data[0].timestamp

        def _window(ta, tb):
            return [s for s in all_data if ta <= (s.timestamp - t0) < tb]

        ab_data      = _window(t_ab_start, t_ab_end)
        recover_data = _window(t_recover_start, t_recover_end)
        bc_data      = _window(t_bc_start, t_bc_end)

        # Drop position error — snapshot nearest to drop timestamp
        drop_snap = None
        if self._drop_timestamp is not None:
            drop_snap = min(all_data, key=lambda s: abs(s.timestamp - self._drop_timestamp),
                            default=None)
        if drop_snap is not None:
            drop_pos_err = float(np.sqrt(
                (drop_snap.pos_x - B[0]) ** 2 +
                (drop_snap.pos_y - B[1]) ** 2 +
                (drop_snap.pos_z - (-B[2])) ** 2   # NED z sign
            ))
        else:
            drop_pos_err = float("nan")

        spike = {}
        if self._drop_timestamp is not None:
            spike = _attitude_spike(all_data, self._drop_timestamp)
        else:
            spike = {"roll": float("nan"), "pitch": float("nan"), "yaw": float("nan")}

        recovery_time = compute_settling_time(recover_data, threshold=thresh, window=2.0)

        self.results = {
            "test":                     "waypoint_drop",
            "controller":               type(self.controller).__name__,
            "drop_success":             self._drop_success,
            "path_ab_rmse":             compute_rmse(ab_data),
            "drop_position_error_m":    drop_pos_err,
            "euler_spike_roll_deg":     spike["roll"],
            "euler_spike_pitch_deg":    spike["pitch"],
            "euler_spike_yaw_deg":      spike["yaw"],
            "recovery_time_s":          recovery_time if recovery_time is not None else float("nan"),
            "post_drop_max_error":      compute_max_error(recover_data) if recover_data else float("nan"),
            "path_bc_rmse":             compute_rmse(bc_data),
            "n_ab":                     len(ab_data),
            "n_recover":                len(recover_data),
            "n_bc":                     len(bc_data),
        }

        logger.info("=== WaypointDrop Results ===")
        for k, v in self.results.items():
            logger.info("  %-36s %s", k + ":",
                        f"{v:.4f}" if isinstance(v, float) else v)

        plot_cfg  = self.config["plotting"]
        ctrl_name = type(self.controller).__name__.lower()
        label     = f"waypoint_drop_{ctrl_name}"
        out       = f"{plot_cfg['output_dir']}/waypoint_drop"

        plot_trajectory(all_data,
                        title=f"Waypoint Drop — {type(self.controller).__name__}",
                        save_path=f"{out}/{label}_trajectory.png",
                        show=plot_cfg["show"], dpi=plot_cfg["dpi"])
        plot_error(all_data,
                   title=f"Waypoint Drop Error — {type(self.controller).__name__}",
                   save_path=f"{out}/{label}_error.png",
                   show=plot_cfg["show"], dpi=plot_cfg["dpi"])
