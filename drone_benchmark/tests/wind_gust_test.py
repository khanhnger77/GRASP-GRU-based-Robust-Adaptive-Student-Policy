"""
Wind Gust Test
==============
Drone hovers at fixed point. Sudden wind gust applied via Gazebo wind plugin.
Measures recovery time and attitude spike.

Phases
------
  Phase 1 — pre_gust:  stable hover, no wind, baseline logging
  Phase 2 — gust_on:   sudden wind applied; drone fights to hold position
  Phase 3 — gust_off:  wind removed; drone recovers to stable hover

Metrics
-------
  pre_gust_rmse:       baseline 3D RMSE before gust
  max_position_error:  peak 3D error during gust phase
  max_roll_deg:        peak |roll| during gust (deg)
  max_pitch_deg:       peak |pitch| during gust (deg)
  max_yaw_deg:         peak |yaw change| during gust (deg)
  recovery_time_s:     seconds after gust-off until error < settle_threshold for 2s
  post_gust_rmse:      3D RMSE during gust-on phase

Model: gz_x500_payload_drop (or any x500 variant)
Wind topic: /world/<world_name>/wind  (configure world_name in config.yaml)
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
        logger.error("gz wind cmd failed (rc=%d): %s", r.returncode, r.stderr.strip())
        return False
    except FileNotFoundError:
        logger.error("gz CLI not found.")
        return False
    except subprocess.TimeoutExpired:
        logger.error("gz wind command timed out.")
        return False


def _attitude_spike(
    snapshots: List[TelemetrySnapshot],
    t_start: float,
    window: float,
) -> Dict[str, float]:
    pre  = [s for s in snapshots if s.timestamp < t_start]
    post = [s for s in snapshots if t_start <= s.timestamp <= t_start + window]
    if not pre or not post:
        return {"max_roll_deg": float("nan"),
                "max_pitch_deg": float("nan"),
                "max_yaw_deg": float("nan")}
    base_roll  = float(np.mean([s.roll  for s in pre[-10:]]))
    base_pitch = float(np.mean([s.pitch for s in pre[-10:]]))
    base_yaw   = float(np.mean([s.yaw   for s in pre[-10:]]))
    return {
        "max_roll_deg":  float(np.degrees(np.max(np.abs([s.roll  - base_roll  for s in post])))),
        "max_pitch_deg": float(np.degrees(np.max(np.abs([s.pitch - base_pitch for s in post])))),
        "max_yaw_deg":   float(np.degrees(np.max(np.abs([s.yaw   - base_yaw   for s in post])))),
    }


class WindGustTest(BaseTest):
    """
    Hover → sudden gust → recover → gust off → recover.

    Timeline:
        t=0                       pre-gust baseline hover
        t=pre_gust_hold           gust applied
        t=pre_gust_hold+gust_dur  gust removed
        t=end                     post-gust recovery logged
    """

    def run_test(self) -> None:
        cfg        = self.config["tests"]["wind_gust"]
        target     = tuple(cfg["target"])          # NED (x, y, z_rel)
        world      = cfg.get("gz_world", "default")
        wind_x     = cfg.get("wind_x", 5.0)
        wind_y     = cfg.get("wind_y", 0.0)
        pre_hold   = cfg["pre_gust_hold"]
        gust_dur   = cfg["gust_duration"]
        post_hold  = cfg["post_gust_hold"]
        rate_hz    = self.config["mavlink"]["setpoint_rate_hz"]
        total      = pre_hold + gust_dur + post_hold

        logger.info(
            "WindGust test | target=%s | wind=(%.1f, %.1f) m/s | "
            "pre=%.0fs gust=%.0fs post=%.0fs",
            target, wind_x, wind_y, pre_hold, gust_dur, post_hold,
        )

        if not self._using_rl:
            self.controller.reset()

        gust_applied = False
        gust_removed = False
        t_gust_on: Optional[float] = None
        t_gust_off: Optional[float] = None

        while self.elapsed < total:
            t = self.elapsed

            if t >= pre_hold and not gust_applied:
                gust_applied = True
                t_gust_on = time.time()
                logger.info("=== GUST ON at t=%.2fs: wind=(%.1f, %.1f, 0) m/s ===",
                            t, wind_x, wind_y)
                _set_wind(world, wind_x, wind_y, 0.0)

            if t >= pre_hold + gust_dur and not gust_removed:
                gust_removed = True
                t_gust_off = time.time()
                logger.info("=== GUST OFF at t=%.2fs ===", t)
                _set_wind(world, 0.0, 0.0, 0.0)

            self.send_setpoint(*target)
            self._sleep_rate(rate_hz)

        # Safety: always remove wind at end
        if not gust_removed:
            _set_wind(world, 0.0, 0.0, 0.0)

        self._compute_metrics(target, pre_hold, gust_dur, post_hold, t_gust_on, t_gust_off)

    def _compute_metrics(
        self,
        target: tuple,
        pre_hold: float,
        gust_dur: float,
        post_hold: float,
        t_gust_on: Optional[float],
        t_gust_off: Optional[float],
    ) -> None:
        cfg     = self.config["tests"]["wind_gust"]
        thresh  = cfg.get("settle_threshold", 0.10)
        all_data = self.tlog.get_data()
        if not all_data:
            self.results = {"error": "no telemetry data"}
            return

        t0 = all_data[0].timestamp
        pre_data  = [s for s in all_data if (s.timestamp - t0) < pre_hold]
        gust_data = [s for s in all_data
                     if pre_hold <= (s.timestamp - t0) < pre_hold + gust_dur]
        post_data = [s for s in all_data if (s.timestamp - t0) >= pre_hold + gust_dur]

        spike = {}
        if t_gust_on is not None:
            spike = _attitude_spike(all_data, t_gust_on, window=gust_dur)
        else:
            spike = {"max_roll_deg": float("nan"),
                     "max_pitch_deg": float("nan"),
                     "max_yaw_deg": float("nan")}

        recovery = compute_settling_time(post_data, threshold=thresh, window=2.0)

        self.results = {
            "test":              "wind_gust",
            "controller":        type(self.controller).__name__,
            "pre_gust_rmse":     compute_rmse(pre_data),
            "gust_on_rmse":      compute_rmse(gust_data),
            "max_position_error": compute_max_error(gust_data) if gust_data else float("nan"),
            "max_roll_deg":      spike["max_roll_deg"],
            "max_pitch_deg":     spike["max_pitch_deg"],
            "max_yaw_deg":       spike["max_yaw_deg"],
            "recovery_time_s":   recovery if recovery is not None else float("nan"),
            "post_gust_rmse":    compute_rmse(post_data),
            "n_pre":             len(pre_data),
            "n_gust":            len(gust_data),
            "n_post":            len(post_data),
        }

        logger.info("=== WindGust Results ===")
        for k, v in self.results.items():
            logger.info("  %-32s %s", k + ":",
                        f"{v:.4f}" if isinstance(v, float) else v)

        plot_cfg  = self.config["plotting"]
        ctrl_name = type(self.controller).__name__.lower()
        label     = f"wind_gust_{ctrl_name}"
        out       = f"{plot_cfg['output_dir']}/wind_gust"

        plot_trajectory(all_data,
                        title=f"Wind Gust — {type(self.controller).__name__}",
                        save_path=f"{out}/{label}_trajectory.png",
                        show=plot_cfg["show"], dpi=plot_cfg["dpi"])
        plot_error(all_data,
                   title=f"Wind Gust Error — {type(self.controller).__name__}",
                   save_path=f"{out}/{label}_error.png",
                   show=plot_cfg["show"], dpi=plot_cfg["dpi"])
