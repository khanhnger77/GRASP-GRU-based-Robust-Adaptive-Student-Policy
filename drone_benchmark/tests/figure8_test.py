"""
Figure-8 Test
=============
Follow a lemniscate trajectory at configurable speed.

PID:  Python PID computes velocity setpoints from position error.
RL:   rl_tools_commander receives updated target positions; policy tracks them.
"""

import logging

from tests.base_test import BaseTest
from trajectories.figure8 import Figure8Trajectory
from metrics import compute_rmse, compute_max_error, compute_control_effort
from utils.plotting import plot_trajectory, plot_error

logger = logging.getLogger(__name__)


class Figure8Test(BaseTest):

    def run_test(self) -> None:
        cfg      = self.config["tests"]["figure8"]
        center   = tuple(cfg["center"])
        speed_key = cfg["speed"]
        speed_val = cfg["speeds"][speed_key]
        duration = cfg["duration"]
        rate_hz  = self.config["mavlink"]["setpoint_rate_hz"]

        traj = Figure8Trajectory.from_speed(
            center=center,
            scale_x=cfg["scale_x"],
            scale_y=cfg["scale_y"],
            speed=speed_val,
            duration=duration,
        )

        logger.info(
            "Figure-8 | controller=%s | speed=%s (%.1f m/s) | duration=%.0fs",
            type(self.controller).__name__, speed_key, speed_val, duration,
        )

        if not self._using_rl:
            self.controller.reset()

        while self.elapsed < duration:
            target = traj.get_setpoint(self.elapsed)
            self.send_setpoint(*target)
            self._sleep_rate(rate_hz)

        self._compute_and_store_metrics()

    def _compute_and_store_metrics(self) -> None:
        data = self.tlog.get_data()
        self.results = {
            "test":           "figure8",
            "controller":     type(self.controller).__name__,
            "speed":          self.config["tests"]["figure8"]["speed"],
            "rmse_3d":        compute_rmse(data),
            "max_error":      compute_max_error(data),
            "control_effort": compute_control_effort(data),
            "n_samples":      len(data),
        }
        logger.info("=== Figure-8 Results ===")
        for k, v in self.results.items():
            logger.info("  %-22s %s", k + ":", f"{v:.4f}" if isinstance(v, float) else v)

        plot_cfg = self.config["plotting"]
        label = f"figure8_{type(self.controller).__name__.lower()}"
        plot_trajectory(data,
            title=f"Figure-8 — {type(self.controller).__name__}",
            save_path=f"{plot_cfg['output_dir']}/{label}_trajectory.png",
            show=plot_cfg["show"], dpi=plot_cfg["dpi"])
        plot_error(data,
            title=f"Figure-8 Error — {type(self.controller).__name__}",
            save_path=f"{plot_cfg['output_dir']}/{label}_error.png",
            show=plot_cfg["show"], dpi=plot_cfg["dpi"])
