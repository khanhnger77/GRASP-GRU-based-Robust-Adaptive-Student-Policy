"""
Hover Test
==========
Hold position at (0, 0, -1.5 m NED) for 30 seconds.

PID:  Python PID → velocity setpoints → PX4 inner loop
RL:   rl_tools_commander activated → policy holds position autonomously @ 500 Hz
"""

import logging
from typing import Any, Dict

from tests.base_test import BaseTest
from metrics import (
    compute_rmse,
    compute_per_axis_rmse,
    compute_max_error,
    compute_control_effort,
)
from utils.plotting import plot_trajectory, plot_error

logger = logging.getLogger(__name__)


class HoverTest(BaseTest):

    def run_test(self) -> None:
        cfg     = self.config["tests"]["hover"]
        target  = tuple(cfg["target"])   # (x, y, z) NED
        duration = cfg["duration"]
        rate_hz = self.config["mavlink"]["setpoint_rate_hz"]

        logger.info(
            "Hover test | controller=%s | target=%s | duration=%.0fs",
            type(self.controller).__name__, target, duration,
        )

        if not self._using_rl:
            self.controller.reset()

        while self.elapsed < duration:
            self.send_setpoint(*target)
            self._sleep_rate(rate_hz)

        self._compute_and_store_metrics(target)

    def _compute_and_store_metrics(self, target) -> None:
        data = self.tlog.get_data()
        rmse_xyz = compute_per_axis_rmse(data)

        self.results = {
            "test":           "hover",
            "controller":     type(self.controller).__name__,
            "rmse_3d":        compute_rmse(data),
            "rmse_x":         rmse_xyz[0],
            "rmse_y":         rmse_xyz[1],
            "rmse_z":         rmse_xyz[2],
            "max_error":      compute_max_error(data),
            "control_effort": compute_control_effort(data),
            "n_samples":      len(data),
        }

        logger.info("=== Hover Results ===")
        for k, v in self.results.items():
            logger.info("  %-22s %s", k + ":", f"{v:.4f}" if isinstance(v, float) else v)

        plot_cfg = self.config["plotting"]
        label = f"hover_{type(self.controller).__name__.lower()}"
        plot_trajectory(data,
            title=f"Hover — {type(self.controller).__name__}",
            save_path=f"{plot_cfg['output_dir']}/{label}_trajectory.png",
            show=plot_cfg["show"], dpi=plot_cfg["dpi"])
        plot_error(data,
            title=f"Hover Error — {type(self.controller).__name__}",
            save_path=f"{plot_cfg['output_dir']}/{label}_error.png",
            show=plot_cfg["show"], dpi=plot_cfg["dpi"])
