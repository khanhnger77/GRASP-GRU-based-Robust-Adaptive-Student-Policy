"""
Trajectory Tracking Test
=========================
Generic test that accepts any trajectory object and tracks it.
Can be used with Circle, Square, or custom trajectories.

Supports both PID and RL controllers.
"""

import logging
from typing import Any, Dict, Protocol

from tests.base_test import BaseTest
from metrics import compute_rmse, compute_max_error, compute_control_effort
from utils.plotting import plot_trajectory, plot_error

logger = logging.getLogger(__name__)


class Trajectory(Protocol):
    """Minimal interface any trajectory must satisfy."""
    duration: float
    def get_setpoint(self, t: float): ...
    def is_done(self, t: float) -> bool: ...


class TrajectoryTrackingTest(BaseTest):
    """
    Generic trajectory tracking test.

    Args:
        trajectory: any object implementing get_setpoint(t) and is_done(t).
    """

    def __init__(self, trajectory: Trajectory, name: str = "trajectory", **kwargs):
        super().__init__(**kwargs)
        self.trajectory = trajectory
        self.test_name = name

    def run_test(self) -> None:
        rate_hz = self.config["mavlink"]["setpoint_rate_hz"]
        controller_type = type(self.controller).__name__

        if controller_type == "PIDController":
            self.controller.reset()

        logger.info("Trajectory tracking test: %s, duration=%.0fs",
                    self.test_name, self.trajectory.duration)

        while not self.trajectory.is_done(self.elapsed):
            t = self.elapsed
            target = self.trajectory.get_setpoint(t)
            self.tlog.update_setpoint(*target)

            snap = self.tlog.get_latest()

            if controller_type == "PIDController":
                vx, vy, vz = self.controller.compute(
                    current=(snap.pos_x, snap.pos_y, snap.pos_z),
                    target=target,
                )
                self.ctrl.send_velocity_setpoint(vx, vy, vz)
            else:
                self.ctrl.send_position_setpoint(*target)

            self._sleep_rate(rate_hz)

        self._compute_and_store_metrics()

    def _compute_and_store_metrics(self) -> None:
        data = self.tlog.get_data()
        self.results = {
            "test": self.test_name,
            "controller": type(self.controller).__name__,
            "rmse_3d": compute_rmse(data),
            "max_error": compute_max_error(data),
            "control_effort": compute_control_effort(data),
            "n_samples": len(data),
        }

        logger.info("=== %s Results ===", self.test_name.capitalize())
        for k, v in self.results.items():
            logger.info("  %s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)

        plot_cfg = self.config["plotting"]
        label = f"{self.test_name}_{type(self.controller).__name__.lower()}"
        plot_trajectory(
            data,
            title=f"{self.test_name.capitalize()} — {type(self.controller).__name__}",
            save_path=f"{plot_cfg['output_dir']}/{label}_trajectory.png",
            show=plot_cfg["show"],
            dpi=plot_cfg["dpi"],
        )
        plot_error(
            data,
            title=f"{self.test_name.capitalize()} Error",
            save_path=f"{plot_cfg['output_dir']}/{label}_error.png",
            show=plot_cfg["show"],
            dpi=plot_cfg["dpi"],
        )
