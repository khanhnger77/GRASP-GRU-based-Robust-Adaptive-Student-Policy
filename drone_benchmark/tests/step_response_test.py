"""
Step Response Test
==================
Stabilise at start → apply step to end → measure transient response.

PID:  velocity setpoints from Python PID
RL:   target position forwarded to rl_tools_commander via shell; policy responds
"""

import logging

from tests.base_test import BaseTest
from metrics import compute_rmse, compute_max_error, compute_settling_time, compute_overshoot
from utils.plotting import plot_trajectory, plot_error

logger = logging.getLogger(__name__)

HOLD_BEFORE = 8.0    # seconds at start position before the step
HOLD_AFTER  = 15.0   # seconds to observe recovery after the step


class StepResponseTest(BaseTest):

    def run_test(self) -> None:
        cfg    = self.config["tests"]["step_response"]
        start  = tuple(cfg["start"])
        end    = tuple(cfg["end"])
        rate_hz = self.config["mavlink"]["setpoint_rate_hz"]

        if not self._using_rl:
            self.controller.reset()

        total = HOLD_BEFORE + HOLD_AFTER
        step_applied = False
        step_time: float = 0.0

        while self.elapsed < total:
            t = self.elapsed

            if t < HOLD_BEFORE:
                target = start
            else:
                if not step_applied:
                    step_applied = True
                    step_time = t
                    logger.info(
                        "Step at t=%.2fs: %s → %s [controller=%s]",
                        t, start, end, type(self.controller).__name__,
                    )
                target = end

            self.send_setpoint(*target)
            self._sleep_rate(rate_hz)

        self._compute_and_store_metrics(step_time, cfg["settle_threshold"])

    def _compute_and_store_metrics(self, step_time: float, thresh: float) -> None:
        data = self.tlog.get_data()
        t0 = data[0].timestamp if data else 0.0
        post_step = [s for s in data if (s.timestamp - t0) >= step_time]

        settling = compute_settling_time(post_step, threshold=thresh)
        self.results = {
            "test":         "step_response",
            "controller":   type(self.controller).__name__,
            "rmse_3d":      compute_rmse(post_step),
            "max_error":    compute_max_error(post_step),
            "settling_time": settling if settling is not None else float("nan"),
            "overshoot_x":  compute_overshoot(post_step, axis="x") or 0.0,
            "overshoot_y":  compute_overshoot(post_step, axis="y") or 0.0,
            "n_samples":    len(data),
        }
        logger.info("=== Step Response Results ===")
        for k, v in self.results.items():
            logger.info("  %-22s %s", k + ":", f"{v:.4f}" if isinstance(v, float) else v)

        plot_cfg = self.config["plotting"]
        label = f"step_{type(self.controller).__name__.lower()}"
        out = f"{plot_cfg['output_dir']}/step"
        plot_trajectory(data,
            title=f"Step Response — {type(self.controller).__name__}",
            save_path=f"{out}/{label}_trajectory.png",
            show=plot_cfg["show"], dpi=plot_cfg["dpi"])
        plot_error(data,
            title=f"Step Response Error — {type(self.controller).__name__}",
            save_path=f"{out}/{label}_error.png",
            show=plot_cfg["show"], dpi=plot_cfg["dpi"])
