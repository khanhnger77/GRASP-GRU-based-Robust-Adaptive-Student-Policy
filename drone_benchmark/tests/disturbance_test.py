"""
Disturbance Rejection Test
===========================
Hold base position → 1-second setpoint offset → release → measure recovery.

For RL: the offset is forwarded to rl_tools_commander as a new target position;
the policy must fly to it and then recover back to base.  This tests exactly
the disturbance-rejection capability that rl-tools trains explicitly.
"""

import logging

from tests.base_test import BaseTest
from metrics import compute_rmse, compute_max_error, compute_settling_time
from utils.plotting import plot_trajectory, plot_error

logger = logging.getLogger(__name__)


class DisturbanceTest(BaseTest):

    def run_test(self) -> None:
        cfg    = self.config["tests"]["disturbance"]
        base   = tuple(cfg["base_position"])
        offset = cfg["disturbance_offset"]
        disturbed = (
            base[0] + offset[0],
            base[1] + offset[1],
            base[2] + offset[2],
        )
        hold_before  = cfg["hold_before"]
        hold_after   = cfg["hold_after"]
        rate_hz = self.config["mavlink"]["setpoint_rate_hz"]
        total   = hold_before + 1.0 + hold_after

        if not self._using_rl:
            self.controller.reset()

        disturbance_active = False
        recovery_start_time: float = 0.0

        while self.elapsed < total:
            t = self.elapsed

            if t < hold_before:
                target = base
            elif t < hold_before + 1.0:
                if not disturbance_active:
                    disturbance_active = True
                    logger.info(
                        "Disturbance ON at t=%.2fs | offset=%s | controller=%s",
                        t, offset, type(self.controller).__name__,
                    )
                target = disturbed
            else:
                if disturbance_active:
                    disturbance_active = False
                    recovery_start_time = t
                    logger.info("Disturbance OFF at t=%.2fs — recovery phase.", t)
                target = base

            self.send_setpoint(*target)
            self._sleep_rate(rate_hz)

        self._compute_and_store_metrics(recovery_start_time)

    def _compute_and_store_metrics(self, recovery_start: float) -> None:
        data = self.tlog.get_data()
        t0 = data[0].timestamp if data else 0.0
        recovery = [s for s in data if (s.timestamp - t0) >= recovery_start]

        thresh = self.config["tests"]["disturbance"].get("settle_threshold", 0.1)
        settling = compute_settling_time(recovery, threshold=thresh)

        self.results = {
            "test":                          "disturbance",
            "controller":                    type(self.controller).__name__,
            "rmse_recovery":                 compute_rmse(recovery),
            "max_error":                     compute_max_error(data),
            "settling_time_after_disturbance": settling if settling is not None else float("nan"),
            "n_samples":                     len(data),
        }
        logger.info("=== Disturbance Results ===")
        for k, v in self.results.items():
            logger.info("  %-35s %s", k + ":", f"{v:.4f}" if isinstance(v, float) else v)

        plot_cfg = self.config["plotting"]
        label = f"disturbance_{type(self.controller).__name__.lower()}"
        plot_trajectory(data,
            title=f"Disturbance — {type(self.controller).__name__}",
            save_path=f"{plot_cfg['output_dir']}/{label}_trajectory.png",
            show=plot_cfg["show"], dpi=plot_cfg["dpi"])
        plot_error(data,
            title=f"Disturbance Error — {type(self.controller).__name__}",
            save_path=f"{plot_cfg['output_dir']}/{label}_error.png",
            show=plot_cfg["show"], dpi=plot_cfg["dpi"])
