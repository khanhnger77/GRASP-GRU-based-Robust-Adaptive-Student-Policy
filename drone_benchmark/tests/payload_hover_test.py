"""
Payload Hover Test
==================
Hold position at (0, 0, -1.5 m NED) for 30 seconds WITH payload attached.

Purpose
-------
Compare PID vs RL stability when the drone is carrying a 0.5 kg payload
suspended on a 0.5 m rope.  The pendulum dynamics introduce periodic
disturbances — this test measures how much each controller is affected.

Compared to the baseline HoverTest (no payload), larger RMSE and attitude
oscillations are expected.  RL was trained with payload dynamics and should
show less attitude coupling than PID.

RL transition
-------------
PID takes off and holds at 1.5 m AGL.  Once the payload pendulum has settled,
RL is activated at the drone's exact current position (zero initial error).
With RESET_PREVIOUS_ACTION_VALUE = 0.32 in the PX4 module the first RL motor
output is at ~66 % throttle (payload hover equilibrium) rather than 50 %,
so the drone does not drop at the switch.

Model: gz_x500_payload_drop
Launch:
    make px4_sitl gz_x500_payload_drop \\
        EXTERNAL_MODULES_LOCATION="$(cd .. && cd external_modules && pwd)" \\
        RL_TOOLS_ROOT="$(cd ../../../ && pwd)" -j8

Payload is NOT dropped during this test.
"""

import logging
import time
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


class PayloadHoverTest(BaseTest):
    """
    Hover with payload attached.

    Overrides run() to use the safe RL activation sequence so the transition
    from PID to RL does not cause a crash when payload is attached.
    """

    def run(self) -> Dict[str, Any]:
        """
        Two entry paths depending on --already-flying flag.

        Normal (default):
            PID takes off → RL activated at exact current pos → stable → test.

        Already-flying (--already-flying):
            User manually armed + took off with RL active before running this script.
            Python connects to telemetry, syncs OFFBOARD keepalive, waits for stable,
            then starts the test without any arming/takeoff/RL-activation step.
        """
        cfg_flight = self.config["flight"]
        cfg_hover  = self.config["tests"]["payload_hover"]

        try:
            if self._already_flying:
                # ── Already-flying path ─────────────────────────────────────
                self._setup_already_flying()

                # Re-sync setpoint stream and RL target to the configured hover position.
                target_z_abs = self._z_rel_to_abs(cfg_hover["target"][2])
                self.ctrl.update_setpoint(
                    cfg_hover["target"][0],
                    cfg_hover["target"][1],
                    target_z_abs,
                )
                if self._using_rl:
                    self.controller.set_target(
                        cfg_hover["target"][0],
                        cfg_hover["target"][1],
                        target_z_abs,
                    )

                logger.info(
                    "Already-flying: waiting for stable hover at target %s...",
                    cfg_hover["target"],
                )
                self._wait_for_stable(
                    target_x=cfg_hover["target"][0],
                    target_y=cfg_hover["target"][1],
                    target_z=target_z_abs,
                    threshold=cfg_flight.get("stable_threshold", 0.15),
                    hold_time=cfg_flight.get("stable_hold", 2.0),
                    timeout=cfg_flight.get("settle_delay", 30.0),
                )

            else:
                # ── Normal path: PID takeoff → RL switch ────────────────────
                self._pre_flight()
                takeoff_ned_z = self._z_rel_to_abs(-cfg_flight["takeoff_altitude"])

                if self._using_rl:
                    msg = self.connection.recv_match("LOCAL_POSITION_NED", timeout=1.0)
                    cur_x = msg.x if msg else 0.0
                    cur_y = msg.y if msg else 0.0
                    cur_z = msg.z if msg else takeoff_ned_z
                    logger.info(
                        "Switching to RL at current pos (%.2f, %.2f, %.2f NED).",
                        cur_x, cur_y, cur_z,
                    )
                    self.controller.activate(
                        cur_x, cur_y, cur_z,
                        warmup_secs=cfg_flight.get("rl_warmup_secs", 1.0),
                    )
                    self.controller.set_target(0.0, 0.0, takeoff_ned_z)
                    self.ctrl.update_setpoint(0.0, 0.0, takeoff_ned_z)
                else:
                    self.ctrl.update_setpoint(0.0, 0.0, takeoff_ned_z)

                self._wait_for_stable(
                    target_x=0.0,
                    target_y=0.0,
                    target_z=takeoff_ned_z,
                    threshold=cfg_flight.get("stable_threshold", 0.08),
                    hold_time=cfg_flight.get("stable_hold", 2.0),
                    timeout=cfg_flight.get("settle_delay", 30.0),
                )

            self.tlog.start()
            self._t_start = time.time()
            self.run_test()

        except KeyboardInterrupt:
            logger.warning("Test interrupted by user (Ctrl-C).")
        except Exception as exc:
            logger.error("Test failed: %s", exc, exc_info=True)
        finally:
            self._post_flight()

        return self.results

    def run_test(self) -> None:
        cfg      = self.config["tests"]["payload_hover"]
        target   = tuple(cfg["target"])   # (x, y, z) NED
        duration = cfg["duration"]
        rate_hz  = self.config["mavlink"]["setpoint_rate_hz"]

        logger.info(
            "PayloadHover test | controller=%s | target=%s | duration=%.0fs",
            type(self.controller).__name__, target, duration,
        )
        logger.info(
            "Payload is ATTACHED (0.5 kg @ 0.5 m rope). Pendulum swing will "
            "appear as attitude disturbances."
        )

        if not self._using_rl:
            self.controller.reset()

        while self.elapsed < duration:
            self.send_setpoint(*target)
            self._sleep_rate(rate_hz)

        self._compute_and_store_metrics(target)

    def _compute_and_store_metrics(self, target) -> None:
        data     = self.tlog.get_data()
        rmse_xyz = compute_per_axis_rmse(data)

        self.results = {
            "test":             "payload_hover",
            "controller":       type(self.controller).__name__,
            "payload_attached": True,
            "rmse_3d":          compute_rmse(data),
            "rmse_x":           rmse_xyz[0],
            "rmse_y":           rmse_xyz[1],
            "rmse_z":           rmse_xyz[2],
            "max_error":        compute_max_error(data),
            "control_effort":   compute_control_effort(data),
            "n_samples":        len(data),
        }

        logger.info("=== PayloadHover Results ===")
        for k, v in self.results.items():
            logger.info(
                "  %-24s %s", k + ":",
                f"{v:.4f}" if isinstance(v, float) else v,
            )

        plot_cfg  = self.config["plotting"]
        ctrl_name = type(self.controller).__name__.lower()
        label = f"payload_hover_{ctrl_name}"
        out = f"{plot_cfg['output_dir']}/payload_hover"
        plot_trajectory(
            data,
            title=f"Payload Hover — {type(self.controller).__name__}",
            save_path=f"{out}/{label}_trajectory.png",
            show=plot_cfg["show"],
            dpi=plot_cfg["dpi"],
        )
        plot_error(
            data,
            title=f"Payload Hover Error — {type(self.controller).__name__}",
            save_path=f"{out}/{label}_error.png",
            show=plot_cfg["show"],
            dpi=plot_cfg["dpi"],
        )
