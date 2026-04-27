"""
Payload Drop Test
=================
Drone hovers with payload attached → drops payload mid-flight → measures
how quickly and smoothly the drone recovers to the original setpoint.

Phases
------
  Phase 1 — Pre-drop hover  (0 … hold_before seconds)
      Drone hovers at drop_position with payload attached.
      Logged data shows baseline stability under payload pendulum disturbances.

  Phase 2 — Drop event  (at t = hold_before)
      Payload is released by publishing to the Gazebo topic:
          gz topic -t /payload/detach -m gz.msgs.Empty -p "unused: true"
      The drone experiences a sudden upward jerk (~0.5 kg mass removed)
      and oscillation from the released rope/pendulum momentum.

  Phase 3 — Post-drop recovery  (hold_before … hold_before + hold_after)
      Drone continues targeting the same setpoint.
      Logged data shows transient response and settling behaviour.

Metrics
-------
  - pre_drop_rmse:     3D RMSE during phase 1  (payload-loaded baseline)
  - post_drop_rmse:    3D RMSE during phase 3  (recovery quality)
  - max_error:         peak position error (worst spike at drop moment)
  - settling_time:     seconds after drop until error < settle_threshold for 2s
  - attitude_spike_*:  peak absolute roll/pitch/yaw change at drop event
  - control_effort_*:  normalised motor squared-sum (pre and post drop)

Model: gz_x500_payload_drop
Launch:
    make px4_sitl gz_x500_payload_drop \\
        EXTERNAL_MODULES_LOCATION="$(cd .. && cd external_modules && pwd)" \\
        RL_TOOLS_ROOT="$(cd ../../../ && pwd)" -j8
"""

import logging
import subprocess
import time
from typing import Any, Dict, List, Optional

import numpy as np

from tests.base_test import BaseTest
from mavlink_interface.telemetry_logger import TelemetrySnapshot
from metrics import compute_rmse, compute_max_error, compute_settling_time, compute_control_effort
from utils.plotting import plot_trajectory, plot_error

logger = logging.getLogger(__name__)

# Gazebo topic command to detach the payload
_GZ_DETACH_CMD = [
    "gz", "topic",
    "-t", "/payload/detach",
    "-m", "gz.msgs.Empty",
    "-p", "unused: true",
]


def _drop_payload() -> bool:
    """
    Publish to /payload/detach via the gz CLI tool.

    Returns True on success, False if gz is not available or the command fails.
    The test continues even on failure so that logs remain consistent.
    """
    try:
        result = subprocess.run(
            _GZ_DETACH_CMD,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        if result.returncode == 0:
            logger.info("Payload detached (gz topic OK).")
            return True
        else:
            logger.error(
                "gz topic /payload/detach failed (rc=%d): %s",
                result.returncode, result.stderr.strip(),
            )
            return False
    except FileNotFoundError:
        logger.error(
            "gz CLI not found — cannot drop payload. "
            "Make sure Gazebo is running and 'gz' is on PATH."
        )
        return False
    except subprocess.TimeoutExpired:
        logger.error("gz topic command timed out.")
        return False


def _attitude_spike(
    snapshots: List[TelemetrySnapshot],
    drop_time: float,
    window: float = 3.0,
) -> Dict[str, float]:
    """
    Compute peak absolute roll/pitch/yaw change in the `window` seconds
    after the drop event.  Baseline is the mean attitude just before the drop.

    Args:
        snapshots:  full telemetry list
        drop_time:  absolute timestamp of the drop event
        window:     seconds after drop to search for peak

    Returns:
        dict with keys max_roll_deg, max_pitch_deg, max_yaw_deg
    """
    pre  = [s for s in snapshots if s.timestamp < drop_time]
    post = [s for s in snapshots if drop_time <= s.timestamp <= drop_time + window]

    if not pre or not post:
        return {"max_roll_deg": float("nan"),
                "max_pitch_deg": float("nan"),
                "max_yaw_deg": float("nan")}

    base_roll  = np.mean([s.roll  for s in pre[-10:]])
    base_pitch = np.mean([s.pitch for s in pre[-10:]])
    base_yaw   = np.mean([s.yaw   for s in pre[-10:]])

    max_roll  = float(np.max(np.abs([s.roll  - base_roll  for s in post])))
    max_pitch = float(np.max(np.abs([s.pitch - base_pitch for s in post])))
    max_yaw   = float(np.max(np.abs([s.yaw   - base_yaw   for s in post])))

    return {
        "max_roll_deg":  float(np.degrees(max_roll)),
        "max_pitch_deg": float(np.degrees(max_pitch)),
        "max_yaw_deg":   float(np.degrees(max_yaw)),
    }


class PayloadDropTest(BaseTest):
    """
    Payload drop test: hover → drop → measure recovery.

    Timeline:
        t=0               : pre-drop hover begins (payload attached)
        t=hold_before     : payload released via gz topic
        t=hold_before+hold_after : test ends

    RL mode
    ───────
    PID takes off and holds at drop_position altitude.  Once the payload
    pendulum has settled, RL activates at the drone's exact current position
    (zero initial error).  RESET_PREVIOUS_ACTION_VALUE = 0.32 in the PX4 module
    ensures the first RL motor output is at payload-hover equilibrium (~66 %
    throttle) so the drone does not drop at the switch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drop_timestamp: Optional[float] = None
        self._drop_success: bool = False

    # ------------------------------------------------------------------
    # Override run(): PID takeoff → pendulum settle → RL switch → test
    # ------------------------------------------------------------------

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
        cfg_drop   = self.config["tests"]["payload_drop"]

        try:
            if self._already_flying:
                # ── Already-flying path ─────────────────────────────────────
                # User pre-conditions (done in PX4 shell):
                #   commander arm
                #   rl_tools_commander set_target_position 0 0 -1.5
                #   rl_tools_commander overwrite on
                #   commander mode offboard
                self._setup_already_flying()

                # Re-sync setpoint stream and RL target to the configured drop position.
                target_z_abs = self._z_rel_to_abs(cfg_drop["drop_position"][2])
                self.ctrl.update_setpoint(
                    cfg_drop["drop_position"][0],
                    cfg_drop["drop_position"][1],
                    target_z_abs,
                )
                if self._using_rl:
                    self.controller.set_target(
                        cfg_drop["drop_position"][0],
                        cfg_drop["drop_position"][1],
                        target_z_abs,
                    )

                logger.info(
                    "Already-flying: waiting for stable hover at drop position %s...",
                    cfg_drop["drop_position"],
                )
                self._wait_for_stable(
                    target_x=cfg_drop["drop_position"][0],
                    target_y=cfg_drop["drop_position"][1],
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

    # ------------------------------------------------------------------

    def run_test(self) -> None:
        cfg          = self.config["tests"]["payload_drop"]
        target       = tuple(cfg["drop_position"])  # NED
        hold_before  = cfg["hold_before"]
        hold_after   = cfg["hold_after"]
        rate_hz      = self.config["mavlink"]["setpoint_rate_hz"]
        total        = hold_before + hold_after

        logger.info(
            "PayloadDrop test | controller=%s | target=%s | "
            "hold_before=%.0fs | hold_after=%.0fs",
            type(self.controller).__name__, target, hold_before, hold_after,
        )

        if not self._using_rl:
            self.controller.reset()

        dropped = False

        while self.elapsed < total:
            t = self.elapsed

            # ── Drop event ──────────────────────────────────────────────
            if t >= hold_before and not dropped:
                dropped = True
                self._drop_timestamp = time.time()
                logger.info(
                    "=== PAYLOAD DROP at t=%.2fs === "
                    "(gz topic -t /payload/detach -m gz.msgs.Empty -p 'unused: true')",
                    t,
                )
                self._drop_success = _drop_payload()
                logger.info(
                    "Post-drop: drone should jerk upward then recover to setpoint %s.",
                    target,
                )

            # ── Control loop ────────────────────────────────────────────
            # Keep commanding the SAME setpoint throughout — the drone must
            # hold position before and after the drop event.
            self.send_setpoint(*target)
            self._sleep_rate(rate_hz)

        self._compute_and_store_metrics(target, hold_before, hold_after)

    def _compute_and_store_metrics(
        self,
        target: tuple,
        hold_before: float,
        hold_after: float,
    ) -> None:
        cfg       = self.config["tests"]["payload_drop"]
        thresh    = cfg.get("settle_threshold", 0.10)

        all_data  = self.tlog.get_data()
        if not all_data:
            self.results = {"error": "no telemetry data"}
            return

        t0 = all_data[0].timestamp

        # Split into pre-drop and post-drop windows
        pre_data  = [s for s in all_data if (s.timestamp - t0) < hold_before]
        post_data = [s for s in all_data if (s.timestamp - t0) >= hold_before]

        # Attitude spike at drop moment
        spike = {}
        if self._drop_timestamp is not None:
            spike = _attitude_spike(all_data, self._drop_timestamp, window=3.0)
        else:
            spike = {
                "max_roll_deg":  float("nan"),
                "max_pitch_deg": float("nan"),
                "max_yaw_deg":   float("nan"),
            }

        # Settling time measured from the drop event
        settling = compute_settling_time(post_data, threshold=thresh, window=2.0)

        self.results = {
            "test":                   "payload_drop",
            "controller":             type(self.controller).__name__,
            "drop_success":           self._drop_success,

            # Pre-drop (payload attached) — baseline under pendulum disturbances
            "pre_drop_rmse":          compute_rmse(pre_data),
            "pre_drop_control_effort": compute_control_effort(pre_data),

            # Drop event — attitude jerk magnitude
            "attitude_spike_roll_deg":  spike["max_roll_deg"],
            "attitude_spike_pitch_deg": spike["max_pitch_deg"],
            "attitude_spike_yaw_deg":   spike["max_yaw_deg"],

            # Post-drop recovery
            "post_drop_rmse":         compute_rmse(post_data),
            "post_drop_max_error":    compute_max_error(post_data),
            "settling_time_s":        settling if settling is not None else float("nan"),
            "post_drop_control_effort": compute_control_effort(post_data),

            "n_samples_pre":          len(pre_data),
            "n_samples_post":         len(post_data),
        }

        logger.info("=== PayloadDrop Results ===")
        for k, v in self.results.items():
            logger.info(
                "  %-34s %s", k + ":",
                f"{v:.4f}" if isinstance(v, float) else v,
            )

        plot_cfg  = self.config["plotting"]
        ctrl_name = type(self.controller).__name__.lower()
        label     = f"payload_drop_{ctrl_name}"
        out       = f"{plot_cfg['output_dir']}/payload_drop"

        # Full trajectory (pre + post drop)
        plot_trajectory(
            all_data,
            title=f"Payload Drop — {type(self.controller).__name__}",
            save_path=f"{out}/{label}_trajectory.png",
            show=plot_cfg["show"],
            dpi=plot_cfg["dpi"],
        )
        # Error magnitude highlighting the drop spike
        plot_error(
            all_data,
            title=f"Payload Drop Error — {type(self.controller).__name__}",
            save_path=f"{out}/{label}_error.png",
            show=plot_cfg["show"],
            dpi=plot_cfg["dpi"],
        )
        # Pre and post drop separately for detailed analysis
        if pre_data:
            plot_error(
                pre_data,
                title=f"Payload Drop — Pre-Drop Phase ({type(self.controller).__name__})",
                save_path=f"{out}/{label}_pre_drop_error.png",
                show=plot_cfg["show"],
                dpi=plot_cfg["dpi"],
            )
        if post_data:
            plot_error(
                post_data,
                title=f"Payload Drop — Post-Drop Recovery ({type(self.controller).__name__})",
                save_path=f"{out}/{label}_post_drop_error.png",
                show=plot_cfg["show"],
                dpi=plot_cfg["dpi"],
            )
