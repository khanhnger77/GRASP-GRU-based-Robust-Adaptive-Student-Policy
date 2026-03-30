"""
Base class for all flight tests.

Lifecycle:
  1. _pre_flight  — arm, take off via PX4 OFFBOARD+PID to a stable hover
  2. _maybe_activate_rl — if controller is RLController, switch RL on
  3. run_test     — test-specific control loop (implemented by subclass)
  4. _maybe_deactivate_rl — if RL was activated, switch it off
  5. _post_flight — land, disarm, flush logs

Two controller modes:
──────────────────────────────────────────────────────────────────────────
PID (PIDController):
  Pre-flight:  OFFBOARD mode → position setpoints → PX4 built-in PID
  Control loop: Python PID → send_velocity_setpoint() at ~50 Hz
  Post-flight: send LAND mode → disarm

RL (RLController):
  Pre-flight:  OFFBOARD mode → position setpoints → PX4 PID takes drone up
  Activation:  shell "rl_tools_commander overwrite on"
               → rl_tools_commander publishes rl_tools_command uORB
               → rl_tools_policy reads sensors + runs neural net @ 500 Hz
               → actuator_motors_multiplexer switches to RL motor outputs
  Control loop: Python calls controller.set_target(x,y,z) via shell command
               → Python ALSO sends position setpoint via MAVLink so PX4 logs
                  reflect the intended target even in RL mode
  Deactivation: shell "rl_tools_commander overwrite off" → smooth transition
  Post-flight: LAND mode → disarm
──────────────────────────────────────────────────────────────────────────
"""

import logging
import math
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

from controllers.rl_controller import RLController
from mavlink_interface.connection import MAVLinkConnection
from mavlink_interface.offboard_control import OffboardController
from mavlink_interface.telemetry_logger import TelemetryLogger

logger = logging.getLogger(__name__)


class BaseTest(ABC):
    """
    Abstract base for all benchmark tests.

    Subclasses implement:
        run_test(self) -> None
    """

    def __init__(
        self,
        connection: MAVLinkConnection,
        controller,          # PIDController | RLController
        config: Dict[str, Any],
        csv_path: str,
    ):
        self.connection = connection
        self.controller = controller
        self.config = config
        self.csv_path = csv_path

        self.ctrl = OffboardController(
            connection,
            setpoint_rate_hz=config["mavlink"]["setpoint_rate_hz"],
        )
        self.tlog = TelemetryLogger(
            connection,
            csv_path,
            flush_interval=config["logging"]["csv_flush_interval"],
        )
        self._t_start: float = 0.0
        self.results: Dict[str, Any] = {}
        self._using_rl = isinstance(controller, RLController)
        self._initial_yaw: float = 0.0  # recorded from ATTITUDE before takeoff

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Full lifecycle. Always lands and disarms, even on exception."""
        try:
            self._pre_flight()

            takeoff_ned_z = -self.config["flight"]["takeoff_altitude"]

            if self._using_rl:
                # Activate RL with current position as initial target (no jerk on switch).
                msg = self.connection.recv_match("LOCAL_POSITION_NED", timeout=1.0)
                if msg:
                    self.controller.activate(msg.x, msg.y, msg.z)
                else:
                    self.controller.activate()
                # Immediately command the test start position so the drone moves
                # there during the settle window rather than staying at the
                # (possibly offset) activation position.
                self.controller.set_target(0.0, 0.0, takeoff_ned_z)

            # Wait until drone is actually stable at start position before logging.
            self._wait_for_stable(
                target_x=0.0,
                target_y=0.0,
                target_z=takeoff_ned_z,
                threshold=self.config["flight"].get("stable_threshold", 0.08),
                hold_time=self.config["flight"].get("stable_hold", 2.0),
                timeout=self.config["flight"].get("settle_delay", 30.0),
            )

            # Logger and timer start only when drone is confirmed stable.
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

    @abstractmethod
    def run_test(self) -> None:
        """Override: inner control loop after takeoff / RL activation."""

    # ------------------------------------------------------------------
    # Pre / post flight
    # ------------------------------------------------------------------

    def _pre_flight(self) -> None:
        """
        Arm and climb to takeoff altitude using PX4 OFFBOARD + position setpoints.

        Sequence:
          1. Request telemetry streams from PX4 (LOCAL_POSITION_NED, ATTITUDE,
             ACTUATOR_OUTPUTS are NOT sent by default over UDP in SITL).
          2. Pre-stream position setpoints for `init_delay` seconds — PX4 requires
             setpoints to already be arriving before accepting the OFFBOARD mode switch.
          3. Arm.
          4. Switch to OFFBOARD.
          5. Wait for drone to reach target altitude (with timeout).
        """
        arm_delay     = self.config["flight"]["arm_delay"]
        init_delay    = self.config["flight"]["offboard_init_delay"]
        takeoff_alt   = self.config["flight"]["takeoff_altitude"]
        takeoff_ned_z = -takeoff_alt  # NED convention: negative Z = up

        # Step 1 — request telemetry streams so we actually receive data
        self.connection.request_streams(
            rate_hz=int(self.config["mavlink"]["setpoint_rate_hz"])
        )
        time.sleep(0.3)

        # Step 2 — start background position setpoint stream
        # With fixed POSITION_MASK this now correctly commands z = -1.5 m NED.
        self.ctrl.start_setpoint_stream(target_pos=(0.0, 0.0, takeoff_ned_z))

        # Step 3 — wait for EKF, set fixed spawn yaw, then arm
        self.ctrl.wait_for_ekf(timeout=30.0)
        self._initial_yaw = math.radians(
            self.config["flight"].get("spawn_yaw_deg", 0.0)
        )
        self.ctrl.update_yaw(self._initial_yaw)
        logger.info("Spawn yaw: %.3f rad (%.1f°)",
                    self._initial_yaw, math.degrees(self._initial_yaw))
        self.ctrl.arm()
        time.sleep(arm_delay)

        # Step 4 — switch to OFFBOARD (setpoints already streaming for init_delay s)
        self.ctrl.set_offboard_mode(init_delay=init_delay)
        logger.info("OFFBOARD active. Climbing to %.1f m AGL...", takeoff_alt)

        # Step 5 — wait for altitude
        self._wait_for_altitude(takeoff_ned_z, tolerance=0.20, timeout=30.0)
        logger.info("Takeoff complete.")
        time.sleep(1.0)  # brief stabilisation

        # Step 6 — fly back to origin and reset yaw, always, for every test.
        # update_yaw is called here unconditionally so yaw is correct even when
        # _fly_to_origin() returns early because the drone is already near origin.
        self.ctrl.update_yaw(self._initial_yaw)
        self._fly_to_origin(takeoff_ned_z)

    def _post_flight(self) -> None:
        """Stop logging first, then deactivate RL, land, disarm."""
        try:
            # Stop logger BEFORE any mode transitions so landing/deactivation
            # transients are never written to the benchmark CSV.
            self.tlog.stop()

            if self._using_rl and self.controller.is_active:
                # Sync MAVLink setpoint to current position before deactivating RL
                # so PX4 PID doesn't snap to the last test setpoint on takeover.
                msg = self.connection.recv_match("LOCAL_POSITION_NED", timeout=0.5)
                if msg:
                    self.ctrl.update_setpoint(msg.x, msg.y, msg.z)
                self.controller.deactivate()
                time.sleep(1.5)

            logger.info("Landing...")
            self.ctrl.land_via_setpoint(timeout=20.0)
            time.sleep(1.0)
            self.ctrl.disarm()
        except Exception as exc:
            logger.error("Error during post-flight: %s", exc)
        finally:
            self.ctrl.stop_setpoint_stream()
            self.tlog.stop()  # no-op if already stopped

    # ------------------------------------------------------------------
    # Helpers used by subclass control loops
    # ------------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since run_test() started."""
        return time.time() - self._t_start

    def send_setpoint(self, x: float, y: float, z: float) -> None:
        """
        Unified setpoint sender — both controllers receive position setpoints.

        Design rationale
        ────────────────
        PID mode:  We send SET_POSITION_TARGET_LOCAL_NED with POSITION_MASK so
                   PX4's own multi-loop PID (position → velocity → attitude →
                   rate → motors) tracks the setpoint.  This is the correct
                   "baseline PID" — it uses the full PX4 controller stack which
                   is what any real OFFBOARD flight would use.

        RL mode:   We send the same position setpoint via MAVLink (so PX4 flight
                   logs show the intended target) AND notify rl_tools_commander
                   via shell.  The RL policy then drives the motors directly at
                   500 Hz, bypassing PX4's PID stack.

        Both paths use the background stream to keep OFFBOARD mode alive.
        Calling send_position_setpoint() here also sends an immediate one-shot
        command in addition to the stream, ensuring minimal latency.
        """
        self.tlog.update_setpoint(x, y, z)
        self.ctrl.update_setpoint(x, y, z)          # update background stream
        self.ctrl.send_position_setpoint(x, y, z)   # immediate one-shot

        if self._using_rl:
            self.controller.set_target(x, y, z)     # notify rl_tools_commander

    def _wait_for_stable(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        threshold: float = 0.08,
        hold_time: float = 2.0,
        timeout: float = 30.0,
    ) -> None:
        """
        Block until the drone holds position within `threshold` metres of
        (target_x, target_y, target_z) for `hold_time` consecutive seconds.

        This is more reliable than a fixed sleep: logging starts only when
        the drone is provably stable, regardless of how long settling takes.
        Falls through after `timeout` seconds with a warning.
        """
        logger.info(
            "Waiting for stable hover at (%.2f, %.2f, %.2f) "
            "err<%.2fm for %.1fs (timeout %.0fs)...",
            target_x, target_y, target_z, threshold, hold_time, timeout,
        )
        deadline = time.time() + timeout
        stable_since: float = 0.0

        while time.time() < deadline:
            msg = self.connection.recv_match("LOCAL_POSITION_NED", timeout=0.2)
            if msg is None:
                stable_since = 0.0
                continue

            err = ((msg.x - target_x) ** 2 +
                   (msg.y - target_y) ** 2 +
                   (msg.z - target_z) ** 2) ** 0.5

            if err < threshold:
                if stable_since == 0.0:
                    stable_since = time.time()
                elif time.time() - stable_since >= hold_time:
                    logger.info("Stable. err=%.3fm — starting log.", err)
                    return
            else:
                if stable_since != 0.0:
                    logger.debug("Unstable again (err=%.3fm), resetting timer.", err)
                stable_since = 0.0

        logger.warning(
            "Stability timeout after %.0fs — starting log anyway.", timeout
        )

    def _fly_to_origin(self, takeoff_ned_z: float, tolerance: float = 0.10, timeout: float = 30.0) -> None:
        """
        Fly to (0, 0) at takeoff altitude and wait until arrived.
        Always executed regardless of current distance.
        """
        msg = self.connection.recv_match("LOCAL_POSITION_NED", timeout=1.0)
        dist = ((msg.x ** 2 + msg.y ** 2) ** 0.5) if msg else 0.0
        logger.info("Flying to origin (current offset: %.2f m)...", dist)
        self.ctrl.update_setpoint(0.0, 0.0, takeoff_ned_z)

        deadline = time.time() + timeout
        while time.time() < deadline:
            m = self.connection.recv_match("LOCAL_POSITION_NED", timeout=0.5)
            if m and (m.x ** 2 + m.y ** 2) ** 0.5 < tolerance:
                break
        logger.info("At origin. Ready to start test.")

    def _wait_for_altitude(
        self, target_z_ned: float, tolerance: float = 0.2, timeout: float = 30.0
    ) -> None:
        """
        Block until pos_z is within tolerance of target_z (NED).

        Reads directly from MAVLink so this works both before and after
        the telemetry logger has started.
        """
        deadline = time.time() + timeout
        last_z = None
        while time.time() < deadline:
            msg = self.connection.recv_match("LOCAL_POSITION_NED", timeout=0.5)
            if msg:
                last_z = msg.z
                err = abs(msg.z - target_z_ned)
                logger.debug("Altitude: z=%.2f  target=%.2f  err=%.2f", msg.z, target_z_ned, err)
                if err < tolerance:
                    return
        z_str = f"{last_z:.2f}" if last_z is not None else "unknown"
        logger.warning(
            "Altitude not reached in %.0fs (last z=%.2s, target=%.2f). Continuing.",
            timeout, z_str, target_z_ned,
        )

    def _sleep_rate(self, rate_hz: float) -> None:
        time.sleep(1.0 / rate_hz)
