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
        already_flying: bool = False,
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
        self._already_flying = already_flying
        self._initial_yaw: float = 0.0  # recorded from ATTITUDE before takeoff
        # NED z offset captured near ground before takeoff.
        # All config z values are treated as relative to this reference.
        self._z_home_ned: float = 0.0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Full lifecycle. Always lands and disarms, even on exception."""
        try:
            self._pre_flight()

            takeoff_ned_z = self._z_rel_to_abs(
                -self.config["flight"]["takeoff_altitude"]
            )

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

        # Safety: always force RL overwrite OFF before any PID-driven takeoff.
        # If a previous RL session was killed (double Ctrl-C), _post_flight may
        # never have sent "overwrite off", leaving RL active and fighting PX4 PID
        # during the next test's takeoff — causing the "Imbalanced propeller"
        # failsafe and a crash.
        if self._using_rl and hasattr(self.controller, "force_overwrite_off"):
            self.controller.force_overwrite_off()
        else:
            # PID controller: send overwrite off directly via shell just in case
            # a prior RL session was hard-killed.
            try:
                from mavlink_interface.px4_shell import PX4Shell
                shell = PX4Shell(self.connection, response_timeout=1.0)
                shell.run("rl_tools_commander overwrite off")
                logger.info("Safety: rl_tools_commander overwrite off (pre-PID takeoff).")
            except Exception as exc:
                logger.debug("Could not send overwrite off (non-critical): %s", exc)

        # Step 1 — request telemetry streams so we actually receive data
        self.connection.request_streams(
            rate_hz=int(self.config["mavlink"]["setpoint_rate_hz"])
        )
        time.sleep(0.3)

        # Step 2 — start background setpoint stream with neutral z first.
        # We will update z after capturing EKF reference altitude.
        self.ctrl.start_setpoint_stream(target_pos=(0.0, 0.0, 0.0))

        # Step 3 — wait for EKF, capture reference z, set yaw, then arm
        self.ctrl.wait_for_ekf(timeout=30.0)
        ekf_msg = self.connection.recv_match("LOCAL_POSITION_NED", timeout=1.0)
        if ekf_msg is not None:
            self._z_home_ned = float(ekf_msg.z)
        else:
            self._z_home_ned = 0.0
            logger.warning("Failed to read LOCAL_POSITION_NED for z-home; using 0.0.")
        takeoff_ned_z = self._z_rel_to_abs(-takeoff_alt)
        logger.info(
            "Takeoff z reference captured: home_z=%.2f -> target_z=%.2f (%.1f m AGL)",
            self._z_home_ned, takeoff_ned_z, takeoff_alt,
        )
        self.ctrl.update_setpoint(0.0, 0.0, takeoff_ned_z)
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

    def _pre_flight_rl_direct(self) -> None:
        """
        Arm and take off directly with RL — no PID takeoff, no mode switch.

        The RL policy is activated BEFORE arming so that when OFFBOARD is
        engaged the policy is already running and drives the motors from the
        first cycle.  This eliminates the PID → RL transition jerk entirely.

        Sequence:
          1. Request telemetry streams.
          2. Start setpoint stream (OFFBOARD keepalive).
          3. Wait for EKF, capture home_z reference.
          4. Activate RL with target = hover altitude (before arming).
          5. Arm.
          6. Switch to OFFBOARD — RL drives motors from t=0.
          7. Wait for drone to reach hover altitude.
          8. Fly to origin.
        """
        arm_delay   = self.config["flight"]["arm_delay"]
        init_delay  = self.config["flight"]["offboard_init_delay"]
        takeoff_alt = self.config["flight"]["takeoff_altitude"]

        # Safety: make sure overwrite is off from any previous run.
        if hasattr(self.controller, "force_overwrite_off"):
            self.controller.force_overwrite_off()

        # Step 1 — telemetry streams
        self.connection.request_streams(
            rate_hz=int(self.config["mavlink"]["setpoint_rate_hz"])
        )
        time.sleep(0.3)

        # Step 2 — start setpoint stream (keeps OFFBOARD alive)
        self.ctrl.start_setpoint_stream(target_pos=(0.0, 0.0, 0.0))

        # Step 3 — wait for EKF, capture z reference
        self.ctrl.wait_for_ekf(timeout=30.0)
        ekf_msg = self.connection.recv_match("LOCAL_POSITION_NED", timeout=1.0)
        if ekf_msg is not None:
            self._z_home_ned = float(ekf_msg.z)
        else:
            self._z_home_ned = 0.0
            logger.warning("Failed to read LOCAL_POSITION_NED for z-home; using 0.0.")

        takeoff_ned_z = self._z_rel_to_abs(-takeoff_alt)
        logger.info(
            "RL direct takeoff: home_z=%.2f -> target_z=%.2f (%.1f m AGL)",
            self._z_home_ned, takeoff_ned_z, takeoff_alt,
        )
        self.ctrl.update_setpoint(0.0, 0.0, takeoff_ned_z)
        self._initial_yaw = math.radians(
            self.config["flight"].get("spawn_yaw_deg", 0.0)
        )
        self.ctrl.update_yaw(self._initial_yaw)

        # Step 4 — activate RL before arming so it is ready when OFFBOARD starts
        logger.info(
            "Activating RL before arm (target %.2f NED = %.1f m AGL)...",
            takeoff_ned_z, takeoff_alt,
        )
        self.controller.activate(0.0, 0.0, takeoff_ned_z, warmup_secs=0.0)

        # Step 5 — arm
        self.ctrl.arm()
        time.sleep(arm_delay)

        # Step 6 — switch to OFFBOARD; RL drives motors immediately
        self.ctrl.set_offboard_mode(init_delay=init_delay)
        logger.info(
            "RL OFFBOARD active — RL handling takeoff to %.1f m AGL...", takeoff_alt
        )

        # Step 7 — wait for altitude
        self._wait_for_altitude(takeoff_ned_z, tolerance=0.20, timeout=30.0)
        logger.info("RL direct takeoff complete.")
        time.sleep(1.0)

        # Step 8 — yaw and origin
        self.ctrl.update_yaw(self._initial_yaw)
        self._fly_to_origin(takeoff_ned_z)

    def _setup_already_flying(self) -> None:
        """
        Set up telemetry and OFFBOARD keepalive when the drone is already airborne
        with RL active.  Called instead of _pre_flight() when --already-flying is set.

        Assumes the user has already:
          1. Started SITL: make px4_sitl gz_x500_payload_drop ...
          2. In the PX4 shell:
               commander arm
               rl_tools_commander set_target_position 0 0 -1.5
               rl_tools_commander overwrite on
               commander mode offboard
          3. Waited until the drone is hovering stably.

        This method only:
          • Requests MAVLink telemetry streams (not sent by default in SITL).
          • Reads current position to capture home_z reference.
          • Starts the background OFFBOARD setpoint stream (keeps PX4 in OFFBOARD).
          • Marks the RLController as active so deactivation works at teardown.
        """
        rate_hz = self.config["mavlink"]["setpoint_rate_hz"]

        # Request telemetry streams so LOCAL_POSITION_NED, ATTITUDE, etc. arrive.
        self.connection.request_streams(rate_hz=int(rate_hz))
        time.sleep(0.3)

        # Read current position; use it as the OFFBOARD keepalive setpoint so we
        # don't inadvertently command the drone to move on stream start.
        ekf_msg = self.connection.recv_match("LOCAL_POSITION_NED", timeout=2.0)
        if ekf_msg is not None:
            cur_x, cur_y, cur_z = float(ekf_msg.x), float(ekf_msg.y), float(ekf_msg.z)
        else:
            cur_x, cur_y, cur_z = 0.0, 0.0, -1.5
            logger.warning(
                "Could not read LOCAL_POSITION_NED; assuming (0, 0, -1.5) NED."
            )

        # In SITL the world origin is at ground level (NED z ≈ 0), so home_z = 0.
        # Config z values (e.g. -1.5) are therefore identical to absolute NED z.
        self._z_home_ned = 0.0
        logger.info(
            "Already-flying: current pos=(%.2f, %.2f, %.2f) NED, home_z=0.0.",
            cur_x, cur_y, cur_z,
        )

        # Start background setpoint stream at current position (OFFBOARD keepalive).
        self.ctrl.start_setpoint_stream(target_pos=(cur_x, cur_y, cur_z))

        # Mark RL as active so controller.deactivate() runs correctly at teardown.
        if self._using_rl:
            self.controller._active = True
            logger.info("RL controller marked as active (pre-activated by user in PX4 shell).")

        self._initial_yaw = math.radians(
            self.config["flight"].get("spawn_yaw_deg", 0.0)
        )
        logger.info("Already-flying setup complete.")

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
            if self._already_flying:
                # Already-flying: switch to PX4 LAND mode so the vehicle uses its
                # own landing controller rather than OFFBOARD position setpoints.
                # This avoids the "PID via OFFBOARD" landing and is the clean way
                # to terminate a session that was started outside Python's control.
                self.ctrl.land()
                time.sleep(10.0)  # give PX4 time to descend and touch down
            else:
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
        z_abs = self._z_rel_to_abs(z)
        self.tlog.update_setpoint(x, y, z_abs)
        self.ctrl.update_setpoint(x, y, z_abs)              # update background stream
        self.ctrl.send_position_setpoint(x, y, z_abs)       # immediate one-shot

        if self._using_rl:
            self.controller.set_target(x, y, z_abs)         # notify rl_tools_commander

    def _z_rel_to_abs(self, z_ned_relative: float) -> float:
        """
        Convert relative NED z (config convention) to absolute LOCAL_POSITION_NED z.
        Example: home_z=-2.98, z_rel=-1.50 -> z_abs=-4.48.
        """
        return self._z_home_ned + z_ned_relative

    def _wait_for_stable(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        threshold: float = 0.08,
        hold_time: float = 2.0,
        timeout: float = 30.0,
    ) -> bool:
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
                    return True
            else:
                if stable_since != 0.0:
                    logger.debug("Unstable again (err=%.3fm), resetting timer.", err)
                stable_since = 0.0

        logger.warning(
            "Stability timeout after %.0fs — starting log anyway.", timeout
        )
        return False

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
        next_log = time.time() + 1.0
        while time.time() < deadline:
            msg = self.connection.recv_match("LOCAL_POSITION_NED", timeout=0.5)
            if msg:
                last_z = msg.z
                err = abs(msg.z - target_z_ned)
                logger.debug("Altitude: z=%.2f  target=%.2f  err=%.2f", msg.z, target_z_ned, err)
                if time.time() >= next_log:
                    logger.info(
                        "Takeoff progress: z=%.2f target=%.2f err=%.2f",
                        msg.z, target_z_ned, err,
                    )
                    next_log = time.time() + 1.0
                if err < tolerance:
                    return
        z_str = f"{last_z:.2f}" if last_z is not None else "unknown"
        logger.warning(
            "Altitude not reached in %.0fs (last z=%s, target=%.2f). Continuing.",
            timeout, z_str, target_z_ned,
        )

    def _sleep_rate(self, rate_hz: float) -> None:
        time.sleep(1.0 / rate_hz)

    # ------------------------------------------------------------------
    # Payload-specific helpers (used by PayloadHoverTest / PayloadDropTest)
    # ------------------------------------------------------------------

    def _wait_for_attitude_stable(
        self,
        max_rate_rad_s: float = 0.10,
        hold_time: float = 2.0,
        timeout: float = 15.0,
    ) -> bool:
        """
        Block until roll/pitch/yaw rates are ALL below `max_rate_rad_s` for
        `hold_time` consecutive seconds.  Used to confirm the payload pendulum
        has stopped swinging before activating RL.

        Returns True if stable achieved, False if timed out.
        """
        logger.info(
            "Waiting for attitude stability (rates < %.2f rad/s for %.1fs, "
            "timeout %.0fs)...",
            max_rate_rad_s, hold_time, timeout,
        )
        deadline = time.time() + timeout
        stable_since: float = 0.0

        while time.time() < deadline:
            msg = self.connection.recv_match("ATTITUDE", timeout=0.2)
            if msg is None:
                stable_since = 0.0
                continue

            max_rate = max(abs(msg.rollspeed), abs(msg.pitchspeed), abs(msg.yawspeed))

            if max_rate < max_rate_rad_s:
                if stable_since == 0.0:
                    stable_since = time.time()
                elif time.time() - stable_since >= hold_time:
                    logger.info(
                        "Attitude stable. max_rate=%.3f rad/s (%.1f deg/s).",
                        max_rate, math.degrees(max_rate),
                    )
                    return True
            else:
                if stable_since != 0.0:
                    logger.debug(
                        "Attitude unstable again (rate=%.3f rad/s), resetting.",
                        max_rate,
                    )
                stable_since = 0.0

        logger.warning(
            "Attitude stability timeout after %.0fs — proceeding anyway.", timeout
        )
        return False

    def _safe_rl_activate_payload(
        self,
        final_target_z_ned: float,
        transition_alt_ned: float = -1.0,
        warmup_secs: float = 2.0,
        jerk_warn_m: float = 0.30,
        jerk_abort_m: float = 0.60,
        monitor_secs: float = 3.0,
    ) -> bool:
        """
        Safe RL activation sequence for payload scenarios.

        Problem with naive activation
        ──────────────────────────────
        PID compensates the 0.5 kg payload by running motors at ~134 % of
        no-payload hover thrust.  When RL takes over, its `previous_action`
        observation is initialised to a neutral value — not to what PID was
        actually sending.  This mismatch causes a sudden thrust discontinuity
        (drone drops or shoots upward) that can crash the vehicle.

        Safe sequence
        ─────────────
        1. Lower drone to `transition_alt_ned` (default −1.0 m NED = 1.0 m AGL).
           If RL jerks it down 0.5 m from here, the drone is still at 0.5 m
           and RL has room to recover — versus crashing from 1.5 m.
        2. Wait for the pendulum to settle (attitude rates quiet).
        3. Activate RL with target = EXACT CURRENT POSITION (zero position
           error at switch-over → RL sees no correction demand on step 1).
        4. Sleep `warmup_secs` so RL runs through several policy cycles and
           its internal state converges.
        5. Monitor `monitor_secs`: log a WARNING if 3D error > `jerk_warn_m`,
           abort with emergency land if error > `jerk_abort_m`.
        6. Command RL to the final test altitude.

        Returns True if activation succeeded, False if aborted (jerk_abort).
        """
        rate_hz = self.config["mavlink"]["setpoint_rate_hz"]
        require_preconditions = self.config["flight"].get(
            "rl_require_preconditions", True
        )
        # Optional split-thresholds for windy scenarios.
        jerk_warn_xy_m = self.config["flight"].get("rl_jerk_warn_xy_m", jerk_warn_m)
        jerk_abort_xy_m = self.config["flight"].get("rl_jerk_abort_xy_m", jerk_abort_m)
        jerk_warn_down_m = self.config["flight"].get("rl_jerk_warn_down_m", jerk_warn_m)
        jerk_abort_down_m = self.config["flight"].get("rl_jerk_abort_down_m", jerk_abort_m)

        # ── Step 1: move to transition altitude (higher = more recovery room) ──
        transition_alt_abs = self._z_rel_to_abs(transition_alt_ned)
        final_target_abs = self._z_rel_to_abs(final_target_z_ned)

        logger.info(
            "Payload RL transition: moving to %.1f m AGL before RL switch...",
            abs(transition_alt_ned),
        )
        self.ctrl.update_setpoint(0.0, 0.0, transition_alt_abs)
        self._wait_for_altitude(transition_alt_abs, tolerance=0.12, timeout=15.0)
        if require_preconditions:
            stable_ok = self._wait_for_stable(
                0.0, 0.0, transition_alt_abs,
                threshold=0.07,
                hold_time=2.0,
                timeout=15.0,
            )
            if not stable_ok:
                logger.error(
                    "RL activation blocked: position not stable at transition altitude."
                )
                return False
        else:
            logger.info(
                "Skipping strict position precondition "
                "(rl_require_preconditions=false)."
            )
            time.sleep(1.0)

        # ── Step 2: wait for pendulum to calm down ────────────────────────
        if require_preconditions:
            attitude_ok = self._wait_for_attitude_stable(
                max_rate_rad_s=0.10,   # ~6 deg/s
                hold_time=2.0,
                timeout=15.0,
            )
            if not attitude_ok:
                logger.error(
                    "RL activation blocked: attitude rates not stable before switch."
                )
                return False
        else:
            logger.info(
                "Skipping strict attitude precondition "
                "(rl_require_preconditions=false)."
            )

        # ── Step 3: activate RL at exact current position ─────────────────
        msg = self.connection.recv_match("LOCAL_POSITION_NED", timeout=1.0)
        cur_x = msg.x if msg else 0.0
        cur_y = msg.y if msg else 0.0
        cur_z = msg.z if msg else transition_alt_abs

        logger.info(
            "Activating RL at exact current pos (%.2f, %.2f, %.2f NED).",
            cur_x, cur_y, cur_z,
        )
        self.controller.activate(cur_x, cur_y, cur_z)

        # ── Step 4: warmup — RL runs several cycles, internal state settles ─
        logger.info("RL warmup %.1fs (letting policy cycles converge)...", warmup_secs)
        time.sleep(warmup_secs)

        # ── Step 5: monitor for jerk ──────────────────────────────────────
        logger.info(
            "Monitoring RL for %.1fs (XY warn/abort=%.2f/%.2f m, "
            "down warn/abort=%.2f/%.2f m)...",
            monitor_secs,
            jerk_warn_xy_m, jerk_abort_xy_m,
            jerk_warn_down_m, jerk_abort_down_m,
        )
        deadline = time.time() + monitor_secs
        max_observed_err = 0.0

        while time.time() < deadline:
            m = self.connection.recv_match("LOCAL_POSITION_NED", timeout=0.1)
            if m is None:
                continue
            dx = m.x - cur_x
            dy = m.y - cur_y
            dz = m.z - cur_z
            err_xy = (dx ** 2 + dy ** 2) ** 0.5
            # NED: z increases when moving down.
            down_err = max(0.0, dz)
            err = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            max_observed_err = max(max_observed_err, err)

            if err_xy > jerk_abort_xy_m or down_err > jerk_abort_down_m:
                logger.error(
                    "RL JERK ABORT — err_xy=%.2f m, down_err=%.2f m "
                    "(limits %.2f / %.2f). Emergency landing.",
                    err_xy, down_err, jerk_abort_xy_m, jerk_abort_down_m,
                )
                return False   # caller will call _post_flight → land

            if err_xy > jerk_warn_xy_m or down_err > jerk_warn_down_m:
                logger.warning(
                    "RL jerk detected (err_xy=%.2f m, down_err=%.2f m) — RL recovering...",
                    err_xy, down_err,
                )

        logger.info(
            "RL stable after activation. max_err_during_monitor=%.3f m.", max_observed_err
        )

        # ── Step 6: command to final test altitude ────────────────────────
        logger.info(
            "Commanding RL to final test altitude %.1f m AGL...",
            abs(final_target_z_ned),
        )
        self.controller.set_target(0.0, 0.0, final_target_abs)
        self.ctrl.update_setpoint(0.0, 0.0, final_target_abs)

        return True
