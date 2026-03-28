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

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Full lifecycle. Always lands and disarms, even on exception."""
        try:
            self._pre_flight()
            self.tlog.start()
            self._t_start = time.time()

            if self._using_rl:
                self.controller.activate()

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
        """Arm and climb to takeoff altitude using PX4's PID via OFFBOARD."""
        arm_delay    = self.config["flight"]["arm_delay"]
        init_delay   = self.config["flight"]["offboard_init_delay"]
        takeoff_alt  = self.config["flight"]["takeoff_altitude"]
        takeoff_ned_z = -takeoff_alt  # NED: negative = up

        # Stream a constant hover setpoint BEFORE switching to OFFBOARD.
        # PX4 refuses the mode switch if no setpoints have arrived yet.
        self.ctrl.start_setpoint_stream(target_pos=(0.0, 0.0, takeoff_ned_z))

        self.ctrl.arm()
        time.sleep(arm_delay)

        self.ctrl.set_offboard_mode(init_delay=init_delay)
        logger.info("OFFBOARD active. Climbing to %.1f m AGL...", takeoff_alt)

        self._wait_for_altitude(takeoff_ned_z, tolerance=0.15, timeout=25.0)
        logger.info("Takeoff complete.")

        # Brief stabilisation pause before the test begins
        time.sleep(1.0)

    def _post_flight(self) -> None:
        """Deactivate RL (if active), land, disarm, stop logging."""
        try:
            if self._using_rl and self.controller.is_active:
                self.controller.deactivate()
                # Let the smooth transition play out before landing
                time.sleep(1.5)

            logger.info("Landing...")
            self.ctrl.land()
            time.sleep(6.0)   # wait for PX4 to touch down
            self.ctrl.disarm()
        except Exception as exc:
            logger.error("Error during post-flight: %s", exc)
        finally:
            self.ctrl.stop_setpoint_stream()
            self.tlog.stop()

    # ------------------------------------------------------------------
    # Helpers used by subclass control loops
    # ------------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since run_test() started."""
        return time.time() - self._t_start

    def send_setpoint(self, x: float, y: float, z: float) -> None:
        """
        Unified setpoint sender for both controller types.

        PID:  calls send_velocity_setpoint — output from controller.compute()
              is a velocity; the position setpoint is also streamed for logging.
        RL:   calls send_position_setpoint — PX4 records the target; the actual
              motor commands come from rl_tools_policy inside PX4.
        """
        snap = self.tlog.get_latest()
        current = (snap.pos_x, snap.pos_y, snap.pos_z)
        target  = (x, y, z)

        self.tlog.update_setpoint(x, y, z)

        if self._using_rl:
            # Inform the on-board RL commander about the target
            self.controller.set_target(x, y, z)
            # Also send a MAVLink position setpoint so PX4 flight logs reflect
            # the desired target (does NOT override RL motor output)
            self.ctrl.send_position_setpoint(x, y, z)
        else:
            # Python PID → velocity command → PX4 inner loop
            vx, vy, vz = self.controller.compute(current=current, target=target)
            self.ctrl.send_velocity_setpoint(vx, vy, vz)

    def _wait_for_altitude(
        self, target_z_ned: float, tolerance: float = 0.2, timeout: float = 25.0
    ) -> None:
        """Block until pos_z is within tolerance of target_z (NED)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            snap = self.tlog.get_latest()
            if abs(snap.pos_z - target_z_ned) < tolerance:
                return
            time.sleep(0.05)
        logger.warning("Altitude target not reached within %.0fs — continuing.", timeout)

    def _sleep_rate(self, rate_hz: float) -> None:
        time.sleep(1.0 / rate_hz)
