"""
RL Controller interface for benchmarking against PX4/GZ-X500.

IMPORTANT — How the RL controller actually works:
─────────────────────────────────────────────────
The RL policy (trained with rl-tools) is compiled as a PX4 *external module*
(`rl_tools_policy`).  It runs at 500 Hz **inside PX4**, not in Python.

When the user issues "rl_tools_commander overwrite on" in the MAVLink console:
  1. `RLtoolsCommander.overwrite` is set True → `command_active` becomes True.
  2. `RLtoolsCommander` publishes `rl_tools_command` uORB messages (target pos).
  3. `RLtoolsPolicy` reads those messages + sensor data and runs the neural net.
  4. `ActuatorMotorsMultiplexer` switches from RC/PID outputs to RL motor outputs.

This class is therefore a *remote controller* that:
  • Activates / deactivates the on-board RL module via MAVLink SERIAL_CONTROL.
  • Sends target position commands to the module via the same shell interface.
  • Does NOT perform any local inference.

Interface contract (mirrors PIDController so tests can swap controllers):
  activate()   — turn on the on-board RL
  deactivate() — turn off the on-board RL (smooth transition)
  set_target(x, y, z) — update the RL commander's position target
"""

import logging
import time
from typing import Optional, Tuple

from mavlink_interface.connection import MAVLinkConnection
from mavlink_interface.px4_shell import PX4Shell

logger = logging.getLogger(__name__)


class RLController:
    """
    Controls the on-board RL module (rl_tools_commander + rl_tools_policy)
    via MAVLink shell commands.

    Usage:
        ctrl = RLController(connection)
        ctrl.activate()
        ctrl.set_target(0.0, 0.0, -1.5)    # NED
        # ... test runs, telemetry logged externally ...
        ctrl.deactivate()
    """

    def __init__(self, connection: MAVLinkConnection):
        self.connection = connection
        self._shell = PX4Shell(connection, response_timeout=1.0)
        self._active = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def activate(self) -> None:
        """
        Activate the on-board RL controller.

        Equivalent to typing "rl_tools_commander overwrite on" in the MAVLink
        console.  The RL module will latch onto the current vehicle position as
        the initial target and begin flying immediately.
        """
        logger.info("Activating on-board RL controller...")
        response = self._shell.run("rl_tools_commander overwrite on")
        logger.debug("Shell response: %r", response)
        self._active = True
        # Give the module a moment to stabilise before tests begin
        time.sleep(0.3)
        logger.info("RL controller active.")

    def deactivate(self) -> None:
        """
        Deactivate the RL controller.

        The commander performs a smooth position-hold transition back to
        neutral before handing control back.
        """
        if not self._active:
            return
        logger.info("Deactivating on-board RL controller...")
        self._shell.run("rl_tools_commander overwrite off")
        self._active = False
        logger.info("RL controller deactivated.")

    def reset(self) -> None:
        """No-op: state is managed on the PX4 side."""
        pass

    # ------------------------------------------------------------------
    # Target position
    # ------------------------------------------------------------------

    def set_target(self, x: float, y: float, z: float) -> None:
        """
        Update the RL commander's position target (NED frame, metres).

        Sends a shell command to the rl_tools_commander module.
        NOTE: requires the `set_target` custom_command to be implemented
        in RLtoolsCommander.  If not present, the target is held at the
        activation position (default POSITION mode behaviour).

        Args:
            x, y, z: NED coordinates (z negative = altitude in metres)
        """
        if not self._active:
            logger.warning("set_target called while RL is not active — ignoring.")
            return
        cmd = f"rl_tools_commander set_target_position {x:.3f} {y:.3f} {z:.3f}"
        self._shell.run_nowait(cmd)

    # ------------------------------------------------------------------
    # Compatibility shim
    # ─────────────────
    # Tests call controller.compute(current, target) for PID and receive a
    # velocity command.  For the RL controller, compute() is a no-op on the
    # Python side — the RL policy already runs inside PX4 and does not need
    # Python to compute anything.  We return (0,0,0) as a sentinel so callers
    # can detect "RL mode" without branching on type().
    # ------------------------------------------------------------------

    def compute(
        self,
        current: Tuple[float, float, float],
        target: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """
        No-op for RL: inference happens inside PX4.

        Side effect: updates the target position of the on-board RL module.
        Returns (0, 0, 0) — callers should NOT send velocity setpoints when
        using the RL controller; use send_position_setpoint instead so PX4
        knows the target for its own logs.
        """
        self.set_target(*target)
        return 0.0, 0.0, 0.0

    @property
    def is_active(self) -> bool:
        return self._active
