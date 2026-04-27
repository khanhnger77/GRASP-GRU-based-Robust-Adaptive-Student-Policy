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
        ctrl.activate(x=0.0, y=0.0, z=-1.5)   # pre-set target to avoid jerk
        ctrl.set_target(0.0, 0.0, -1.5)
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

    def activate(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        warmup_secs: float = 0.3,
    ) -> None:
        """
        Activate the on-board RL controller.

        If (x, y, z) are provided, the RL target is pre-set to that position
        *before* overwrite is enabled.  This prevents a jerk when the RL policy
        first takes control because it will already be targeting where the drone
        currently hovers.

        Equivalent to:
            rl_tools_commander set_target_position x y z
            rl_tools_commander overwrite on
        """
        logger.info("Activating on-board RL controller...")

        if x is not None and y is not None and z is not None:
            cmd = f"rl_tools_commander set_target_position {x:.3f} {y:.3f} {z:.3f}"
            self._shell.run(cmd)
            logger.debug("Pre-set RL target to (%.3f, %.3f, %.3f)", x, y, z)
            time.sleep(0.05)

        response = self._shell.run("rl_tools_commander overwrite on")
        logger.debug("Shell response: %r", response)
        self._active = True
        # Give the module time to run several policy cycles before tests begin.
        # Baseline (no payload): 0.3 s is sufficient.
        # Payload scenarios: caller should pass warmup_secs >= 2.0 via
        #   _safe_rl_activate_payload() which handles this with active monitoring.
        time.sleep(warmup_secs)
        logger.info("RL controller active (warmup=%.1fs).", warmup_secs)

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

    def force_overwrite_off(self) -> None:
        """
        Best-effort safety guard: force overwrite OFF on PX4 even if this
        Python object thinks RL is inactive.

        Useful at test startup after previous interrupted runs.
        """
        logger.info("Ensuring RL overwrite is OFF before takeoff...")
        try:
            self._shell.run("rl_tools_commander overwrite off")
        finally:
            self._active = False

    def reset(self) -> None:
        """No-op: state is managed on the PX4 side."""
        pass

    # ------------------------------------------------------------------
    # Target position
    # ------------------------------------------------------------------

    def set_target(self, x: float, y: float, z: float) -> None:
        """
        Update the RL commander's position target (NED frame, metres).

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
        using the RL controller.
        """
        self.set_target(*target)
        return 0.0, 0.0, 0.0

    @property
    def is_active(self) -> bool:
        return self._active
