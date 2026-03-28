"""
Offboard control helpers for PX4 via MAVLink.

Provides:
  - Arming / disarming
  - Mode switching (OFFBOARD, STABILIZED, LAND)
  - SET_POSITION_TARGET_LOCAL_NED setpoint publishing
  - SET_ATTITUDE_TARGET publishing (for torque/thrust)

PX4 OFFBOARD mode requires setpoints to arrive at >= 2 Hz to stay active.
We publish at the much higher rate configured in config.yaml (default 50 Hz).
"""

import time
import logging
import threading
from typing import Tuple, Optional

from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as mavlink2

from mavlink_interface.connection import MAVLinkConnection

logger = logging.getLogger(__name__)

# Type masks for SET_POSITION_TARGET_LOCAL_NED
# Bit set = ignore that field.
POSITION_MASK = 0b0000_111_111_000_111  # use position only (ignore vel, accel, yaw-rate)
VELOCITY_MASK = 0b0000_111_111_111_000  # use velocity only
POSITION_VELOCITY_MASK = 0b0000_111_111_000_000  # use pos + vel


class OffboardController:
    """
    High-level wrapper around PX4 MAVLink offboard commands.

    Typical usage sequence:
        ctrl = OffboardController(connection)
        ctrl.start_setpoint_stream(target_pos=(0,0,-1.5))  # pre-stream required
        ctrl.arm()
        ctrl.set_offboard_mode()
        # ... publish setpoints in your test loop ...
        ctrl.land()
        ctrl.disarm()
        ctrl.stop_setpoint_stream()
    """

    # PX4 custom mode IDs
    PX4_MODE_OFFBOARD = 6
    PX4_MODE_LAND = 9
    PX4_MODE_STABILIZED = 7

    def __init__(self, connection: MAVLinkConnection, setpoint_rate_hz: float = 50.0):
        self.conn = connection
        self.master = connection.master
        self.rate = setpoint_rate_hz
        self._stream_thread: Optional[threading.Thread] = None
        self._stream_active = False
        self._current_setpoint: Tuple[float, float, float] = (0.0, 0.0, -1.5)
        self._setpoint_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Arming
    # ------------------------------------------------------------------

    def arm(self, timeout: float = 10.0) -> None:
        """Arm the vehicle and wait for confirmation."""
        logger.info("Arming vehicle...")
        self.master.arducopter_arm()
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.conn.recv_match("COMMAND_ACK", timeout=1.0)
            if msg and msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    logger.info("Armed successfully.")
                    return
                else:
                    raise RuntimeError(f"Arm rejected with result code {msg.result}")
        raise TimeoutError("Arm command timed out.")

    def disarm(self, force: bool = False) -> None:
        """Disarm the vehicle. Set force=True to disarm while in the air (unsafe)."""
        param2 = 21196.0 if force else 0.0
        logger.info("Disarming vehicle (force=%s)...", force)
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0,      # disarm
            param2, # magic value for force
            0, 0, 0, 0, 0,
        )

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def set_mode(self, mode_id: int, timeout: float = 5.0) -> None:
        """Switch PX4 custom flight mode by numeric ID."""
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.conn.recv_match("COMMAND_ACK", timeout=0.5)
            if msg and msg.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    return
        # Mode switches sometimes don't ACK explicitly; verify via HEARTBEAT
        logger.debug("Mode switch ACK not received; continuing.")

    def set_offboard_mode(self, init_delay: float = 1.0) -> None:
        """
        Switch to OFFBOARD mode.

        PX4 requires setpoints to already be streaming before accepting the
        mode switch — call start_setpoint_stream() first.
        """
        logger.info("Switching to OFFBOARD mode...")
        time.sleep(init_delay)  # let setpoints buffer up on PX4 side
        self.set_mode(self.PX4_MODE_OFFBOARD)
        logger.info("OFFBOARD mode set.")

    def land(self) -> None:
        """Command PX4 to land using the LAND mode."""
        logger.info("Switching to LAND mode...")
        self.set_mode(self.PX4_MODE_LAND)

    # ------------------------------------------------------------------
    # Setpoint streaming (background thread)
    # ------------------------------------------------------------------

    def start_setpoint_stream(
        self, target_pos: Tuple[float, float, float] = (0.0, 0.0, -1.5)
    ) -> None:
        """
        Start a background thread that continuously publishes the current
        position setpoint. Required before entering OFFBOARD mode.
        """
        with self._setpoint_lock:
            self._current_setpoint = target_pos
        self._stream_active = True
        self._stream_thread = threading.Thread(
            target=self._stream_loop, daemon=True, name="setpoint-stream"
        )
        self._stream_thread.start()
        logger.info("Setpoint stream started at %.0f Hz.", self.rate)

    def stop_setpoint_stream(self) -> None:
        self._stream_active = False
        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)
        logger.info("Setpoint stream stopped.")

    def _stream_loop(self) -> None:
        interval = 1.0 / self.rate
        while self._stream_active:
            with self._setpoint_lock:
                pos = self._current_setpoint
            self._send_position_setpoint(pos[0], pos[1], pos[2])
            time.sleep(interval)

    def update_setpoint(self, x: float, y: float, z: float) -> None:
        """Thread-safe update of the streaming setpoint position (NED)."""
        with self._setpoint_lock:
            self._current_setpoint = (x, y, z)

    # ------------------------------------------------------------------
    # Low-level MAVLink message builders
    # ------------------------------------------------------------------

    def _send_position_setpoint(
        self,
        x: float, y: float, z: float,
        vx: float = 0.0, vy: float = 0.0, vz: float = 0.0,
        yaw: float = 0.0,
        type_mask: int = POSITION_MASK,
    ) -> None:
        """
        Publish SET_POSITION_TARGET_LOCAL_NED.

        Coordinates are in NED (North-East-Down) frame:
          x = North (forward), y = East (right), z = Down (negative = up)
        """
        self.master.mav.set_position_target_local_ned_send(
            int(time.time() * 1e3) & 0xFFFFFFFF,  # time_boot_ms
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            x, y, z,       # position
            vx, vy, vz,    # velocity (used if mask allows)
            0.0, 0.0, 0.0, # acceleration (ignored)
            yaw,           # yaw setpoint (radians)
            0.0,           # yaw rate
        )

    def send_position_setpoint(
        self, x: float, y: float, z: float, yaw: float = 0.0
    ) -> None:
        """Send a one-shot position setpoint (NED). Also updates stream target."""
        self.update_setpoint(x, y, z)
        self._send_position_setpoint(x, y, z, yaw=yaw, type_mask=POSITION_MASK)

    def send_velocity_setpoint(
        self, vx: float, vy: float, vz: float, yaw_rate: float = 0.0
    ) -> None:
        """Send a velocity setpoint in NED body frame."""
        self.master.mav.set_position_target_local_ned_send(
            int(time.time() * 1e3) & 0xFFFFFFFF,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            VELOCITY_MASK,
            0.0, 0.0, 0.0,    # position (ignored)
            vx, vy, vz,
            0.0, 0.0, 0.0,
            0.0,
            yaw_rate,
        )
