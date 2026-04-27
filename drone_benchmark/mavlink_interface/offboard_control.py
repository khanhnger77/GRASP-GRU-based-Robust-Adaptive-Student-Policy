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

# Type masks for SET_POSITION_TARGET_LOCAL_NED (MAVLink POSITION_TARGET_TYPEMASK).
# A bit value of 1 means "ignore this field".
#
# Bit layout (0-indexed from LSB):
#   bit 0 = x pos   bit 1 = y pos   bit 2 = z pos
#   bit 3 = vx      bit 4 = vy      bit 5 = vz
#   bit 6 = ax      bit 7 = ay      bit 8 = az
#   bit 9 = force   bit 10 = yaw    bit 11 = yaw_rate
#
# POSITION only  → ignore vel(3-5), accel(6-8), force(9), yaw(10), yaw_rate(11)
#                  bits 3..11 set, bits 0..2 clear
POSITION_MASK = 0b0000_111_111_111_000  # = 0xDF8 = 3576  ← USE x y z

# VELOCITY only  → ignore pos(0-2), accel(6-8), force(9), yaw(10), yaw_rate(11)
#                  bits 0..2 and 6..11 set, bits 3..5 clear
VELOCITY_MASK = 0b0000_111_111_000_111  # = 0xDC7 = 3527  ← USE vx vy vz

# POSITION + VELOCITY  → ignore accel(6-8), force(9), yaw(10), yaw_rate(11)
POSITION_VELOCITY_MASK = 0b0000_111_111_000_000  # = 0xDC0 = 3520  ← USE pos + vel

# POSITION + YAW  → bit 10 cleared so PX4 tracks the yaw setpoint
POSITION_YAW_MASK = 0b0000_101_111_111_000       # = 0xBF8 = 3064  ← USE x y z yaw


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
        self._current_yaw: float = 0.0
        self._setpoint_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Arming
    # ------------------------------------------------------------------

    def wait_for_ekf(self, timeout: float = 30.0) -> None:
        """
        Wait until EKF2 has a valid position estimate.

        PX4 will reject arm with MAV_RESULT_TEMPORARILY_REJECTED (code 1) if
        EKF2 hasn't converged yet.  We wait for LOCAL_POSITION_NED messages
        to start arriving — that indicates EKF is running and position is valid.
        """
        logger.info("Waiting for EKF2 position estimate (timeout=%.0fs)...", timeout)
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.conn.recv_match("LOCAL_POSITION_NED", timeout=1.0)
            if msg is not None:
                logger.info("EKF2 ready — pos (%.2f, %.2f, %.2f)", msg.x, msg.y, msg.z)
                return
        logger.warning("EKF2 ready-check timed out — trying to arm anyway.")

    def arm(self, timeout: float = 20.0, retry_interval: float = 1.0) -> None:
        """
        Arm the vehicle, retrying on MAV_RESULT_TEMPORARILY_REJECTED.

        PX4 returns code 1 (TEMPORARILY_REJECTED) when pre-arm checks are
        still running (EKF not ready, sensors warming up, etc.).  We retry
        at `retry_interval` seconds until `timeout` expires.
        """
        logger.info("Arming vehicle...")
        deadline = time.time() + timeout
        attempt  = 0

        while time.time() < deadline:
            attempt += 1
            # Use COMMAND_LONG directly for reliable cross-firmware arming
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,      # confirmation
                1.0,    # param1: 1 = arm
                0.0,    # param2: 0 = normal arm (not force)
                0, 0, 0, 0, 0,
            )

            msg = self.conn.recv_match("COMMAND_ACK", timeout=2.0)
            if msg and msg.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    logger.info("Armed successfully (attempt %d).", attempt)
                    return
                elif msg.result == 1:  # MAV_RESULT_TEMPORARILY_REJECTED
                    logger.info(
                        "Arm temporarily rejected (attempt %d) — EKF/pre-arm "
                        "checks not ready, retrying in %.1fs...", attempt, retry_interval
                    )
                    time.sleep(retry_interval)
                    continue
                else:
                    # Permanent rejection (wrong mode, safety switch, etc.)
                    raise RuntimeError(
                        f"Arm permanently rejected (result={msg.result}). "
                        "Check PX4 pre-arm status in QGroundControl."
                    )
            # No ACK yet — keep waiting
            time.sleep(0.2)

        raise TimeoutError(
            f"Arm timed out after {timeout}s ({attempt} attempts). "
            "Tip: check PX4 console with 'commander status'."
        )

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
        """
        Switch PX4 custom flight mode using MAV_CMD_DO_SET_MODE (COMMAND_LONG).

        WHY NOT set_mode_send():
          The old SET_MODE message encodes custom_mode as a uint32 where PX4
          reads main_mode = (custom_mode >> 16) & 0xFF.  Passing mode_id=6
          gives main_mode = 0 (MANUAL), not OFFBOARD.

        MAV_CMD_DO_SET_MODE via COMMAND_LONG passes main_mode directly in
        param2 as a plain integer — no bit-shifting needed.

          param1 = MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
          param2 = main_mode  (6=OFFBOARD, 9=LAND, 7=STABILIZED …)
          param3 = sub_mode   (0 for all modes we use)
        """
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE,
            0,                                                    # confirmation
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,   # param1 base mode
            float(mode_id),                                       # param2 main mode
            0.0,                                                  # param3 sub mode
            0.0, 0.0, 0.0, 0.0,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.conn.recv_match("COMMAND_ACK", timeout=0.5)
            if msg and msg.command == mavutil.mavlink.MAV_CMD_DO_SET_MODE:
                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    logger.debug("Mode %d accepted.", mode_id)
                    return
                else:
                    logger.warning("Mode switch rejected (result=%d).", msg.result)
                    return
        logger.debug("Mode switch ACK timeout — assuming accepted.")

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

    def land_via_setpoint(self, timeout: float = 20.0) -> None:
        """
        Land by sending a position setpoint at ground level (z=0 NED) while
        staying in OFFBOARD mode — no mode switch, no jerk.

        Reads current x/y so the drone descends straight down.
        Returns once z > -0.25 m (near ground) or timeout.
        """
        logger.info("Landing via position setpoint (no mode switch)...")
        msg = self.conn.recv_match("LOCAL_POSITION_NED", timeout=1.0)
        cur_x = msg.x if msg else 0.0
        cur_y = msg.y if msg else 0.0

        # Target: stay over current x/y, descend to ground (z=0 NED)
        self.update_setpoint(cur_x, cur_y, 0.0)

        deadline = time.time() + timeout
        while time.time() < deadline:
            m = self.conn.recv_match("LOCAL_POSITION_NED", timeout=0.5)
            if m and m.z > -0.25:   # within 25 cm of ground
                break
        logger.info("Landed (setpoint method).")

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
                yaw = self._current_yaw
            self._send_position_setpoint(
                pos[0], pos[1], pos[2], yaw=yaw, type_mask=POSITION_YAW_MASK
            )
            time.sleep(interval)

    def update_setpoint(self, x: float, y: float, z: float) -> None:
        """Thread-safe update of the streaming setpoint position (NED)."""
        with self._setpoint_lock:
            self._current_setpoint = (x, y, z)

    def update_yaw(self, yaw_rad: float) -> None:
        """Thread-safe update of the yaw setpoint (radians, NED convention: 0=North)."""
        with self._setpoint_lock:
            self._current_yaw = yaw_rad

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
