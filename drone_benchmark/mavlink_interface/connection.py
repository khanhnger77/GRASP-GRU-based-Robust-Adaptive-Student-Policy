"""
MAVLink connection management.

Handles connecting to PX4 via pymavlink, waiting for heartbeat,
and providing a shared MAVLink master object to all subsystems.
"""

import time
import threading
import logging
from typing import Optional

from pymavlink import mavutil

logger = logging.getLogger(__name__)


class MAVLinkConnection:
    """
    Manages a single MAVLink UDP connection to PX4.

    Usage:
        conn = MAVLinkConnection("udp:localhost:14550")
        conn.connect()
        master = conn.master
        conn.close()
    """

    def __init__(self, connection_string: str, heartbeat_timeout: float = 10.0):
        self.connection_string = connection_string
        self.heartbeat_timeout = heartbeat_timeout
        self.master: Optional[mavutil.mavlink_connection] = None
        self._lock = threading.Lock()
        self._connected = False

    def connect(self) -> None:
        """Open connection and wait for first heartbeat from the vehicle."""
        logger.info("Connecting to %s ...", self.connection_string)
        self.master = mavutil.mavlink_connection(self.connection_string)

        logger.info("Waiting for heartbeat (timeout=%ss)...", self.heartbeat_timeout)
        msg = self.master.wait_heartbeat(timeout=self.heartbeat_timeout)
        if msg is None:
            raise ConnectionError(
                f"No heartbeat received within {self.heartbeat_timeout}s "
                f"from {self.connection_string}"
            )

        self._connected = True
        logger.info(
            "Heartbeat received — system %d component %d",
            self.master.target_system,
            self.master.target_component,
        )

    def is_connected(self) -> bool:
        return self._connected and self.master is not None

    def close(self) -> None:
        if self.master is not None:
            self.master.close()
            self._connected = False
            logger.info("MAVLink connection closed.")

    def request_streams(self, rate_hz: int = 50) -> None:
        """
        Ask PX4 to stream telemetry messages at the requested rate.

        PX4 SITL does NOT push all messages by default over UDP — we must
        explicitly request them.  Uses both the legacy REQUEST_DATA_STREAM
        and the newer SET_MESSAGE_INTERVAL so it works across PX4 versions.

        Call this once after connect(), before starting tests.
        """
        if self.master is None:
            return

        interval_us = int(1e6 / rate_hz)  # microseconds between messages
        sys  = self.master.target_system
        comp = self.master.target_component

        # ── Legacy REQUEST_DATA_STREAM (works on all PX4 versions) ──────────
        stream_ids = [
            mavutil.mavlink.MAV_DATA_STREAM_ALL,            # everything
            mavutil.mavlink.MAV_DATA_STREAM_POSITION,       # LOCAL_POSITION_NED
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,         # ATTITUDE
            mavutil.mavlink.MAV_DATA_STREAM_RAW_CONTROLLER, # ACTUATOR_CONTROL_TARGET
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA3,         # ACTUATOR_OUTPUTS (PX4)
        ]
        for stream_id in stream_ids:
            self.master.mav.request_data_stream_send(
                sys, comp, stream_id, rate_hz, 1  # 1 = start
            )
            time.sleep(0.02)

        # ── MAVLink 2 SET_MESSAGE_INTERVAL for key message IDs ───────────────
        # These IDs are stable across PX4 versions.
        message_ids = {
            "LOCAL_POSITION_NED":      32,
            "ATTITUDE":                30,
            "ACTUATOR_CONTROL_TARGET": 140,
            "ACTUATOR_OUTPUT_STATUS":  375,   # PX4-specific (ACTUATOR_OUTPUTS)
            "BATTERY_STATUS":          147,
        }
        for name, msg_id in message_ids.items():
            self.master.mav.command_long_send(
                sys, comp,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,              # confirmation
                float(msg_id),  # param1: message ID
                float(interval_us),  # param2: interval in µs
                0, 0, 0, 0, 0,
            )
            time.sleep(0.02)

        logger.info("Telemetry streams requested at %d Hz.", rate_hz)

    def recv_match(self, msg_type: str, blocking: bool = True, timeout: float = 1.0):
        """
        Helper: receive a specific MAVLink message.

        Args:
            msg_type: e.g. "LOCAL_POSITION_NED"
            blocking: whether to block until received
            timeout: seconds before returning None
        """
        if self.master is None:
            return None
        return self.master.recv_match(type=msg_type, blocking=blocking, timeout=timeout)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()
