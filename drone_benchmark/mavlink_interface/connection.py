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
