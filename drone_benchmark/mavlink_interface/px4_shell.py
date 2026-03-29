"""
PX4 NSH Shell access via MAVLink SERIAL_CONTROL.

PX4 exposes an interactive shell over MAVLink using the SERIAL_CONTROL
message (ID 126) with device=SERIAL_CONTROL_DEV_SHELL (10).

This is how QGroundControl's MAVLink Console works, and how we can send
module commands like "rl_tools_commander overwrite on" from Python.

Wire protocol:
  - Send: SERIAL_CONTROL with count=N, data=cmd_bytes[:70], padded to 70 bytes
  - Response: one or more SERIAL_CONTROL messages echoing output back
"""

import logging
import time
from typing import Optional

from pymavlink import mavutil

from mavlink_interface.connection import MAVLinkConnection

logger = logging.getLogger(__name__)

# SERIAL_CONTROL device IDs (from mavlink common.xml)
SERIAL_CONTROL_DEV_SHELL = 10

# SERIAL_CONTROL flags (MAVLink spec)
# REPLY=1     → set by PX4 on *responses*, NOT by GCS on commands
# RESPOND=2   → GCS sets this to request PX4 to send output back
# EXCLUSIVE=4 → take exclusive shell access (prevents QGC interference)
SERIAL_CONTROL_FLAG_REPLY     = 1
SERIAL_CONTROL_FLAG_RESPOND   = 2
SERIAL_CONTROL_FLAG_EXCLUSIVE = 4

_MAX_DATA = 70  # SERIAL_CONTROL.data field is fixed 70 bytes


class PX4Shell:
    """
    Send commands to PX4's NSH shell via MAVLink SERIAL_CONTROL.

    Usage:
        shell = PX4Shell(connection)
        output = shell.run("rl_tools_commander overwrite on")
    """

    def __init__(self, connection: MAVLinkConnection, response_timeout: float = 2.0):
        self.conn = connection
        self.master = connection.master
        self.response_timeout = response_timeout

    def run(self, command: str, wait_for_response: bool = True) -> str:
        """
        Send a shell command to PX4 and return the response string.

        Args:
            command:           NSH command string, e.g. "rl_tools_commander overwrite on"
            wait_for_response: if True, block and collect output until timeout

        Returns:
            Response text from PX4 (may be empty if command produces no output).
        """
        # Append newline so PX4 shell executes the line
        cmd = (command.strip() + "\n").encode("ascii")

        # MAVLink SERIAL_CONTROL.data is exactly 70 bytes; split into chunks if needed
        chunks = [cmd[i : i + _MAX_DATA] for i in range(0, len(cmd), _MAX_DATA)]

        for chunk in chunks:
            data = list(chunk) + [0] * (_MAX_DATA - len(chunk))
            self.master.mav.serial_control_send(
                device=SERIAL_CONTROL_DEV_SHELL,
                flags=SERIAL_CONTROL_FLAG_EXCLUSIVE | SERIAL_CONTROL_FLAG_RESPOND,
                timeout=0,
                baudrate=0,
                count=len(chunk),
                data=data,
            )

        logger.debug("PX4 shell → %r", command)

        if not wait_for_response:
            return ""

        return self._collect_response()

    def _collect_response(self) -> str:
        """
        Collect SERIAL_CONTROL response messages until timeout.
        Returns accumulated response text.
        """
        output = []
        deadline = time.time() + self.response_timeout
        while time.time() < deadline:
            msg = self.conn.recv_match("SERIAL_CONTROL", blocking=True, timeout=0.1)
            if msg is None:
                break
            if msg.count > 0:
                text = bytes(msg.data[: msg.count]).decode("ascii", errors="replace")
                output.append(text)
                logger.debug("PX4 shell ← %r", text)
        return "".join(output)

    def run_nowait(self, command: str) -> None:
        """Fire-and-forget: send command without waiting for response."""
        self.run(command, wait_for_response=False)
