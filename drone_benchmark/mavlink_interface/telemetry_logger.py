"""
Telemetry logging via MAVLink.

Runs a background thread that continuously reads MAVLink messages
and stores them in a thread-safe buffer.  A CSV writer periodically
flushes buffered rows to disk.

Logged messages:
  LOCAL_POSITION_NED   — position (x,y,z) + velocity (vx,vy,vz)
  ATTITUDE             — roll, pitch, yaw + rates
  ACTUATOR_OUTPUTS     — raw PWM / normalised motor commands
  BATTERY_STATUS       — voltage, current
"""

import csv
import logging
import os
import threading
import time
from dataclasses import dataclass, field, fields, astuple
from typing import Dict, List, Optional

from mavlink_interface.connection import MAVLinkConnection

logger = logging.getLogger(__name__)


@dataclass
class TelemetrySnapshot:
    timestamp: float = 0.0

    # LOCAL_POSITION_NED
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    vel_z: float = 0.0

    # ATTITUDE
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    rollspeed: float = 0.0
    pitchspeed: float = 0.0
    yawspeed: float = 0.0

    # Actuators (first 4 outputs — motors)
    motor0: float = 0.0
    motor1: float = 0.0
    motor2: float = 0.0
    motor3: float = 0.0

    # Setpoint (filled by tests, not from MAVLink)
    sp_x: float = 0.0
    sp_y: float = 0.0
    sp_z: float = 0.0


_CSV_HEADER = [f.name for f in fields(TelemetrySnapshot)]


class TelemetryLogger:
    """
    Continuously reads MAVLink telemetry and writes it to a CSV file.

    Usage:
        tlog = TelemetryLogger(connection, "logs/hover_pid.csv")
        tlog.start()
        # ... run test ...
        tlog.stop()
        snapshots = tlog.get_data()
    """

    def __init__(
        self,
        connection: MAVLinkConnection,
        csv_path: str,
        flush_interval: int = 10,
    ):
        self.conn = connection
        self.csv_path = csv_path
        self.flush_interval = flush_interval

        self._running = False
        self._lock = threading.Lock()
        self._latest = TelemetrySnapshot()
        self._buffer: List[TelemetrySnapshot] = []
        self._all_data: List[TelemetrySnapshot] = []

        self._recv_thread: Optional[threading.Thread] = None
        self._write_thread: Optional[threading.Thread] = None
        self._csv_file = None
        self._csv_writer = None

        # Current setpoint — updated externally by tests
        self.setpoint: tuple = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(_CSV_HEADER)
        self._running = True

        self._recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="telem-recv"
        )
        self._write_thread = threading.Thread(
            target=self._write_loop, daemon=True, name="telem-write"
        )
        self._recv_thread.start()
        self._write_thread.start()
        logger.info("Telemetry logger started → %s", self.csv_path)

    def stop(self) -> None:
        if not self._running and self._csv_file is None:
            return  # already stopped — idempotent
        self._running = False
        if self._recv_thread:
            self._recv_thread.join(timeout=2.0)
        if self._write_thread:
            self._write_thread.join(timeout=2.0)
        self._flush_buffer()
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
        logger.info("Telemetry logger stopped. %d rows written.", len(self._all_data))

    def get_latest(self) -> TelemetrySnapshot:
        with self._lock:
            import copy
            return copy.copy(self._latest)

    def get_data(self) -> List[TelemetrySnapshot]:
        with self._lock:
            return list(self._all_data)

    def update_setpoint(self, x: float, y: float, z: float) -> None:
        self.setpoint = (x, y, z)

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        """
        Receives MAVLink messages and merges them into the latest snapshot.
        Non-blocking approach: iterate over any available messages quickly.
        """
        master = self.conn.master
        while self._running:
            try:
                msg = master.recv_match(blocking=True, timeout=0.05)
                if msg is None:
                    continue
                self._process_message(msg)
            except Exception as exc:
                if self._running:
                    logger.warning("Telemetry recv error: %s", exc)

    def _process_message(self, msg) -> None:
        mtype = msg.get_type()
        with self._lock:
            snap = self._latest
            snap.timestamp = time.time()
            sp = self.setpoint
            snap.sp_x, snap.sp_y, snap.sp_z = sp

            if mtype == "LOCAL_POSITION_NED":
                snap.pos_x = msg.x
                snap.pos_y = msg.y
                snap.pos_z = msg.z
                snap.vel_x = msg.vx
                snap.vel_y = msg.vy
                snap.vel_z = msg.vz
                # Snapshot on every position update (primary loop rate)
                import copy
                row = copy.copy(snap)
                self._buffer.append(row)
                self._all_data.append(row)

            elif mtype == "ATTITUDE":
                snap.roll = msg.roll
                snap.pitch = msg.pitch
                snap.yaw = msg.yaw
                snap.rollspeed = msg.rollspeed
                snap.pitchspeed = msg.pitchspeed
                snap.yawspeed = msg.yawspeed

            elif mtype == "ACTUATOR_OUTPUTS":
                # PX4 publishes up to 16 actuator outputs; first 4 are motors
                outputs = msg.output
                snap.motor0 = outputs[0] if len(outputs) > 0 else 0.0
                snap.motor1 = outputs[1] if len(outputs) > 1 else 0.0
                snap.motor2 = outputs[2] if len(outputs) > 2 else 0.0
                snap.motor3 = outputs[3] if len(outputs) > 3 else 0.0

    def _write_loop(self) -> None:
        while self._running:
            time.sleep(0.5)
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        with self._lock:
            rows = self._buffer[:]
            self._buffer.clear()
        if rows and self._csv_writer:
            for snap in rows:
                self._csv_writer.writerow(astuple(snap))
            self._csv_file.flush()
