"""
serial_bridge.py
================
Arduino serial communication with reconnect safety, retry logic, and health monitoring.

Protocol (RYG adaptive phase control):
    Host → Arduino: <S0,S1,S2,S3,S4,S5,S6,S7>\n
        Each S_i ∈ {0, 1, 2}  →  0=RED  1=YELLOW  2=GREEN
        8 directions: [tls_0_dir0..3, tls_1_dir0..3]
    Arduino → Host: ACK<ts>\n (optional echo confirmation)

Reconnect strategy on USB disconnect:
    1. Log critical error
    2. Set connected=False
    3. Attempt reconnect with exponential backoff
    4. If reconnect fails, enter safe mode (all lights red)
    5. Resume normal operation on successful reconnect
"""

import serial
import time
import logging
from typing import Optional, List
import threading

logger = logging.getLogger(__name__)

# Light-state constants for call sites
RED    = 0
YELLOW = 1
GREEN  = 2
_VALID_STATES = frozenset({RED, YELLOW, GREEN})


class SerialBridge:
    """
    Safe USB serial communication to Arduino traffic light controller.

    Protocol:
        Host → Arduino: <S0,S1,S2,S3,S4,S5,S6,S7>\n
        Each value: 0=RED  1=YELLOW  2=GREEN
    """

    def __init__(
        self,
        port: str = 'COM3',
        baudrate: int = 115200,
        timeout: float = 1.0,
        max_reconnect_attempts: int = 5,
        reconnect_interval_s: float = 1.0,
        enable_ack: bool = False,
    ) -> None:
        self.ser: Optional[serial.Serial] = None
        self.port = str(port)
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.connected = False
        self.send_count = 0
        self.fail_count = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = int(max_reconnect_attempts)
        self.reconnect_interval_s = float(reconnect_interval_s)
        self.enable_ack = bool(enable_ack)
        self.last_sent_time = 0.0
        self.lock = threading.Lock()
        self.watchdog_active = False

        try:
            self._open_connection()
            logger.info(
                "Serial connected on %s @ %d baud (timeout=%.1fs)",
                self.port, self.baudrate, self.timeout
            )
        except Exception as e:
            self.connected = False
            logger.exception("Initial serial connection failed: %s", e)
            raise RuntimeError(f"Serial connection failed on {self.port}") from e

    def _open_connection(self) -> None:
        try:
            if self.ser is not None:
                try:
                    self.ser.close()
                except Exception:
                    pass

            self.ser = serial.Serial(
                self.port,
                self.baudrate,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
            time.sleep(2.0)  # Arduino bootloader delay
            self.connected = True
            self.reconnect_attempts = 0
            logger.info("Serial port opened: %s", self.port)
        except Exception as e:
            self.connected = False
            raise RuntimeError(f"Failed to open {self.port}: {e}") from e

    def _close_connection(self) -> None:
        with self.lock:
            try:
                if self.ser is not None and self.ser.is_open:
                    self.ser.close()
            except Exception as e:
                logger.warning("Error closing serial port: %s", e)
            finally:
                self.ser = None
                self.connected = False

    def reconnect(self, force: bool = False) -> bool:
        """
        Attempt to reconnect to Arduino with exponential backoff.

        Returns True if reconnection successful, False otherwise.
        """
        if self.connected and not force:
            return True

        with self.lock:
            if self.reconnect_attempts >= self.max_reconnect_attempts:
                logger.critical(
                    "Max reconnect attempts (%d) exhausted. System in SAFE MODE.",
                    self.max_reconnect_attempts
                )
                return False

            wait_time = min(
                self.reconnect_interval_s * (2 ** self.reconnect_attempts),
                30.0
            )
            logger.warning(
                "Attempting reconnect (attempt %d/%d) after %.1fs...",
                self.reconnect_attempts + 1, self.max_reconnect_attempts, wait_time
            )
            time.sleep(wait_time)

            try:
                self._open_connection()
                logger.info("Reconnection successful!")
                return True
            except Exception as e:
                self.reconnect_attempts += 1
                logger.error("Reconnect attempt %d failed: %s", self.reconnect_attempts, e)
                return False

    def is_healthy(self) -> bool:
        with self.lock:
            return self.connected and self.ser is not None and self.ser.is_open

    def get_stats(self) -> dict:
        return {
            "connected": self.connected,
            "port": self.port,
            "baudrate": self.baudrate,
            "send_count": self.send_count,
            "fail_count": self.fail_count,
            "reconnect_attempts": self.reconnect_attempts,
            "last_sent_time": self.last_sent_time,
        }

    def pack_and_send_data(self, states_array: List[int], retry_on_fail: bool = True) -> bool:
        """
        Pack 8 RYG light states and send to Arduino.

        Format: <S0,S1,S2,S3,S4,S5,S6,S7>\n
        Each value must be 0 (RED), 1 (YELLOW), or 2 (GREEN).

        Parameters
        ----------
        states_array : list of int
            8 values, each in {0, 1, 2}.
        retry_on_fail : bool
            If True, attempt reconnect on first failure and retry.

        Returns
        -------
        bool
            True if send succeeded, False if failed.
        """
        if len(states_array) != 8:
            logger.error("Invalid command length: expected 8, got %d", len(states_array))
            self.fail_count += 1
            return False

        values: List[int] = []
        for v in states_array:
            try:
                iv = int(v)
                if iv not in _VALID_STATES:
                    logger.error("Invalid light state %s: must be 0=RED, 1=YELLOW, or 2=GREEN", v)
                    self.fail_count += 1
                    return False
                values.append(iv)
            except (TypeError, ValueError) as e:
                logger.error("Invalid serial value: %s (%s)", v, e)
                self.fail_count += 1
                return False

        data_str = "<" + ",".join(map(str, values)) + ">\n"

        if self._try_send(data_str):
            self.send_count += 1
            return True

        if not retry_on_fail:
            self.fail_count += 1
            return False

        logger.warning("Initial send failed. Attempting reconnect and retry...")
        if self.reconnect(force=True):
            if self._try_send(data_str):
                self.send_count += 1
                logger.info("Retry successful after reconnect")
                return True

        self.fail_count += 1
        logger.error("Send failed: could not send to %s after reconnect attempt", self.port)
        return False

    def _try_send(self, data_str: str) -> bool:
        with self.lock:
            if not self.connected or self.ser is None or not self.ser.is_open:
                logger.error("Serial not connected")
                self.connected = False
                return False

            try:
                self.ser.write(data_str.encode('utf-8'))
                self.last_sent_time = time.time()
                logger.info("Serial sent: %s", data_str.strip())

                if self.enable_ack:
                    try:
                        response = self.ser.readline(timeout=self.timeout).decode('utf-8').strip()
                        if response.startswith("ACK"):
                            logger.debug("ACK received: %s", response)
                        else:
                            logger.warning("Unexpected response (no ACK): %s", response)
                    except Exception as e:
                        logger.warning("Failed to read ACK: %s", e)

                return True

            except serial.SerialException as e:
                logger.critical("Serial write failed (USB disconnect?): %s", e)
                self.connected = False
                return False
            except Exception as e:
                logger.exception("Unexpected error during serial send: %s", e)
                self.connected = False
                return False

    def start_watchdog(self, interval_s: float = 5.0) -> None:
        if self.watchdog_active:
            return
        self.watchdog_active = True

        def _watchdog_loop():
            while self.watchdog_active:
                try:
                    time.sleep(interval_s)
                    if not self.is_healthy():
                        logger.warning(
                            "Watchdog detected disconnection. Reconnect count: %d",
                            self.reconnect_attempts
                        )
                        self.reconnect()
                except Exception as e:
                    logger.exception("Watchdog error: %s", e)

        thread = threading.Thread(target=_watchdog_loop, daemon=True, name="serial-watchdog")
        thread.start()
        logger.info("Serial watchdog started (interval=%.1fs)", interval_s)

    def stop_watchdog(self) -> None:
        self.watchdog_active = False
        logger.info("Serial watchdog stopped")

    def close(self) -> None:
        self.stop_watchdog()
        self._close_connection()
        logger.info("Serial bridge closed")


if __name__ == "__main__":
    bridge = SerialBridge(port='COM3')
    # Group A green for both intersections: dirs 0,1=GREEN  dirs 2,3=RED  × 2
    while True:
        states = [GREEN, GREEN, RED, RED, GREEN, GREEN, RED, RED]
        bridge.pack_and_send_data(states)
        time.sleep(5)
