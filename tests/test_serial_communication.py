"""
tests/test_serial_communication.py
===================================
Test serial bridge safety, reconnect logic, and RYG state validation.

Protocol: <S0,S1,...,S7>\n  where each S_i in {0=RED, 1=YELLOW, 2=GREEN}
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import time

from src.utils.serial_bridge import SerialBridge, RED, YELLOW, GREEN


class TestSerialBridge:
    """Serial communication safety tests."""

    def test_initialization_valid_port(self):
        with patch('serial.Serial') as mock_serial_class:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            mock_serial_class.return_value = mock_instance

            bridge = SerialBridge(port='COM3', baudrate=115200, timeout=1.0)

            assert bridge.port == 'COM3'
            assert bridge.baudrate == 115200
            assert bridge.connected == True
            assert bridge.send_count == 0
            assert bridge.fail_count == 0

    def test_pack_and_send_valid(self):
        """Group A green for both TLS: dirs 0,1=GREEN  dirs 2,3=RED × 2."""
        with patch('serial.Serial') as mock_serial_class:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            mock_serial_class.return_value = mock_instance

            bridge = SerialBridge(port='COM3')

            # Phase 0 both TLS: Group A (dirs 0,1) GREEN, Group B (dirs 2,3) RED
            states = [GREEN, GREEN, RED, RED, GREEN, GREEN, RED, RED]
            ok = bridge.pack_and_send_data(states, retry_on_fail=False)

            assert ok == True
            assert bridge.send_count == 1
            assert bridge.fail_count == 0

            mock_instance.write.assert_called()
            call_bytes = mock_instance.write.call_args[0][0]
            assert b"<2,2,0,0,2,2,0,0>" in call_bytes

    def test_pack_and_send_phase1(self):
        """Group B green for both TLS: dirs 0,1=RED  dirs 2,3=GREEN × 2."""
        with patch('serial.Serial') as mock_serial_class:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            mock_serial_class.return_value = mock_instance

            bridge = SerialBridge(port='COM3')

            states = [RED, RED, GREEN, GREEN, RED, RED, GREEN, GREEN]
            ok = bridge.pack_and_send_data(states, retry_on_fail=False)

            assert ok == True
            call_bytes = mock_instance.write.call_args[0][0]
            assert b"<0,0,2,2,0,0,2,2>" in call_bytes

    def test_pack_and_send_yellow_transition(self):
        """Yellow state is valid."""
        with patch('serial.Serial') as mock_serial_class:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            mock_serial_class.return_value = mock_instance

            bridge = SerialBridge(port='COM3')

            states = [YELLOW, YELLOW, RED, RED, GREEN, GREEN, RED, RED]
            ok = bridge.pack_and_send_data(states, retry_on_fail=False)
            assert ok == True

    def test_pack_and_send_invalid_length(self):
        """Wrong number of values must fail."""
        with patch('serial.Serial') as mock_serial_class:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            mock_serial_class.return_value = mock_instance

            bridge = SerialBridge(port='COM3')

            ok = bridge.pack_and_send_data([GREEN, GREEN, RED], retry_on_fail=False)

            assert ok == False
            assert bridge.fail_count == 1
            mock_instance.write.assert_not_called()

    def test_pack_and_send_invalid_state_value(self):
        """Value outside {0,1,2} must fail."""
        with patch('serial.Serial') as mock_serial_class:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            mock_serial_class.return_value = mock_instance

            bridge = SerialBridge(port='COM3')

            # 3 is not a valid RYG state
            states = [GREEN, GREEN, RED, RED, GREEN, GREEN, RED, 3]
            ok = bridge.pack_and_send_data(states, retry_on_fail=False)

            assert ok == False
            assert bridge.fail_count == 1

    def test_pack_and_send_negative_value(self):
        """Negative values are not valid RYG states."""
        with patch('serial.Serial') as mock_serial_class:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            mock_serial_class.return_value = mock_instance

            bridge = SerialBridge(port='COM3')

            states = [GREEN, GREEN, RED, RED, GREEN, GREEN, RED, -1]
            ok = bridge.pack_and_send_data(states, retry_on_fail=False)

            assert ok == False
            assert bridge.fail_count == 1

    def test_reconnect_on_disconnect(self):
        with patch('serial.Serial') as mock_serial_class:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            mock_serial_class.return_value = mock_instance

            bridge = SerialBridge(
                port='COM3',
                max_reconnect_attempts=3,
                reconnect_interval_s=0.1,
            )

            import serial
            mock_instance.write.side_effect = [
                serial.SerialException("USB disconnect"),
                None,
            ]

            states = [GREEN, GREEN, RED, RED, GREEN, GREEN, RED, RED]
            ok = bridge.pack_and_send_data(states, retry_on_fail=True)

            assert bridge.connected == True or ok == True

    def test_is_healthy(self):
        with patch('serial.Serial') as mock_serial_class:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            mock_serial_class.return_value = mock_instance

            bridge = SerialBridge(port='COM3')
            assert bridge.is_healthy() == True

            mock_instance.is_open = False
            bridge.connected = False
            assert bridge.is_healthy() == False

    def test_get_stats(self):
        with patch('serial.Serial') as mock_serial_class:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            mock_serial_class.return_value = mock_instance

            bridge = SerialBridge(port='COM3')
            states = [GREEN, GREEN, RED, RED, GREEN, GREEN, RED, RED]
            bridge.pack_and_send_data(states, retry_on_fail=False)

            stats = bridge.get_stats()
            assert stats['connected'] == True
            assert stats['port'] == 'COM3'
            assert stats['send_count'] == 1
            assert stats['fail_count'] == 0

    def test_close_graceful(self):
        with patch('serial.Serial') as mock_serial_class:
            mock_instance = MagicMock()
            mock_instance.is_open = True
            mock_serial_class.return_value = mock_instance

            bridge = SerialBridge(port='COM3')
            bridge.stop_watchdog()
            bridge.close()

            assert bridge.connected == False
            mock_instance.close.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
