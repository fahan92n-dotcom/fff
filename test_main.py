"""Tests for process wiring extracted from the compatibility module."""

import unittest
from unittest.mock import Mock, patch

import main as runtime


class TestIntegrationWiring(unittest.TestCase):
    def test_configure_integrations_registers_all_callbacks(self):
        handler = Mock()
        with patch.object(runtime, "set_telegram_sender") as data_sender, patch.object(
            runtime,
            "set_signal_handler",
        ) as signal_handler, patch.object(
            runtime,
            "set_command_handler",
        ) as command_handler:
            runtime.configure_integrations(handler)

        data_sender.assert_called_once_with(runtime.send_telegram)
        signal_handler.assert_called_once_with(runtime._fire_signal)
        command_handler.assert_called_once_with(handler)


class TestScanPair(unittest.TestCase):
    def test_full_scan_pair_runs_both_sides(self):
        workers = []

        class ImmediateThread:
            def __init__(self, target, daemon):
                self.target = target
                self.daemon = daemon
                workers.append(self)

            def start(self):
                self.target()

            def join(self):
                return

        with patch.object(runtime, "run_cascade_scan") as long_scan, patch.object(
            runtime,
            "run_short_cascade_scan",
        ) as short_scan, patch.object(
            runtime.threading,
            "Thread",
            ImmediateThread,
        ), patch.object(runtime, "trim_memory"), patch.object(
            runtime,
            "_ribbon_cache",
            {},
        ):
            ran = runtime._run_full_scan_pair()

        self.assertTrue(ran)
        self.assertEqual(len(workers), 2)
        long_scan.assert_called_once_with()
        short_scan.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
