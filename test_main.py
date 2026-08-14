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
            "clear_ribbon_cache",
        ) as clear_ribbon:
            ran = runtime._run_full_scan_pair()

        self.assertTrue(ran)
        self.assertEqual(len(workers), 2)
        long_scan.assert_called_once_with()
        short_scan.assert_called_once_with()
        clear_ribbon.assert_called_once_with()


class TestBackgroundWatchers(unittest.TestCase):
    def test_start_background_services_starts_cascade_and_quick_check(self):
        started = []

        def fake_run_forever(target, name):
            started.append((target, name))

        with patch.object(runtime, "HTTPServer") as http, patch.object(
            runtime,
            "delete_webhook",
        ), patch.object(
            runtime,
            "_start_market_threads",
        ), patch.object(
            runtime,
            "run_forever",
            fake_run_forever,
        ), patch.object(runtime, "send_telegram"):
            http.return_value = Mock()
            runtime.start_background_services()

        names = [name for _target, name in started]
        self.assertEqual(
            names,
            [
                "poll_telegram_commands",
                "cascade_watcher",
                "quick_check_watcher",
            ],
        )
        by_name = {name: target for target, name in started}
        self.assertIs(by_name["cascade_watcher"], runtime.cascade_watcher)
        self.assertIs(by_name["quick_check_watcher"], runtime.quick_check_watcher)
        self.assertIs(
            by_name["poll_telegram_commands"],
            runtime.poll_telegram_commands,
        )


if __name__ == "__main__":
    unittest.main()
