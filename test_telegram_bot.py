"""Tests for Telegram transport boundaries and signal wiring."""

import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pandas as pd

import telegram_bot


class TestTelegramDispatch(unittest.TestCase):
    def test_authorized_update_is_dispatched_in_worker(self):
        handler = Mock()
        worker = Mock()
        original_handler = telegram_bot._command_handler
        telegram_bot.set_command_handler(handler)
        update = {
            "message": {
                "text": "/status",
                "chat": {"id": 123},
            }
        }
        try:
            with patch.object(
                telegram_bot,
                "ALLOWED_CHAT_IDS",
                frozenset({"123"}),
            ), patch.object(
                telegram_bot.threading,
                "Thread",
                return_value=worker,
            ) as thread:
                telegram_bot._dispatch_update(update)
        finally:
            telegram_bot.set_command_handler(original_handler)

        thread.assert_called_once_with(
            target=handler,
            args=("/status", "123"),
            daemon=True,
        )
        worker.start.assert_called_once_with()

    def test_unauthorized_update_is_ignored(self):
        handler = Mock()
        original_handler = telegram_bot._command_handler
        telegram_bot.set_command_handler(handler)
        try:
            with patch.object(
                telegram_bot,
                "ALLOWED_CHAT_IDS",
                frozenset({"999"}),
            ), patch.object(telegram_bot.threading, "Thread") as thread:
                telegram_bot._dispatch_update(
                    {
                        "message": {
                            "text": "/status",
                            "chat": {"id": 123},
                        }
                    }
                )
        finally:
            telegram_bot.set_command_handler(original_handler)

        thread.assert_not_called()
        handler.assert_not_called()

    def test_authorized_channel_post_is_dispatched(self):
        handler = Mock()
        worker = Mock()
        original_handler = telegram_bot._command_handler
        telegram_bot.set_command_handler(handler)
        update = {
            "channel_post": {
                "text": "/status",
                "chat": {"id": -1003968771145},
            }
        }
        try:
            with patch.object(
                telegram_bot,
                "ALLOWED_CHAT_IDS",
                frozenset({"-1003968771145"}),
            ), patch.object(
                telegram_bot.threading,
                "Thread",
                return_value=worker,
            ) as thread:
                telegram_bot._dispatch_update(update)
        finally:
            telegram_bot.set_command_handler(original_handler)

        thread.assert_called_once_with(
            target=handler,
            args=("/status", "-1003968771145"),
            daemon=True,
        )
        worker.start.assert_called_once_with()

    def test_channel_post_without_message_key_is_not_dropped(self):
        handler = Mock()
        original_handler = telegram_bot._command_handler
        telegram_bot.set_command_handler(handler)
        try:
            with patch.object(
                telegram_bot,
                "ALLOWED_CHAT_IDS",
                frozenset({"-1003968771145"}),
            ), patch.object(telegram_bot.threading, "Thread") as thread:
                telegram_bot._dispatch_update(
                    {
                        "update_id": 1,
                        "channel_post": {
                            "text": "/سبب_شراء",
                            "chat": {"id": -1003968771145},
                        },
                    }
                )
        finally:
            telegram_bot.set_command_handler(original_handler)

        thread.assert_called_once()
        self.assertEqual(
            thread.call_args.kwargs["args"],
            ("/سبب_شراء", "-1003968771145"),
        )


class TestSignalNotification(unittest.TestCase):
    def test_signal_is_claimed_saved_cleared_and_sent(self):
        ready_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        candle_ts = datetime(2024, 1, 1, 11, 57, 0, tzinfo=timezone.utc)
        frame = pd.DataFrame(
            {
                "ts": [candle_ts, ready_at],
                "close": [12.5, 14.0],
            }
        )
        with patch.object(
            telegram_bot,
            "claim_signal",
            return_value=ready_at,
        ) as claim, patch.object(
            telegram_bot,
            "save_signal",
        ) as save, patch.object(
            telegram_bot,
            "_clear_waiting_candidate",
        ) as clear, patch.object(
            telegram_bot,
            "send_telegram",
            return_value=True,
        ) as send:
            emitted = telegram_bot._fire_signal(
                "BATUSDT",
                15,
                45,
                5,
                frame,
                signal_type="buy",
                price=12.5,
                candle_ts=candle_ts,
            )

        self.assertTrue(emitted)
        claim.assert_called_once_with(("BATUSDT", 15, 45, 5, "buy"))
        save.assert_called_once_with(
            "BATUSDT",
            12.5,
            15,
            45,
            5,
            signal_type="buy",
        )
        clear.assert_called_once_with(
            "BATUSDT",
            15,
            45,
            5,
            signal_type="buy",
        )
        send.assert_called_once()
        message = send.call_args.args[0]
        self.assertIn("12.5", message)
        self.assertIn("شمعة التحقق: 2024-01-01 11:57:00 UTC", message)
        self.assertNotIn("14", message)


if __name__ == "__main__":
    unittest.main()
