"""Regression tests for /سبب_شراء and /سبب_بيع cascade diag commands."""

import unittest
from unittest.mock import patch

import fahadal92 as bot


class TestNormalizeCommandText(unittest.TestCase):
    def test_strips_bot_username_suffix(self):
        self.assertEqual(
            bot._normalize_command_text("/سبب_بيع@Fahaod92"),
            "/سبب_بيع",
        )
        self.assertEqual(
            bot._normalize_command_text("/check5@Bot BTCUSDT"),
            "/check5 BTCUSDT",
        )


class TestCascadeDiagSellCommand(unittest.TestCase):
    def test_sell_diag_sends_html_safe_report(self):
        sent = []

        def fake_send(message, chat_id=None):
            sent.append((message, chat_id))
            return True

        with patch.object(bot, "send_telegram", side_effect=fake_send):
            bot._dispatch_command("/سبب_بيع", "123")

        self.assertEqual(len(sent), 1)
        message, chat_id = sent[0]
        self.assertEqual(chat_id, "123")
        self.assertIn("البيع SHORT", message)
        self.assertIn("Stoch&lt;80", message)
        self.assertNotIn("Stoch<80", message)

    def test_score_command_uses_stored_tv_totals(self):
        sent = []

        def fake_send(message, chat_id=None):
            sent.append((message, chat_id))
            return True

        with patch.object(bot, "send_telegram", side_effect=fake_send), patch(
            "tv_webhook.format_score_message",
            return_value="📊 نتائج تجريبية",
        ):
            bot._dispatch_command("/نتائج", "123")

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][1], "123")
        self.assertIn("نتائج", sent[0][0])

    def test_buy_diag_still_works(self):
        sent = []

        def fake_send(message, chat_id=None):
            sent.append((message, chat_id))
            return True

        with patch.object(bot, "send_telegram", side_effect=fake_send):
            bot._dispatch_command("/سبب_شراء", "123")

        self.assertEqual(len(sent), 1)
        self.assertIn("الشراء LONG", sent[0][0])


if __name__ == "__main__":
    unittest.main()
