"""Tests for TradingView webhook score store and on-demand Telegram totals."""

import json
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import tv_webhook


class ScoreStoreTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "tv_score.json")
        self.env = patch.dict(os.environ, {"TV_SCORE_PATH": self.path})
        self.env.start()
        tv_webhook.reset_score()

    def tearDown(self):
        tv_webhook.reset_score()
        self.env.stop()
        self.tmp.cleanup()


class TestScoreCommands(unittest.TestCase):
    def test_arabic_and_slash_commands(self):
        self.assertTrue(tv_webhook.is_score_command("/نتائج"))
        self.assertTrue(tv_webhook.is_score_command("/نتائج@MyBot"))
        self.assertTrue(tv_webhook.is_score_command("نتائج"))
        self.assertTrue(tv_webhook.is_score_command("/score"))
        self.assertTrue(tv_webhook.is_score_command("/SCORE"))
        self.assertFalse(tv_webhook.is_score_command("/week"))
        self.assertFalse(tv_webhook.is_score_command("3"))


class TestFormatScoreMessage(ScoreStoreTestCase):
    def test_empty_explains_webhook(self):
        text = tv_webhook.format_score_message()
        self.assertIn("ما وصله نتائج بعد", text)
        self.assertIn("/نتائج", text)
        self.assertIn("/tv?token=", text)

    def test_shows_counts_after_results(self):
        tv_webhook.apply_event(
            {
                "event": "result",
                "side": "buy",
                "tag": "60/20",
                "symbol": "BTCUSDT",
                "price": 100,
                "win": True,
                "wins": 12,
                "losses": 8,
                "open": 1,
            }
        )
        text = tv_webhook.format_score_message()
        self.assertIn("ناجحة", text)
        self.assertIn("12", text)
        self.assertIn("فاشلة", text)
        self.assertIn("8", text)
        self.assertIn("مفتوحة", text)
        self.assertIn("0.67", text)
        self.assertIn("0.53", text)
        self.assertNotIn("خطوة", text)


class TestIngest(ScoreStoreTestCase):
    def test_stores_result_without_telegram(self):
        send = Mock(return_value=True)
        body = json.dumps(
            {
                "event": "result",
                "side": "buy",
                "tag": "60/20",
                "symbol": "BTCUSDT",
                "price": 63071,
                "win": True,
                "wins": 4,
                "losses": 2,
                "open": 0,
            }
        ).encode()
        with patch.object(tv_webhook, "webhook_secret", return_value="s3cret"):
            status, reply = tv_webhook.ingest(body, {"token": ["s3cret"]}, send)
        self.assertEqual(status, 200)
        self.assertEqual(reply, b"ok")
        send.assert_not_called()
        snap = tv_webhook.snapshot()
        self.assertEqual(snap["wins"], 4)
        self.assertEqual(snap["losses"], 2)

    def test_entry_does_not_message(self):
        send = Mock(return_value=True)
        body = json.dumps(
            {
                "event": "entry",
                "side": "buy",
                "tag": "60/20",
                "symbol": "BTCUSDT",
                "price": 63071,
                "wins": 3,
                "losses": 1,
                "open": 1,
            }
        ).encode()
        with patch.object(tv_webhook, "webhook_secret", return_value="s3cret"):
            status, reply = tv_webhook.ingest(body, {"token": ["s3cret"]}, send)
        self.assertEqual(status, 200)
        self.assertEqual(reply, b"ok")
        send.assert_not_called()
        self.assertEqual(tv_webhook.snapshot()["open"], 1)

    def test_rejects_bad_token(self):
        send = Mock(return_value=True)
        body = b'{"event":"entry","side":"buy","price":1}'
        with patch.object(tv_webhook, "webhook_secret", return_value="s3cret"):
            status, reply = tv_webhook.ingest(body, {"token": ["nope"]}, send)
        self.assertEqual(status, 401)
        self.assertEqual(reply, b"unauthorized")
        send.assert_not_called()

    def test_rejects_when_secret_missing(self):
        send = Mock(return_value=True)
        body = b'{"event":"entry","side":"buy","price":1}'
        with patch.object(tv_webhook, "webhook_secret", return_value=""):
            status, _reply = tv_webhook.ingest(body, {"token": ["x"]}, send)
        self.assertEqual(status, 401)
        send.assert_not_called()

    def test_rejects_invalid_json(self):
        send = Mock(return_value=True)
        with patch.object(tv_webhook, "webhook_secret", return_value="s3cret"):
            status, reply = tv_webhook.ingest(b"not-json", {"token": ["s3cret"]}, send)
        self.assertEqual(status, 400)
        self.assertEqual(reply, b"bad payload")


class TestHandleHttp(ScoreStoreTestCase):
    def test_post_tv_ok_silent(self):
        handler = Mock()
        handler.path = "/tv?token=s3cret"
        payload = json.dumps(
            {
                "event": "entry",
                "side": "sell",
                "tag": "15/5",
                "symbol": "BTCUSDT",
                "price": 1.5,
                "wins": 0,
                "losses": 0,
                "open": 1,
            }
        ).encode()
        handler.headers = {"Content-Length": str(len(payload))}
        handler.rfile = BytesIO(payload)
        handler.wfile = BytesIO()
        send = Mock(return_value=True)
        with patch.object(tv_webhook, "webhook_secret", return_value="s3cret"):
            tv_webhook.handle_http(handler, send)
        handler.send_response.assert_called_once_with(200)
        self.assertEqual(handler.wfile.getvalue(), b"ok")
        send.assert_not_called()

    def test_unknown_path_is_404(self):
        handler = Mock()
        handler.path = "/health"
        handler.wfile = BytesIO()
        tv_webhook.handle_http(handler, Mock())
        handler.send_response.assert_called_once_with(404)


class TestHandleScoreCommand(ScoreStoreTestCase):
    def test_sends_current_totals(self):
        tv_webhook.apply_event(
            {
                "event": "result",
                "side": "sell",
                "tag": "15/5",
                "symbol": "BTCUSDT",
                "price": 100,
                "win": False,
                "wins": 5,
                "losses": 9,
                "open": 0,
            }
        )
        send = Mock(return_value=True)
        tv_webhook.handle_score_command("42", send)
        send.assert_called_once()
        message, chat_id = send.call_args[0]
        self.assertEqual(chat_id, "42")
        self.assertIn("5", message)
        self.assertIn("9", message)


if __name__ == "__main__":
    unittest.main()
