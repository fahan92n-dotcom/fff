"""Tests for TradingView webhook → Telegram results."""

import json
import unittest
from io import BytesIO
from unittest.mock import Mock, patch

import tv_webhook


class TestFormatResultMessage(unittest.TestCase):
    def test_entry_buy(self):
        text = tv_webhook.format_result_message(
            {
                "event": "entry",
                "side": "buy",
                "tag": "60/20",
                "symbol": "BTCUSDT",
                "price": 63071.74,
            }
        )
        self.assertIn("صفقة شراء", text)
        self.assertIn("60/20", text)
        self.assertIn("BTCUSDT", text)
        self.assertNotIn("خطوة", text)

    def test_result_win_and_loss(self):
        win = tv_webhook.format_result_message(
            {
                "event": "result",
                "side": "sell",
                "tag": "15/5",
                "symbol": "BTCUSDT",
                "price": 100,
                "win": True,
            }
        )
        loss = tv_webhook.format_result_message(
            {
                "event": "result",
                "side": "buy",
                "tag": "15/5",
                "symbol": "BTCUSDT",
                "price": 100,
                "win": False,
            }
        )
        self.assertIn("ناجحة", win)
        self.assertIn("بيع", win)
        self.assertIn("فاشلة", loss)
        self.assertIn("شراء", loss)

    def test_rejects_unknown_event(self):
        with self.assertRaises(ValueError):
            tv_webhook.format_result_message(
                {"event": "steps", "side": "buy", "price": 1}
            )


class TestIngest(unittest.TestCase):
    def test_forwards_authorized_entry(self):
        send = Mock(return_value=True)
        body = json.dumps(
            {
                "event": "entry",
                "side": "buy",
                "tag": "60/20",
                "symbol": "BTCUSDT",
                "price": 63071,
            }
        ).encode()
        with patch.object(tv_webhook, "webhook_secret", return_value="s3cret"):
            status, reply = tv_webhook.ingest(body, {"token": ["s3cret"]}, send)
        self.assertEqual(status, 200)
        self.assertEqual(reply, b"ok")
        send.assert_called_once()
        self.assertIn("صفقة شراء", send.call_args[0][0])

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


class TestHandleHttp(unittest.TestCase):
    def test_post_tv_ok(self):
        handler = Mock()
        handler.path = "/tv?token=s3cret"
        payload = json.dumps(
            {
                "event": "entry",
                "side": "sell",
                "tag": "15/5",
                "symbol": "BTCUSDT",
                "price": 1.5,
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

    def test_unknown_path_is_404(self):
        handler = Mock()
        handler.path = "/health"
        handler.wfile = BytesIO()
        tv_webhook.handle_http(handler, Mock())
        handler.send_response.assert_called_once_with(404)


if __name__ == "__main__":
    unittest.main()
