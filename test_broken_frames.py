"""Tests for broken-frames audit and Telegram report command."""

import unittest
from unittest.mock import patch

import pandas as pd

import cascade_pipeline as pipeline
import fahadal92 as bot
from cascade_steps import TRIPLING_PAIRS
from indicators import MIN_CANDLES


def _ohlcv(rows=500, freq="1min"):
    times = pd.date_range("2024-01-01", periods=rows, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "ts": times,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "vol": 10.0,
        }
    )


class TestAuditBrokenFrames(unittest.TestCase):
    def test_not_ready_when_prefetch_incomplete(self):
        with patch.object(pipeline.fast_prefetch_done, "is_set", return_value=False):
            report = pipeline.audit_broken_frames(symbols=["BTCUSDT"])

        self.assertFalse(report["ready"])
        self.assertEqual(report["broken_frame_count"], 0)
        self.assertEqual(report["broken_by_symbol"], {})

    def test_reports_missing_raw_and_min_candles(self):
        # Deep 1m so small 1m-sourced frames pass; empty 30m/60m breaks the rest.
        raw_1m = _ohlcv(rows=MIN_CANDLES * 45 + 200, freq="1min")

        def fake_cached(symbol, tf):
            if tf == "1m":
                return raw_1m.copy()
            return pd.DataFrame()

        with (
            patch.object(pipeline.fast_prefetch_done, "is_set", return_value=True),
            patch.object(pipeline, "get_cached", side_effect=fake_cached),
        ):
            report = pipeline.audit_broken_frames(symbols=["AAAUSDT"])

        self.assertTrue(report["ready"])
        self.assertIn("AAAUSDT", report["broken_by_symbol"])
        issues = report["broken_by_symbol"]["AAAUSDT"]
        self.assertGreater(len(issues), 0)

        reasons = {item["reason"] for item in issues}
        self.assertIn("missing_raw_base", reasons)

        # Frames built purely from 1m with enough candles should stay healthy.
        small_1m_bases = {
            base
            for base, _c, _t, api, _tapi in TRIPLING_PAIRS
            if api == "1m" and _tapi == "1m"
        }
        broken_bases = {item["base_frame"] for item in issues}
        healthy_small = small_1m_bases - broken_bases
        self.assertTrue(healthy_small)
        self.assertIn(9, healthy_small)
        self.assertIn(45, healthy_small)

    def test_all_ok_when_cache_is_deep(self):
        raw_1m = _ohlcv(rows=MIN_CANDLES * 90 + 100, freq="1min")
        raw_30m = _ohlcv(rows=MIN_CANDLES * 10 + 50, freq="30min")
        raw_60m = _ohlcv(rows=MIN_CANDLES * 10 + 50, freq="60min")

        def fake_cached(symbol, tf):
            return {
                "1m": raw_1m,
                "30m": raw_30m,
                "60m": raw_60m,
            }[tf].copy()

        with (
            patch.object(pipeline.fast_prefetch_done, "is_set", return_value=True),
            patch.object(pipeline, "get_cached", side_effect=fake_cached),
        ):
            report = pipeline.audit_broken_frames(symbols=["BTCUSDT"])

        self.assertTrue(report["ready"])
        self.assertEqual(report["broken_by_symbol"], {})
        self.assertEqual(report["ok_symbols"], ["BTCUSDT"])
        self.assertEqual(report["broken_frame_count"], 0)


class TestBrokenFramesCommand(unittest.TestCase):
    def test_command_lists_symbol_and_frames(self):
        fake_report = {
            "ready": True,
            "symbols_checked": 2,
            "total_pairs": len(TRIPLING_PAIRS),
            "broken_frame_count": 2,
            "ok_symbols": ["ETHUSDT"],
            "broken_by_symbol": {
                "BTCUSDT": [
                    {
                        "base_frame": 240,
                        "confirm_frame": 720,
                        "triple_frame": 80,
                        "base_api": "60m",
                        "triple_api": "1m",
                        "reason": "min_candles",
                        "detail": "شموع غير كافية على الأساسي (45/300)",
                        "candle_count": 45,
                    },
                    {
                        "base_frame": 210,
                        "confirm_frame": 630,
                        "triple_frame": 70,
                        "base_api": "30m",
                        "triple_api": "1m",
                        "reason": "missing_raw_base",
                        "detail": "بيانات المصدر 30m ناقصة",
                        "candle_count": 0,
                    },
                ]
            },
        }
        sent = []

        with (
            patch.object(bot, "audit_broken_frames", return_value=fake_report),
            patch.object(
                bot,
                "send_telegram",
                side_effect=lambda message, chat_id=None: sent.append(
                    (message, chat_id)
                ),
            ),
        ):
            bot._dispatch_command("/broken_frames", "42")

        self.assertEqual(len(sent), 1)
        message, chat_id = sent[0]
        self.assertEqual(chat_id, "42")
        self.assertIn("BTCUSDT", message)
        self.assertIn("فريمان معطوبان", message)
        self.assertIn("240m / 720m / 80m", message)
        self.assertIn("210m / 630m / 70m", message)

    def test_arabic_alias_accepts_optional_symbol(self):
        captured = {}

        def fake_handle(chat_id, symbol=None):
            captured["chat_id"] = chat_id
            captured["symbol"] = symbol

        with patch.object(bot, "handle_broken_frames_command", side_effect=fake_handle):
            bot._dispatch_command("/فريمات ethusdt", "9")

        self.assertEqual(captured["chat_id"], "9")
        self.assertEqual(captured["symbol"], "ethusdt")

    def test_format_all_ok_message(self):
        chunks = bot.format_broken_frames_report(
            {
                "ready": True,
                "symbols_checked": 3,
                "total_pairs": 16,
                "broken_by_symbol": {},
                "ok_symbols": ["A", "B", "C"],
                "broken_frame_count": 0,
            }
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("كل الفريمات شغّالة", chunks[0])


if __name__ == "__main__":
    unittest.main()
