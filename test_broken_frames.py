"""Tests for broken-frames audit and Telegram report command."""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

import cascade_pipeline as pipeline
import fahadal92 as bot
import state_manager as state
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

        ok_frames = report["ok_frames_by_symbol"]["AAAUSDT"]
        self.assertGreater(len(ok_frames), 0)
        ok_bases = {item["base_frame"] for item in ok_frames}
        self.assertIn(9, ok_bases)
        self.assertIn(45, ok_bases)
        self.assertEqual(
            len(ok_frames) + len(issues),
            len(TRIPLING_PAIRS),
        )

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
        self.assertEqual(report["ok_frame_count"], len(TRIPLING_PAIRS))
        self.assertEqual(
            len(report["ok_frames_by_symbol"]["BTCUSDT"]),
            len(TRIPLING_PAIRS),
        )


class TestBrokenFramesCommand(unittest.TestCase):
    def test_command_lists_ok_and_broken_frames(self):
        fake_report = {
            "ready": True,
            "symbols_checked": 2,
            "total_pairs": len(TRIPLING_PAIRS),
            "broken_frame_count": 2,
            "ok_frame_count": len(TRIPLING_PAIRS) - 2 + len(TRIPLING_PAIRS),
            "ok_symbols": ["ETHUSDT"],
            "ok_frames_by_symbol": {
                "BTCUSDT": [
                    {
                        "base_frame": 9,
                        "confirm_frame": 27,
                        "triple_frame": 3,
                        "base_api": "1m",
                        "triple_api": "1m",
                    },
                    {
                        "base_frame": 60,
                        "confirm_frame": 180,
                        "triple_frame": 20,
                        "base_api": "60m",
                        "triple_api": "1m",
                    },
                ],
                "ETHUSDT": [
                    {
                        "base_frame": pair[0],
                        "confirm_frame": pair[1],
                        "triple_frame": pair[2],
                        "base_api": pair[3],
                        "triple_api": pair[4],
                    }
                    for pair in TRIPLING_PAIRS
                ],
            },
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
        self.assertIn("فريمان صالحان", message)
        self.assertIn("فريمان معطوبان", message)
        self.assertIn("❌", message)
        self.assertIn("240m / 720m / 80m", message)
        self.assertIn("210m / 630m / 70m", message)
        self.assertIn("✅ الصالح:", message)
        self.assertIn("9m", message)
        self.assertIn("ETHUSDT", message)

    def test_single_symbol_lists_healthy_triples(self):
        fake_report = {
            "ready": True,
            "symbols_checked": 1,
            "total_pairs": len(TRIPLING_PAIRS),
            "broken_frame_count": 1,
            "ok_frame_count": 1,
            "ok_symbols": [],
            "ok_frames_by_symbol": {
                "BTCUSDT": [
                    {
                        "base_frame": 9,
                        "confirm_frame": 27,
                        "triple_frame": 3,
                        "base_api": "1m",
                        "triple_api": "1m",
                    }
                ]
            },
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
                    }
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
            bot._dispatch_command("/broken_frames BTCUSDT", "7")

        message = sent[0][0]
        self.assertIn("الصالحة:", message)
        self.assertIn("✅ <code>9m / 27m / 3m</code>", message)

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
                "ok_frames_by_symbol": {
                    "A": [],
                    "B": [],
                    "C": [],
                },
                "ok_symbols": ["A", "B", "C"],
                "broken_frame_count": 0,
                "ok_frame_count": 48,
            }
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("كل الفريمات صالحة", chunks[0])
        self.assertIn("صالح / معطوب", chunks[0])

    def test_week_alias_routes_to_week_handler(self):
        captured = {}

        def fake_handle(chat_id, symbol=None, *, week=False):
            captured["chat_id"] = chat_id
            captured["symbol"] = symbol
            captured["week"] = week

        with patch.object(bot, "handle_broken_frames_command", side_effect=fake_handle):
            bot._dispatch_command("/broken_frames week", "11")
            self.assertTrue(captured["week"])
            bot._dispatch_command("/فريمات اسبوع", "11")
            self.assertTrue(captured["week"])
            bot._dispatch_command("/فريمات_اسبوع", "11")
            self.assertTrue(captured["week"])


class TestBrokenFramesHistory(unittest.TestCase):
    def setUp(self):
        with state.broken_frames_history_lock:
            state.broken_frames_history.clear()
            state.last_broken_frames_snapshot_at["at"] = None

    def test_aggregate_counts_occurrences_across_snapshots(self):
        t1 = datetime(2026, 8, 1, 12, tzinfo=timezone.utc)
        t2 = t1 + timedelta(hours=6)
        snapshots = [
            {
                "time": t1,
                "broken_by_symbol": {
                    "BTCUSDT": [
                        {
                            "base_frame": 240,
                            "confirm_frame": 720,
                            "triple_frame": 80,
                            "detail": "شموع غير كافية",
                        }
                    ]
                },
            },
            {
                "time": t2,
                "broken_by_symbol": {
                    "BTCUSDT": [
                        {
                            "base_frame": 240,
                            "confirm_frame": 720,
                            "triple_frame": 80,
                            "detail": "شموع غير كافية (40/300)",
                        }
                    ],
                    "ETHUSDT": [
                        {
                            "base_frame": 90,
                            "confirm_frame": 270,
                            "triple_frame": 30,
                            "detail": "بيانات المصدر 30m ناقصة",
                        }
                    ],
                },
            },
        ]
        aggregate = state.aggregate_broken_frames_history(snapshots)
        self.assertEqual(aggregate["snapshots"], 2)
        self.assertEqual(aggregate["by_symbol"]["BTCUSDT"][0]["count"], 2)
        self.assertEqual(
            aggregate["by_symbol"]["BTCUSDT"][0]["last_detail"],
            "شموع غير كافية (40/300)",
        )
        self.assertEqual(aggregate["by_symbol"]["ETHUSDT"][0]["count"], 1)

    def test_week_report_uses_history(self):
        fake_live = {
            "ready": True,
            "symbols_checked": 1,
            "total_pairs": len(TRIPLING_PAIRS),
            "broken_frame_count": 0,
            "ok_frame_count": len(TRIPLING_PAIRS),
            "broken_by_symbol": {},
            "ok_frames_by_symbol": {"BTCUSDT": []},
            "ok_symbols": ["BTCUSDT"],
        }
        t1 = datetime.now(timezone.utc) - timedelta(days=1)
        with state.broken_frames_history_lock:
            state.broken_frames_history.append(
                {
                    "time": t1,
                    "symbols_checked": 1,
                    "broken_frame_count": 1,
                    "broken_by_symbol": {
                        "BTCUSDT": [
                            {
                                "base_frame": 240,
                                "confirm_frame": 720,
                                "triple_frame": 80,
                                "detail": "شموع غير كافية",
                            }
                        ]
                    },
                }
            )

        sent = []
        with (
            patch.object(bot, "audit_broken_frames", return_value=fake_live),
            patch.object(
                bot,
                "send_telegram",
                side_effect=lambda message, chat_id=None: sent.append(message),
            ),
        ):
            bot._dispatch_command("/broken_frames week", "5")

        self.assertTrue(sent)
        self.assertIn("آخر 7 أيام", sent[0])
        self.assertIn("BTCUSDT", sent[0])
        self.assertIn("240m / 720m / 80m", sent[0])


if __name__ == "__main__":
    unittest.main()
