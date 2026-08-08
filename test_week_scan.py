"""Tests for historical /week strategy scan and win/loss classification."""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

import fahadal92 as bot
import week_scan


def _bars(start, count, minutes=1, price=100.0, drift=0.0, high_off=0.5, low_off=0.5):
    rows = []
    px = price
    for index in range(count):
        ts = start + timedelta(minutes=minutes * index)
        open_px = px
        close_px = px + drift
        high_px = max(open_px, close_px) + high_off
        low_px = min(open_px, close_px) - low_off
        rows.append(
            {
                "ts": ts,
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "vol": 1.0,
            }
        )
        px = close_px
    return pd.DataFrame(rows)


class TestOutcomeLevels(unittest.TestCase):
    def test_short_frames_use_tight_levels(self):
        for frame in (9, 15, 27):
            self.assertEqual(
                week_scan.outcome_levels(frame),
                (week_scan.SHORT_WIN_PCT, week_scan.SHORT_LOSS_PCT),
            )

    def test_long_frames_use_wide_levels(self):
        for frame in (30, 150, 240):
            self.assertEqual(
                week_scan.outcome_levels(frame),
                (week_scan.LONG_WIN_PCT, week_scan.LONG_LOSS_PCT),
            )


class TestEvaluateOutcome(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2026, 8, 1, tzinfo=timezone.utc)

    def test_buy_win_at_one_percent(self):
        # Entry 100 → TP 101. First bar high reaches 101.2 before any SL.
        future = pd.DataFrame(
            [
                {
                    "ts": self.start + timedelta(minutes=1),
                    "open": 100.0,
                    "high": 101.2,
                    "low": 99.8,
                    "close": 101.0,
                    "vol": 1,
                }
            ]
        )
        outcome, exit_price, _ = week_scan.evaluate_outcome("buy", 100.0, future)
        self.assertEqual(outcome, "win")
        self.assertAlmostEqual(exit_price, 101.0)

    def test_buy_loss_at_zero_point_eight(self):
        future = pd.DataFrame(
            [
                {
                    "ts": self.start + timedelta(minutes=1),
                    "open": 100.0,
                    "high": 100.2,
                    "low": 99.1,
                    "close": 99.2,
                    "vol": 1,
                }
            ]
        )
        outcome, exit_price, _ = week_scan.evaluate_outcome("buy", 100.0, future)
        self.assertEqual(outcome, "loss")
        self.assertAlmostEqual(exit_price, 99.2)

    def test_buy_win_short_frame_levels(self):
        # Entry 100 → TP 100.67 with short-frame 0.67%.
        future = pd.DataFrame(
            [
                {
                    "ts": self.start + timedelta(minutes=1),
                    "open": 100.0,
                    "high": 100.8,
                    "low": 99.9,
                    "close": 100.7,
                    "vol": 1,
                }
            ]
        )
        win_pct, loss_pct = week_scan.outcome_levels(15)
        outcome, exit_price, _ = week_scan.evaluate_outcome(
            "buy", 100.0, future, win_pct=win_pct, loss_pct=loss_pct
        )
        self.assertEqual(outcome, "win")
        self.assertAlmostEqual(exit_price, 100.67)

    def test_buy_loss_short_frame_levels(self):
        # Entry 100 → SL 99.49 with short-frame 0.51%.
        future = pd.DataFrame(
            [
                {
                    "ts": self.start + timedelta(minutes=1),
                    "open": 100.0,
                    "high": 100.1,
                    "low": 99.4,
                    "close": 99.45,
                    "vol": 1,
                }
            ]
        )
        win_pct, loss_pct = week_scan.outcome_levels(9)
        outcome, exit_price, _ = week_scan.evaluate_outcome(
            "buy", 100.0, future, win_pct=win_pct, loss_pct=loss_pct
        )
        self.assertEqual(outcome, "loss")
        self.assertAlmostEqual(exit_price, 99.49)

    def test_sell_win_at_one_percent(self):
        future = pd.DataFrame(
            [
                {
                    "ts": self.start + timedelta(minutes=1),
                    "open": 100.0,
                    "high": 100.1,
                    "low": 98.8,
                    "close": 99.0,
                    "vol": 1,
                }
            ]
        )
        outcome, exit_price, _ = week_scan.evaluate_outcome("sell", 100.0, future)
        self.assertEqual(outcome, "win")
        self.assertAlmostEqual(exit_price, 99.0)

    def test_sell_loss_at_zero_point_eight(self):
        future = pd.DataFrame(
            [
                {
                    "ts": self.start + timedelta(minutes=1),
                    "open": 100.0,
                    "high": 100.9,
                    "low": 99.9,
                    "close": 100.5,
                    "vol": 1,
                }
            ]
        )
        outcome, exit_price, _ = week_scan.evaluate_outcome("sell", 100.0, future)
        self.assertEqual(outcome, "loss")
        self.assertAlmostEqual(exit_price, 100.8)

    def test_same_bar_both_levels_counts_loss(self):
        future = pd.DataFrame(
            [
                {
                    "ts": self.start + timedelta(minutes=1),
                    "open": 100.0,
                    "high": 101.5,
                    "low": 99.0,
                    "close": 100.5,
                    "vol": 1,
                }
            ]
        )
        outcome, _, _ = week_scan.evaluate_outcome("buy", 100.0, future)
        self.assertEqual(outcome, "loss")

    def test_open_when_neither_hit(self):
        future = pd.DataFrame(
            [
                {
                    "ts": self.start + timedelta(minutes=1),
                    "open": 100.0,
                    "high": 100.4,
                    "low": 99.6,
                    "close": 100.1,
                    "vol": 1,
                }
            ]
        )
        outcome, exit_price, exit_ts = week_scan.evaluate_outcome(
            "buy", 100.0, future
        )
        self.assertEqual(outcome, "open")
        self.assertIsNone(exit_price)
        self.assertIsNone(exit_ts)


class TestFormatWeekReport(unittest.TestCase):
    def test_splits_wins_and_losses(self):
        now = datetime(2026, 8, 6, tzinfo=timezone.utc)
        result = {
            "ready": True,
            "start": now - timedelta(days=7),
            "end": now,
            "symbols_scanned": 2,
            "total": 2,
            "wins": [
                {
                    "symbol": "BTCUSDT",
                    "type": "buy",
                    "base_frame": 60,
                    "confirm_frame": 180,
                    "triple_frame": 20,
                    "time": now - timedelta(days=1),
                    "price": 100.0,
                    "outcome": "win",
                }
            ],
            "losses": [
                {
                    "symbol": "ETHUSDT",
                    "type": "sell",
                    "base_frame": 30,
                    "confirm_frame": 90,
                    "triple_frame": 10,
                    "time": now - timedelta(days=2),
                    "price": 50.0,
                    "outcome": "loss",
                }
            ],
            "opens": [],
        }
        chunks = week_scan.format_week_trades_report(result)
        text = "\n".join(chunks)
        self.assertIn("الناجحون", text)
        self.assertIn("الخاسرون", text)
        self.assertIn("BTCUSDT", text)
        self.assertIn("ETHUSDT", text)
        self.assertIn("0.67%", text)
        self.assertIn("0.51%", text)
        self.assertIn("1%", text)
        self.assertIn("0.8%", text)


class TestDedupeAndCommand(unittest.TestCase):
    def test_dedupe_keeps_earliest_within_window(self):
        t0 = datetime(2026, 8, 1, tzinfo=timezone.utc)
        signals = [
            {
                "symbol": "BTCUSDT",
                "type": "buy",
                "base_frame": 60,
                "confirm_frame": 180,
                "triple_frame": 20,
                "time": t0,
                "price": 1,
                "outcome": "win",
            },
            {
                "symbol": "BTCUSDT",
                "type": "buy",
                "base_frame": 60,
                "confirm_frame": 180,
                "triple_frame": 20,
                "time": t0 + timedelta(hours=1),
                "price": 2,
                "outcome": "loss",
            },
            {
                "symbol": "BTCUSDT",
                "type": "buy",
                "base_frame": 60,
                "confirm_frame": 180,
                "triple_frame": 20,
                "time": t0 + timedelta(hours=5),
                "price": 3,
                "outcome": "win",
            },
        ]
        kept = week_scan._dedupe_signals(signals, hours=4)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["price"], 1)
        self.assertEqual(kept[1]["price"], 3)

    def test_week_command_routes_to_market_scan(self):
        called = {}

        def fake_handle(chat_id, send_fn):
            called["chat_id"] = chat_id
            called["send"] = send_fn

        with patch.object(bot, "handle_week_command", side_effect=fake_handle):
            bot._dispatch_command("/week", "42")
            bot._dispatch_command("3", "42")

        self.assertEqual(called["chat_id"], "42")
        self.assertIs(called["send"], bot.send_telegram)

    def test_handle_week_sends_formatted_report(self):
        now = datetime(2026, 8, 6, tzinfo=timezone.utc)
        fake_result = {
            "ready": True,
            "start": now - timedelta(days=7),
            "end": now,
            "symbols_scanned": 1,
            "total": 1,
            "wins": [
                {
                    "symbol": "BTCUSDT",
                    "type": "buy",
                    "base_frame": 60,
                    "confirm_frame": 180,
                    "triple_frame": 20,
                    "time": now - timedelta(hours=3),
                    "price": 100.0,
                    "outcome": "win",
                }
            ],
            "losses": [],
            "opens": [],
        }
        sent = []
        with (
            patch.object(week_scan, "fast_prefetch_done") as done,
            patch.object(
                week_scan,
                "scan_week_trades",
                return_value=fake_result,
            ),
        ):
            done.is_set.return_value = True
            week_scan.handle_week_command(
                "9",
                lambda message, chat_id=None: sent.append(message),
            )

        self.assertTrue(sent)
        joined = "\n".join(sent)
        self.assertIn("صفقات الاستراتيجية", joined)
        self.assertIn("الناجحون", joined)
        self.assertIn("BTCUSDT", joined)


class TestSliceClosed(unittest.TestCase):
    def test_keeps_only_fully_closed_candles(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        df = _bars(start, 5, minutes=10, price=10.0)
        # asof exactly at end of 3rd candle (index 2): ts=start+20m, end=start+30m
        asof = start + timedelta(minutes=30)
        sliced = week_scan._slice_closed(df, 10, asof)
        self.assertEqual(len(sliced), 3)
        self.assertEqual(_utc_safe(sliced["ts"].iloc[-1]), start + timedelta(minutes=20))


def _utc_safe(ts):
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


if __name__ == "__main__":
    unittest.main()
