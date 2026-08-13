"""Tests for RSI-only main 50 + entry 45/55 scan."""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

import pullback_bot.rsi_scan as rs


def _bars(start, count, minutes=1, price=100.0, drift=0.0):
    rows = []
    px = price
    for index in range(count):
        ts = start + timedelta(minutes=minutes * index)
        close_px = px + drift
        rows.append(
            {
                "ts": ts,
                "open": px,
                "high": max(px, close_px) + 1.0,
                "low": min(px, close_px) - 1.0,
                "close": close_px,
                "vol": 1.0,
            }
        )
        px = close_px
    return pd.DataFrame(rows)


def _empty_stepped(grid):
    return {
        "rsi_buy": np.zeros(len(grid), dtype=bool),
        "rsi_sell": np.zeros(len(grid), dtype=bool),
    }


def _fill_levels(stepped, grid):
    for main, _entry, _win, _loss, _group in rs.LEVELS:
        stepped.setdefault(main, _empty_stepped(grid))
    return stepped


class TestLevels(unittest.TestCase):
    def test_user_table(self):
        self.assertEqual(rs.LEVELS[0], (45, 5, 0.50, 0.37, "a"))
        self.assertEqual(rs.LEVELS[1], (60, 5, 0.50, 0.37, "a"))
        self.assertEqual(rs.LEVELS[2], (90, 9, 0.67, 0.54, "b"))
        self.assertEqual(rs.LEVELS[3], (120, 10, 0.67, 0.54, "b"))
        self.assertEqual(rs.LEVELS[4], (150, 11, 0.67, 0.54, "b"))
        self.assertEqual(rs.MAIN_BUY, 50.0)
        self.assertEqual(rs.ENTRY_BUY, 45.0)
        self.assertEqual(rs.ENTRY_SELL, 55.0)


class TestScanSide(unittest.TestCase):
    def test_buy_after_entry_rsi_clears_then_holds(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=5 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=5 * (i + 1)) for i in range(3)],
                "close": [99.0, 100.0, 101.0],
                "rsi_buy": [False, False, True],
                "rsi_sell": [True, True, False],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=5 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[45]["rsi_buy"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=0.4)
        signals = rs._scan_side(
            "buy", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "BTCUSDT",
        )
        buys = [s for s in signals if s["type"] == "buy" and s["base_frame"] == 45]
        self.assertTrue(buys)
        self.assertAlmostEqual(buys[0]["price"], 101.0)

    def test_no_buy_when_main_rsi_not_above_50(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=5 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=5 * (i + 1)) for i in range(3)],
                "close": [99.0, 100.0, 101.0],
                "rsi_buy": [False, False, True],
                "rsi_sell": [True, True, False],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=5 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=0.4)
        signals = rs._scan_side(
            "buy", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "BTCUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 45 for s in signals))

    def test_sell_needs_entry_rsi_below_55(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=5 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=5 * (i + 1)) for i in range(3)],
                "close": [101.0, 100.0, 99.0],
                "rsi_buy": [True, True, False],
                "rsi_sell": [False, False, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=5 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[45]["rsi_sell"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = rs._scan_side(
            "sell", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "ETHUSDT",
        )
        sells = [s for s in signals if s["type"] == "sell" and s["base_frame"] == 45]
        self.assertTrue(sells)


class TestReport(unittest.TestCase):
    def test_report_is_rsi_only(self):
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=30)
        result = {
            "ready": True,
            "start": start,
            "end": end,
            "days": 30,
            "wins": [],
            "losses": [],
            "opens": [],
            "total": 0,
            "symbols": ["BTCUSDT"],
        }
        text = rs.format_plain_report(result)
        self.assertIn("RSI", text)
        self.assertIn("بدون EMA", text)
        self.assertIn("50", text)
        self.assertIn("45", text)


if __name__ == "__main__":
    unittest.main()
