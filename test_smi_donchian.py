"""Tests for reverse-sat + 3× Donchian + MACD confirm."""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

import pullback_bot.smi_donchian as sd
from pullback_bot.strategy import evaluate_outcome, fetch_btc_1m_vision


def _bars(start, count, minutes=1, price=100.0, drift=0.0, high_off=1.0, low_off=1.0):
    rows = []
    px = price
    for index in range(count):
        ts = start + timedelta(minutes=minutes * index)
        open_px = px
        close_px = px + drift
        rows.append(
            {
                "ts": ts,
                "open": open_px,
                "high": max(open_px, close_px) + high_off,
                "low": min(open_px, close_px) - low_off,
                "close": close_px,
                "vol": 1.0,
            }
        )
        px = close_px
    return pd.DataFrame(rows)


def _empty_stepped(grid):
    return {
        "sell_sat": np.zeros(len(grid), dtype=bool),
        "buy_sat": np.zeros(len(grid), dtype=bool),
        "don_green": np.zeros(len(grid), dtype=bool),
        "don_red": np.zeros(len(grid), dtype=bool),
        "buy_macd": np.ones(len(grid), dtype=bool),
        "sell_macd": np.ones(len(grid), dtype=bool),
    }


def _fill_levels(stepped, grid):
    for main, reverse_min, reverse_last, reverse_abort, don_tf, entry, _w, _l in sd.LEVELS:
        stepped.setdefault(main, _empty_stepped(grid))
        stepped.setdefault(don_tf, _empty_stepped(grid))
        stepped.setdefault(reverse_abort, _empty_stepped(grid))
        stepped.setdefault(entry, _empty_stepped(grid))
        for minutes in range(reverse_min, reverse_last + 1):
            stepped.setdefault(minutes, _empty_stepped(grid))
    return stepped


class TestLevels(unittest.TestCase):
    def test_confirm_is_three_times_main(self):
        for main, _rmin, _rlast, _abort, don_tf, _entry, _w, _l in sd.LEVELS:
            self.assertEqual(don_tf, main * 3)

    def test_user_table_and_tp_sl(self):
        self.assertEqual(sd.LEVELS[0], (60, 10, 23, 24, 180, 5, 0.50, 0.37))
        self.assertEqual(sd.LEVELS[1], (90, 15, 35, 36, 270, 9, 0.67, 0.53))
        self.assertEqual(sd.LEVELS[2], (120, 20, 46, 48, 360, 10, 0.67, 0.53))
        self.assertEqual(sd.LEVELS[3], (150, 25, 59, 60, 450, 11, 0.67, 0.53))
        self.assertEqual([lvl[0] for lvl in sd.LEVELS], [60, 90, 120, 150])
        self.assertEqual(sd.MACD_FAST, 12)
        self.assertEqual(sd.MACD_SLOW, 26)
        self.assertEqual(sd.MACD_SIGNAL, 9)
        self.assertEqual(sd.SYMBOLS, ("BTCUSDT",))

    def test_two_hour_reverse_skips_47(self):
        _main, reverse_min, reverse_last, reverse_abort, _don, entry, win_pct, loss_pct = sd.LEVELS[2]
        accepted = list(range(reverse_min, reverse_last + 1))
        self.assertEqual(accepted[0], 20)
        self.assertEqual(accepted[-1], 46)
        self.assertNotIn(47, accepted)
        self.assertEqual(reverse_abort, 48)
        self.assertEqual(entry, 10)
        self.assertEqual(win_pct, 0.67)
        self.assertEqual(loss_pct, 0.53)


class TestFetchWrapper(unittest.TestCase):
    def test_btc_fetch_still_targets_btcusdt(self):
        with patch("pullback_bot.strategy.fetch_1m_vision", return_value=pd.DataFrame()) as mocked:
            fetch_btc_1m_vision(target=10)
        mocked.assert_called_once_with("BTCUSDT", target=10)


class TestScanSide(unittest.TestCase):
    def test_sell_waits_for_reverse_sat_then_entry_flip(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=5 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=5 * (i + 1)) for i in range(3)],
                "close": [101.0, 100.0, 99.0],
                "don_green": [True, True, False],
                "don_red": [False, False, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=5 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[60]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[10]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[180]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "BTCUSDT",
        )
        sells = [s for s in signals if s["type"] == "sell" and s["base_frame"] == 60]
        self.assertTrue(sells)
        self.assertAlmostEqual(sells[0]["price"], 99.0)
        self.assertAlmostEqual(sells[0]["win_pct"], 0.50)
        self.assertAlmostEqual(sells[0]["loss_pct"], 0.37)

    def test_ninety_uses_067_053(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=9 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=9 * (i + 1)) for i in range(3)],
                "close": [101.0, 100.0, 99.0],
                "don_green": [True, True, False],
                "don_red": [False, False, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=9 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[90]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[15]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[270]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 80, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {9: entry}, grid, start, start + timedelta(hours=2),
            raw_1m, "BTCUSDT",
        )
        sells = [s for s in signals if s["type"] == "sell" and s["base_frame"] == 90]
        self.assertTrue(sells)
        self.assertAlmostEqual(sells[0]["win_pct"], 0.67)
        self.assertAlmostEqual(sells[0]["loss_pct"], 0.53)

    def test_no_entry_without_reverse_sat(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=5 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=5 * (i + 1)) for i in range(3)],
                "close": [101.0, 100.0, 99.0],
                "don_green": [True, True, False],
                "don_red": [False, False, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=5 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[60]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[180]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "BTCUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 60 for s in signals))

    def test_larger_main_cancels_smaller(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=5 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=5 * (i + 1)) for i in range(3)],
                "close": [101.0, 100.0, 99.0],
                "don_green": [True, True, False],
                "don_red": [False, False, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=5 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[60]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[90]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[10]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[180]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "BTCUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 60 for s in signals))

    def test_no_sell_when_confirm_not_red(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=5 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=5 * (i + 1)) for i in range(3)],
                "close": [101.0, 100.0, 99.0],
                "don_green": [True, True, False],
                "don_red": [False, False, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=5 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[60]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[10]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[180]["don_green"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "BTCUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 60 for s in signals))

    def test_buy_requires_confirm_green(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=5 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=5 * (i + 1)) for i in range(3)],
                "close": [99.0, 100.0, 101.0],
                "don_green": [False, False, True],
                "don_red": [True, True, False],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=5 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[60]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[10]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[180]["don_green"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=0.5)
        signals = sd._scan_side(
            "buy", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "BTCUSDT",
        )
        buys = [s for s in signals if s["type"] == "buy" and s["base_frame"] == 60]
        self.assertTrue(buys)

    def test_sell_ignored_when_macd_histogram_not_red(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=5 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=5 * (i + 1)) for i in range(3)],
                "close": [101.0, 100.0, 99.0],
                "don_green": [True, True, False],
                "don_red": [False, False, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=5 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[60]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[10]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[180]["don_red"] = np.ones(len(grid), dtype=bool)
        stepped[180]["sell_macd"] = np.zeros(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "BTCUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 60 for s in signals))

    def test_buy_ignored_when_macd_histogram_not_green(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=5 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=5 * (i + 1)) for i in range(3)],
                "close": [99.0, 100.0, 101.0],
                "don_green": [False, False, True],
                "don_red": [True, True, False],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=5 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[60]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[10]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[180]["don_green"] = np.ones(len(grid), dtype=bool)
        stepped[180]["buy_macd"] = np.zeros(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=0.5)
        signals = sd._scan_side(
            "buy", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "BTCUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 60 for s in signals))

    def test_no_entry_without_entry_donchian_flip(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=5 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=5 * (i + 1)) for i in range(3)],
                "close": [101.0, 100.0, 99.0],
                "don_green": [False, False, False],
                "don_red": [True, True, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=5 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[60]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[10]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[180]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "BTCUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 60 for s in signals))


class TestEvaluateUsesLevelPct(unittest.TestCase):
    def test_loss_at_037(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        future = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=1)],
                "high": [100.2],
                "low": [99.6],
            }
        )
        outcome, _, _ = evaluate_outcome(
            "buy", 100.0, future, win_pct=0.50, loss_pct=0.37
        )
        self.assertEqual(outcome, "loss")

    def test_loss_at_053(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        future = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=1)],
                "high": [100.2],
                "low": [99.4],
            }
        )
        outcome, _, _ = evaluate_outcome(
            "buy", 100.0, future, win_pct=0.67, loss_pct=0.53
        )
        self.assertEqual(outcome, "loss")


class TestScanAll(unittest.TestCase):
    def test_merges_and_omits_rsi_ema(self):
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=30)
        trade = {
            "symbol": "BTCUSDT",
            "type": "buy",
            "time": start + timedelta(days=1),
            "price": 65000.0,
            "base_frame": 60,
            "confirm_frame": 180,
            "triple_frame": 5,
            "win_pct": 0.50,
            "loss_pct": 0.37,
            "outcome": "win",
            "exit_price": 65325.0,
            "exit_ts": start + timedelta(days=1, hours=1),
        }

        def fake_scan(symbol, **_kwargs):
            return {
                "ready": True,
                "start": start,
                "end": end,
                "days": 30,
                "wins": [trade],
                "losses": [],
                "opens": [],
                "total": 1,
                "market": "spot-vision",
                "symbol": symbol,
            }

        with patch.object(sd, "scan_symbol", side_effect=fake_scan):
            result = sd.scan_all(("BTCUSDT",), days=30, now=end)
        grouped = sd.group_results(result)
        self.assertEqual(grouped["all"]["total"], 1)
        text = sd.format_plain_report(result)
        self.assertIn("BTCUSDT", text)
        self.assertIn("Donchian", text)
        self.assertIn("MACD", text)
        self.assertNotIn("RSI", text)
        self.assertNotIn("EMA50", text)


if __name__ == "__main__":
    unittest.main()
