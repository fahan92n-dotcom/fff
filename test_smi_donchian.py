"""Tests for raised reverse-sat floors + 3× Donchian (no EMA60 / RSI)."""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

import pullback_bot.smi_donchian as sd
from pullback_bot.strategy import evaluate_outcome


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
    }


def _fill_levels(stepped, grid):
    for main, reverse_min, reverse_stop, _entry, don_tf, _win, _loss, _group in sd.LEVELS:
        stepped.setdefault(main, _empty_stepped(grid))
        stepped.setdefault(don_tf, _empty_stepped(grid))
        stepped.setdefault(reverse_stop, _empty_stepped(grid))
        for minutes in range(reverse_min, reverse_stop):
            stepped.setdefault(minutes, _empty_stepped(grid))
    for main in sd.MAINS:
        stepped.setdefault(main, _empty_stepped(grid))
    return stepped


class TestLevels(unittest.TestCase):
    def test_don_confirm_is_three_times_main(self):
        for main, _rmin, _rstop, _entry, don_tf, _win, _loss, _group in sd.LEVELS:
            self.assertEqual(don_tf, main * 3)

    def test_raised_reverse_floors(self):
        by_main_a = {
            lvl[0]: lvl for lvl in sd.LEVELS if lvl[7] == "a"
        }
        self.assertEqual(by_main_a[45][:4], (45, 10, 18, 6))
        self.assertEqual(by_main_a[60][:4], (60, 12, 24, 5))
        self.assertEqual(by_main_a[90][:4], (90, 18, 36, 8))
        self.assertEqual(by_main_a[120][:4], (120, 25, 48, 10))
        self.assertEqual(by_main_a[150][:4], (150, 30, 60, 13))
        self.assertEqual(by_main_a[180][:4], (180, 35, 72, 15))

    def test_group_b_entries(self):
        by_main_b = {lvl[0]: lvl for lvl in sd.LEVELS if lvl[7] == "b"}
        self.assertEqual(by_main_b[90][3], 9)
        self.assertEqual(by_main_b[120][3], 10)
        self.assertEqual(by_main_b[150][3], 11)
        self.assertEqual(by_main_b[90][5:7], (0.67, 0.54))


class TestScanSide(unittest.TestCase):
    def test_sell_enters_after_raised_reverse_and_don_red(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=6 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=6 * (i + 1)) for i in range(3)],
                "close": [101.0, 100.0, 99.0],
                "don_green": [True, True, False],
                "don_red": [False, False, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=6 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[45]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[10]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = sd._scan_side(
            "sell", stepped, {6: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "BTCUSDT",
        )
        sells = [s for s in signals if s["type"] == "sell" and s["base_frame"] == 45]
        self.assertTrue(sells)
        first = sells[0]
        self.assertEqual(first["confirm_frame"], 135)
        self.assertEqual(first["reverse_min"], 10)
        self.assertEqual(first["triple_frame"], 6)
        self.assertAlmostEqual(first["price"], 99.0)
        self.assertAlmostEqual(first["win_pct"], 0.50)

    def test_small_reverse_sat_below_floor_does_not_enter(self):
        """45m old floor was 8; sat only on 8 must not count."""
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=6 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=6 * (i + 1)) for i in range(3)],
                "close": [101.0, 100.0, 99.0],
                "don_green": [True, True, False],
                "don_red": [False, False, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=6 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped.setdefault(8, _empty_stepped(grid))
        stepped[45]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[8]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = sd._scan_side(
            "sell", stepped, {6: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "BTCUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 45 for s in signals))

    def test_reverse_stop_frame_blocks(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=6 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=6 * (i + 1)) for i in range(3)],
                "close": [101.0, 100.0, 99.0],
                "don_green": [True, True, False],
                "don_red": [False, False, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=6 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[45]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[10]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[18]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = sd._scan_side(
            "sell", stepped, {6: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "BTCUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 45 for s in signals))

    def test_no_sell_when_3x_don_not_red(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=6 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=6 * (i + 1)) for i in range(3)],
                "close": [101.0, 100.0, 99.0],
                "don_green": [True, True, False],
                "don_red": [False, False, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=6 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[45]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[10]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_green"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = sd._scan_side(
            "sell", stepped, {6: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "BTCUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 45 for s in signals))

    def test_buy_requires_3x_green(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=6 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=6 * (i + 1)) for i in range(3)],
                "close": [99.0, 100.0, 101.0],
                "don_green": [False, False, True],
                "don_red": [True, True, False],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=6 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[45]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[10]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_green"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=0.4)
        signals = sd._scan_side(
            "buy", stepped, {6: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "ETHUSDT",
        )
        buys = [s for s in signals if s["type"] == "buy" and s["base_frame"] == 45]
        self.assertTrue(buys)
        self.assertEqual(buys[0]["symbol"], "ETHUSDT")

    def test_no_entry_if_donchian_never_clears(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=6 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=6 * (i + 1)) for i in range(3)],
                "close": [99.0, 98.0, 97.0],
                "don_green": [False, False, False],
                "don_red": [True, True, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=6 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[45]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[10]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = sd._scan_side(
            "sell", stepped, {6: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "BTCUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 45 for s in signals))

    def test_larger_main_sat_cancels_smaller(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry6 = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=6 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=6 * (i + 1)) for i in range(3)],
                "close": [101.0, 100.0, 99.0],
                "don_green": [True, True, False],
                "don_red": [False, False, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=6 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[45]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[10]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_red"] = np.ones(len(grid), dtype=bool)
        stepped[90]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[18]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[270]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = sd._scan_side(
            "sell",
            stepped,
            {6: entry6, 8: entry6.copy(), 9: entry6.copy()},
            grid,
            start,
            start + timedelta(hours=1),
            raw_1m,
            "BTCUSDT",
        )
        self.assertTrue(signals)
        self.assertTrue(all(s["base_frame"] == 90 for s in signals))

    def test_group_b_uses_067_054(self):
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
        stepped[18]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[270]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 50, minutes=1, price=100.0, drift=-0.4)
        signals = sd._scan_side(
            "sell", stepped, {9: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "XRPUSDT",
        )
        group_b = [s for s in signals if s.get("group") == "b"]
        self.assertTrue(group_b)
        self.assertAlmostEqual(group_b[0]["win_pct"], 0.67)
        self.assertAlmostEqual(group_b[0]["loss_pct"], 0.54)
        self.assertEqual(group_b[0]["triple_frame"], 9)


class TestEvaluateUsesLevelPct(unittest.TestCase):
    def test_short_group_loss_037(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        future = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=1)],
                "high": [100.4],
                "low": [99.6],
            }
        )
        outcome, _, _ = evaluate_outcome("buy", 100.0, future, win_pct=0.50, loss_pct=0.37)
        self.assertEqual(outcome, "loss")

    def test_long_group_win_067(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        future = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=1)],
                "high": [100.70],
                "low": [99.50],
            }
        )
        outcome, _, _ = evaluate_outcome("buy", 100.0, future, win_pct=0.67, loss_pct=0.54)
        self.assertEqual(outcome, "win")


class TestFeaturesNoEma(unittest.TestCase):
    def test_smi_don_has_no_ema_or_rsi(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        raw = _bars(start, 400, minutes=1, price=100.0, drift=0.05)
        feat = sd._smi_don_features(raw, 1)
        self.assertIsNotNone(feat)
        self.assertNotIn("ema", feat.columns)
        self.assertNotIn("rsi", feat.columns)
        self.assertIn("sell_sat", feat.columns)
        self.assertIn("don_green", feat.columns)


class TestScanAll(unittest.TestCase):
    def test_merges_symbols_and_groups(self):
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=30)
        btc_trade = {
            "symbol": "BTCUSDT",
            "type": "buy",
            "time": start + timedelta(days=1),
            "price": 100.0,
            "base_frame": 45,
            "confirm_frame": 135,
            "reverse_min": 10,
            "reverse_stop": 18,
            "triple_frame": 6,
            "win_pct": 0.50,
            "loss_pct": 0.37,
            "group": "a",
            "outcome": "win",
            "exit_price": 100.5,
            "exit_ts": start + timedelta(days=1, hours=1),
        }
        eth_trade = {
            "symbol": "ETHUSDT",
            "type": "sell",
            "time": start + timedelta(days=2),
            "price": 200.0,
            "base_frame": 90,
            "confirm_frame": 270,
            "reverse_min": 18,
            "reverse_stop": 36,
            "triple_frame": 9,
            "win_pct": 0.67,
            "loss_pct": 0.54,
            "group": "b",
            "outcome": "loss",
            "exit_price": 201.08,
            "exit_ts": start + timedelta(days=2, hours=1),
        }

        def fake_scan(symbol, **_kwargs):
            trade = btc_trade if symbol == "BTCUSDT" else eth_trade
            wins = [trade] if trade["outcome"] == "win" else []
            losses = [trade] if trade["outcome"] == "loss" else []
            return {
                "ready": True,
                "start": start,
                "end": end,
                "days": 30,
                "wins": wins,
                "losses": losses,
                "opens": [],
                "total": 1,
                "market": "spot-vision",
                "symbol": symbol,
            }

        with patch.object(sd, "scan_symbol", side_effect=fake_scan):
            result = sd.scan_all(("BTCUSDT", "ETHUSDT"), days=30, now=end)
        self.assertEqual(result["total"], 2)
        grouped = sd.group_results(result)
        self.assertEqual(grouped["group_a"]["total"], 1)
        self.assertEqual(grouped["group_b"]["total"], 1)
        self.assertAlmostEqual(grouped["all"]["pnl"], 0.50 - 0.54)
        text = sd.format_plain_report(result)
        self.assertIn("135", text)
        self.assertIn("بدون EMA60/RSI", text)
        self.assertIn("BTCUSDT", text)
        self.assertIn("ETHUSDT", text)
        self.assertIn("عكس", text)


if __name__ == "__main__":
    unittest.main()
