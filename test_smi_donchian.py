"""Tests for SMI-only + Donchian confirm (no EMA60 / RSI)."""

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
    for main, confirm, _entry, _win, _loss in sd.LEVELS:
        stepped.setdefault(main, _empty_stepped(grid))
        stepped.setdefault(confirm, _empty_stepped(grid))
    return stepped


class TestLevels(unittest.TestCase):
    def test_confirm_is_three_times_main(self):
        for main, confirm, _entry, _win, _loss in sd.LEVELS:
            self.assertEqual(confirm, main * 3)

    def test_user_table(self):
        self.assertEqual(sd.LEVELS[0], (45, 135, 5, 0.50, 0.37))
        self.assertEqual(sd.LEVELS[1], (60, 180, 5, 0.50, 0.37))
        self.assertEqual(sd.LEVELS[2], (90, 270, 9, 0.67, 0.54))
        self.assertEqual(sd.LEVELS[3], (120, 360, 10, 0.67, 0.54))
        self.assertEqual(sd.LEVELS[4], (150, 450, 11, 0.67, 0.54))


class TestScanSide(unittest.TestCase):
    def test_sell_enters_after_confirm_red_and_entry_flip(self):
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
        stepped[45]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = sd._scan_side(
            "sell", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "BTCUSDT",
        )
        self.assertTrue(any(s["type"] == "sell" for s in signals))
        first = next(s for s in signals if s["type"] == "sell")
        self.assertEqual(first["base_frame"], 45)
        self.assertEqual(first["confirm_frame"], 135)
        self.assertEqual(first["triple_frame"], 5)
        self.assertAlmostEqual(first["price"], 99.0)
        self.assertAlmostEqual(first["win_pct"], 0.50)
        self.assertAlmostEqual(first["loss_pct"], 0.37)

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
        stepped[45]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_green"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = sd._scan_side(
            "sell", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "BTCUSDT",
        )
        self.assertEqual(signals, [])

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
        stepped[45]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_green"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=0.4)
        signals = sd._scan_side(
            "buy", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "ETHUSDT",
        )
        self.assertTrue(any(s["type"] == "buy" for s in signals))
        first = next(s for s in signals if s["type"] == "buy")
        self.assertEqual(first["symbol"], "ETHUSDT")
        self.assertAlmostEqual(first["price"], 101.0)

    def test_no_entry_if_donchian_never_clears(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=5 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=5 * (i + 1)) for i in range(3)],
                "close": [99.0, 98.0, 97.0],
                "don_green": [False, False, False],
                "don_red": [True, True, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=5 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[45]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = sd._scan_side(
            "sell", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "BTCUSDT",
        )
        self.assertEqual(signals, [])

    def test_larger_main_sat_cancels_smaller(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry5 = pd.DataFrame(
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
        stepped[45]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_red"] = np.ones(len(grid), dtype=bool)
        stepped[90]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[270]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = sd._scan_side(
            "sell",
            stepped,
            {5: entry5, 9: entry5.copy()},
            grid,
            start,
            start + timedelta(hours=1),
            raw_1m,
            "BTCUSDT",
        )
        self.assertTrue(signals)
        self.assertTrue(all(s["base_frame"] == 90 for s in signals))

    def test_long_level_uses_067_054(self):
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
        stepped[270]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 50, minutes=1, price=100.0, drift=-0.4)
        signals = sd._scan_side(
            "sell", stepped, {9: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "XRPUSDT",
        )
        self.assertTrue(signals)
        self.assertAlmostEqual(signals[0]["win_pct"], 0.67)
        self.assertAlmostEqual(signals[0]["loss_pct"], 0.54)


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

    def test_entry_has_no_ema(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        raw = _bars(start, 80, minutes=1, price=100.0, drift=0.1)
        feat = sd._entry_features(raw, 1)
        self.assertIsNotNone(feat)
        self.assertNotIn("ema", feat.columns)
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
            "triple_frame": 5,
            "win_pct": 0.50,
            "loss_pct": 0.37,
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
            "triple_frame": 9,
            "win_pct": 0.67,
            "loss_pct": 0.54,
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
        self.assertEqual(grouped["short"]["total"], 1)
        self.assertEqual(grouped["long"]["total"], 1)
        self.assertAlmostEqual(grouped["all"]["pnl"], 0.50 - 0.54)
        text = sd.format_plain_report(result)
        self.assertIn("135", text)
        self.assertIn("بدون EMA60/RSI", text)
        self.assertIn("BTCUSDT", text)
        self.assertIn("ETHUSDT", text)


if __name__ == "__main__":
    unittest.main()
