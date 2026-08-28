"""Tests for SMI + Donchian 3× AND MACD 3× combined confirm scan."""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

import pullback_bot.smi_don_macd as sdm
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
        "macd_green": np.zeros(len(grid), dtype=bool),
        "macd_red": np.zeros(len(grid), dtype=bool),
    }


def _fill_levels(stepped, grid):
    for main, confirm, _entry, _win, _loss, _group in sdm.LEVELS:
        stepped.setdefault(main, _empty_stepped(grid))
        stepped.setdefault(confirm, _empty_stepped(grid))
    return stepped


def _entry_flip_sell(start):
    return pd.DataFrame(
        {
            "ts": [start + timedelta(minutes=5 * i) for i in range(3)],
            "end_ts": [start + timedelta(minutes=5 * (i + 1)) for i in range(3)],
            "close": [101.0, 100.0, 99.0],
            "don_green": [True, True, False],
            "don_red": [False, False, True],
            "macd_green": [True, True, False],
            "macd_red": [False, False, True],
        }
    )


def _entry_flip_buy(start):
    return pd.DataFrame(
        {
            "ts": [start + timedelta(minutes=5 * i) for i in range(3)],
            "end_ts": [start + timedelta(minutes=5 * (i + 1)) for i in range(3)],
            "close": [99.0, 100.0, 101.0],
            "don_green": [False, False, True],
            "don_red": [True, True, False],
            "macd_green": [False, False, True],
            "macd_red": [True, True, False],
        }
    )


class TestLevels(unittest.TestCase):
    def test_confirm_is_three_times_main(self):
        for main, confirm, _entry, _win, _loss, _group in sdm.LEVELS:
            self.assertEqual(confirm, main * 3)

    def test_user_table(self):
        self.assertEqual(sdm.LEVELS[0], (45, 135, 5, 0.50, 0.37, "a"))
        self.assertEqual(sdm.LEVELS[1], (60, 180, 5, 0.50, 0.37, "a"))
        self.assertEqual(sdm.LEVELS[2], (90, 270, 9, 0.67, 0.54, "b"))
        self.assertEqual(sdm.LEVELS[3], (120, 360, 10, 0.67, 0.54, "b"))
        self.assertEqual(sdm.LEVELS[4], (150, 450, 11, 0.67, 0.54, "b"))


class TestScanSideSynthetic(unittest.TestCase):
    def test_sell_needs_both_confirm_red(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=5 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[45]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_red"] = np.ones(len(grid), dtype=bool)
        stepped[135]["macd_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = sdm._scan_side(
            "sell", stepped, {5: _entry_flip_sell(start)}, grid,
            start, start + timedelta(hours=1), raw_1m, "BTCUSDT",
        )
        sells = [s for s in signals if s["type"] == "sell" and s["base_frame"] == 45]
        self.assertTrue(sells)
        self.assertAlmostEqual(sells[0]["price"], 99.0)

    def test_no_sell_when_only_donchian_red(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=5 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[45]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = sdm._scan_side(
            "sell", stepped, {5: _entry_flip_sell(start)}, grid,
            start, start + timedelta(hours=1), raw_1m, "BTCUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 45 for s in signals))

    def test_no_sell_when_only_macd_red(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=5 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[45]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["macd_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = sdm._scan_side(
            "sell", stepped, {5: _entry_flip_sell(start)}, grid,
            start, start + timedelta(hours=1), raw_1m, "BTCUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 45 for s in signals))

    def test_no_sell_when_entry_macd_not_red(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = _entry_flip_sell(start).copy()
        entry["macd_red"] = [False, False, False]
        entry["macd_green"] = [True, True, True]
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=5 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[45]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_red"] = np.ones(len(grid), dtype=bool)
        stepped[135]["macd_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = sdm._scan_side(
            "sell", stepped, {5: entry}, grid,
            start, start + timedelta(hours=1), raw_1m, "BTCUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 45 for s in signals))

    def test_buy_needs_both_confirm_green(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=5 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[45]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_green"] = np.ones(len(grid), dtype=bool)
        stepped[135]["macd_green"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=0.4)
        signals = sdm._scan_side(
            "buy", stepped, {5: _entry_flip_buy(start)}, grid,
            start, start + timedelta(hours=1), raw_1m, "ETHUSDT",
        )
        buys = [s for s in signals if s["type"] == "buy" and s["base_frame"] == 45]
        self.assertTrue(buys)


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

        with patch.object(sdm, "scan_symbol", side_effect=fake_scan):
            result = sdm.scan_all(("BTCUSDT", "ETHUSDT"), days=30, now=end)
        grouped = sdm.group_results(result)
        self.assertEqual(grouped["group_a"]["total"], 1)
        self.assertEqual(grouped["group_b"]["total"], 1)
        text = sdm.format_plain_report(result)
        self.assertIn("Donchian", text)
        self.assertIn("MACD", text)
        self.assertIn("BTCUSDT", text)
        self.assertIn("بدون EMA60", text)


if __name__ == "__main__":
    unittest.main()
