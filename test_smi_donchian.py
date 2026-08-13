"""Tests for SMI + reverse-sat + 3× Donchian confirm (no EMA/RSI)."""

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
    }


def _fill_levels(stepped, grid):
    for main, cmin, cstop, don_tf, entry in sd.LEVELS:
        stepped.setdefault(main, _empty_stepped(grid))
        stepped.setdefault(don_tf, _empty_stepped(grid))
        stepped.setdefault(cstop, _empty_stepped(grid))
        stepped.setdefault(entry, _empty_stepped(grid))
        for minutes in range(cmin, cstop):
            stepped.setdefault(minutes, _empty_stepped(grid))
    stepped.setdefault(sd.HALT_MAIN_MINUTES, _empty_stepped(grid))
    return stepped


class TestLevels(unittest.TestCase):
    def test_confirm_is_three_times_main(self):
        for main, _cmin, _cstop, don_tf, _entry in sd.LEVELS:
            self.assertEqual(don_tf, main * 3)

    def test_user_table_and_halt(self):
        self.assertEqual(sd.LEVELS[0], (45, 8, 18, 135, 5))
        self.assertEqual(sd.LEVELS[1], (60, 10, 24, 180, 5))
        self.assertEqual(sd.LEVELS[2], (90, 15, 36, 270, 9))
        self.assertEqual(sd.LEVELS[3], (120, 20, 48, 360, 10))
        self.assertEqual(sd.LEVELS[4], (150, 25, 60, 450, 11))
        self.assertEqual(sd.LEVELS[5], (180, 30, 72, 540, 12))
        self.assertEqual(sd.LEVELS[6], (210, 35, 84, 630, 14))
        self.assertEqual(sd.LEVELS[7], (240, 40, 96, 720, 16))
        self.assertEqual(sd.HALT_MAIN_MINUTES, 300)
        self.assertEqual(sd.WIN_PCT, 1.0)
        self.assertEqual(sd.LOSS_PCT, 0.77)

    def test_requested_symbols(self):
        self.assertEqual(
            sd.SYMBOLS,
            (
                "ADAUSDT",
                "SUIUSDT",
                "HYPEUSDT",
                "AVAXUSDT",
                "LINKUSDT",
                "AAVEUSDT",
                "TAOUSDT",
                "XLMUSDT",
                "HBARUSDT",
                "DOTUSDT",
            ),
        )


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
        stepped[45]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[8]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "ADAUSDT",
        )
        sells = [s for s in signals if s["type"] == "sell" and s["base_frame"] == 45]
        self.assertTrue(sells)
        self.assertAlmostEqual(sells[0]["price"], 99.0)

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
        stepped[45]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "ADAUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 45 for s in signals))

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
        stepped[45]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[120]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[8]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_red"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "ADAUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 45 for s in signals))

    def test_five_hour_sat_halts_side(self):
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
        stepped[8]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_red"] = np.ones(len(grid), dtype=bool)
        stepped[300]["sell_sat"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "ADAUSDT",
        )
        self.assertFalse(signals)

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
        stepped[8]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_green"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "ADAUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 45 for s in signals))

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
        stepped[8]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[135]["don_green"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=0.5)
        signals = sd._scan_side(
            "buy", stepped, {5: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "SUIUSDT",
        )
        buys = [s for s in signals if s["type"] == "buy" and s["base_frame"] == 45]
        self.assertTrue(buys)


class TestEvaluateUsesLevelPct(unittest.TestCase):
    def test_loss_at_077(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        future = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=1)],
                "high": [100.4],
                "low": [99.2],
            }
        )
        outcome, _, _ = evaluate_outcome(
            "buy", 100.0, future, win_pct=1.0, loss_pct=0.77
        )
        self.assertEqual(outcome, "loss")


class TestScanAll(unittest.TestCase):
    def test_merges_symbols(self):
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=30)
        ada_trade = {
            "symbol": "ADAUSDT",
            "type": "buy",
            "time": start + timedelta(days=1),
            "price": 1.0,
            "base_frame": 45,
            "confirm_frame": 135,
            "triple_frame": 5,
            "win_pct": 1.0,
            "loss_pct": 0.77,
            "outcome": "win",
            "exit_price": 1.01,
            "exit_ts": start + timedelta(days=1, hours=1),
        }
        sui_trade = {
            "symbol": "SUIUSDT",
            "type": "sell",
            "time": start + timedelta(days=2),
            "price": 2.0,
            "base_frame": 90,
            "confirm_frame": 270,
            "triple_frame": 9,
            "win_pct": 1.0,
            "loss_pct": 0.77,
            "outcome": "loss",
            "exit_price": 2.0154,
            "exit_ts": start + timedelta(days=2, hours=1),
        }

        def fake_scan(symbol, **_kwargs):
            trade = ada_trade if symbol == "ADAUSDT" else sui_trade
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
            result = sd.scan_all(("ADAUSDT", "SUIUSDT"), days=30, now=end)
        grouped = sd.group_results(result)
        self.assertEqual(grouped["all"]["total"], 2)
        text = sd.format_plain_report(result)
        self.assertIn("ADAUSDT", text)
        self.assertIn("الأكبر يلغي الأصغر", text)


if __name__ == "__main__":
    unittest.main()
