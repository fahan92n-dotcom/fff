"""Tests for the standalone pullback bot strategy."""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

import pullback_bot.strategy as pb
from pullback_bot import main as pullback_main


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


class TestPullbackLevels(unittest.TestCase):
    def test_levels_match_agreed_table(self):
        self.assertEqual(pb.LEVELS[0], (30, 5, 12, 2))
        self.assertEqual(pb.LEVELS[9], (270, 45, 84, 18))
        self.assertEqual(pb.LEVELS[11], (330, 55, 132, 22))
        self.assertEqual(pb.LEVELS[-1], (360, 60, 144, 24))
        self.assertEqual(pb.HALT_MAIN_MINUTES, 420)
        self.assertEqual(pb.EMA_SPAN, 60)

    def test_confirm_range_is_min_to_stop_exclusive(self):
        _main, confirm_min, confirm_stop, _entry = pb.LEVELS[0]
        accepted = list(range(confirm_min, confirm_stop))
        self.assertEqual(accepted[0], 5)
        self.assertEqual(accepted[-1], 11)
        self.assertNotIn(12, accepted)


class TestMainEmaFilter(unittest.TestCase):
    def test_buy_main_requires_close_above_ema60(self):
        """Main buy sat needs close above EMA60 on the formation candle."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        n = 400
        closes = np.concatenate(
            [np.full(300, 100.0), np.linspace(100.0, 130.0, n - 300)]
        )
        rows = []
        for i, close in enumerate(closes):
            ts = start + timedelta(minutes=i)
            rows.append(
                {
                    "ts": ts,
                    "open": float(close),
                    "high": float(close) + 1.0,
                    "low": float(close) - 1.0,
                    "close": float(close),
                    "vol": 1.0,
                }
            )
        raw = pd.DataFrame(rows)
        smi = np.concatenate([np.full(350, 0.0), np.full(n - 350, 50.0)])
        with patch.object(pb, "calc_smi", return_value=(
            pd.Series(smi),
            pd.Series(smi),
            pd.Series(smi),
        )):
            feat = pb._frame_features(raw, 1)
        self.assertIsNotNone(feat)
        self.assertTrue(bool(feat["buy_main"].iloc[-1]))
        self.assertFalse(bool(feat["sell_main"].iloc[-1]))

    def test_sell_main_requires_close_below_ema60(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        n = 400
        closes = np.concatenate(
            [np.full(300, 100.0), np.linspace(100.0, 70.0, n - 300)]
        )
        rows = []
        for i, close in enumerate(closes):
            ts = start + timedelta(minutes=i)
            rows.append(
                {
                    "ts": ts,
                    "open": float(close),
                    "high": float(close) + 1.0,
                    "low": float(close) - 1.0,
                    "close": float(close),
                    "vol": 1.0,
                }
            )
        raw = pd.DataFrame(rows)
        smi = np.concatenate([np.full(350, 0.0), np.full(n - 350, -50.0)])
        with patch.object(pb, "calc_smi", return_value=(
            pd.Series(smi),
            pd.Series(smi),
            pd.Series(smi),
        )):
            feat = pb._frame_features(raw, 1)
        self.assertIsNotNone(feat)
        self.assertTrue(bool(feat["sell_main"].iloc[-1]))
        self.assertFalse(bool(feat["buy_main"].iloc[-1]))

    def test_buy_main_rejected_when_close_below_ema60(self):
        """Buy sat born below EMA60 is dead for the whole episode."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        n = 400
        closes = np.concatenate(
            [np.full(300, 100.0), np.linspace(100.0, 70.0, n - 300)]
        )
        rows = []
        for i, close in enumerate(closes):
            ts = start + timedelta(minutes=i)
            rows.append(
                {
                    "ts": ts,
                    "open": float(close),
                    "high": float(close) + 1.0,
                    "low": float(close) - 1.0,
                    "close": float(close),
                    "vol": 1.0,
                }
            )
        raw = pd.DataFrame(rows)
        smi = np.concatenate([np.full(350, 0.0), np.full(n - 350, 50.0)])
        with patch.object(pb, "calc_smi", return_value=(
            pd.Series(smi),
            pd.Series(smi),
            pd.Series(smi),
        )):
            feat = pb._frame_features(raw, 1)
        self.assertIsNotNone(feat)
        self.assertFalse(bool(feat["buy_main"].iloc[-1]))


class TestSatFormationEma(unittest.TestCase):
    def test_episode_valid_only_when_formation_ok(self):
        sat = np.array([False, True, True, False, True, True])
        # Episode1 forms with EMA ok; episode2 forms with EMA bad.
        formation_ok = np.array([False, True, True, False, False, True])
        out = pb._sat_episode_formation_valid(sat, formation_ok)
        np.testing.assert_array_equal(
            out, np.array([False, True, True, False, False, False])
        )

    def test_buy_main_ok_even_without_rsi_check(self):
        """RSI is ignored — buy sat above EMA60 is valid for main entry."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        n = 400
        closes = np.concatenate(
            [np.full(300, 100.0), np.linspace(100.0, 130.0, n - 300)]
        )
        rows = []
        for i, close in enumerate(closes):
            ts = start + timedelta(minutes=i)
            rows.append(
                {
                    "ts": ts,
                    "open": float(close),
                    "high": float(close) + 1.0,
                    "low": float(close) - 1.0,
                    "close": float(close),
                    "vol": 1.0,
                }
            )
        raw = pd.DataFrame(rows)
        # Sat forms only after price has risen above EMA60.
        smi = np.concatenate([np.full(350, 0.0), np.full(n - 350, 50.0)])
        with patch.object(pb, "calc_smi", return_value=(
            pd.Series(smi),
            pd.Series(smi),
            pd.Series(smi),
        )):
            feat = pb._frame_features(raw, 1)
        self.assertIsNotNone(feat)
        self.assertTrue(bool(feat["buy_sat"].iloc[-1]))
        self.assertTrue(bool(feat["buy_main"].iloc[-1]))

    def test_sell_main_ok_even_without_rsi_check(self):
        """RSI is ignored — sell sat below EMA60 is valid for main entry."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        n = 400
        closes = np.concatenate(
            [np.full(300, 100.0), np.linspace(100.0, 70.0, n - 300)]
        )
        rows = []
        for i, close in enumerate(closes):
            ts = start + timedelta(minutes=i)
            rows.append(
                {
                    "ts": ts,
                    "open": float(close),
                    "high": float(close) + 1.0,
                    "low": float(close) - 1.0,
                    "close": float(close),
                    "vol": 1.0,
                }
            )
        raw = pd.DataFrame(rows)
        smi = np.concatenate([np.full(350, 0.0), np.full(n - 350, -50.0)])
        with patch.object(pb, "calc_smi", return_value=(
            pd.Series(smi),
            pd.Series(smi),
            pd.Series(smi),
        )):
            feat = pb._frame_features(raw, 1)
        self.assertIsNotNone(feat)
        self.assertTrue(bool(feat["sell_sat"].iloc[-1]))
        self.assertTrue(bool(feat["sell_main"].iloc[-1]))

    def test_buy_main_ok_when_sat_forms_above_ema60(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        n = 400
        closes = np.concatenate(
            [np.full(300, 100.0), np.linspace(100.0, 130.0, n - 300)]
        )
        rows = []
        for i, close in enumerate(closes):
            ts = start + timedelta(minutes=i)
            rows.append(
                {
                    "ts": ts,
                    "open": float(close),
                    "high": float(close) + 1.0,
                    "low": float(close) - 1.0,
                    "close": float(close),
                    "vol": 1.0,
                }
            )
        raw = pd.DataFrame(rows)
        # Sat starts after close is above EMA60.
        smi = np.concatenate([np.full(350, 0.0), np.full(n - 350, 50.0)])
        with patch.object(pb, "calc_smi", return_value=(
            pd.Series(smi),
            pd.Series(smi),
            pd.Series(smi),
        )):
            feat = pb._frame_features(raw, 1)
        self.assertIsNotNone(feat)
        self.assertTrue(bool(feat["buy_main"].iloc[-1]))

    def test_main_stays_valid_if_live_ema_breaks_after_formation(self):
        """After a good first sat close, later EMA breaks do not kill main."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        n = 400
        # Rally (formation above EMA) then drop back below EMA.
        closes = np.concatenate(
            [
                np.full(280, 100.0),
                np.linspace(100.0, 130.0, 40),
                np.linspace(130.0, 90.0, n - 320),
            ]
        )
        rows = []
        for i, close in enumerate(closes):
            ts = start + timedelta(minutes=i)
            rows.append(
                {
                    "ts": ts,
                    "open": float(close),
                    "high": float(close) + 1.0,
                    "low": float(close) - 1.0,
                    "close": float(close),
                    "vol": 1.0,
                }
            )
        raw = pd.DataFrame(rows)
        # Sat forms near the top (above EMA), then stays on while price drops.
        smi = np.concatenate([np.full(310, 0.0), np.full(n - 310, 50.0)])
        with patch.object(pb, "calc_smi", return_value=(
            pd.Series(smi),
            pd.Series(smi),
            pd.Series(smi),
        )):
            feat = pb._frame_features(raw, 1)
        self.assertIsNotNone(feat)
        self.assertTrue(bool(feat["buy_sat"].iloc[-1]))
        self.assertLess(float(feat["close"].iloc[-1]), float(feat["ema"].iloc[-1]))
        self.assertTrue(bool(feat["buy_main"].iloc[-1]))


class TestBoolStep(unittest.TestCase):
    def test_ffill_latest_closed(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        ends = np.array([start + timedelta(minutes=30), start + timedelta(minutes=60)])
        values = np.array([True, False])
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=m) for m in range(15, 75, 15)]
        )
        out = pb._bool_step(ends, values, grid)
        self.assertEqual(list(out), [False, True, True, False])


def _empty_stepped(grid):
    return {
        "sell_main": np.zeros(len(grid), dtype=bool),
        "buy_main": np.zeros(len(grid), dtype=bool),
        "sell_sat": np.zeros(len(grid), dtype=bool),
        "buy_sat": np.zeros(len(grid), dtype=bool),
    }


def _fill_levels(stepped, grid):
    for main, cmin, cstop, _entry in pb.LEVELS:
        stepped.setdefault(main, _empty_stepped(grid))
        for minutes in range(cmin, cstop + 1):
            stepped.setdefault(minutes, _empty_stepped(grid))
    return stepped


class TestScanSideSynthetic(unittest.TestCase):
    def test_sell_entry_after_counter_then_donchian_flip(self):
        """Main sat + reverse sat → EMA60 cross down + green→red flip."""
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=2 * i) for i in range(6)],
                "end_ts": [start + timedelta(minutes=2 * (i + 1)) for i in range(6)],
                "close": [100.0, 101.0, 102.0, 99.0, 98.0, 97.0],
                "ema": [99.0, 99.5, 100.0, 100.5, 100.2, 99.8],
                "don": [1, 1, 1, -1, -1, -1],
                "above_ema": [True, True, True, False, False, False],
                "below_ema": [False, False, False, True, True, True],
                "don_green": [True, True, True, False, False, False],
                "don_red": [False, False, False, True, True, True],
            }
        )
        grid_start = start + timedelta(minutes=2)
        grid = pd.DatetimeIndex(
            [grid_start + timedelta(minutes=i) for i in range(0, 12)]
        )
        stepped = _fill_levels({}, grid)
        stepped[30]["sell_main"] = np.ones(len(grid), dtype=bool)
        stepped[30]["sell_sat"] = np.ones(len(grid), dtype=bool)
        for minutes in range(5, 12):
            stepped[minutes]["buy_sat"] = np.ones(len(grid), dtype=bool)

        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = pb._scan_side(
            "sell",
            stepped,
            {2: entry},
            grid,
            start,
            start + timedelta(hours=1),
            raw_1m,
        )
        self.assertTrue(any(s["type"] == "sell" for s in signals))
        first = next(s for s in signals if s["type"] == "sell")
        self.assertEqual(first["base_frame"], 30)
        self.assertAlmostEqual(first["price"], 99.0)

    def test_sell_entry_cross_and_flip_on_different_candles(self):
        """Conditions complete at different times; enter when both hold."""
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=2 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=2 * (i + 1)) for i in range(3)],
                "close": [101.0, 99.5, 99.0],
                "ema": [100.0, 100.0, 100.2],
                "don": [1, 1, -1],
                "above_ema": [True, False, False],
                "below_ema": [False, True, True],
                "don_green": [True, True, False],
                "don_red": [False, False, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=2 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[30]["sell_main"] = np.ones(len(grid), dtype=bool)
        stepped[30]["sell_sat"] = np.ones(len(grid), dtype=bool)
        for minutes in range(5, 12):
            stepped[minutes]["buy_sat"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = pb._scan_side(
            "sell",
            stepped,
            {2: entry},
            grid,
            start,
            start + timedelta(hours=1),
            raw_1m,
        )
        self.assertTrue(any(s["type"] == "sell" for s in signals))
        first = next(s for s in signals if s["type"] == "sell")
        self.assertAlmostEqual(first["price"], 99.0)

    def test_sell_entry_partial_start_does_not_enter(self):
        """Partial at counter start (one condition true) never arms without both-clear."""
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=2 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=2 * (i + 1)) for i in range(3)],
                "close": [99.0, 98.0, 97.0],
                "ema": [100.0, 100.5, 100.2],
                "don": [1, -1, -1],
                "above_ema": [False, False, False],
                "below_ema": [True, True, True],
                "don_green": [True, False, False],
                "don_red": [False, True, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=2 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[30]["sell_main"] = np.ones(len(grid), dtype=bool)
        stepped[30]["sell_sat"] = np.ones(len(grid), dtype=bool)
        for minutes in range(5, 12):
            stepped[minutes]["buy_sat"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = pb._scan_side(
            "sell",
            stepped,
            {2: entry},
            grid,
            start,
            start + timedelta(hours=1),
            raw_1m,
        )
        self.assertEqual(signals, [])

    def test_sell_entry_requires_both_clear_before_both_hold(self):
        """After reverse sat: both unmet, then both hold → enter."""
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=2 * i) for i in range(4)],
                "end_ts": [start + timedelta(minutes=2 * (i + 1)) for i in range(4)],
                # partial → both clear (green+above) → both hold (red+below)
                "close": [99.0, 101.0, 101.0, 98.0],
                "ema": [100.0, 100.0, 100.0, 100.2],
                "don": [1, 1, 1, -1],
                "above_ema": [False, True, True, False],
                "below_ema": [True, False, False, True],
                "don_green": [True, True, True, False],
                "don_red": [False, False, False, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=2 * (i + 1)) for i in range(4)]
        )
        stepped = _fill_levels({}, grid)
        stepped[30]["sell_main"] = np.ones(len(grid), dtype=bool)
        stepped[30]["sell_sat"] = np.ones(len(grid), dtype=bool)
        for minutes in range(5, 12):
            stepped[minutes]["buy_sat"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = pb._scan_side(
            "sell",
            stepped,
            {2: entry},
            grid,
            start,
            start + timedelta(hours=1),
            raw_1m,
        )
        self.assertTrue(any(s["type"] == "sell" for s in signals))
        first = next(s for s in signals if s["type"] == "sell")
        self.assertAlmostEqual(first["price"], 98.0)

    def test_no_entry_without_donchian_flip(self):
        """Donchian stays green — sell conditions never both hold."""
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=2 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=2 * (i + 1)) for i in range(3)],
                "close": [101.0, 99.5, 99.0],
                "ema": [100.0, 100.0, 100.2],
                "don": [1, 1, 1],
                "above_ema": [True, False, False],
                "below_ema": [False, True, True],
                "don_green": [True, True, True],
                "don_red": [False, False, False],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=2 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[30]["sell_main"] = np.ones(len(grid), dtype=bool)
        stepped[30]["sell_sat"] = np.ones(len(grid), dtype=bool)
        for minutes in range(5, 12):
            stepped[minutes]["buy_sat"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = pb._scan_side(
            "sell",
            stepped,
            {2: entry},
            grid,
            start,
            start + timedelta(hours=1),
            raw_1m,
        )
        self.assertEqual(signals, [])

    def test_buy_entry_red_below_then_green_above(self):
        """Buy: Donchian red + below EMA60 first, enter on green + above."""
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=2 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=2 * (i + 1)) for i in range(3)],
                "close": [99.0, 99.5, 101.0],
                "ema": [100.0, 100.0, 100.2],
                "don": [-1, -1, 1],
                "above_ema": [False, False, True],
                "below_ema": [True, True, False],
                "don_green": [False, False, True],
                "don_red": [True, True, False],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=2 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[30]["buy_main"] = np.ones(len(grid), dtype=bool)
        stepped[30]["buy_sat"] = np.ones(len(grid), dtype=bool)
        for minutes in range(5, 12):
            stepped[minutes]["sell_sat"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=0.4)
        signals = pb._scan_side(
            "buy",
            stepped,
            {2: entry},
            grid,
            start,
            start + timedelta(hours=1),
            raw_1m,
        )
        self.assertTrue(any(s["type"] == "buy" for s in signals))
        first = next(s for s in signals if s["type"] == "buy")
        self.assertAlmostEqual(first["price"], 101.0)

    def test_both_hold_at_start_waits_for_clear_then_enters(self):
        """Both already true at reverse sat → wait for clear, then enter on hold."""
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=2 * i) for i in range(4)],
                "end_ts": [start + timedelta(minutes=2 * (i + 1)) for i in range(4)],
                # 0: both hold (no reject) → 1: still hold → 2: both clear → 3: both hold
                "close": [99.0, 98.5, 101.0, 98.0],
                "ema": [100.0, 100.0, 100.0, 100.2],
                "don": [-1, -1, 1, -1],
                "above_ema": [False, False, True, False],
                "below_ema": [True, True, False, True],
                "don_green": [False, False, True, False],
                "don_red": [True, True, False, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=2 * (i + 1)) for i in range(4)]
        )
        stepped = _fill_levels({}, grid)
        stepped[30]["sell_main"] = np.ones(len(grid), dtype=bool)
        stepped[30]["sell_sat"] = np.ones(len(grid), dtype=bool)
        for minutes in range(5, 12):
            stepped[minutes]["buy_sat"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = pb._scan_side(
            "sell",
            stepped,
            {2: entry},
            grid,
            start,
            start + timedelta(hours=1),
            raw_1m,
        )
        self.assertTrue(any(s["type"] == "sell" for s in signals))
        first = next(s for s in signals if s["type"] == "sell")
        self.assertAlmostEqual(first["price"], 98.0)

    def test_no_entry_while_both_hold_never_clears(self):
        """If conditions stay aligned and never go unmet, no entry."""
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=2 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=2 * (i + 1)) for i in range(3)],
                "close": [99.0, 98.0, 97.0],
                "ema": [100.0, 100.5, 100.2],
                "don": [-1, -1, -1],
                "above_ema": [False, False, False],
                "below_ema": [True, True, True],
                "don_green": [False, False, False],
                "don_red": [True, True, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=2 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[30]["sell_main"] = np.ones(len(grid), dtype=bool)
        stepped[30]["sell_sat"] = np.ones(len(grid), dtype=bool)
        for minutes in range(5, 12):
            stepped[minutes]["buy_sat"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = pb._scan_side(
            "sell",
            stepped,
            {2: entry},
            grid,
            start,
            start + timedelta(hours=1),
            raw_1m,
        )
        self.assertEqual(signals, [])

    def test_entry_survives_counter_gap_after_clear_wait(self):
        """After reverse sat, clear-wait survives a counter gap then enters."""
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=2 * i) for i in range(4)],
                "end_ts": [start + timedelta(minutes=2 * (i + 1)) for i in range(4)],
                # 0: both hold; 1: gap (still hold); 2: both clear; 3: both hold → enter
                "close": [99.0, 98.5, 101.0, 98.0],
                "ema": [100.0, 100.0, 100.0, 100.2],
                "don": [-1, -1, 1, -1],
                "above_ema": [False, False, True, False],
                "below_ema": [True, True, False, True],
                "don_green": [False, False, True, False],
                "don_red": [True, True, False, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=2 * (i + 1)) for i in range(4)]
        )
        stepped = _fill_levels({}, grid)
        stepped[30]["sell_main"] = np.ones(len(grid), dtype=bool)
        stepped[30]["sell_sat"] = np.ones(len(grid), dtype=bool)
        for minutes in range(5, 12):
            stepped[minutes]["buy_sat"] = np.array([True, False, True, True])
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = pb._scan_side(
            "sell",
            stepped,
            {2: entry},
            grid,
            start,
            start + timedelta(hours=1),
            raw_1m,
        )
        self.assertTrue(any(s["type"] == "sell" for s in signals))
        first = next(s for s in signals if s["type"] == "sell")
        self.assertAlmostEqual(first["price"], 98.0)

    def test_no_entry_without_reverse_sat(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=2 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=2 * (i + 1)) for i in range(3)],
                "close": [101.0, 99.0, 98.0],
                "ema": [100.0, 100.5, 100.2],
                "don": [1, -1, -1],
                "above_ema": [True, False, False],
                "below_ema": [False, True, True],
                "don_green": [True, False, False],
                "don_red": [False, True, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=2 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[30]["sell_main"] = np.ones(len(grid), dtype=bool)
        stepped[30]["sell_sat"] = np.ones(len(grid), dtype=bool)
        # No counter buy-sat on 5..11
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = pb._scan_side(
            "sell",
            stepped,
            {2: entry},
            grid,
            start,
            start + timedelta(hours=1),
            raw_1m,
        )
        self.assertEqual(signals, [])

    def test_entry_allowed_after_confirm_clears(self):
        """Reverse sat must confirm once; entry flip may happen after it clears."""
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=2 * i) for i in range(4)],
                "end_ts": [start + timedelta(minutes=2 * (i + 1)) for i in range(4)],
                "close": [101.0, 102.0, 99.0, 98.0],
                "ema": [99.5, 100.0, 100.5, 100.2],
                "don": [1, 1, -1, -1],
                "above_ema": [True, True, False, False],
                "below_ema": [False, False, True, True],
                "don_green": [True, True, False, False],
                "don_red": [False, False, True, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=2 * (i + 1)) for i in range(4)]
        )
        stepped = _fill_levels({}, grid)
        stepped[30]["sell_main"] = np.ones(len(grid), dtype=bool)
        stepped[30]["sell_sat"] = np.ones(len(grid), dtype=bool)
        for minutes in range(5, 12):
            stepped[minutes]["buy_sat"] = np.array([True, True, False, False])

        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = pb._scan_side(
            "sell",
            stepped,
            {2: entry},
            grid,
            start,
            start + timedelta(hours=1),
            raw_1m,
        )
        self.assertTrue(any(s["type"] == "sell" for s in signals))
        first = next(s for s in signals if s["type"] == "sell")
        self.assertAlmostEqual(first["price"], 99.0)

    def test_confirm_stop_blocks_entry(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start],
                "end_ts": [start + timedelta(minutes=2)],
                "close": [99.0],
                "ema": [100.0],
                "don": [-1],
                "above_ema": [False],
                "below_ema": [True],
                "don_green": [False],
                "don_red": [True],
            }
        )
        grid = pd.DatetimeIndex([start + timedelta(minutes=2)])
        stepped = {}
        for main, cmin, cstop, _entry in pb.LEVELS:
            stepped[main] = {
                "sell_main": np.array([main == 30]),
                "buy_main": np.array([False]),
                "sell_sat": np.array([main == 30]),
                "buy_sat": np.array([False]),
            }
            for minutes in range(cmin, cstop + 1):
                stepped[minutes] = {
                    "sell_main": np.array([False]),
                    "buy_main": np.array([False]),
                    "sell_sat": np.array([False]),
                    "buy_sat": np.array([minutes == 12 or 5 <= minutes <= 11]),
                }
        raw_1m = _bars(start, 10, minutes=1, price=100.0, drift=-0.5)
        signals = pb._scan_side(
            "sell",
            stepped,
            {2: entry},
            grid,
            start,
            start + timedelta(hours=1),
            raw_1m,
        )
        self.assertEqual(signals, [])

    def test_larger_smi_sat_cancels_smaller_even_without_ema(self):
        """90m SMI sat alone cancels 60m; without EMA on 90m → no entry."""
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        # 60m entry TF is 4m; build candles that would otherwise sell-enter.
        entry4 = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=4 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=4 * (i + 1)) for i in range(3)],
                "close": [101.0, 99.0, 98.0],
                "ema": [100.0, 100.5, 100.2],
                "don": [1, -1, -1],
                "above_ema": [True, False, False],
                "below_ema": [False, True, True],
                "don_green": [True, False, False],
                "don_red": [False, True, True],
            }
        )
        entry6 = entry4.copy()
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=4 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        # 60m fully ready to enter; 90m has SMI sat only (no sell_main).
        stepped[60]["sell_main"] = np.ones(len(grid), dtype=bool)
        stepped[60]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[90]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[90]["sell_main"] = np.zeros(len(grid), dtype=bool)
        for minutes in range(10, 24):
            stepped[minutes]["buy_sat"] = np.ones(len(grid), dtype=bool)
        for minutes in range(15, 36):
            stepped[minutes]["buy_sat"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = pb._scan_side(
            "sell",
            stepped,
            {4: entry4, 6: entry6},
            grid,
            start,
            start + timedelta(hours=1),
            raw_1m,
        )
        self.assertEqual(signals, [])

    def test_smaller_enters_when_larger_has_no_smi_sat(self):
        """Without larger SMI sat, 60m full main can still enter."""
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry4 = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=4 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=4 * (i + 1)) for i in range(3)],
                "close": [101.0, 99.0, 98.0],
                "ema": [100.0, 100.5, 100.2],
                "don": [1, -1, -1],
                "above_ema": [True, False, False],
                "below_ema": [False, True, True],
                "don_green": [True, False, False],
                "don_red": [False, True, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=4 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[60]["sell_main"] = np.ones(len(grid), dtype=bool)
        stepped[60]["sell_sat"] = np.ones(len(grid), dtype=bool)
        for minutes in range(10, 24):
            stepped[minutes]["buy_sat"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = pb._scan_side(
            "sell",
            stepped,
            {4: entry4},
            grid,
            start,
            start + timedelta(hours=1),
            raw_1m,
        )
        self.assertTrue(any(s["base_frame"] == 60 for s in signals))


class TestStandaloneBot(unittest.TestCase):
    def test_format_empty(self):
        result = {
            "ready": True,
            "start": datetime(2026, 8, 1, tzinfo=timezone.utc),
            "end": datetime(2026, 8, 8, tzinfo=timezone.utc),
            "wins": [],
            "losses": [],
            "opens": [],
            "total": 0,
            "symbol": "BTCUSDT",
            "market": "spot-vision",
        }
        chunks = pb.format_pullback_week_report(result)
        self.assertEqual(len(chunks), 1)
        self.assertIn("لا توجد صفقات", chunks[0])

    def test_pullback_not_advertised_in_cascade_help(self):
        import fahadal92 as cascade

        calls = []
        with patch.object(cascade, "send_telegram", side_effect=lambda m, c=None: calls.append(m)):
            cascade._dispatch_command_inner("/help", "1")
        self.assertTrue(calls)
        self.assertNotIn("Pullback", calls[0])
        self.assertNotIn("week_pullback", calls[0])

    def test_standalone_dispatch_week(self):
        calls = []

        def fake_send(msg, chat_id=None):
            calls.append(msg)

        with patch.object(pb, "scan_pullback_week", return_value={
            "ready": True,
            "start": datetime(2026, 8, 1, tzinfo=timezone.utc),
            "end": datetime(2026, 8, 8, tzinfo=timezone.utc),
            "wins": [],
            "losses": [],
            "opens": [],
            "total": 0,
            "symbol": "BTCUSDT",
            "market": "spot-vision",
        }), patch.object(pullback_main, "send_telegram", side_effect=fake_send):
            pullback_main._dispatch_command("/week", "1")
        self.assertTrue(any("Pullback" in c for c in calls))

    def test_help_lists_main_ema60_rule(self):
        calls = []

        def fake_send(msg, chat_id=None):
            calls.append(msg)

        with patch.object(pullback_main, "send_telegram", side_effect=fake_send):
            pullback_main._dispatch_command("/help", "1")
        self.assertEqual(len(calls), 1)
        help_text = calls[0]
        self.assertIn("EMA60", help_text)
        self.assertIn("تشبّع", help_text)
        self.assertIn("إلغاء", help_text)
        self.assertIn("عكس", help_text)
        self.assertIn("/month", help_text)
        self.assertNotIn("RSI", help_text)

    def test_standalone_dispatch_month(self):
        calls = []

        def fake_send(msg, chat_id=None):
            calls.append(msg)

        with patch.object(
            pb,
            "scan_pullback_week",
            return_value={
                "ready": True,
                "start": datetime(2026, 7, 11, tzinfo=timezone.utc),
                "end": datetime(2026, 8, 10, tzinfo=timezone.utc),
                "days": 30,
                "wins": [],
                "losses": [],
                "opens": [],
                "total": 0,
                "symbol": "BTCUSDT",
                "market": "spot-vision",
            },
        ) as scan, patch.object(pullback_main, "send_telegram", side_effect=fake_send):
            pullback_main._dispatch_command("/month", "1")
        scan.assert_called_once()
        self.assertEqual(scan.call_args.kwargs.get("days"), 30)
        self.assertTrue(any("30" in c for c in calls))

    def test_format_report_uses_days_label(self):
        result = {
            "ready": True,
            "start": datetime(2026, 7, 11, tzinfo=timezone.utc),
            "end": datetime(2026, 8, 10, tzinfo=timezone.utc),
            "days": 30,
            "wins": [],
            "losses": [],
            "opens": [],
            "total": 0,
            "symbol": "BTCUSDT",
            "market": "spot-vision",
        }
        chunks = pb.format_pullback_week_report(result)
        self.assertIn("آخر 30 يومًا", chunks[0])

    def test_scan_side_uses_passed_symbol(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=2 * i) for i in range(3)],
                "end_ts": [start + timedelta(minutes=2 * (i + 1)) for i in range(3)],
                "close": [101.0, 99.0, 98.0],
                "ema": [100.0, 100.5, 100.2],
                "don": [1, -1, -1],
                "above_ema": [True, False, False],
                "below_ema": [False, True, True],
                "don_green": [True, False, False],
                "don_red": [False, True, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=2 * (i + 1)) for i in range(3)]
        )
        stepped = _fill_levels({}, grid)
        stepped[30]["sell_main"] = np.ones(len(grid), dtype=bool)
        stepped[30]["sell_sat"] = np.ones(len(grid), dtype=bool)
        for minutes in range(5, 12):
            stepped[minutes]["buy_sat"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.4)
        signals = pb._scan_side(
            "sell",
            stepped,
            {2: entry},
            grid,
            start,
            start + timedelta(hours=1),
            raw_1m,
            "ETHUSDT",
        )
        self.assertTrue(signals)
        self.assertTrue(all(s["symbol"] == "ETHUSDT" for s in signals))

    def test_multi_scan_merges_symbols(self):
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=30)
        btc = {
            "symbol": "BTCUSDT",
            "type": "buy",
            "time": start + timedelta(days=1),
            "price": 100.0,
            "base_frame": 45,
            "confirm_frame": 8,
            "triple_frame": 3,
            "confirm_stop": 18,
            "outcome": "win",
            "exit_price": 101.0,
            "exit_ts": start + timedelta(days=1, hours=1),
        }
        eth = {
            "symbol": "ETHUSDT",
            "type": "sell",
            "time": start + timedelta(days=2),
            "price": 200.0,
            "base_frame": 60,
            "confirm_frame": 10,
            "triple_frame": 4,
            "confirm_stop": 24,
            "outcome": "loss",
            "exit_price": 201.4,
            "exit_ts": start + timedelta(days=2, hours=1),
        }

        def fake_scan(*, symbol, **_kwargs):
            trade = btc if symbol == "BTCUSDT" else eth
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

        with patch.object(pb, "scan_pullback_week", side_effect=fake_scan):
            result = pb.scan_pullback_symbols(
                ("BTCUSDT", "ETHUSDT"), days=30, now=end
            )
        self.assertEqual(result["total"], 2)
        text = "\n".join(pb.format_pullback_multi_report(result))
        self.assertIn("BTCUSDT", text)
        self.assertIn("ETHUSDT", text)
        self.assertIn("EMA60", text)


if __name__ == "__main__":
    unittest.main()
