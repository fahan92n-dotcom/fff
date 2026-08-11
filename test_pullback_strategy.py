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
        """Main buy sat needs SMI/RSI AND close above EMA60 on that candle."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        # Flat then ramp so EMA60 lags below close near the end.
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
        with patch.object(pb, "calc_smi", return_value=(
            pd.Series(np.full(n, 50.0)),
            pd.Series(np.full(n, 50.0)),
            pd.Series(np.full(n, 50.0)),
        )), patch.object(pb, "calc_rsi_tv", return_value=pd.Series(np.full(n, 60.0))):
            feat = pb._frame_features(raw, 1)
        self.assertIsNotNone(feat)
        # Last bar: close above EMA after the ramp → buy_main True.
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
        with patch.object(pb, "calc_smi", return_value=(
            pd.Series(np.full(n, -50.0)),
            pd.Series(np.full(n, -50.0)),
            pd.Series(np.full(n, -50.0)),
        )), patch.object(pb, "calc_rsi_tv", return_value=pd.Series(np.full(n, 40.0))):
            feat = pb._frame_features(raw, 1)
        self.assertIsNotNone(feat)
        self.assertTrue(bool(feat["sell_main"].iloc[-1]))
        self.assertFalse(bool(feat["buy_main"].iloc[-1]))

    def test_buy_main_rejected_when_close_below_ema60(self):
        """SMI/RSI buy-sat alone is not enough if close is below EMA60."""
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
        with patch.object(pb, "calc_smi", return_value=(
            pd.Series(np.full(n, 50.0)),
            pd.Series(np.full(n, 50.0)),
            pd.Series(np.full(n, 50.0)),
        )), patch.object(pb, "calc_rsi_tv", return_value=pd.Series(np.full(n, 60.0))):
            feat = pb._frame_features(raw, 1)
        self.assertIsNotNone(feat)
        self.assertFalse(bool(feat["buy_main"].iloc[-1]))


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

    def test_sell_entry_partial_start_state_is_fine(self):
        """Start green+below (one condition already true); enter on red."""
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

    def test_reject_if_both_conditions_true_at_start(self):
        """Both conditions already true on the first watched candle → reject."""
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

    def test_fresh_counter_after_rejection_rechecks(self):
        """Rejected episode ends with the counter; a fresh one can enter."""
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=2 * i) for i in range(4)],
                "end_ts": [start + timedelta(minutes=2 * (i + 1)) for i in range(4)],
                "close": [99.0, 98.5, 99.5, 98.0],
                "ema": [100.0, 100.0, 100.0, 100.2],
                "don": [-1, -1, 1, -1],
                "above_ema": [False, False, False, False],
                "below_ema": [True, True, True, True],
                "don_green": [False, False, True, False],
                "don_red": [True, True, False, True],
            }
        )
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=2 * (i + 1)) for i in range(4)]
        )
        stepped = _fill_levels({}, grid)
        stepped[30]["sell_main"] = np.ones(len(grid), dtype=bool)
        # Counter sat on candle 0 (rejected: red+below already), gap on 1,
        # fresh counter on 2 (green+below → watch), entry on 3 (red+below).
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
        self.assertIn("رفض", help_text)


if __name__ == "__main__":
    unittest.main()
