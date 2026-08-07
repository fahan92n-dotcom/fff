"""Tests for the pullback saturation strategy week scan."""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

import pullback_strategy as pb


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
        self.assertEqual(pb.LEVELS[6], (180, 30, 72, 12))
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


class TestBoolStep(unittest.TestCase):
    def test_ffill_latest_closed(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        ends = np.array(
            [start + timedelta(minutes=30), start + timedelta(minutes=60)],
            dtype="datetime64[ns]",
        )
        # pandas needs tz-aware via DatetimeIndex in helper; pass python datetimes
        ends = np.array([start + timedelta(minutes=30), start + timedelta(minutes=60)])
        values = np.array([True, False])
        grid = pd.DatetimeIndex(
            [start + timedelta(minutes=m) for m in range(15, 75, 15)]
        )
        out = pb._bool_step(ends, values, grid)
        # 15m: before first close → False; 30/45 → True; 60 → False
        self.assertEqual(list(out), [False, True, True, False])


class TestScanSideSynthetic(unittest.TestCase):
    def test_sell_entry_requires_arm_then_flip(self):
        """Green+above then red+below inside an open window yields one sell."""
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        # 6 entry candles on 2m
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
        # Force level 0 active + confirm ok for whole grid
        stepped = {
            30: {
                "sell_main": np.ones(len(grid), dtype=bool),
                "buy_main": np.zeros(len(grid), dtype=bool),
                "sell_sat": np.ones(len(grid), dtype=bool),
                "buy_sat": np.zeros(len(grid), dtype=bool),
            },
            5: {
                "sell_main": np.zeros(len(grid), dtype=bool),
                "buy_main": np.zeros(len(grid), dtype=bool),
                "sell_sat": np.zeros(len(grid), dtype=bool),
                "buy_sat": np.ones(len(grid), dtype=bool),
            },
            12: {
                "sell_main": np.zeros(len(grid), dtype=bool),
                "buy_main": np.zeros(len(grid), dtype=bool),
                "sell_sat": np.zeros(len(grid), dtype=bool),
                "buy_sat": np.zeros(len(grid), dtype=bool),
            },
        }
        def _empty():
            return {
                "sell_main": np.zeros(len(grid), dtype=bool),
                "buy_main": np.zeros(len(grid), dtype=bool),
                "sell_sat": np.zeros(len(grid), dtype=bool),
                "buy_sat": np.zeros(len(grid), dtype=bool),
            }

        for main, cmin, cstop, _entry in pb.LEVELS:
            stepped.setdefault(main, _empty())
            for minutes in range(cmin, cstop + 1):
                stepped.setdefault(minutes, _empty())
        # ensure confirm 5..11 buy sat for level 0
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
        self.assertEqual(first["triple_frame"], 2)
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
                    # 12m buy sat active → stop
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


class TestReportAndCommand(unittest.TestCase):
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

    def test_command_routes(self):
        import fahadal92 as bot

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
        }), patch.object(bot, "send_telegram", side_effect=fake_send):
            bot._dispatch_command_inner("/week_pullback", "1")
        self.assertTrue(any("Pullback" in c for c in calls))


if __name__ == "__main__":
    unittest.main()
