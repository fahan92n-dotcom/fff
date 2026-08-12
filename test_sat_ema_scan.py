"""Tests for SMI sat + EMA60 close only (no Donchian / reverse sat)."""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

import pullback_bot.sat_ema_scan as se


def _feat(start, n, sat, above, minutes=30, price=100.0):
    ts = [start + timedelta(minutes=minutes * i) for i in range(n)]
    end_ts = [t + timedelta(minutes=minutes) for t in ts]
    return pd.DataFrame(
        {
            "ts": ts,
            "end_ts": end_ts,
            "close": np.full(n, price),
            "ema": np.full(n, price),
            "buy_sat": np.array(sat, dtype=bool),
            "sell_sat": np.zeros(n, dtype=bool),
            "above_ema": np.array(above, dtype=bool),
            "below_ema": np.zeros(n, dtype=bool),
        }
    )


class TestSatEmaScan(unittest.TestCase):
    def test_buy_enters_on_first_close_above_ema_during_sat(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        feat = _feat(
            start,
            4,
            sat=[False, True, True, True],
            above=[False, False, True, True],
        )
        raw = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=i) for i in range(200)],
                "high": np.linspace(100, 102, 200),
                "low": np.linspace(99.5, 101.5, 200),
            }
        )
        signals = se._scan_side(
            "buy", feat, start, start + timedelta(days=1), raw, "BTCUSDT", 30
        )
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["type"], "buy")
        self.assertEqual(signals[0]["time"], start + timedelta(minutes=90))

    def test_one_entry_per_episode(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        feat = _feat(
            start,
            5,
            sat=[True, True, False, True, True],
            above=[True, True, False, True, True],
        )
        raw = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=i) for i in range(300)],
                "high": np.full(300, 101.2),
                "low": np.full(300, 99.5),
            }
        )
        signals = se._scan_side(
            "buy", feat, start, start + timedelta(days=1), raw, "ETHUSDT", 30
        )
        self.assertEqual(len(signals), 2)

    def test_no_entry_if_sat_never_closes_above_ema(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        feat = _feat(start, 3, sat=[True, True, True], above=[False, False, False])
        raw = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=i) for i in range(120)],
                "high": np.full(120, 100.0),
                "low": np.full(120, 99.0),
            }
        )
        signals = se._scan_side(
            "buy", feat, start, start + timedelta(days=1), raw, "XRPUSDT", 30
        )
        self.assertEqual(signals, [])

    def test_sell_needs_close_below_ema(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        n = 3
        ts = [start + timedelta(minutes=30 * i) for i in range(n)]
        feat = pd.DataFrame(
            {
                "ts": ts,
                "end_ts": [t + timedelta(minutes=30) for t in ts],
                "close": [100.0, 99.0, 98.0],
                "ema": [100.0, 100.0, 100.0],
                "buy_sat": np.zeros(n, dtype=bool),
                "sell_sat": np.array([True, True, True]),
                "above_ema": np.array([True, False, False]),
                "below_ema": np.array([False, True, True]),
            }
        )
        raw = pd.DataFrame(
            {
                "ts": [start + timedelta(minutes=i) for i in range(200)],
                "high": np.full(200, 100.8),
                "low": np.full(200, 98.5),
            }
        )
        signals = se._scan_side(
            "sell", feat, start, start + timedelta(days=1), raw, "BTCUSDT", 30
        )
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["type"], "sell")
        self.assertEqual(signals[0]["time"], start + timedelta(minutes=60))


if __name__ == "__main__":
    unittest.main()
