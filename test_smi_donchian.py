"""Tests for SMI-against-MACD-color paper scanner + Donchian/EMA50 entry."""

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
    for main, entry, halt in sd.LEVELS:
        stepped.setdefault(main, _empty_stepped(grid))
        stepped.setdefault(entry, _empty_stepped(grid))
        stepped.setdefault(halt, _empty_stepped(grid))
    return stepped


def _buy_entry(start, minutes=1):
    return pd.DataFrame(
        {
            "ts": [start + timedelta(minutes=minutes * i) for i in range(3)],
            "end_ts": [start + timedelta(minutes=minutes * (i + 1)) for i in range(3)],
            "close": [99.0, 100.0, 101.0],
            "don_green": [False, False, True],
            "don_red": [True, True, False],
            "above_ema": [False, False, True],
            "below_ema": [True, True, False],
        }
    )


def _sell_entry(start, minutes=1):
    return pd.DataFrame(
        {
            "ts": [start + timedelta(minutes=minutes * i) for i in range(3)],
            "end_ts": [start + timedelta(minutes=minutes * (i + 1)) for i in range(3)],
            "close": [101.0, 100.0, 99.0],
            "don_green": [True, True, False],
            "don_red": [False, False, True],
            "above_ema": [True, True, False],
            "below_ema": [False, False, True],
        }
    )


class TestLevels(unittest.TestCase):
    def test_user_table_tp_sl_and_unique_mains(self):
        self.assertEqual(sd.LEVELS[0], (15, 1, 18))
        self.assertEqual(sd.LEVELS[1], (18, 1, 21))
        self.assertEqual(sd.LEVELS[2], (21, 1, 24))
        self.assertEqual(sd.LEVELS[3], (24, 1, 27))
        self.assertEqual(sd.LEVELS[4], (27, 1, 30))
        self.assertEqual(sd.LEVELS[5], (30, 2, 33))
        mains = [lvl[0] for lvl in sd.LEVELS]
        self.assertEqual(mains, [15, 18, 21, 24, 27, 30])
        self.assertEqual(len(mains), len(set(mains)))
        self.assertEqual(sd.WIN_PCT, 0.67)
        self.assertEqual(sd.LOSS_PCT, 0.53)
        self.assertEqual(sd.EMA_SPAN, 50)
        self.assertEqual(sd.MACD_FAST, 12)
        self.assertEqual(sd.MACD_SLOW, 26)
        self.assertEqual(sd.MACD_SIGNAL, 9)

    def test_thirty_uses_two_minute_entry(self):
        self.assertEqual(sd.LEVELS[-1][1], 2)
        self.assertTrue(all(lvl[1] == 1 for lvl in sd.LEVELS[:-1]))

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


class TestMacdColor(unittest.TestCase):
    def test_rising_close_histogram_green(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        df = _bars(start, 80, minutes=1, price=100.0, drift=0.5)
        _line, _signal, hist = sd._calc_macd_full(df["close"])
        self.assertTrue(hist.iloc[-1] > 0)

    def test_falling_close_histogram_red(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        df = _bars(start, 80, minutes=1, price=100.0, drift=-0.5)
        _line, _signal, hist = sd._calc_macd_full(df["close"])
        self.assertTrue(hist.iloc[-1] < 0)


class TestScanSide(unittest.TestCase):
    def test_buy_on_sell_sat_and_green_macd(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = _buy_entry(start)
        grid = pd.DatetimeIndex(entry["end_ts"])
        stepped = _fill_levels({}, grid)
        stepped[15]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[15]["buy_macd"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=0.5)
        signals = sd._scan_side(
            "buy", stepped, {1: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "SUIUSDT",
        )
        buys = [s for s in signals if s["type"] == "buy" and s["base_frame"] == 15]
        self.assertTrue(buys)
        self.assertAlmostEqual(buys[0]["price"], 101.0)
        self.assertAlmostEqual(buys[0]["win_pct"], 0.67)
        self.assertAlmostEqual(buys[0]["loss_pct"], 0.53)

    def test_no_buy_when_macd_not_green(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = _buy_entry(start)
        grid = pd.DatetimeIndex(entry["end_ts"])
        stepped = _fill_levels({}, grid)
        stepped[15]["sell_sat"] = np.ones(len(grid), dtype=bool)
        stepped[15]["buy_macd"] = np.zeros(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=0.5)
        signals = sd._scan_side(
            "buy", stepped, {1: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "SUIUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 15 for s in signals))

    def test_sell_on_buy_sat_and_red_macd(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = _sell_entry(start)
        grid = pd.DatetimeIndex(entry["end_ts"])
        stepped = _fill_levels({}, grid)
        stepped[15]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[15]["sell_macd"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {1: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "ADAUSDT",
        )
        sells = [s for s in signals if s["type"] == "sell" and s["base_frame"] == 15]
        self.assertTrue(sells)
        self.assertAlmostEqual(sells[0]["price"], 99.0)

    def test_no_sell_when_macd_not_red(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = _sell_entry(start)
        grid = pd.DatetimeIndex(entry["end_ts"])
        stepped = _fill_levels({}, grid)
        stepped[15]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[15]["sell_macd"] = np.zeros(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {1: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "ADAUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 15 for s in signals))

    def test_no_sell_without_ema50_cross(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = _sell_entry(start)
        entry["below_ema"] = [False, False, False]
        entry["above_ema"] = [True, True, True]
        grid = pd.DatetimeIndex(entry["end_ts"])
        stepped = _fill_levels({}, grid)
        stepped[15]["buy_sat"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {1: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "ADAUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 15 for s in signals))

    def test_no_sell_when_donchian_not_red(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = _sell_entry(start)
        entry["don_red"] = [False, False, False]
        entry["don_green"] = [True, True, True]
        grid = pd.DatetimeIndex(entry["end_ts"])
        stepped = _fill_levels({}, grid)
        stepped[15]["buy_sat"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {1: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "ADAUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 15 for s in signals))

    def test_larger_main_cancels_smaller_when_macd_matches(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = _sell_entry(start)
        grid = pd.DatetimeIndex(entry["end_ts"])
        stepped = _fill_levels({}, grid)
        stepped[15]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[15]["sell_macd"] = np.ones(len(grid), dtype=bool)
        stepped[18]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[18]["sell_macd"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {1: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "ADAUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 15 for s in signals))
        self.assertTrue(any(s["base_frame"] == 18 for s in signals))

    def test_larger_without_matching_macd_does_not_cancel(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = _sell_entry(start)
        grid = pd.DatetimeIndex(entry["end_ts"])
        stepped = _fill_levels({}, grid)
        stepped[15]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[15]["sell_macd"] = np.ones(len(grid), dtype=bool)
        stepped[18]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[18]["sell_macd"] = np.zeros(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {1: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "ADAUSDT",
        )
        self.assertTrue(any(s["base_frame"] == 15 for s in signals))

    def test_thirty_halted_by_thirty_three(self):
        start = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        entry = _sell_entry(start, minutes=2)
        grid = pd.DatetimeIndex(entry["end_ts"])
        stepped = _fill_levels({}, grid)
        stepped[30]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[30]["sell_macd"] = np.ones(len(grid), dtype=bool)
        stepped[33]["buy_sat"] = np.ones(len(grid), dtype=bool)
        stepped[33]["sell_macd"] = np.ones(len(grid), dtype=bool)
        raw_1m = _bars(start, 40, minutes=1, price=100.0, drift=-0.5)
        signals = sd._scan_side(
            "sell", stepped, {2: entry}, grid, start, start + timedelta(hours=1),
            raw_1m, "ADAUSDT",
        )
        self.assertFalse(any(s["base_frame"] == 30 for s in signals))


class TestEvaluateUsesLevelPct(unittest.TestCase):
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
    def test_merges_symbols(self):
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        end = start + timedelta(days=30)
        ada_trade = {
            "symbol": "ADAUSDT",
            "type": "buy",
            "time": start + timedelta(days=1),
            "price": 1.0,
            "base_frame": 15,
            "confirm_frame": 18,
            "triple_frame": 1,
            "win_pct": 0.67,
            "loss_pct": 0.53,
            "outcome": "win",
            "exit_price": 1.0067,
            "exit_ts": start + timedelta(days=1, hours=1),
        }
        sui_trade = {
            "symbol": "SUIUSDT",
            "type": "sell",
            "time": start + timedelta(days=2),
            "price": 2.0,
            "base_frame": 30,
            "confirm_frame": 33,
            "triple_frame": 2,
            "win_pct": 0.67,
            "loss_pct": 0.53,
            "outcome": "loss",
            "exit_price": 2.0106,
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
        self.assertIn("EMA50", text)
        self.assertIn("MACD", text)
        self.assertIn("0.67", text)
        self.assertIn("0.53", text)


if __name__ == "__main__":
    unittest.main()
