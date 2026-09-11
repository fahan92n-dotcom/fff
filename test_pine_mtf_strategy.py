"""Unit tests for the Pine sequential MTF week scan."""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

import pine_mtf_strategy as pine


class TestRsiGates(unittest.TestCase):
    def test_buy_gate_requires_confirm_band_and_main_gap(self):
        self.assertTrue(pine.buy_rsi_gate(55.0, 50.0))
        self.assertFalse(pine.buy_rsi_gate(49.9, 45.0))
        self.assertFalse(pine.buy_rsi_gate(60.1, 55.0))
        self.assertFalse(pine.buy_rsi_gate(55.0, 53.0))  # gap < 3
        self.assertFalse(pine.buy_rsi_gate(55.0, 44.0))  # gap > 10

    def test_sell_gate_requires_confirm_band_and_main_gap(self):
        self.assertTrue(pine.sell_rsi_gate(45.0, 50.0))
        self.assertFalse(pine.sell_rsi_gate(39.9, 45.0))
        self.assertFalse(pine.sell_rsi_gate(50.1, 55.0))
        self.assertFalse(pine.sell_rsi_gate(45.0, 47.0))  # gap < 3
        self.assertFalse(pine.sell_rsi_gate(45.0, 56.0))  # gap > 10


class TestEntryLevels(unittest.TestCase):
    def test_long_uses_one_percent_tp_and_point_eight_sl(self):
        tp, sl = pine._entry_levels("buy", 100.0)
        self.assertAlmostEqual(tp, 101.0)
        self.assertAlmostEqual(sl, 99.2)

    def test_short_sl_matches_pine_tp_pct_bug(self):
        tp, sl = pine._entry_levels("sell", 100.0)
        self.assertAlmostEqual(tp, 99.0)
        self.assertAlmostEqual(sl, 101.0)


class TestEvaluateOutcome(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2026, 8, 1, tzinfo=timezone.utc)

    def _future(self, high, low):
        return pd.DataFrame(
            [
                {
                    "ts": self.start + timedelta(minutes=1),
                    "open": 100.0,
                    "high": high,
                    "low": low,
                    "close": 100.0,
                    "vol": 1,
                }
            ]
        )

    def test_buy_win(self):
        outcome, price, _ = pine.evaluate_outcome(
            "buy", 100.0, 101.0, 99.2, self._future(101.2, 99.8)
        )
        self.assertEqual(outcome, "win")
        self.assertAlmostEqual(price, 101.0)

    def test_buy_loss(self):
        outcome, price, _ = pine.evaluate_outcome(
            "buy", 100.0, 101.0, 99.2, self._future(100.4, 99.1)
        )
        self.assertEqual(outcome, "loss")
        self.assertAlmostEqual(price, 99.2)

    def test_same_bar_both_counts_loss(self):
        outcome, _, _ = pine.evaluate_outcome(
            "buy", 100.0, 101.0, 99.2, self._future(101.5, 98.5)
        )
        self.assertEqual(outcome, "loss")

    def test_sell_win(self):
        outcome, price, _ = pine.evaluate_outcome(
            "sell", 100.0, 99.0, 101.0, self._future(100.2, 98.8)
        )
        self.assertEqual(outcome, "win")
        self.assertAlmostEqual(price, 99.0)


class TestWeekFilter(unittest.TestCase):
    def test_keeps_fills_inside_window(self):
        now = datetime(2026, 9, 5, tzinfo=timezone.utc)
        start, end = pine.period_bounds(now=now, days=7)
        trades = [
            {"fill_ts": now - timedelta(days=8), "outcome": "win", "type": "buy"},
            {"fill_ts": now - timedelta(days=2), "outcome": "loss", "type": "buy"},
            {"fill_ts": now - timedelta(hours=1), "outcome": "open", "type": "sell"},
        ]
        week = pine.filter_week(trades, start, end)
        self.assertEqual(len(week), 2)
        summary = pine.summarize(week)
        self.assertEqual(len(summary["losses"]), 1)
        self.assertEqual(len(summary["opens"]), 1)
        self.assertEqual(len(summary["wins"]), 0)


class TestLookaheadMap(unittest.TestCase):
    def test_chart_bars_see_containing_completed_htf_bar(self):
        start = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
        chart = pd.DataFrame(
            {
                "ts": [
                    start,
                    start + timedelta(minutes=5),
                    start + timedelta(minutes=10),
                    start + timedelta(minutes=15),
                ]
            }
        )
        htf = pd.DataFrame(
            {
                "ts": [start, start + timedelta(minutes=15)],
                "smi": [-50.0, 10.0],
            }
        )
        mapped = pine.map_htf_lookahead(chart, htf, ["smi"])
        self.assertEqual(list(mapped["smi"]), [-50.0, -50.0, -50.0, 10.0])


def _blank_chart(rows, start):
    return pd.DataFrame(
        {
            "ts": [start + timedelta(minutes=5 * i) for i in range(rows)],
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "vol": 1.0,
            "smi": 0.0,
            "macd": 0.0,
            "hist": 0.0,
            "trend": 0.0,
            "ema50": 100.0,
            "rsi": 50.0,
            "rsi_ma": 50.0,
            "stoch_k": 50.0,
            "smi_main": 0.0,
            "macd_main": 0.0,
            "hist_main": 0.0,
            "trend_main": 0.0,
            "ema50_main": 100.0,
            "close_main": 100.0,
            "rsi_main": 50.0,
            "macd_confirm": 0.0,
            "hist_confirm": 0.0,
            "rsi_confirm": 55.0,
        }
    )


class TestReplaySequence(unittest.TestCase):
    def test_buy_path_emits_one_winning_trade(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        rows = pine.WARMUP_BARS + 10
        chart = _blank_chart(rows, start)
        i0 = pine.WARMUP_BARS
        chart.loc[i0 - 1, "smi_main"] = -39.0
        chart.loc[i0, "smi_main"] = -41.0  # c1
        chart.loc[i0 + 1, ["hist_main", "macd_main"]] = [-1.0, 0.0]  # c2
        chart.loc[i0 + 2, "trend_main"] = 1.0  # c3
        chart.loc[i0 + 2, ["close_main", "ema50_main"]] = [100.0, 99.0]
        chart.loc[i0 + 3, ["close_main", "ema50_main"]] = [98.0, 99.0]  # c4
        chart.loc[i0 + 4, ["macd_main", "hist_confirm"]] = [-0.5, 0.4]  # c5
        chart.loc[i0 + 5, "trend"] = -1.0  # c6
        chart.loc[i0 + 5, "smi"] = -39.0
        chart.loc[i0 + 6, "smi"] = -41.0  # c7
        chart.loc[i0 + 6, ["rsi", "rsi_ma"]] = [30.0, 40.0]
        chart.loc[i0 + 7, ["rsi", "rsi_ma"]] = [32.0, 31.0]  # touch + RSI cross
        chart.loc[i0 + 7, "stoch_k"] = 15.0
        chart.loc[i0 + 8, "stoch_k"] = 25.0  # stoch cross + RSI gate
        chart.loc[i0 + 8, ["rsi_confirm", "rsi_main"]] = [55.0, 50.0]
        chart.loc[i0 + 8, "close"] = 100.0
        chart.loc[i0 + 9, "open"] = 100.1
        fill_ts = chart.loc[i0 + 9, "ts"]
        raw_1m = pd.DataFrame(
            [
                {
                    "ts": fill_ts,
                    "open": 100.1,
                    "high": 101.5,
                    "low": 99.8,
                    "close": 101.0,
                    "vol": 1.0,
                }
            ]
        )
        trades = pine.replay_signals(chart, raw_1m)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["type"], "buy")
        self.assertEqual(trades[0]["outcome"], "win")
        self.assertAlmostEqual(trades[0]["price"], 100.1)

    def test_sell_path_emits_one_losing_trade(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        rows = pine.WARMUP_BARS + 10
        chart = _blank_chart(rows, start)
        i0 = pine.WARMUP_BARS
        chart.loc[i0 - 1, "smi_main"] = 39.0
        chart.loc[i0, "smi_main"] = 41.0  # c1
        chart.loc[i0 + 1, ["hist_main", "macd_main"]] = [1.0, 0.0]  # c2
        chart.loc[i0 + 2, "trend_main"] = -1.0  # c3
        chart.loc[i0 + 2, ["close_main", "ema50_main"]] = [100.0, 101.0]
        chart.loc[i0 + 3, ["close_main", "ema50_main"]] = [102.0, 101.0]  # c4
        chart.loc[i0 + 4, ["macd_main", "hist_confirm"]] = [0.5, -0.4]  # c5
        chart.loc[i0 + 5, "trend"] = 1.0  # c6
        chart.loc[i0 + 5, "smi"] = 39.0
        chart.loc[i0 + 6, "smi"] = 41.0  # c7
        chart.loc[i0 + 6, ["rsi", "rsi_ma"]] = [70.0, 60.0]
        chart.loc[i0 + 7, ["rsi", "rsi_ma"]] = [58.0, 59.0]  # touch + RSI cross
        chart.loc[i0 + 7, "stoch_k"] = 85.0
        chart.loc[i0 + 8, "stoch_k"] = 75.0  # stoch cross + RSI gate
        chart.loc[i0 + 8, ["rsi_confirm", "rsi_main"]] = [45.0, 50.0]
        chart.loc[i0 + 8, "close"] = 100.0
        chart.loc[i0 + 9, "open"] = 99.9
        fill_ts = chart.loc[i0 + 9, "ts"]
        raw_1m = pd.DataFrame(
            [
                {
                    "ts": fill_ts,
                    "open": 99.9,
                    "high": 101.2,
                    "low": 99.5,
                    "close": 100.8,
                    "vol": 1.0,
                }
            ]
        )
        trades = pine.replay_signals(chart, raw_1m)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["type"], "sell")
        self.assertEqual(trades[0]["outcome"], "loss")
        self.assertAlmostEqual(trades[0]["price"], 99.9)

    def test_buy_path_skips_when_rsi_gate_fails(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        rows = pine.WARMUP_BARS + 10
        chart = _blank_chart(rows, start)
        i0 = pine.WARMUP_BARS
        chart.loc[i0 - 1, "smi_main"] = -39.0
        chart.loc[i0, "smi_main"] = -41.0
        chart.loc[i0 + 1, ["hist_main", "macd_main"]] = [-1.0, 0.0]
        chart.loc[i0 + 2, "trend_main"] = 1.0
        chart.loc[i0 + 2, ["close_main", "ema50_main"]] = [100.0, 99.0]
        chart.loc[i0 + 3, ["close_main", "ema50_main"]] = [98.0, 99.0]
        chart.loc[i0 + 4, ["macd_main", "hist_confirm"]] = [-0.5, 0.4]
        chart.loc[i0 + 5, "trend"] = -1.0
        chart.loc[i0 + 5, "smi"] = -39.0
        chart.loc[i0 + 6, "smi"] = -41.0
        chart.loc[i0 + 6, ["rsi", "rsi_ma"]] = [30.0, 40.0]
        chart.loc[i0 + 7, ["rsi", "rsi_ma"]] = [32.0, 31.0]
        chart.loc[i0 + 7, "stoch_k"] = 15.0
        chart.loc[i0 + 8, "stoch_k"] = 25.0
        chart.loc[i0 + 8, ["rsi_confirm", "rsi_main"]] = [55.0, 54.0]  # gap < 3
        trades = pine.replay_signals(chart, pd.DataFrame(columns=["ts"]))
        self.assertEqual(trades, [])


class TestFormatReport(unittest.TestCase):
    def test_splits_wins_and_losses(self):
        start = datetime(2026, 9, 4, tzinfo=timezone.utc)
        result = {
            "symbol": "BTCUSDT",
            "market": "futures",
            "start": start,
            "end": start + timedelta(days=7),
            "trades": [],
            "wins": [
                {
                    "type": "buy",
                    "fill_ts": start + timedelta(days=1),
                    "price": 110000.0,
                    "outcome": "win",
                    "rsi_confirm": 55.0,
                    "rsi_main": 50.0,
                    "exit_ts": start + timedelta(days=1, hours=2),
                    "exit_price": 111100.0,
                }
            ],
            "losses": [
                {
                    "type": "sell",
                    "fill_ts": start + timedelta(days=2),
                    "price": 111000.0,
                    "outcome": "loss",
                    "rsi_confirm": 45.0,
                    "rsi_main": 50.0,
                    "exit_ts": start + timedelta(days=2, hours=1),
                    "exit_price": 112110.0,
                }
            ],
            "opens": [],
            "net_pct": 0.2,
        }
        result["trades"] = result["wins"] + result["losses"]
        text = pine.format_report(result)
        self.assertIn("رابحة: 1", text)
        self.assertIn("خاسرة: 1", text)
        self.assertIn("الصفقات الرابحة:", text)
        self.assertIn("الصفقات الخاسرة:", text)
        self.assertIn("شراء @ 110000.00", text)
        self.assertIn("بيع @ 111000.00", text)


class TestScanWeekUsesFetchedFrame(unittest.TestCase):
    def test_raises_when_ohlcv_missing(self):
        with self.assertRaises(RuntimeError):
            pine.scan_week(raw_1m=pd.DataFrame())

    def test_fetch_prefers_futures(self):
        fake = pd.DataFrame(
            {
                "ts": [datetime(2026, 9, 1, tzinfo=timezone.utc)],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "vol": [1.0],
            }
        )
        with patch.object(pine, "get_ohlcv_full_futures", return_value=fake):
            df, market = pine.fetch_btc_1m(target=1)
        self.assertEqual(market, "futures")
        self.assertEqual(len(df), 1)


if __name__ == "__main__":
    unittest.main()
