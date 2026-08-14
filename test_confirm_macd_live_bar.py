"""Confirm MACD (step ⑤) must follow the chart's current confirm bar.

Live cascade used to drop the incomplete 27m candle, so a buy could fire
while TradingView already showed a red histogram on the forming confirm bar.
"""

import unittest
from datetime import datetime, timezone
from dataclasses import replace

import pandas as pd

import cascade_steps as strategy
import indicators as ind
from cascade_steps import LONG_RULES, SHORT_RULES


def _1m_range(start, minutes, close=20.0):
    times = pd.date_range(start=start, periods=minutes, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "ts": times,
            "open": close,
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
            "vol": 1.0,
        }
    )


class TestConfirmMacdLiveBar(unittest.TestCase):
    def test_confirm_frame_keeps_in_progress_bar_from_closed_source(self):
        start = datetime(2026, 8, 13, 20, 0, tzinfo=timezone.utc)
        raw = _1m_range(start, minutes=4 * 60 + 18)
        asof = datetime(2026, 8, 14, 0, 18, tzinfo=timezone.utc)

        full = ind.resample_ohlcv_closed(raw, 27)
        closed_only = full[ind.candle_period_ends(full["ts"], 27) <= asof]
        live = ind.confirm_macd_frame(raw, "1m", 27, now=asof)

        self.assertEqual(
            pd.Timestamp(live["ts"].iloc[-1]),
            pd.Timestamp("2026-08-14 00:00:00+00:00"),
        )
        # UTC-day 27m last closed bar is the 23:51 remainder, not epoch 23:33.
        self.assertEqual(
            pd.Timestamp(closed_only["ts"].iloc[-1]),
            pd.Timestamp("2026-08-13 23:51:00+00:00"),
        )

    def test_confirm_frame_drops_unclosed_source_minute(self):
        start = datetime(2026, 8, 14, 0, 0, tzinfo=timezone.utc)
        raw = _1m_range(start, minutes=19)
        asof = datetime(2026, 8, 14, 0, 18, tzinfo=timezone.utc)
        live = ind.confirm_macd_frame(raw, "1m", 27, now=asof)
        # 00:18 1m has not closed yet; last source minute is 00:17.
        self.assertEqual(float(live["close"].iloc[-1]), 20.0)
        self.assertEqual(
            pd.Timestamp(live["ts"].iloc[-1]),
            pd.Timestamp("2026-08-14 00:00:00+00:00"),
        )

    def test_buy_step5_rejects_when_current_confirm_bar_is_red(self):
        start = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)
        asof = datetime(2026, 8, 14, 0, 18, tzinfo=timezone.utc)
        n = int((asof - start).total_seconds() // 60)
        raw = _1m_range(start, minutes=n, close=20.0)
        # Last closed 27m (23:51 remainder) stays high; forming 00:00 bar dumps.
        dump = raw["ts"] >= pd.Timestamp("2026-08-14 00:00:00+00:00")
        raw.loc[dump, ["open", "high", "low", "close"]] = [18.0, 18.1, 17.9, 18.0]

        closed_confirm = ind.resample_ohlcv_closed(raw, 27)
        closed_confirm = closed_confirm[
            closed_confirm["ts"] < pd.Timestamp("2026-08-14 00:00:00+00:00")
        ]

        seen = []

        def spy(df):
            seen.append(pd.Timestamp(df["ts"].iloc[-1]))
            hist = ind._calc_macd_hist(df["close"])
            return bool(hist.iloc[-1] > 0)

        candidate = {
            "sym": "KORUUSDT",
            "base_api": "1m",
            "confirm_frame": 27,
            "df_confirm": closed_confirm,
            "raw_base": raw,
            "asof": asof,
        }
        rules = replace(LONG_RULES, confirm_histogram_check=spy)
        ok, reason = strategy._step5(candidate, rules)
        self.assertEqual(
            seen[-1],
            pd.Timestamp("2026-08-14 00:00:00+00:00"),
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "macd_confirm")

    def test_sell_step5_uses_current_confirm_bar(self):
        start = datetime(2026, 8, 13, 20, 0, tzinfo=timezone.utc)
        raw = _1m_range(start, minutes=4 * 60 + 18)
        asof = datetime(2026, 8, 14, 0, 18, tzinfo=timezone.utc)
        seen = []

        def spy(df):
            seen.append(pd.Timestamp(df["ts"].iloc[-1]))
            return True

        candidate = {
            "sym": "KORUUSDT",
            "base_api": "1m",
            "confirm_frame": 27,
            "df_confirm": pd.DataFrame({"ts": [pd.Timestamp("2026-08-13 23:33:00+00:00")], "close": [1.0]}),
            "raw_base": raw,
            "asof": asof,
        }
        rules = replace(SHORT_RULES, confirm_histogram_check=spy)
        ok, reason = strategy._step5(candidate, rules)
        self.assertTrue(ok)
        self.assertEqual(reason, "passed")
        self.assertEqual(seen[-1], pd.Timestamp("2026-08-14 00:00:00+00:00"))


if __name__ == "__main__":
    unittest.main()
