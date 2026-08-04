"""
Step 8:
1) إغلاق تشبع SMI
2) لمس RSI على شمعة مغلقة (35 تشبع بيعي / 65 تشبع شرائي) بدون متوسط
3) الإشارة عند تقاطع Stochastic — أي وقت بعد اللمس (بلا قيد 3 شموع)
"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import indicators as ind


def _base_df(n=220):
    ts = pd.date_range("2024-01-01", periods=n, freq="20min", tz="UTC")
    close = np.linspace(100, 90, n)
    return pd.DataFrame({
        "ts": ts,
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "vol": np.ones(n),
    })


class TestRsiTouchThenStochCross(unittest.TestCase):
    def test_signal_candle_is_stoch_cross_after_rsi_touch(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rsi_i, cross_i = 170, 190  # فجوة كبيرة — مسموحة

        rsi = pd.Series(np.full(n, 40.0))
        rsi.iloc[rsi_i] = 34.0
        k = pd.Series(np.full(n, 10.0))
        k.iloc[cross_i - 1] = 18.0
        k.iloc[cross_i] = 25.0  # تقاطع فوق 20
        d = pd.Series(np.full(n, 50.0))

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)):
            self.assertTrue(ind.check_rsi_stoch(df, since_ts))
            self.assertEqual(
                ind.find_rsi_stoch_entry_index(df, since_ts, side="long"),
                cross_i,
            )

    def test_fails_when_stoch_is_above_20_without_cross(self):
        """مجرد البقاء فوق 20 لا يكفي — لازم تقاطع."""
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rsi = pd.Series(np.full(n, 40.0))
        rsi.iloc[170] = 34.0
        k = pd.Series(np.full(n, 25.0))  # فوق 20 من البداية — بلا تقاطع
        d = pd.Series(np.full(n, 50.0))

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)):
            self.assertFalse(ind.check_rsi_stoch(df, since_ts))

    def test_fails_when_stoch_cross_before_rsi_touch(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rsi = pd.Series(np.full(n, 40.0))
        rsi.iloc[190] = 34.0  # اللمس بعد التقاطع
        k = pd.Series(np.full(n, 10.0))
        k.iloc[170] = 18.0
        k.iloc[171] = 25.0
        d = pd.Series(np.full(n, 50.0))

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)):
            self.assertFalse(ind.check_rsi_stoch(df, since_ts))

    def test_short_signal_on_stoch_cross_below_80_after_rsi_65(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rsi = pd.Series(np.full(n, 50.0))
        rsi.iloc[170] = 66.0
        k = pd.Series(np.full(n, 90.0))
        k.iloc[185] = 82.0
        k.iloc[186] = 75.0
        d = pd.Series(np.full(n, 50.0))

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)):
            self.assertTrue(ind.check_rsi_stoch_short(df, since_ts))
            self.assertEqual(
                ind.find_rsi_stoch_entry_index(df, since_ts, side="short"),
                186,
            )


class TestStep8SmiThenRsiThenStoch(unittest.TestCase):
    def test_fails_when_events_before_smi_close(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rsi = pd.Series(np.full(n, 40.0))
        rsi.iloc[170] = 30.0
        k = pd.Series(np.full(n, 10.0))
        k.iloc[171] = 18.0
        k.iloc[172] = 25.0
        d = pd.Series(np.full(n, 50.0))
        smi = pd.Series(np.full(n, -10.0))
        smi.iloc[190] = -45.0

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)), \
             patch.object(ind, "calc_smi", return_value=(smi, smi, smi)):
            self.assertIsNone(
                ind.find_step8_entry_index(
                    df,
                    since_ts,
                    smi_threshold=-40,
                    rsi_threshold=35,
                    direction="long",
                )
            )

    def test_passes_smi_then_rsi_touch_then_late_stoch_cross(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        smi = pd.Series(np.full(n, -10.0))
        smi.iloc[160] = -45.0
        rsi = pd.Series(np.full(n, 40.0))
        rsi.iloc[165] = 35.0
        k = pd.Series(np.full(n, 10.0))
        k.iloc[200] = 19.0
        k.iloc[201] = 22.0  # تقاطع متأخر بعد اللمس — مسموح
        d = pd.Series(np.full(n, 50.0))

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)), \
             patch.object(ind, "calc_smi", return_value=(smi, smi, smi)):
            self.assertEqual(
                ind.find_step8_entry_index(
                    df,
                    since_ts,
                    smi_threshold=-40,
                    rsi_threshold=35,
                    direction="long",
                ),
                201,
            )


if __name__ == "__main__":
    unittest.main()
