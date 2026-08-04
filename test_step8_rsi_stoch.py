"""
اختبارات Step 8:
1) إغلاق تشبع SMI أولًا
2) بعدها RSI وصل المستوى (بدون تقاطع المتوسط) و Stochastic بالاتجاه خلال ±3 شموع
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


class TestRsiStochNoMaCross(unittest.TestCase):
    def test_long_passes_on_rsi_touch_and_stoch_above_20_within_gap(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rsi_i, stoch_i = 180, 182

        rsi = pd.Series(np.full(n, 40.0))
        rsi.iloc[rsi_i] = 34.0  # وصل 35 بدون أي متوسط
        k = pd.Series(np.full(n, 10.0))
        k.iloc[stoch_i] = 21.0  # فوق 20
        d = pd.Series(np.full(n, 50.0))

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)):
            self.assertTrue(ind.check_rsi_stoch(df, since_ts, max_gap=3))
            self.assertEqual(
                ind.find_rsi_stoch_entry_index(df, since_ts, max_gap=3, side="long"),
                stoch_i,
            )

    def test_long_fails_when_stoch_never_above_20_near_rsi(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rsi = pd.Series(np.full(n, 40.0))
        rsi.iloc[180] = 34.0
        k = pd.Series(np.full(n, 10.0))  # دائمًا تحت/يساوي — مو فوق 20
        d = pd.Series(np.full(n, 50.0))

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)):
            self.assertFalse(ind.check_rsi_stoch(df, since_ts, max_gap=3))

    def test_long_fails_when_gap_exceeds_3(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rsi = pd.Series(np.full(n, 40.0))
        rsi.iloc[180] = 34.0
        k = pd.Series(np.full(n, 10.0))
        k.iloc[185] = 25.0  # gap=5
        d = pd.Series(np.full(n, 50.0))

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)):
            self.assertFalse(ind.check_rsi_stoch(df, since_ts, max_gap=3))

    def test_short_passes_on_rsi_touch_and_stoch_below_80(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rsi = pd.Series(np.full(n, 50.0))
        rsi.iloc[180] = 66.0
        k = pd.Series(np.full(n, 90.0))
        k.iloc[181] = 79.0
        d = pd.Series(np.full(n, 50.0))

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)):
            self.assertTrue(ind.check_rsi_stoch_short(df, since_ts, max_gap=3))


class TestStep8SmiThenRsiStochOrder(unittest.TestCase):
    def test_fails_when_rsi_stoch_complete_before_smi_close(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rsi_i, stoch_i, smi_i = 170, 172, 190

        rsi = pd.Series(np.full(n, 40.0))
        rsi.iloc[rsi_i] = 30.0
        k = pd.Series(np.full(n, 10.0))
        k.iloc[stoch_i] = 25.0
        d = pd.Series(np.full(n, 50.0))
        smi = pd.Series(np.full(n, -10.0))
        smi.iloc[smi_i] = -45.0

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
                    max_gap=3,
                )
            )

    def test_passes_when_smi_close_precedes_rsi_and_stoch(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        smi_i, rsi_i, stoch_i = 160, 180, 182

        rsi = pd.Series(np.full(n, 40.0))
        rsi.iloc[rsi_i] = 30.0
        k = pd.Series(np.full(n, 10.0))
        k.iloc[stoch_i] = 25.0
        d = pd.Series(np.full(n, 50.0))
        smi = pd.Series(np.full(n, -10.0))
        smi.iloc[smi_i] = -45.0

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)), \
             patch.object(ind, "calc_smi", return_value=(smi, smi, smi)):
            index = ind.find_step8_entry_index(
                df,
                since_ts,
                smi_threshold=-40,
                rsi_threshold=35,
                direction="long",
                max_gap=3,
            )

        self.assertEqual(index, stoch_i)

    def test_ignores_rsi_ma_cross_requirement(self):
        """حتى لو RSI فوق متوسطه طول الوقت، يكفي لمس 35 + Stoch>20 بعد SMI."""
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        smi = pd.Series(np.full(n, -10.0))
        smi.iloc[160] = -45.0
        rsi = pd.Series(np.full(n, 50.0))
        rsi.iloc[175] = 35.0
        k = pd.Series(np.full(n, 5.0))
        k.iloc[176] = 22.0
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
                    max_gap=3,
                ),
                176,
            )


if __name__ == "__main__":
    unittest.main()
