"""
Step 8:
1) إغلاق تشبع SMI
2) لمس RSI على شمعة مغلقة (35 / 65)
3) تقاطع RSI مع متوسطه
4) بعده خلال 3 شموع: Stoch فوق 20 / تحت 80
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


def _series_with_fixed_ma(values, ma_values):
    s = pd.Series(values, dtype=float)

    class _Roll:
        def mean(self, *_a, **_k):
            return pd.Series(ma_values, dtype=float)

    s.rolling = lambda *_a, **_k: _Roll()
    return s


class TestRsiCrossThenStochWithin3(unittest.TestCase):
    def test_passes_when_stoch_above_20_within_3_after_rsi_cross(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        touch_i, cross_i, stoch_i = 160, 180, 182  # gap=2 بعد التقاطع

        rsi_vals = np.full(n, 40.0)
        ma_vals = np.full(n, 45.0)
        rsi_vals[touch_i] = 34.0
        rsi_vals[cross_i - 1] = 44.0
        rsi_vals[cross_i] = 46.0  # تقاطع فوق المتوسط 45
        rsi = _series_with_fixed_ma(rsi_vals, ma_vals)

        k = pd.Series(np.full(n, 10.0))
        k.iloc[stoch_i] = 25.0
        d = pd.Series(np.full(n, 50.0))

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)):
            self.assertTrue(ind.check_rsi_stoch(df, since_ts, max_gap=3))
            self.assertEqual(
                ind.find_rsi_stoch_entry_index(df, since_ts, max_gap=3, side="long"),
                stoch_i,
            )

    def test_fails_when_stoch_above_20_after_more_than_3_candles(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rsi_vals = np.full(n, 40.0)
        ma_vals = np.full(n, 45.0)
        rsi_vals[160] = 34.0
        rsi_vals[179] = 44.0
        rsi_vals[180] = 46.0
        rsi = _series_with_fixed_ma(rsi_vals, ma_vals)

        k = pd.Series(np.full(n, 10.0))
        k.iloc[185] = 25.0  # gap=5 من تقاطع 180
        d = pd.Series(np.full(n, 50.0))

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)):
            # تأكد أن التقاطع عند 180
            self.assertEqual(
                ind.find_rsi_ma_cross_index(df, since_ts, side="long", at_or_after=160),
                180,
            )
            self.assertFalse(ind.check_rsi_stoch(df, since_ts, max_gap=3))

    def test_fails_when_stoch_only_before_rsi_cross(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rsi_vals = np.full(n, 40.0)
        ma_vals = np.full(n, 45.0)
        rsi_vals[160] = 34.0
        rsi_vals[179] = 44.0
        rsi_vals[180] = 46.0
        rsi = _series_with_fixed_ma(rsi_vals, ma_vals)

        k = pd.Series(np.full(n, 10.0))
        k.iloc[175] = 25.0  # قبل التقاطع فقط
        d = pd.Series(np.full(n, 50.0))

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)):
            self.assertFalse(ind.check_rsi_stoch(df, since_ts, max_gap=3))

    def test_short_stoch_below_80_within_3_after_rsi_cross_down(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rsi_vals = np.full(n, 60.0)
        ma_vals = np.full(n, 55.0)
        rsi_vals[160] = 66.0
        rsi_vals[179] = 56.0
        rsi_vals[180] = 54.0  # تقاطع لتحت المتوسط
        rsi = _series_with_fixed_ma(rsi_vals, ma_vals)

        k = pd.Series(np.full(n, 90.0))
        k.iloc[182] = 70.0
        d = pd.Series(np.full(n, 50.0))

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)):
            self.assertTrue(ind.check_rsi_stoch_short(df, since_ts, max_gap=3))
            self.assertEqual(
                ind.find_rsi_stoch_entry_index(df, since_ts, max_gap=3, side="short"),
                182,
            )


class TestStep8FullOrder(unittest.TestCase):
    def test_requires_smi_before_rsi_path(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rsi_vals = np.full(n, 40.0)
        ma_vals = np.full(n, 45.0)
        rsi_vals[170] = 30.0
        rsi_vals[171] = 44.0
        rsi_vals[172] = 46.0
        rsi = _series_with_fixed_ma(rsi_vals, ma_vals)
        k = pd.Series(np.full(n, 10.0))
        k.iloc[173] = 25.0
        d = pd.Series(np.full(n, 50.0))
        smi = pd.Series(np.full(n, -10.0))
        smi.iloc[190] = -45.0  # SMI بعد الأحداث — يفشل

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

    def test_rejects_rsi_ma_cross_before_raw_touch_35(self):
        """التقاطع مع المتوسط قبل لمس 35 (قيمة RSI الخام) يجب أن يُرفض."""
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rsi_vals = np.full(n, 40.0)
        ma_vals = np.full(n, 45.0)
        # تقاطع مبكر عند 165 — قبل لمس 35
        rsi_vals[164] = 44.0
        rsi_vals[165] = 46.0
        # لمس 35 لاحقًا بدون تقاطع جديد بعده
        rsi_vals[190] = 34.0
        rsi = _series_with_fixed_ma(rsi_vals, ma_vals)
        k = pd.Series(np.full(n, 10.0))
        k.iloc[191] = 25.0
        d = pd.Series(np.full(n, 50.0))

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)):
            self.assertIsNone(
                ind.find_rsi_stoch_entry_index(
                    df,
                    since_ts,
                    max_gap=3,
                    side="long",
                )
            )

    def test_rsi_touch_uses_raw_value_not_ma(self):
        """لمس 35 يُقاس على RSI الخام حتى لو المتوسط أعلى بكثير."""
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rsi_vals = np.full(n, 40.0)
        ma_vals = np.full(n, 70.0)  # المتوسط بعيد — لا يؤثر على اللمس
        rsi_vals[170] = 35.0
        index = None
        rsi = pd.Series(rsi_vals)
        with patch.object(ind, "calc_rsi_tv", return_value=rsi):
            index = ind.find_rsi_touch_index(
                df,
                since_ts,
                threshold=35,
                direction="long",
            )
        self.assertEqual(index, 170)


if __name__ == "__main__":
    unittest.main()
