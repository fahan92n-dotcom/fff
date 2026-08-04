"""
اختبارات منطق الخطوة 8: تقاطع RSI/المتوسط + Stochastic %K خلال ±3 شموع،
دون اشتراط الحالة على الشمعة الحالية.
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


def _series_with_fixed_signal(values, signal_values):
    """Series بحيث .rolling(...).mean() يُرجع signal_values."""
    s = pd.Series(values, dtype=float)

    class _Roll:
        def mean(self, *_a, **_k):
            return pd.Series(signal_values, dtype=float)

    s.rolling = lambda *_a, **_k: _Roll()
    return s


class TestCheckRsiStochLong(unittest.TestCase):
    def test_passes_when_crosses_within_gap_even_if_current_stoch_back_below_20(self):
        """
        تقاطع RSI لفوق متوسطه، Stoch يطلع فوق 20 خلال فجوة=2،
        ثم يعود Stoch تحت 20 الآن — يجب النجاح.
        """
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rc, sc = 180, 182

        rsi_vals = np.full(n, 40.0)
        sig_vals = np.full(n, 45.0)
        rsi_vals[rc - 1] = 44.0
        rsi_vals[rc] = 46.0  # cross up vs sig=45

        k = pd.Series(np.full(n, 10.0))
        k.iloc[sc - 1] = 18.0
        k.iloc[sc] = 25.0  # stoch cross, gap=2
        k.iloc[-1] = 15.0  # الآن تحت 20
        d = pd.Series(np.full(n, 50.0))

        rsi = _series_with_fixed_signal(rsi_vals, sig_vals)

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)):
            self.assertTrue(ind.check_rsi_stoch(df, since_ts, max_gap=3))

    def test_fails_when_stoch_cross_farther_than_gap(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rc, sc = 180, 185  # gap=5 > 3

        rsi_vals = np.full(n, 40.0)
        sig_vals = np.full(n, 45.0)
        rsi_vals[rc - 1] = 44.0
        rsi_vals[rc] = 46.0

        k = pd.Series(np.full(n, 10.0))
        k.iloc[sc - 1] = 18.0
        k.iloc[sc] = 25.0
        d = pd.Series(np.full(n, 50.0))

        rsi = _series_with_fixed_signal(rsi_vals, sig_vals)

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)):
            self.assertFalse(ind.check_rsi_stoch(df, since_ts, max_gap=3))


class TestCheckRsiStochShort(unittest.TestCase):
    def test_passes_when_crosses_within_gap_even_if_current_stoch_back_above_80(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rc, sc = 180, 181

        rsi_vals = np.full(n, 60.0)
        sig_vals = np.full(n, 55.0)
        rsi_vals[rc - 1] = 56.0
        rsi_vals[rc] = 54.0  # cross down vs sig=55

        k = pd.Series(np.full(n, 90.0))
        k.iloc[sc - 1] = 82.0
        k.iloc[sc] = 75.0  # stoch cross below 80, gap=1
        k.iloc[-1] = 85.0  # الآن فوق 80
        d = pd.Series(np.full(n, 50.0))

        rsi = _series_with_fixed_signal(rsi_vals, sig_vals)

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)):
            self.assertTrue(ind.check_rsi_stoch_short(df, since_ts, max_gap=3))


class TestFindRsiStochEntryIndex(unittest.TestCase):
    def test_returns_completion_candle_not_latest_bar(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rc, sc = 180, 182

        rsi_vals = np.full(n, 40.0)
        sig_vals = np.full(n, 45.0)
        rsi_vals[rc - 1] = 44.0
        rsi_vals[rc] = 46.0

        k = pd.Series(np.full(n, 10.0))
        k.iloc[sc - 1] = 18.0
        k.iloc[sc] = 25.0
        k.iloc[-1] = 15.0
        d = pd.Series(np.full(n, 50.0))
        rsi = _series_with_fixed_signal(rsi_vals, sig_vals)

        with patch.object(ind, "calc_rsi_tv", return_value=rsi), \
             patch.object(ind, "calc_stoch_tv", return_value=(k, d)):
            index = ind.find_rsi_stoch_entry_index(
                df,
                since_ts,
                max_gap=3,
                side="long",
            )

        self.assertEqual(index, sc)
        self.assertNotEqual(index, n - 1)


class TestStep8SmiThenRsiStochOrder(unittest.TestCase):
    def test_fails_when_rsi_stoch_complete_before_smi_touch(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        rc, sc, smi_i = 170, 172, 190

        rsi_vals = np.full(n, 40.0)
        sig_vals = np.full(n, 45.0)
        rsi_vals[rc - 1] = 44.0
        rsi_vals[rc] = 46.0
        rsi_vals[sc] = 30.0  # لمس RSI قبل SMI فقط

        k = pd.Series(np.full(n, 10.0))
        k.iloc[sc - 1] = 18.0
        k.iloc[sc] = 25.0
        d = pd.Series(np.full(n, 50.0))
        rsi = _series_with_fixed_signal(rsi_vals, sig_vals)

        smi = pd.Series(np.full(n, -10.0))
        smi.iloc[smi_i] = -45.0  # تشبع SMI بعد اكتمال RSI/Stoch

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

    def test_passes_when_smi_touch_precedes_rsi_stoch(self):
        n = 220
        df = _base_df(n)
        since_ts = df["ts"].iloc[150]
        smi_i, rc, sc = 160, 180, 182

        rsi_vals = np.full(n, 40.0)
        sig_vals = np.full(n, 45.0)
        rsi_vals[rc - 1] = 44.0
        rsi_vals[rc] = 46.0
        rsi_vals[170] = 30.0  # لمس RSI بعد SMI

        k = pd.Series(np.full(n, 10.0))
        k.iloc[sc - 1] = 18.0
        k.iloc[sc] = 25.0
        d = pd.Series(np.full(n, 50.0))
        rsi = _series_with_fixed_signal(rsi_vals, sig_vals)

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

        self.assertEqual(index, sc)


if __name__ == "__main__":
    unittest.main()
