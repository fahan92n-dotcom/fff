"""
اختبارات شرط MACD 40٪ بمقياس خط الصفر + نافذة يوم/3 أيام.
"""
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd

import indicators as ind


def _df_with_hours(n=300, end=None):
    end = end or datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    # فريم 60 دقيقة
    ts = pd.date_range(end=end, periods=n, freq="60min")
    close = np.linspace(100, 100, n) + np.random.RandomState(0).randn(n) * 0.01
    return pd.DataFrame({
        "ts": ts,
        "open": close,
        "high": close + 0.1,
        "low": close - 0.1,
        "close": close,
        "vol": np.ones(n),
    })


class TestMacdWindowHours(unittest.TestCase):
    def test_small_frames_use_one_day(self):
        for tf in (9, 15, 30, 45, 60):
            self.assertEqual(ind._get_macd_window_hours(tf), 24)

    def test_large_frames_use_three_days(self):
        for tf in (90, 120, 180, 240):
            self.assertEqual(ind._get_macd_window_hours(tf), 72)


class TestMacdLineLong(unittest.TestCase):
    def test_rejects_macd_below_histogram(self):
        df = _df_with_hours()
        n = len(df)
        macd = pd.Series(np.full(n, -1.0))
        hist = pd.Series(np.full(n, -0.5))  # macd < hist
        signal = macd - hist

        with patch.object(ind, "_calc_macd_full", return_value=(macd, signal, hist)):
            self.assertFalse(ind.check_macd_line_long(df, pct=0.40, base_frame=60))

    def test_allows_macd_touching_histogram(self):
        df = _df_with_hours()
        n = len(df)
        # قمة موجبة 100 خلال النافذة → سقف 40؛ الحالي 30 وفوق hist
        macd = pd.Series(np.full(n, 30.0))
        macd.iloc[-5] = 100.0
        hist = pd.Series(np.full(n, -1.0))
        macd.iloc[-1] = 30.0  # 30 >= -1 و 30 <= 40
        signal = macd - hist

        with patch.object(ind, "_calc_macd_full", return_value=(macd, signal, hist)):
            self.assertTrue(ind.check_macd_line_long(df, pct=0.40, base_frame=60))

    def test_rejects_above_40_percent_of_peak_above_zero(self):
        df = _df_with_hours()
        n = len(df)
        macd = pd.Series(np.full(n, 10.0))
        macd.iloc[-10] = 100.0  # قمة 100 → سقف 40
        macd.iloc[-1] = 50.0    # فوق السقف
        hist = pd.Series(np.full(n, -1.0))
        signal = macd - hist

        with patch.object(ind, "_calc_macd_full", return_value=(macd, signal, hist)):
            self.assertFalse(ind.check_macd_line_long(df, pct=0.40, base_frame=60))


class TestMacdLineShort(unittest.TestCase):
    def test_rejects_macd_above_histogram(self):
        df = _df_with_hours()
        n = len(df)
        macd = pd.Series(np.full(n, 1.0))
        hist = pd.Series(np.full(n, 0.5))  # macd > hist
        signal = macd - hist

        with patch.object(ind, "_calc_macd_full", return_value=(macd, signal, hist)):
            self.assertFalse(ind.check_macd_line_short(df, pct=0.40, base_frame=60))

    def test_rejects_deeper_than_40_percent_of_trough_below_zero(self):
        df = _df_with_hours()
        n = len(df)
        macd = pd.Series(np.full(n, -10.0))
        macd.iloc[-10] = -100.0  # قاع -100 → أرضية -40
        macd.iloc[-1] = -50.0    # أعمق من -40
        hist = pd.Series(np.full(n, 1.0))
        # اجعل الحالي تحت الهوستقرام: -50 <= 1
        signal = macd - hist

        with patch.object(ind, "_calc_macd_full", return_value=(macd, signal, hist)):
            self.assertFalse(ind.check_macd_line_short(df, pct=0.40, base_frame=60))

    def test_allows_within_40_percent_band(self):
        df = _df_with_hours()
        n = len(df)
        macd = pd.Series(np.full(n, -10.0))
        macd.iloc[-10] = -100.0
        macd.iloc[-1] = -30.0  # فوق الأرضية -40 وتحت hist
        hist = pd.Series(np.full(n, 1.0))
        signal = macd - hist

        with patch.object(ind, "_calc_macd_full", return_value=(macd, signal, hist)):
            self.assertTrue(ind.check_macd_line_short(df, pct=0.40, base_frame=60))


if __name__ == "__main__":
    unittest.main()
