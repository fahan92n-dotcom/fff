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


class TestMacdZeroBand(unittest.TestCase):
    def test_band_is_40_percent_of_peak_and_trough_from_zero(self):
        window = pd.Series([-100.0, -10.0, 20.0, 100.0])
        floor, ceiling = ind._macd_zero_band(window, pct=0.40)
        self.assertEqual(floor, -40.0)
        self.assertEqual(ceiling, 40.0)

    def test_missing_positive_side_sets_ceiling_to_zero(self):
        window = pd.Series([-100.0, -10.0])
        floor, ceiling = ind._macd_zero_band(window, pct=0.40)
        self.assertEqual(floor, -40.0)
        self.assertEqual(ceiling, 0.0)

    def test_missing_negative_side_sets_floor_to_zero(self):
        window = pd.Series([10.0, 100.0])
        floor, ceiling = ind._macd_zero_band(window, pct=0.40)
        self.assertEqual(floor, 0.0)
        self.assertEqual(ceiling, 40.0)


class TestMacdLineLong(unittest.TestCase):
    def _patch(self, df, macd, hist):
        signal = macd - hist
        return patch.object(ind, "_calc_macd_full", return_value=(macd, signal, hist))

    def test_buy_rejects_when_macd_is_below_histogram(self):
        """شراء: الحد السفلي = خط MACD أكبر من الهوستقرام."""
        df = _df_with_hours()
        n = len(df)
        macd = pd.Series(np.full(n, -10.0))
        macd.iloc[-10] = 100.0
        macd.iloc[-8] = -100.0
        macd.iloc[-1] = -10.0
        hist = pd.Series(np.full(n, -5.0))  # macd < hist
        with self._patch(df, macd, hist):
            self.assertFalse(ind.check_macd_line_long(df, pct=0.40, base_frame=60))

    def test_buy_allows_macd_above_histogram_within_40_percent(self):
        df = _df_with_hours()
        n = len(df)
        macd = pd.Series(np.full(n, 30.0))
        macd.iloc[-10] = 100.0
        macd.iloc[-8] = -100.0
        macd.iloc[-1] = 30.0  # 30 > hist(-1) وداخل [−40, +40]
        hist = pd.Series(np.full(n, -1.0))
        with self._patch(df, macd, hist):
            self.assertTrue(ind.check_macd_line_long(df, pct=0.40, base_frame=60))

    def test_buy_allows_touching_histogram(self):
        df = _df_with_hours()
        n = len(df)
        macd = pd.Series(np.full(n, -20.0))
        macd.iloc[-10] = 100.0
        macd.iloc[-8] = -100.0
        macd.iloc[-1] = -20.0
        hist = pd.Series(np.full(n, -20.0))  # يلامس
        with self._patch(df, macd, hist):
            self.assertTrue(ind.check_macd_line_long(df, pct=0.40, base_frame=60))

    def test_buy_rejects_above_40_percent_of_peak_above_zero(self):
        df = _df_with_hours()
        n = len(df)
        macd = pd.Series(np.full(n, 50.0))
        macd.iloc[-10] = 100.0  # قمة 100 → سقف 40
        macd.iloc[-8] = -100.0
        macd.iloc[-1] = 50.0    # أكبر من الهوستقرام لكن فوق 40٪
        hist = pd.Series(np.full(n, -1.0))
        with self._patch(df, macd, hist):
            self.assertFalse(ind.check_macd_line_long(df, pct=0.40, base_frame=60))


class TestMacdLineShort(unittest.TestCase):
    def _patch(self, df, macd, hist):
        signal = macd - hist
        return patch.object(ind, "_calc_macd_full", return_value=(macd, signal, hist))

    def test_sell_rejects_when_macd_is_above_green_histogram(self):
        """بيع: خط MACD أقل من الهوستقرام الأخضر."""
        df = _df_with_hours()
        n = len(df)
        macd = pd.Series(np.full(n, 10.0))
        macd.iloc[-10] = 100.0
        macd.iloc[-8] = -100.0
        macd.iloc[-1] = 10.0
        hist = pd.Series(np.full(n, 5.0))  # macd > hist
        with self._patch(df, macd, hist):
            self.assertFalse(ind.check_macd_line_short(df, pct=0.40, base_frame=60))

    def test_sell_allows_macd_below_green_histogram_within_40_percent(self):
        df = _df_with_hours()
        n = len(df)
        macd = pd.Series(np.full(n, -10.0))
        macd.iloc[-10] = 100.0
        macd.iloc[-8] = -100.0
        macd.iloc[-1] = -10.0  # −10 < hist(50) وداخل [−40, +40]
        hist = pd.Series(np.full(n, 50.0))
        with self._patch(df, macd, hist):
            self.assertTrue(ind.check_macd_line_short(df, pct=0.40, base_frame=60))

    def test_sell_allows_touching_green_histogram(self):
        df = _df_with_hours()
        n = len(df)
        macd = pd.Series(np.full(n, 20.0))
        macd.iloc[-10] = 100.0
        macd.iloc[-8] = -100.0
        macd.iloc[-1] = 20.0
        hist = pd.Series(np.full(n, 20.0))  # يلامس
        with self._patch(df, macd, hist):
            self.assertTrue(ind.check_macd_line_short(df, pct=0.40, base_frame=60))

    def test_sell_rejects_deeper_than_40_percent_of_trough_below_zero(self):
        df = _df_with_hours()
        n = len(df)
        macd = pd.Series(np.full(n, -50.0))
        macd.iloc[-10] = -100.0  # قاع −100 → أرضية −40
        macd.iloc[-8] = 100.0
        macd.iloc[-1] = -50.0    # أقل من الهوستقرام لكن أعمق من 40٪
        hist = pd.Series(np.full(n, 1.0))
        with self._patch(df, macd, hist):
            self.assertFalse(ind.check_macd_line_short(df, pct=0.40, base_frame=60))


def _smi_series(values):
    series = pd.Series(values, dtype=float)
    return series, series, series


class TestFindSaturationStartIndex(unittest.TestCase):
    def test_returns_first_candle_of_current_run(self):
        df = _df_with_hours(n=300)
        smi = np.zeros(300)
        smi[295:] = -50.0  # نوبة تشبع من 295 حتى النهاية
        with patch.object(ind, "calc_smi", return_value=_smi_series(smi)):
            self.assertEqual(
                ind.find_saturation_start_index(df, threshold=-40, direction="long"),
                295,
            )

    def test_ignores_older_disconnected_run(self):
        df = _df_with_hours(n=300)
        smi = np.zeros(300)
        smi[100:110] = -50.0  # نوبة قديمة منفصلة
        smi[297:] = -50.0
        with patch.object(ind, "calc_smi", return_value=_smi_series(smi)):
            self.assertEqual(
                ind.find_saturation_start_index(df, threshold=-40, direction="long"),
                297,
            )

    def test_returns_none_when_last_candle_not_saturated(self):
        df = _df_with_hours(n=300)
        smi = np.zeros(300)
        smi[100:110] = -50.0
        with patch.object(ind, "calc_smi", return_value=_smi_series(smi)):
            self.assertIsNone(
                ind.find_saturation_start_index(df, threshold=-40, direction="long")
            )

    def test_short_direction_uses_upper_threshold(self):
        df = _df_with_hours(n=300)
        smi = np.zeros(300)
        smi[290:] = 45.0
        with patch.object(ind, "calc_smi", return_value=_smi_series(smi)):
            self.assertEqual(
                ind.find_saturation_start_index(df, threshold=40, direction="short"),
                290,
            )


class TestCheckMacdAtSaturationStart(unittest.TestCase):
    """القرار يُتخذ على أول شمعة متشبعة ولا يتأثر بالشموع اللاحقة."""

    def test_evaluates_on_sliced_frame_ending_at_saturation_start(self):
        df = _df_with_hours(n=300)
        seen_lengths = {}

        def fake_red(df_eval):
            seen_lengths["hist"] = len(df_eval)
            return True

        def fake_line(df_eval, pct=0.40, base_frame=60):
            seen_lengths["line"] = len(df_eval)
            return True

        with patch.object(ind, "find_saturation_start_index", return_value=250), \
                patch.object(ind, "check_macd_red", side_effect=fake_red), \
                patch.object(ind, "check_macd_line_long", side_effect=fake_line):
            self.assertTrue(
                ind.check_macd_at_saturation_start(df, 60, direction="long")
            )
        self.assertEqual(seen_lengths, {"hist": 251, "line": 251})

    def test_rejects_without_current_saturation(self):
        df = _df_with_hours(n=300)
        with patch.object(ind, "find_saturation_start_index", return_value=None):
            self.assertFalse(
                ind.check_macd_at_saturation_start(df, 60, direction="long")
            )

    def test_rejects_when_saturation_start_lacks_warmup(self):
        df = _df_with_hours(n=300)
        with patch.object(ind, "find_saturation_start_index", return_value=100):
            self.assertFalse(
                ind.check_macd_at_saturation_start(df, 60, direction="long")
            )

    def test_short_direction_uses_short_checks(self):
        df = _df_with_hours(n=300)
        with patch.object(ind, "find_saturation_start_index", return_value=250), \
                patch.object(ind, "check_macd_green", return_value=True) as green, \
                patch.object(ind, "check_macd_line_short", return_value=True) as line:
            self.assertTrue(
                ind.check_macd_at_saturation_start(df, 60, direction="short")
            )
        green.assert_called_once()
        line.assert_called_once()


if __name__ == "__main__":
    unittest.main()
