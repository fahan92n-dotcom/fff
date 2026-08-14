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

    def test_open_upper_band_allows_above_40_percent(self):
        df = _df_with_hours()
        n = len(df)
        macd = pd.Series(np.full(n, 10.0))
        macd.iloc[-10] = 100.0  # قمة 100 → سقف 40 في النسخة الحية
        macd.iloc[-1] = 50.0    # فوق السقف، وما زال فوق الهوستقرام
        hist = pd.Series(np.full(n, -1.0))
        signal = macd - hist

        with patch.object(ind, "_calc_macd_full", return_value=(macd, signal, hist)):
            self.assertTrue(ind.check_macd_line_long(df, pct=None, base_frame=60))

    def test_open_upper_band_still_rejects_below_histogram(self):
        df = _df_with_hours()
        n = len(df)
        macd = pd.Series(np.full(n, -1.0))
        hist = pd.Series(np.full(n, -0.5))  # macd < hist
        signal = macd - hist

        with patch.object(ind, "_calc_macd_full", return_value=(macd, signal, hist)):
            self.assertFalse(ind.check_macd_line_long(df, pct=None, base_frame=60))


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

    def test_open_lower_band_allows_deeper_than_40_percent(self):
        df = _df_with_hours()
        n = len(df)
        macd = pd.Series(np.full(n, -10.0))
        macd.iloc[-10] = -100.0  # قاع -100 → أرضية -40 في النسخة الحية
        macd.iloc[-1] = -50.0    # أعمق من -40 وتحت الهوستقرام
        hist = pd.Series(np.full(n, 1.0))
        signal = macd - hist

        with patch.object(ind, "_calc_macd_full", return_value=(macd, signal, hist)):
            self.assertTrue(ind.check_macd_line_short(df, pct=None, base_frame=60))

    def test_open_lower_band_still_rejects_above_histogram(self):
        df = _df_with_hours()
        n = len(df)
        macd = pd.Series(np.full(n, 1.0))
        hist = pd.Series(np.full(n, 0.5))  # macd > hist
        signal = macd - hist

        with patch.object(ind, "_calc_macd_full", return_value=(macd, signal, hist)):
            self.assertFalse(ind.check_macd_line_short(df, pct=None, base_frame=60))


class TestResolveMacdLinePct(unittest.TestCase):
    def test_default_is_forty_percent(self):
        self.assertEqual(ind.resolve_macd_line_pct(None), 0.40)
        self.assertEqual(ind.resolve_macd_line_pct({}), 0.40)

    def test_explicit_none_opens_the_far_side(self):
        self.assertIsNone(ind.resolve_macd_line_pct({"macd_line_pct": None}))


if __name__ == "__main__":
    unittest.main()
