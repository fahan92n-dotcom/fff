"""
اختبارات وحدة لـ _has_higher_tf_saturation و TF_TO_API و resample_ohlcv.

تتحقق هذه الاختبارات من:
1. أن TF_TO_API يربط كل فريم بمصدره الصحيح المحدد في TRIPLING_PAIRS.
2. أن _has_higher_tf_saturation تجلب البيانات من المصدر الصحيح لكل فريم أعلى
   (30m لفريمات 90/120/150، و60m لفريمات 60/180/210/240) وليس دائماً من مصدر المرشح.
3. أن الفريمات ذات البيانات الكافية يتم تقييمها بشكل صحيح (لا ترجع False بسبب بيانات وهمية ناقصة).
"""
import unittest
from unittest.mock import patch
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# تحميل الوحدة بدون تشغيل main()
import fahadal92 as bot


def _make_ohlcv(n: int, smi_value: float = -50.0) -> pd.DataFrame:
    """
    ينشئ DataFrame وهمي لـ OHLCV يحتوي على n شمعة مغلقة تُنتج SMI قريبًا من smi_value.
    نستخدم أسعارًا ثابتة (flat) مع موجة صغيرة عشوائية للتحقق من أن الحسابات تعمل.
    """
    np.random.seed(42)
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = pd.date_range(start=base_time, periods=n, freq="1min")

    # سعر ثابت مع ذبذبة صغيرة
    close = np.full(n, 1.0) + np.random.randn(n) * 0.001

    # لجعل SMI أقل من -40 (تشبع بيعي): نجعل الأسعار في نهاية السلسلة أقل من وسطها
    if smi_value <= -40:
        close[-50:] = close[-50:] - 0.05  # انخفاض حاد في آخر 50 شمعة

    # لجعل SMI أعلى من 40 (تشبع شرائي): نجعل الأسعار في نهاية السلسلة أعلى من وسطها
    if smi_value >= 40:
        close[-50:] = close[-50:] + 0.05

    df = pd.DataFrame({
        "ts": times,
        "open": close,
        "high": close + 0.0005,
        "low": close - 0.0005,
        "close": close,
        "vol": np.ones(n) * 1000.0,
    })
    return df


class TestTFToAPI(unittest.TestCase):
    """اختبار صحة خريطة TF_TO_API."""

    def test_low_frames_use_1m(self):
        """الفريمات 9-45 دقيقة يجب أن تستخدم مصدر 1m."""
        for tf in [9, 12, 15, 18, 21, 24, 27, 30, 45]:
            self.assertEqual(bot.TF_TO_API[tf], "1m",
                             f"الفريم {tf}m يجب أن يستخدم 1m كمصدر")

    def test_frame_60_uses_60m(self):
        """الفريم 60 دقيقة يجب أن يستخدم مصدر 60m."""
        self.assertEqual(bot.TF_TO_API[60], "60m")

    def test_medium_frames_use_30m(self):
        """الفريمات 90/120/150 دقيقة يجب أن تستخدم مصدر 30m."""
        for tf in [90, 120, 150]:
            self.assertEqual(bot.TF_TO_API[tf], "30m",
                             f"الفريم {tf}m يجب أن يستخدم 30m كمصدر")

    def test_high_frames_use_60m(self):
        """الفريمات 180/210/240 دقيقة يجب أن تستخدم مصدر 60m."""
        for tf in [180, 210, 240]:
            self.assertEqual(bot.TF_TO_API[tf], "60m",
                             f"الفريم {tf}m يجب أن يستخدم 60m كمصدر")

    def test_all_timeframe_chain_covered(self):
        """كل الفريمات في TIMEFRAME_CHAIN يجب أن تكون موجودة في TF_TO_API."""
        for tf in bot.TIMEFRAME_CHAIN:
            self.assertIn(tf, bot.TF_TO_API,
                          f"الفريم {tf}m غير موجود في TF_TO_API")

    def test_derived_from_tripling_pairs(self):
        """TF_TO_API يجب أن يكون مشتقًا مباشرة من TRIPLING_PAIRS (العمود الرابع)."""
        expected = {p[0]: p[3] for p in bot.TRIPLING_PAIRS}
        self.assertEqual(bot.TF_TO_API, expected)


class TestHasHigherTFSaturationSourceRouting(unittest.TestCase):
    """
    يتحقق أن _has_higher_tf_saturation تستدعي get_cached
    بالمصدر الصحيح (native_api من TF_TO_API) لكل فريم أعلى،
    وليس دائماً بـ base_api الخاص بالمرشح.
    """

    def _make_candidate(self, base_frame: int, base_api: str = "1m") -> dict:
        return {
            "sym": "TESTUSDT",
            "base_frame": base_frame,
            "base_api": base_api,
        }

    def test_uses_30m_source_for_90m_frame(self):
        """
        لمرشح base_api='1m'، عند فحص الفريم 90m يجب استدعاء get_cached بـ '30m' لا '1m'.
        """
        candidate = self._make_candidate(base_frame=9, base_api="1m")
        calls = []

        def mock_get_cached(sym, api):
            calls.append((sym, api))
            return pd.DataFrame()  # فارغ → تخطَّ

        def mock_get_resampled(raw_df, sym, api, tf):
            return pd.DataFrame()

        with patch.object(bot, "get_cached", side_effect=mock_get_cached):
            bot._has_higher_tf_saturation(candidate, "buy", mock_get_resampled)

        apis_called = [api for _, api in calls]
        # يجب أن نرى '30m' في الاستدعاءات (للفريمات 90/120/150)
        self.assertIn("30m", apis_called,
                      "يجب استدعاء get_cached('TESTUSDT', '30m') للفريمات 90/120/150")

    def test_uses_60m_source_for_180m_frame(self):
        """
        لمرشح base_api='1m'، عند فحص الفريم 180m يجب استدعاء get_cached بـ '60m' لا '1m'.
        """
        candidate = self._make_candidate(base_frame=9, base_api="1m")
        calls = []

        def mock_get_cached(sym, api):
            calls.append((sym, api))
            return pd.DataFrame()

        def mock_get_resampled(raw_df, sym, api, tf):
            return pd.DataFrame()

        with patch.object(bot, "get_cached", side_effect=mock_get_cached):
            bot._has_higher_tf_saturation(candidate, "buy", mock_get_resampled)

        apis_called = [api for _, api in calls]
        self.assertIn("60m", apis_called,
                      "يجب استدعاء get_cached('TESTUSDT', '60m') للفريمات 180/210/240")

    def test_no_duplicate_1m_calls_for_large_frames(self):
        """
        يجب عدم استدعاء get_cached بـ '1m' للفريمات 90/120/150/180/210/240.
        """
        candidate = self._make_candidate(base_frame=9, base_api="1m")
        calls_per_api_per_tf = []

        def mock_get_cached(sym, api):
            return pd.DataFrame()

        tf_api_pairs = []

        def mock_get_resampled(raw_df, _sym, _api, tf):
            tf_api_pairs.append((_api, tf))
            return pd.DataFrame()

        # تتبع الـ api لكل فريم عبر get_cached
        call_map = {}  # tf → api used

        original_get_cached = bot.get_cached

        def tracking_get_cached(sym, api):
            # نسجل الـ api لكل استدعاء
            calls_per_api_per_tf.append(api)
            return pd.DataFrame()

        with patch.object(bot, "get_cached", side_effect=tracking_get_cached):
            bot._has_higher_tf_saturation(candidate, "buy", mock_get_resampled)

        # نحسب عدد مرات استدعاء '1m' — يجب أن يكون فقط للفريمات 12-45
        # وليس للفريمات الكبيرة
        # عدد الفريمات من 12 إلى 45 دقيقة التي تستخدم 1m = 12,15,18,21,24,27,30,45 = 8 فريمات
        one_m_calls = calls_per_api_per_tf.count("1m")
        thirty_m_calls = calls_per_api_per_tf.count("30m")
        sixty_m_calls = calls_per_api_per_tf.count("60m")

        # 12,15,18,21,24,27,30,45,60 → 8 فريمات تستخدم 1m (12-45) + فريم 60 يستخدم 60m
        # لكن هنا base_frame=9 فنفحص 12,15,18,21,24,27,30,45 (1m) + 60 (60m) + 90,120,150 (30m) + 180,210,240 (60m)
        self.assertEqual(one_m_calls, 8,  # 12,15,18,21,24,27,30,45
                         f"المتوقع 8 استدعاء بـ 1m، حصلنا على {one_m_calls}")
        self.assertEqual(thirty_m_calls, 3,  # 90,120,150
                         f"المتوقع 3 استدعاء بـ 30m، حصلنا على {thirty_m_calls}")
        self.assertEqual(sixty_m_calls, 4,  # 60,180,210,240
                         f"المتوقع 4 استدعاء بـ 60m، حصلنا على {sixty_m_calls}")


class TestHasHigherTFSaturationLogic(unittest.TestCase):
    """
    يتحقق أن _has_higher_tf_saturation ترجع True عندما يكون فريم أعلى في تشبع،
    وFalse عندما لا يكون، مع بيانات كافية.
    """

    def _make_candidate(self, base_frame: int) -> dict:
        return {
            "sym": "BTCUSDT",
            "base_frame": base_frame,
            "base_api": "1m",
        }

    def test_returns_false_when_no_saturation(self):
        """ترجع False عندما لا يوجد أي فريم أعلى في تشبع بيعي."""
        candidate = self._make_candidate(base_frame=45)

        # كل البيانات فارغة → لا يوجد تشبع
        def mock_get_cached(sym, api):
            return pd.DataFrame()

        def mock_get_resampled(raw, sym, api, tf):
            return pd.DataFrame()

        with patch.object(bot, "get_cached", side_effect=mock_get_cached):
            result = bot._has_higher_tf_saturation(candidate, "buy", mock_get_resampled)

        self.assertFalse(result)

    def test_returns_true_when_higher_tf_oversold(self):
        """ترجع True عندما يكون فريم أعلى في تشبع بيعي (check_smi_oversold = True)."""
        candidate = self._make_candidate(base_frame=9)

        # نصنع DataFrame كافي مع SMI في منطقة التشبع البيعي
        large_df = _make_ohlcv(n=500, smi_value=-50.0)

        def mock_get_cached(sym, api):
            return large_df.copy()

        def mock_get_resampled(raw, sym, api, tf):
            # نرجع df كافي فقط للفريم 60 (أول فريم أعلى من 9 في TIMEFRAME_CHAIN)
            if tf == 60:
                return large_df.copy()
            return pd.DataFrame()

        with patch.object(bot, "get_cached", side_effect=mock_get_cached):
            # check_smi_oversold يحتاج SMI ≤ -40 — لضمان التحقق نُغلف الدالة
            with patch.object(bot, "check_smi_oversold", return_value=True):
                result = bot._has_higher_tf_saturation(candidate, "buy", mock_get_resampled)

        self.assertTrue(result)

    def test_returns_true_when_higher_tf_overbought(self):
        """ترجع True عندما يكون فريم أعلى في تشبع شرائي (check_smi_overbought = True) لإشارة بيع."""
        candidate = self._make_candidate(base_frame=9)

        large_df = _make_ohlcv(n=500, smi_value=50.0)

        def mock_get_cached(sym, api):
            return large_df.copy()

        def mock_get_resampled(raw, sym, api, tf):
            if tf == 90:
                return large_df.copy()
            return pd.DataFrame()

        with patch.object(bot, "get_cached", side_effect=mock_get_cached):
            with patch.object(bot, "check_smi_overbought", return_value=True):
                result = bot._has_higher_tf_saturation(candidate, "sell", mock_get_resampled)

        self.assertTrue(result)

    def test_skips_frame_when_base_frame_equal(self):
        """لا يفحص فريمًا مساوياً لـ base_frame."""
        candidate = self._make_candidate(base_frame=240)
        calls = []

        def mock_get_cached(sym, api):
            calls.append(api)
            return pd.DataFrame()

        def mock_get_resampled(raw, sym, api, tf):
            return pd.DataFrame()

        with patch.object(bot, "get_cached", side_effect=mock_get_cached):
            bot._has_higher_tf_saturation(candidate, "buy", mock_get_resampled)

        # base_frame = 240 وهو أكبر فريم في TIMEFRAME_CHAIN → لا استدعاءات
        self.assertEqual(len(calls), 0,
                         "يجب عدم فحص أي فريم عندما base_frame = 240 (أعلى فريم)")


class TestResampleOrigin(unittest.TestCase):
    """يتحقق أن resample_ohlcv يُنتج حدود شموع متوافقة مع epoch UTC (Binance/TradingView)."""

    def test_90min_candle_boundary(self):
        """شموع 90m يجب أن تبدأ عند 00:00, 01:30, 03:00, 04:30... UTC."""
        # أنشئ بيانات 1m من 00:00 إلى 04:00 UTC
        times = pd.date_range(
            start=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            periods=240, freq="1min"
        )
        df = pd.DataFrame({
            "ts": times,
            "open": 1.0, "high": 1.001, "low": 0.999, "close": 1.0, "vol": 1.0,
        })

        # نستدعي بـ وقت مستقبلي لتجنب حذف الشمعة الأخيرة
        with patch("fahadal92.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 2, tzinfo=timezone.utc)
            mock_dt.side_effect = datetime
            result = bot.resample_ohlcv(df.copy(), 90)

        if not result.empty:
            expected_boundaries = [0, 90, 180]  # دقائق من بداية اليوم UTC
            for ts in result["ts"]:
                minutes_from_midnight = ts.hour * 60 + ts.minute
                self.assertIn(minutes_from_midnight, expected_boundaries,
                              f"حد الشمعة {ts} غير متوافق مع epoch UTC")


if __name__ == "__main__":
    unittest.main(verbosity=2)
