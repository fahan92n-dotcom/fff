"""
اختبارات وحدة لـ _has_higher_tf_saturation و TF_TO_API و resample_ohlcv.

تتحقق هذه الاختبارات من:
1. أن TF_TO_API يربط كل فريم بمصدره الصحيح المحدد في TRIPLING_PAIRS.
2. أن _has_higher_tf_saturation تفحص الفريم التالي مباشرة وتستخدم مصدره الصحيح
   (مثل 30m لفريم 90، و60m لفريم 180) وليس دائماً مصدر المرشح.
3. أن الفريمات ذات البيانات الكافية يتم تقييمها بشكل صحيح (لا ترجع False بسبب بيانات وهمية ناقصة).
"""
import unittest
from unittest.mock import patch
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# تحميل الوحدة بدون تشغيل main()
import cascade_steps
import cascade_pipeline as pipeline
import fahadal92 as bot
import indicators as ind


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

    def test_high_frames_use_native_divisible_sources(self):
        """الفريمات الكبيرة تستخدم مصدرًا ينقسم عليها صحيحًا."""
        self.assertEqual(bot.TF_TO_API[180], "60m")
        self.assertEqual(bot.TF_TO_API[210], "30m")  # 210 لا ينقسم على 60
        self.assertEqual(bot.TF_TO_API[240], "60m")
        self.assertEqual(bot.TF_TO_API[cascade_steps.CANCEL_ONLY_HIGHER_TF], "60m")

    def test_all_tripling_targets_divisible_by_source(self):
        """كل base/confirm/triple يجب أن ينقسم على مصدره."""
        for base, confirm, triple, base_api, triple_api in bot.TRIPLING_PAIRS:
            base_minutes = int(base_api.replace("m", ""))
            triple_minutes = int(triple_api.replace("m", ""))
            self.assertEqual(base % base_minutes, 0, f"{base}m من {base_api}")
            self.assertEqual(confirm % base_minutes, 0, f"{confirm}m من {base_api}")
            self.assertEqual(triple % triple_minutes, 0, f"{triple}m من {triple_api}")

    def test_all_timeframe_chain_covered(self):
        """كل الفريمات في TIMEFRAME_CHAIN يجب أن تكون موجودة في TF_TO_API."""
        for tf in bot.TIMEFRAME_CHAIN:
            self.assertIn(tf, bot.TF_TO_API,
                          f"الفريم {tf}m غير موجود في TF_TO_API")

    def test_derived_from_tripling_pairs(self):
        """TF_TO_API = مصادر TRIPLING_PAIRS + سقف الإلغاء 5 ساعات."""
        expected = {p[0]: p[3] for p in bot.TRIPLING_PAIRS}
        expected[cascade_steps.CANCEL_ONLY_HIGHER_TF] = (
            cascade_steps.CANCEL_ONLY_HIGHER_API
        )
        self.assertEqual(bot.TF_TO_API, expected)

    def test_five_hour_ceiling_cancels_240_and_does_not_trade(self):
        """5 ساعات يلغي 240m ولا يدخل سلسلة الإشارات."""
        self.assertEqual(bot.NEXT_TF[240], cascade_steps.CANCEL_ONLY_HIGHER_TF)
        self.assertNotIn(cascade_steps.CANCEL_ONLY_HIGHER_TF, bot.TIMEFRAME_CHAIN)
        self.assertNotIn(
            cascade_steps.CANCEL_ONLY_HIGHER_TF,
            [pair[0] for pair in bot.TRIPLING_PAIRS],
        )
        self.assertEqual(
            cascade_steps.CANCEL_ONLY_HIGHER_TF
            % int(cascade_steps.CANCEL_ONLY_HIGHER_API.replace("m", "")),
            0,
        )


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

    def test_uses_30m_source_for_immediate_90m_frame(self):
        """
        لمرشح base_api='1m'، عند فحص الفريم 90m يجب استدعاء get_cached بـ '30m' لا '1m'.
        """
        candidate = self._make_candidate(base_frame=60, base_api="60m")
        calls = []

        def mock_get_cached(sym, api):
            calls.append((sym, api))
            return pd.DataFrame()  # فارغ → تخطَّ

        def mock_get_resampled(raw_df, sym, api, tf):
            return pd.DataFrame()

        with patch.object(cascade_steps, "get_cached", side_effect=mock_get_cached):
            bot._has_higher_tf_saturation(candidate, "buy", mock_get_resampled)

        apis_called = [api for _, api in calls]
        self.assertEqual(apis_called, ["30m"])

    def test_uses_60m_source_for_immediate_180m_frame(self):
        """
        لمرشح base_api='1m'، عند فحص الفريم 180m يجب استدعاء get_cached بـ '60m' لا '1m'.
        """
        candidate = self._make_candidate(base_frame=150, base_api="30m")
        calls = []

        def mock_get_cached(sym, api):
            calls.append((sym, api))
            return pd.DataFrame()

        def mock_get_resampled(raw_df, sym, api, tf):
            return pd.DataFrame()

        with patch.object(cascade_steps, "get_cached", side_effect=mock_get_cached):
            bot._has_higher_tf_saturation(candidate, "buy", mock_get_resampled)

        apis_called = [api for _, api in calls]
        self.assertEqual(apis_called, ["60m"])

    def test_checks_only_the_immediate_next_frame(self):
        """
        يجب عدم استدعاء get_cached بـ '1m' للفريمات 90/120/150/180/210/240.
        """
        candidate = self._make_candidate(base_frame=9, base_api="1m")
        calls_per_api_per_tf = []

        def mock_get_resampled(raw_df, _sym, _api, tf):
            return pd.DataFrame()

        def tracking_get_cached(sym, api):
            calls_per_api_per_tf.append(api)
            return pd.DataFrame()

        with patch.object(cascade_steps, "get_cached", side_effect=tracking_get_cached):
            bot._has_higher_tf_saturation(candidate, "buy", mock_get_resampled)

        self.assertEqual(calls_per_api_per_tf, ["1m"])


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

        with patch.object(cascade_steps, "get_cached", side_effect=mock_get_cached):
            result = bot._has_higher_tf_saturation(candidate, "buy", mock_get_resampled)

        self.assertFalse(result)

    def test_returns_true_when_higher_tf_oversold(self):
        """ترجع True عندما يكون فريم أعلى في تشبع بيعي الآن."""
        candidate = self._make_candidate(base_frame=9)
        large_df = _make_ohlcv(n=500, smi_value=-50.0)
        smi = pd.Series([-50.0] * len(large_df))

        def mock_get_cached(sym, api):
            return large_df.copy()

        def mock_get_resampled(raw, sym, api, tf):
            if tf == 12:
                return large_df.copy()
            return pd.DataFrame()

        with patch.object(cascade_steps, "get_cached", side_effect=mock_get_cached):
            with patch.object(
                cascade_steps,
                "calc_smi",
                return_value=(smi, smi, smi),
            ):
                result = bot._has_higher_tf_saturation(
                    candidate,
                    "buy",
                    mock_get_resampled,
                )

        self.assertTrue(result)

    def test_returns_true_when_higher_tf_overbought(self):
        """ترجع True عندما يكون فريم أعلى في تشبع شرائي الآن لإشارة بيع."""
        candidate = self._make_candidate(base_frame=60)
        large_df = _make_ohlcv(n=500, smi_value=50.0)
        smi = pd.Series([50.0] * len(large_df))

        def mock_get_cached(sym, api):
            return large_df.copy()

        def mock_get_resampled(raw, sym, api, tf):
            if tf == 90:
                return large_df.copy()
            return pd.DataFrame()

        with patch.object(cascade_steps, "get_cached", side_effect=mock_get_cached):
            with patch.object(
                cascade_steps,
                "calc_smi",
                return_value=(smi, smi, smi),
            ):
                result = bot._has_higher_tf_saturation(
                    candidate,
                    "sell",
                    mock_get_resampled,
                )

        self.assertTrue(result)

    def test_skips_smaller_tf_on_first_closed_candle_after_higher_exit_buy(self):
        """بعد خروج 30m من التشبع وإغلاق أول شمعة → نلغي تشبع 27m (شراء)."""
        candidate = self._make_candidate(base_frame=27)
        large_df = _make_ohlcv(n=200, smi_value=0.0)
        # السابقة متشبعة، الحالية خرجت
        smi = pd.Series([0.0] * 198 + [-50.0, -10.0])

        def mock_get_cached(sym, api):
            return large_df.copy()

        def mock_get_resampled(raw, sym, api, tf):
            self.assertEqual(tf, 30)
            return large_df.copy()

        with patch.object(cascade_steps, "get_cached", side_effect=mock_get_cached):
            with patch.object(
                cascade_steps,
                "calc_smi",
                return_value=(smi, smi, smi),
            ):
                result = bot._has_higher_tf_saturation(
                    candidate,
                    "buy",
                    mock_get_resampled,
                )

        self.assertTrue(result)

    def test_skips_smaller_tf_on_first_closed_candle_after_higher_exit_sell(self):
        """بعد خروج 30m من التشبع وإغلاق أول شمعة → نلغي تشبع 27m (بيع)."""
        candidate = self._make_candidate(base_frame=27)
        large_df = _make_ohlcv(n=200, smi_value=0.0)
        smi = pd.Series([0.0] * 198 + [50.0, 10.0])

        def mock_get_cached(sym, api):
            return large_df.copy()

        def mock_get_resampled(raw, sym, api, tf):
            self.assertEqual(tf, 30)
            return large_df.copy()

        with patch.object(cascade_steps, "get_cached", side_effect=mock_get_cached):
            with patch.object(
                cascade_steps,
                "calc_smi",
                return_value=(smi, smi, smi),
            ):
                result = bot._has_higher_tf_saturation(
                    candidate,
                    "sell",
                    mock_get_resampled,
                )

        self.assertTrue(result)

    def test_skips_smaller_tf_on_second_closed_candle_after_higher_exit(self):
        """ثاني شمعة بعد خروج الأكبر من التشبع ما زالت تلغي الأصغر."""
        candidate = self._make_candidate(base_frame=27)
        large_df = _make_ohlcv(n=200, smi_value=0.0)
        smi = pd.Series([0.0] * 197 + [-50.0, -10.0, -5.0])

        def mock_get_cached(sym, api):
            return large_df.copy()

        def mock_get_resampled(raw, sym, api, tf):
            return large_df.copy()

        with patch.object(cascade_steps, "get_cached", side_effect=mock_get_cached):
            with patch.object(
                cascade_steps,
                "calc_smi",
                return_value=(smi, smi, smi),
            ):
                result = bot._has_higher_tf_saturation(
                    candidate,
                    "buy",
                    mock_get_resampled,
                )

        self.assertTrue(result)

    def test_allows_smaller_tf_after_third_non_saturated_higher_candle(self):
        """من ثالث شمعة بدون تشبع على الأكبر، الأصغر يُسمح له."""
        candidate = self._make_candidate(base_frame=27)
        large_df = _make_ohlcv(n=200, smi_value=0.0)
        smi = pd.Series([0.0] * 196 + [-50.0, -10.0, -5.0, -2.0])

        def mock_get_cached(sym, api):
            return large_df.copy()

        def mock_get_resampled(raw, sym, api, tf):
            return large_df.copy()

        with patch.object(cascade_steps, "get_cached", side_effect=mock_get_cached):
            with patch.object(
                cascade_steps,
                "calc_smi",
                return_value=(smi, smi, smi),
            ):
                result = bot._has_higher_tf_saturation(
                    candidate,
                    "buy",
                    mock_get_resampled,
                )

        self.assertFalse(result)

    def test_240m_checks_five_hour_ceiling_on_60m_source(self):
        """مرشح 240m يفحص تشبع 5 ساعات من مصدر 60m."""
        candidate = self._make_candidate(base_frame=240)
        calls = []
        resampled_tfs = []

        def mock_get_cached(sym, api):
            calls.append(api)
            return pd.DataFrame({"ts": [1]})

        def mock_get_resampled(raw_df, sym, api, tf):
            resampled_tfs.append(tf)
            return pd.DataFrame()

        with patch.object(cascade_steps, "get_cached", side_effect=mock_get_cached):
            bot._has_higher_tf_saturation(candidate, "buy", mock_get_resampled)

        self.assertEqual(calls, ["60m"])
        self.assertEqual(resampled_tfs, [cascade_steps.CANCEL_ONLY_HIGHER_TF])

    def test_returns_true_when_five_hour_ceiling_oversold(self):
        """تشبع 5 ساعات يلغي إشارة 240m."""
        candidate = self._make_candidate(base_frame=240)
        large_df = _make_ohlcv(n=500, smi_value=-50.0)
        smi = pd.Series([-50.0] * len(large_df))

        def mock_get_cached(sym, api):
            return large_df.copy()

        def mock_get_resampled(raw, sym, api, tf):
            if tf == cascade_steps.CANCEL_ONLY_HIGHER_TF:
                return large_df.copy()
            return pd.DataFrame()

        with patch.object(cascade_steps, "get_cached", side_effect=mock_get_cached):
            with patch.object(
                cascade_steps,
                "calc_smi",
                return_value=(smi, smi, smi),
            ):
                result = bot._has_higher_tf_saturation(
                    candidate,
                    "buy",
                    mock_get_resampled,
                )

        self.assertTrue(result)


class TestResampleOrigin(unittest.TestCase):
    """Closed candles must match TradingView's UTC session grid."""

    def _day_1m(self, day):
        times = pd.date_range(start=day, periods=24 * 60, freq="1min")
        return pd.DataFrame(
            {
                "ts": times,
                "open": 1.0,
                "high": 1.001,
                "low": 0.999,
                "close": 1.0,
                "vol": 1.0,
            }
        )

    def test_90min_candle_boundary(self):
        """شموع 90m يجب أن تبدأ عند 00:00, 01:30, 03:00, 04:30... UTC."""
        df = self._day_1m(datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)).iloc[:240]
        with patch("fahadal92.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 2, tzinfo=timezone.utc)
            mock_dt.side_effect = datetime
            result = bot.resample_ohlcv(df.copy(), 90)

        if not result.empty:
            expected_boundaries = [0, 90, 180]
            for ts in result["ts"]:
                minutes_from_midnight = ts.hour * 60 + ts.minute
                self.assertIn(
                    minutes_from_midnight,
                    expected_boundaries,
                    f"حد الشمعة {ts} غير متوافق مع epoch UTC",
                )

    def test_150min_matches_tradingview_utc_midnight_grid(self):
        """150m: 00:00, 02:30, 05:00, 07:30, 10:00, 12:30 — ليس 12:00 epoch."""
        df = self._day_1m(datetime(2026, 8, 12, 0, 0, tzinfo=timezone.utc))
        result = ind.resample_ohlcv_closed(df, 150)
        opens = [
            ts.hour * 60 + ts.minute
            for ts in result["ts"]
            if ts.date() == datetime(2026, 8, 12).date()
        ]
        self.assertEqual(
            opens,
            list(range(0, 24 * 60, 150)),
        )
        self.assertIn(10 * 60, opens)
        self.assertIn(12 * 60 + 30, opens)
        self.assertNotIn(12 * 60, opens)

    def test_27min_on_epoch_offset_day_starts_at_utc_midnight(self):
        """13 Aug 2026: epoch 27m ينزاح 18 دقيقة؛ الشارت يبدأ 00:00."""
        self.assertEqual(
            int(
                (
                    datetime(2026, 8, 13, tzinfo=timezone.utc)
                    - datetime(1970, 1, 1, tzinfo=timezone.utc)
                ).total_seconds()
                // 60
            )
            % 27,
            18,
        )
        df = self._day_1m(datetime(2026, 8, 13, 0, 0, tzinfo=timezone.utc))
        result = ind.resample_ohlcv_closed(df, 27)
        opens = [
            ts.hour * 60 + ts.minute
            for ts in result["ts"]
            if ts.date() == datetime(2026, 8, 13).date()
        ]
        self.assertEqual(opens[0], 0)
        self.assertIn(27, opens)
        self.assertNotIn(18, opens)
        self.assertEqual(opens, list(range(0, 24 * 60, 27)))

    def test_every_cascade_frame_from_its_source_matches_utc_session(self):
        """كل شموع الأساس والتأكيد والدخول، من مصدر 1m/30m/60m الفعلي."""
        day = datetime(2026, 8, 13, 0, 0, tzinfo=timezone.utc)
        seen = []
        for minutes, source_api, role in cascade_steps.iter_cascade_frames():
            source_minutes = int(str(source_api).replace("m", ""))
            periods = 24 * 60 // source_minutes
            times = pd.date_range(
                start=day, periods=periods, freq=f"{source_minutes}min"
            )
            raw = pd.DataFrame(
                {
                    "ts": times,
                    "open": 1.0,
                    "high": 1.001,
                    "low": 0.999,
                    "close": 1.0,
                    "vol": 1.0,
                }
            )
            result = ind.resample_ohlcv_closed(raw, minutes)
            opens = [
                ts.hour * 60 + ts.minute
                for ts in result["ts"]
                if ts.normalize() == pd.Timestamp(day)
            ]
            expected = list(range(0, 24 * 60, int(minutes)))
            with self.subTest(role=role, minutes=minutes, source=source_api):
                self.assertEqual(
                    opens,
                    expected,
                    f"{role} {minutes}m from {source_api} drifted from UTC session",
                )
            seen.append((role, minutes, source_api))
        self.assertEqual(len(seen), len(cascade_steps.TRIPLING_PAIRS) * 3)

    def test_remainder_close_for_every_non_tiling_cascade_frame(self):
        midnight = pd.Timestamp("2026-08-14 00:00:00+00:00")
        frames = {minutes for minutes, _source, _role in cascade_steps.iter_cascade_frames()}
        for minutes in sorted(frames):
            last_open_min = list(range(0, 24 * 60, minutes))[-1]
            last_open = datetime(2026, 8, 13, tzinfo=timezone.utc) + pd.Timedelta(
                minutes=last_open_min
            )
            end = ind.candle_period_end(last_open, minutes)
            with self.subTest(minutes=minutes, last_open=last_open_min):
                if last_open_min + minutes > 24 * 60:
                    self.assertEqual(end, midnight)
                else:
                    self.assertEqual(
                        end,
                        pd.Timestamp(last_open) + pd.Timedelta(minutes=minutes),
                    )

    def test_remainder_bar_closes_at_utc_midnight(self):
        stub_open = datetime(2026, 8, 13, 23, 51, tzinfo=timezone.utc)
        end = ind.candle_period_end(stub_open, 27)
        self.assertEqual(end, pd.Timestamp("2026-08-14 00:00:00+00:00"))
        full_open = datetime(2026, 8, 13, 23, 24, tzinfo=timezone.utc)
        self.assertEqual(
            ind.candle_period_end(full_open, 27),
            pd.Timestamp("2026-08-13 23:51:00+00:00"),
        )
        tiled = datetime(2026, 8, 13, 23, 51, tzinfo=timezone.utc)
        self.assertEqual(
            ind.candle_period_end(tiled, 9),
            pd.Timestamp("2026-08-14 00:00:00+00:00"),
        )

    def test_resample_ohlcv_keeps_remainder_after_midnight(self):
        """After 00:00 UTC the 23:51 27m stub is closed and must stay."""
        df = self._day_1m(datetime(2026, 8, 13, 0, 0, tzinfo=timezone.utc))
        with patch.object(ind, "datetime") as mock_dt:
            mock_dt.now.return_value = datetime(
                2026, 8, 14, 0, 10, tzinfo=timezone.utc
            )
            result = ind.resample_ohlcv(df.copy(), 27)
        stub = result[result["ts"] == pd.Timestamp("2026-08-13 23:51:00+00:00")]
        self.assertEqual(len(stub), 1)


class TestQuickCheckWatcherInterval(unittest.TestCase):
    """يتحقق أن quick_check_watcher يفحص بسرعة كافية لحذف المرشحات الصغيرة فورًا."""

    def test_quick_check_interval_is_three_seconds(self):
        self.assertEqual(bot.QUICK_CHECK_INTERVAL_SECONDS, 3)

    def test_quick_check_watcher_uses_configured_interval(self):
        with patch.object(
            pipeline.time,
            "sleep",
            side_effect=SystemExit,
        ) as mock_sleep:
            with self.assertRaises(SystemExit):
                bot.quick_check_watcher()
        mock_sleep.assert_called_once_with(bot.QUICK_CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
