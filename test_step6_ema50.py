"""
اختبارات EMA50 للخطوة 6: إغلاق تحت/فوق الخط أثناء تشبع SMI فقط.
"""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import indicators as ind


def _make_df(closes, start="2024-01-01", freq="60min"):
    n = len(closes)
    ts = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    close = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "ts": ts,
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "vol": np.ones(n),
    })


def _fake_smi(values):
    series = pd.Series(values, dtype=float)

    def _calc(high, low, close, **_kwargs):
        aligned = series.reindex(close.index).fillna(0.0)
        return aligned, aligned, aligned

    return _calc


class TestEma50ClosedBelowSince(unittest.TestCase):
    def test_true_when_earlier_saturated_candle_closed_below(self):
        closes = [100.0] * 100 + [90.0, 90.0, 90.0] + [110.0, 110.0]
        df = _make_df(closes)
        since_ts = df["ts"].iloc[100]
        smi_vals = [0.0] * 100 + [-50.0, -50.0, -50.0] + [0.0, 0.0]
        with patch.object(ind, "calc_smi", side_effect=_fake_smi(smi_vals)):
            self.assertTrue(ind.check_ema50_closed_below_since(df, since_ts))
            self.assertFalse(ind.check_ema50_below(df))

    def test_false_when_close_below_but_not_during_saturation(self):
        closes = [100.0] * 100 + [90.0, 90.0, 90.0] + [110.0, 110.0]
        df = _make_df(closes)
        since_ts = df["ts"].iloc[100]
        # هبوط تحت EMA لكن بدون تشبع SMI
        smi_vals = [0.0] * len(df)
        with patch.object(ind, "calc_smi", side_effect=_fake_smi(smi_vals)):
            self.assertFalse(ind.check_ema50_closed_below_since(df, since_ts))

    def test_false_when_no_close_below_since(self):
        closes = [50.0] * 100 + list(np.linspace(200, 250, 25))
        df = _make_df(closes)
        since_ts = df["ts"].iloc[100]
        smi_vals = [-50.0] * len(df)
        with patch.object(ind, "calc_smi", side_effect=_fake_smi(smi_vals)):
            self.assertFalse(ind.check_ema50_closed_below_since(df, since_ts))

    def test_false_without_since_ts(self):
        df = _make_df([100.0] * 120)
        self.assertFalse(ind.check_ema50_closed_below_since(df, None))

    def test_wick_above_ema_does_not_count_for_short(self):
        """فتيل فوق EMA50 بدون إغلاق فوقه لا يمرر شرط البيع."""
        closes = [100.0] * 100 + [99.0, 99.0]
        df = _make_df(closes)
        df.loc[df.index[-1], "high"] = 200.0  # فتيل طويل فوق EMA
        since_ts = df["ts"].iloc[100]
        smi_vals = [0.0] * 100 + [50.0, 50.0]
        with patch.object(ind, "calc_smi", side_effect=_fake_smi(smi_vals)):
            self.assertFalse(ind.check_ema50_closed_above_since(df, since_ts))


class TestEma50ClosedAboveSince(unittest.TestCase):
    def test_true_when_earlier_saturated_candle_closed_above(self):
        closes = [100.0] * 100 + [120.0, 120.0, 120.0] + [80.0, 80.0]
        df = _make_df(closes)
        since_ts = df["ts"].iloc[100]
        smi_vals = [0.0] * 100 + [50.0, 50.0, 50.0] + [0.0, 0.0]
        with patch.object(ind, "calc_smi", side_effect=_fake_smi(smi_vals)):
            self.assertTrue(ind.check_ema50_closed_above_since(df, since_ts))
            self.assertFalse(ind.check_ema50_above(df))

    def test_false_when_close_above_after_saturation_ended(self):
        closes = [100.0] * 100 + [90.0, 90.0] + [120.0, 120.0]
        df = _make_df(closes)
        since_ts = df["ts"].iloc[100]
        # التشبع على الشموع تحت EMA فقط؛ الإغلاق فوق EMA بعد انتهاء التشبع
        smi_vals = [0.0] * 100 + [50.0, 50.0] + [0.0, 0.0]
        with patch.object(ind, "calc_smi", side_effect=_fake_smi(smi_vals)):
            self.assertFalse(ind.check_ema50_closed_above_since(df, since_ts))


class TestAbandonWhenBaseSaturationEnds(unittest.TestCase):
    def test_waiting_stage6_abandoned_when_smi_leaves_overbought(self):
        import cascade_pipeline as pipeline
        import state_manager as state

        candidate = {
            "sym": "ADAUSDT",
            "base_api": "1m",
            "triple_api": "1m",
            "base_frame": 15,
            "confirm_frame": 45,
            "triple_frame": 5,
            "df_base": _make_df([100.0] * 120),
            "df_confirm": _make_df([100.0] * 120),
            "df_triple": _make_df([100.0] * 120),
        }
        with state.last_complete_short_lock:
            state.last_complete_short_survivors.clear()
            state.last_complete_short_survivors[6] = [candidate]
        state.mark_stage_ready("sell", 1, [candidate])

        # SMI الحالي خارج التشبع الشرائي
        smi_vals = [10.0] * 120
        with patch.object(
            pipeline,
            "_refresh_waiting_candidate",
            return_value=candidate,
        ), patch.object(
            pipeline,
            "calc_smi",
            side_effect=_fake_smi(smi_vals),
        ), patch.object(
            pipeline,
            "_has_higher_tf_saturation",
            return_value=False,
        ):
            kept = pipeline._waiting_transition_candidates(
                "sell",
                6,
                get_resampled=lambda *a, **k: candidate["df_base"],
            )

        self.assertEqual(kept, [])
        self.assertEqual(state.get_stage_candidates("sell", 6), [])
        self.assertIsNone(
            state.get_step1_ready_since("ADAUSDT", 15, 45, 5, "sell")
        )


if __name__ == "__main__":
    unittest.main()
