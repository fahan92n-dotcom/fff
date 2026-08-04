"""
اختبارات EMA50 للخطوة 6: أي إغلاق تحت/فوق الخط منذ تشبع الرئيسي.
"""
import unittest
from datetime import datetime, timezone

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


class TestEma50ClosedBelowSince(unittest.TestCase):
    def test_true_when_earlier_candle_closed_below_even_if_last_is_above(self):
        # سلسلة طويلة لـ EMA50، ثم هبوط تحت المتوسط ثم صعود فوقه في الآخر
        closes = [100.0] * 60 + [90.0, 90.0, 90.0] + [110.0, 110.0]
        df = _make_df(closes)
        since_ts = df["ts"].iloc[60]  # من بداية الهبوط
        # آخر إغلاق فوق EMA غالبًا، لكن الشموع عند 90 تحت المتوسط
        self.assertTrue(ind.check_ema50_closed_below_since(df, since_ts))
        # النسخة القديمة (آخر شمعة فقط) قد تفشل
        self.assertFalse(ind.check_ema50_below(df))

    def test_false_when_no_close_below_since(self):
        # صعود حاد بعد since يضمن الإغلاق فوق EMA طوال النافذة
        closes = [50.0] * 55 + list(np.linspace(200, 250, 25))
        df = _make_df(closes)
        since_ts = df["ts"].iloc[55]
        ema = df["close"].ewm(span=50, adjust=False).mean()
        mask = df["ts"] >= since_ts
        self.assertFalse((df.loc[mask, "close"] < ema.loc[mask]).any())
        self.assertFalse(ind.check_ema50_closed_below_since(df, since_ts))

    def test_false_without_since_ts(self):
        df = _make_df([100.0] * 60)
        self.assertFalse(ind.check_ema50_closed_below_since(df, None))


class TestEma50ClosedAboveSince(unittest.TestCase):
    def test_true_when_earlier_candle_closed_above_even_if_last_is_below(self):
        closes = [100.0] * 60 + [120.0, 120.0, 120.0] + [80.0, 80.0]
        df = _make_df(closes)
        since_ts = df["ts"].iloc[60]
        self.assertTrue(ind.check_ema50_closed_above_since(df, since_ts))
        self.assertFalse(ind.check_ema50_above(df))


if __name__ == "__main__":
    unittest.main()
