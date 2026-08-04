"""
اختبارات وحدة لطبقة binance_data (الكاش وربط الحالة المشتركة).

تتحقق من:
1. أن cache_merge / get_cached يعملان على نفس كائن ohlcv_cache.
2. أن الحالة المشتركة المستوردة في fahadal92 هي نفس كائنات binance_data (لا نسخ مكررة).
3. أن cleanup_old_symbols_cache يحذف مفاتيح الرموز غير النشطة فقط.
"""
import unittest
from datetime import datetime, timezone

import pandas as pd

import binance_data as bd
import fahadal92 as bot


def _make_df(n=3, start_price=100.0):
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = pd.date_range(start=base, periods=n, freq="1min")
    close = [start_price + i for i in range(n)]
    return pd.DataFrame({
        "ts": times,
        "open": close,
        "high": [c + 0.5 for c in close],
        "low": [c - 0.5 for c in close],
        "close": close,
        "vol": [1000.0] * n,
    })


class TestSharedStateIdentity(unittest.TestCase):
    """الكاش والرموز يجب أن تكون نفس الكائنات بين الموديولين."""

    def test_ohlcv_cache_is_same_object(self):
        self.assertIs(bot.ohlcv_cache, bd.ohlcv_cache)
        self.assertIs(bot.ohlcv_cache_lock, bd.ohlcv_cache_lock)

    def test_symbols_cache_is_same_object(self):
        self.assertIs(bot.symbols_cache, bd.symbols_cache)
        self.assertIs(bot.symbols_cache_lock, bd.symbols_cache_lock)

    def test_prefetch_events_are_same_objects(self):
        self.assertIs(bot.fast_prefetch_done, bd.fast_prefetch_done)
        self.assertIs(bot.prefetch_done, bd.prefetch_done)
        self.assertIs(bot.cache_updated_event, bd.cache_updated_event)

    def test_telegram_sender_wired(self):
        self.assertIs(bd._telegram_sender, bot.send_telegram)


class TestCacheMergeAndGet(unittest.TestCase):
    """اختبار دمج وقراءة الكاش."""

    def setUp(self):
        with bd.ohlcv_cache_lock:
            bd.ohlcv_cache.clear()

    def tearDown(self):
        with bd.ohlcv_cache_lock:
            bd.ohlcv_cache.clear()

    def test_cache_merge_empty_noop(self):
        bd.cache_merge("BTCUSDT", "1m", pd.DataFrame())
        self.assertEqual(len(bd.ohlcv_cache), 0)

    def test_cache_merge_and_get_cached(self):
        df = _make_df(5)
        bd.cache_merge("BTCUSDT", "1m", df)
        got = bd.get_cached("BTCUSDT", "1m")
        self.assertEqual(len(got), 5)
        self.assertEqual(float(got["close"].iloc[-1]), 104.0)

    def test_get_cached_returns_copy(self):
        df = _make_df(3)
        bd.cache_merge("ETHUSDT", "30m", df)
        got = bd.get_cached("ETHUSDT", "30m")
        got.loc[got.index[0], "close"] = -1
        again = bd.get_cached("ETHUSDT", "30m")
        self.assertNotEqual(float(again["close"].iloc[0]), -1)

    def test_cache_merge_dedupes_by_ts(self):
        df1 = _make_df(3, start_price=10)
        df2 = _make_df(3, start_price=20)  # نفس timestamps، أسعار مختلفة
        bd.cache_merge("SOLUSDT", "1m", df1)
        bd.cache_merge("SOLUSDT", "1m", df2)
        got = bd.get_cached("SOLUSDT", "1m")
        self.assertEqual(len(got), 3)
        # keep="last" → قيم df2
        self.assertEqual(float(got["close"].iloc[0]), 20.0)

    def test_get_cached_missing_returns_empty(self):
        got = bd.get_cached("MISSINGUSDT", "1m")
        self.assertTrue(got.empty)


class TestCleanupOldSymbolsCache(unittest.TestCase):
    """تنظيف مفاتيح الكاش للرموز غير النشطة."""

    def setUp(self):
        with bd.ohlcv_cache_lock:
            bd.ohlcv_cache.clear()
        with bd.symbols_cache_lock:
            bd.symbols_cache.clear()

    def tearDown(self):
        with bd.ohlcv_cache_lock:
            bd.ohlcv_cache.clear()
        with bd.symbols_cache_lock:
            bd.symbols_cache.clear()

    def test_removes_stale_symbol_keys(self):
        bd.cache_merge("BTCUSDT", "1m", _make_df(2))
        bd.cache_merge("OLDUSDT", "1m", _make_df(2))
        with bd.symbols_cache_lock:
            bd.symbols_cache[:] = ["BTCUSDT"]
        bd.cleanup_old_symbols_cache()
        self.assertIn(("BTCUSDT", "1m"), bd.ohlcv_cache)
        self.assertNotIn(("OLDUSDT", "1m"), bd.ohlcv_cache)


if __name__ == "__main__":
    unittest.main()
