"""Tests for TradingView-compatible Donchian Trend Ribbon hue."""

import unittest

import numpy as np
import pandas as pd

import indicators as ind


def _mixed_green_dataframe():
    """
    Build data where dchannel(20) stays bullish but dchannel(11) turns bearish.

    TradingView still displays green because the 11-period disagreement only
    changes that band's opacity.  The bot must therefore return green.
    """
    n = 32
    close = np.full(n, 95.0)
    high = np.full(n, 100.0)
    low = np.full(n, 90.0)

    # This deep low remains inside the final 20-bar window but is outside the
    # final 11-bar window.
    low[15] = 80.0

    # Break above the prior high so every Donchian trend becomes bullish.
    close[20] = 110.0
    high[20] = 110.0
    low[20] = 109.0

    # Recent lows are 95; final close 94 flips dchannel(11) bearish, while
    # dchannel(20) remains bullish because its prior low is still 80.
    low[21:31] = 95.0
    close[21:31] = 97.0
    close[31] = 94.0
    high[31] = 96.0
    low[31] = 94.0

    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "ts": ts,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "vol": np.ones(n),
        }
    )


class TestTradingViewDonchianHue(unittest.TestCase):
    """The binary condition follows dchannel(20), not all opacity bands."""

    def setUp(self):
        with ind._ribbon_cache_lock:
            ind._ribbon_cache.clear()

    def tearDown(self):
        with ind._ribbon_cache_lock:
            ind._ribbon_cache.clear()

    def test_default_period_matches_tradingview_input(self):
        self.assertEqual(ind.DONCHIAN_DLEN, 20)

    def test_green_uses_maintrend_even_when_short_band_disagrees(self):
        df = _mixed_green_dataframe()

        maintrend = ind.calc_donchian_trend_pine(
            df["close"], df["high"], df["low"], 20
        )
        short_trend = ind.calc_donchian_trend_pine(
            df["close"], df["high"], df["low"], 11
        )

        self.assertEqual(maintrend, 1)
        self.assertEqual(short_trend, -1)
        self.assertTrue(ind.check_donchian_trend_ribbon(df, "green"))
        self.assertFalse(ind.check_donchian_trend_ribbon(df, "red"))

    def test_cache_keeps_same_maintrend_result(self):
        df = _mixed_green_dataframe()
        key = ("BTCUSDT", "1m", 20)

        self.assertTrue(
            ind.check_donchian_trend_ribbon(df, "green", cache_key=key)
        )
        self.assertEqual(ind._ribbon_cache[key], 1)


if __name__ == "__main__":
    unittest.main()
