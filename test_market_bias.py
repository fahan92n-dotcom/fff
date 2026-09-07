"""Market Bias (CEREBR) as used in pine/six_indicators_strategy.pine.

Screenshot defaults: period 50, smoothing 10, oscillator 7.
Main TF. Long = dark green (osc_bias > 0 and >= osc_smooth).
Short = dark red (osc_bias < 0 and <= osc_smooth).
"""
import unittest

import numpy as np
import pandas as pd

from research.six_indicators_backtest import indicators_for, market_bias_osc, run_engine


class TestMarketBiasOsc(unittest.TestCase):
    def test_rising_series_is_dark_green(self):
        n = 120
        close = 100.0 * (1.01 ** np.arange(n))
        high = close * 1.002
        low = close * 0.998
        op = close * 0.999
        bias, sm = market_bias_osc(
            pd.Series(op), pd.Series(high), pd.Series(low), pd.Series(close)
        )
        self.assertTrue(np.isfinite(bias[-1]))
        self.assertTrue(np.isfinite(sm[-1]))
        self.assertGreater(bias[-1], 0)
        self.assertGreaterEqual(bias[-1], sm[-1])

    def test_falling_series_is_dark_red(self):
        # تسارع الهبوط يجعل osc_bias أكثر سلبية من متوسطه = أحمر غامق
        n = 200
        t = np.arange(n)
        close = 100.0 - 0.001 * (t ** 2)
        high = close * 1.002
        low = close * 0.998
        op = close * 1.001
        bias, sm = market_bias_osc(
            pd.Series(op), pd.Series(high), pd.Series(low), pd.Series(close)
        )
        self.assertTrue(np.isfinite(bias[-1]))
        self.assertTrue(np.isfinite(sm[-1]))
        self.assertLess(bias[-1], 0)
        self.assertLessEqual(bias[-1], sm[-1])

    def test_light_green_is_not_buy(self):
        """Weak/light lime: bias > 0 but below its EMA — not dark green."""
        bias = np.array([5.0, 4.0, 3.0, 2.0])
        sm = np.array([3.0, 3.2, 3.1, 2.8])
        mbU = (bias > 0) & (bias >= sm)
        mbD = (bias < 0) & (bias <= sm)
        self.assertFalse(mbU[-1])
        self.assertFalse(mbD[-1])

    def test_light_red_is_not_sell(self):
        """Weak/light red: bias < 0 but above its EMA — not dark red."""
        bias = np.array([-5.0, -4.0, -3.0, -2.0])
        sm = np.array([-3.0, -3.2, -3.1, -2.8])
        mbU = (bias > 0) & (bias >= sm)
        mbD = (bias < 0) & (bias <= sm)
        self.assertFalse(mbU[-1])
        self.assertFalse(mbD[-1])

    def test_indicators_for_exposes_mb_flags(self):
        n = 200
        close = 50_000 * (1.002 ** np.arange(n))
        tfd = pd.DataFrame({
            "open": close * 0.999,
            "high": close * 1.001,
            "low": close * 0.998,
            "close": close,
        })
        z = indicators_for(tfd)
        self.assertTrue(z["mbU"][-1])
        self.assertFalse(z["mbD"][-1])
        self.assertFalse(np.any(z["mbU"] & z["mbD"]))


def _all(n, val=True):
    return np.full(n, val, dtype=bool)


def _ones(n, *idxs):
    a = np.zeros(n, dtype=bool)
    for i in idxs:
        a[i] = True
    return a


class TestEngineMarketBiasStep(unittest.TestCase):
    def test_does_not_fire_until_dark_bias(self):
        n = 6
        always = _all(n)
        never = _all(n, False)
        mb = _ones(n, 5)
        fires, reach, _ = run_engine(
            n, never, always,
            always, always, always, always, always,
            mb, always, always,
            always, always, always, always, always,
        )
        self.assertEqual(fires, [5])
        self.assertGreater(reach[6], 0)

    def test_stays_when_bias_false(self):
        n = 6
        always = _all(n)
        never = _all(n, False)
        fires, reach, _ = run_engine(
            n, never, always,
            always, always, always, always, always,
            never, always, always,
            always, always, always, always, always,
        )
        self.assertGreater(reach[5], 0)
        self.assertEqual(reach[6], 0)
        self.assertEqual(fires, [])


if __name__ == "__main__":
    unittest.main()
