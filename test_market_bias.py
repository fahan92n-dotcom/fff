"""Market Bias (CEREBR) as used in pine/six_indicators_strategy.pine.

Screenshot defaults: period 50, smoothing 10. Long when osc_bias>0.
"""
import unittest

import numpy as np
import pandas as pd

from research.six_indicators_backtest import indicators_for, market_bias_osc, run_engine


class TestMarketBiasOsc(unittest.TestCase):
    def test_rising_series_is_bullish(self):
        n = 120
        close = 100.0 * (1.01 ** np.arange(n))
        high = close * 1.002
        low = close * 0.998
        op = close * 0.999
        bias = market_bias_osc(
            pd.Series(op), pd.Series(high), pd.Series(low), pd.Series(close)
        )
        self.assertTrue(np.isfinite(bias[-1]))
        self.assertGreater(bias[-1], 0)

    def test_falling_series_is_bearish(self):
        n = 120
        close = 100.0 * (0.99 ** np.arange(n))
        high = close * 1.002
        low = close * 0.998
        op = close * 1.001
        bias = market_bias_osc(
            pd.Series(op), pd.Series(high), pd.Series(low), pd.Series(close)
        )
        self.assertTrue(np.isfinite(bias[-1]))
        self.assertLess(bias[-1], 0)

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
    def test_does_not_fire_until_confirm_bias(self):
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
