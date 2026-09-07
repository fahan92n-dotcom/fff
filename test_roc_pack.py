"""ROC length-48 pack used by research/roc_nine_month_backtest.py."""
import unittest

import numpy as np
import pandas as pd

from research.roc_nine_month_backtest import roc_pack, attach_roc
from research.six_indicators_backtest import indicators_for


class TestRocPack(unittest.TestCase):
    def test_rising_series_maroc_above_zero(self):
        n = 200
        close = 100.0 * (1.01 ** np.arange(n))
        tfd = pd.DataFrame({
            "open": close * 0.999,
            "high": close * 1.001,
            "low": close * 0.998,
            "close": close,
        })
        _roc, maroc, ok, maroc_up, above_ma = roc_pack(tfd)
        self.assertTrue(ok[-1])
        self.assertGreater(maroc[-1], 0)
        self.assertTrue(maroc_up[-1])
        self.assertTrue(above_ma[-1])

    def test_falling_series_maroc_not_above_zero(self):
        n = 200
        close = 100.0 * (0.99 ** np.arange(n))
        tfd = pd.DataFrame({
            "open": close * 1.001,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
        })
        _roc, maroc, ok, maroc_up, above_ma = roc_pack(tfd)
        self.assertTrue(ok[-1])
        self.assertLess(maroc[-1], 0)
        self.assertFalse(maroc_up[-1])
        self.assertFalse(above_ma[-1])

    def test_attach_roc_mutually_exclusive(self):
        n = 180
        close = 50_000 * (1.002 ** np.arange(n))
        tfd = pd.DataFrame({
            "open": close * 0.999,
            "high": close * 1.001,
            "low": close * 0.998,
            "close": close,
        })
        z = attach_roc(indicators_for(tfd), tfd)
        self.assertFalse(np.any(z["rocU"] & z["rocD"]))
        self.assertTrue(z["rocU"][-1])
        self.assertFalse(z["rocD"][-1])


if __name__ == "__main__":
    unittest.main()
