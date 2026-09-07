"""ROC (نالان) as used in pine/six_indicators_strategy.pine.

User formula:
    roc = (close - close[48]) / close[48] * 100
    maroc = ema(roc, 48)
    expr = maroc > 0   # orange long / blue short
"""
import unittest

import numpy as np
import pandas as pd

from indicators import ema_tv
from research.six_indicators_backtest import at, indicators_for, run_engine


def _roc_maroc(close, length=48):
    close = np.asarray(close, dtype=float)
    prev = np.roll(close, length)
    prev[:length] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        roc = np.where(prev != 0, (close - prev) / prev * 100.0, 0.0)
    maroc = ema_tv(pd.Series(roc), length).to_numpy()
    finite = np.isfinite(maroc)
    return roc, maroc, finite & (maroc > 0), finite & ~(maroc > 0)


class TestRocFormula(unittest.TestCase):
    def test_matches_user_script_on_known_bars(self):
        n = 120
        close = 100.0 * (1.01 ** np.arange(n))
        roc, maroc, up, dn = _roc_maroc(close, 48)
        # First 48 bars have no close[48]; maroc needs another 48 after ROC exists.
        self.assertTrue(np.all(np.isnan(roc[:48])))
        i = 48
        expected = (close[i] - close[i - 48]) / close[i - 48] * 100.0
        self.assertAlmostEqual(roc[i], expected, places=10)
        last = -1
        self.assertTrue(np.isfinite(maroc[last]))
        self.assertTrue(up[last])
        self.assertFalse(dn[last])

    def test_declining_series_is_blue_not_orange(self):
        close = 100.0 * (0.99 ** np.arange(120))
        _, _, up, dn = _roc_maroc(close, 48)
        self.assertFalse(up[-1])
        self.assertTrue(dn[-1])

    def test_missing_maroc_is_neither_side(self):
        close = np.linspace(100, 110, 30)
        _, _, up, dn = _roc_maroc(close, 48)
        self.assertFalse(np.any(up))
        self.assertFalse(np.any(dn))

    def test_indicators_for_exposes_confirm_roc_flags(self):
        n = 200
        close = 50_000 * (1.002 ** np.arange(n))
        tfd = pd.DataFrame({
            "close": close,
            "high": close * 1.001,
            "low": close * 0.999,
        })
        z = indicators_for(tfd)
        self.assertIn("rocU", z)
        self.assertIn("rocD", z)
        self.assertTrue(z["rocU"][-1])
        self.assertFalse(z["rocD"][-1])
        self.assertFalse(np.any(z["rocU"] & z["rocD"]))


def _ones(n, *idxs):
    a = np.zeros(n, dtype=bool)
    for i in idxs:
        a[i] = True
    return a


def _all(n, val=True):
    return np.full(n, val, dtype=bool)


class TestEngineRocStep(unittest.TestCase):
    def test_does_not_fire_until_confirm_roc(self):
        n = 6
        always = _all(n)
        never = _all(n, False)
        roc = _ones(n, 5)
        fires, reach, _ = run_engine(
            n, never, always,
            always, always, always, always, always,
            roc, always, always,
            always, always, always, always, always,
        )
        self.assertEqual(fires, [5])
        self.assertGreater(reach[6], 0)

    def test_stays_at_macd_confirm_when_roc_false(self):
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

    def test_at_does_not_treat_nan_as_true(self):
        cond = np.array([np.nan, 1.0, 0.0])
        idx = np.array([0, 1, 2])
        self.assertEqual(at(cond, idx).tolist(), [False, True, False])


if __name__ == "__main__":
    unittest.main()
