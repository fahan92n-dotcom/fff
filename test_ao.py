"""Awesome Oscillator: SMA(mid, 5) − SMA(mid, 34) and Cascade setup sides."""

import unittest

import numpy as np
import pandas as pd

import cascade_steps as steps
import indicators as ind


def _df_from_mid(mids, start="2024-01-01", freq="60min"):
    mid = np.asarray(mids, dtype=float)
    ts = pd.date_range(start=start, periods=len(mid), freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "ts": ts,
            "open": mid,
            "high": mid,
            "low": mid,
            "close": mid,
            "vol": np.ones(len(mid)),
        }
    )


class TestCalcAo(unittest.TestCase):
    def test_matches_sma5_minus_sma34_on_midpoint(self):
        high = pd.Series([10.0 + i for i in range(40)])
        low = high - 2.0
        mid = (high + low) / 2.0
        expected = mid.rolling(5).mean() - mid.rolling(34).mean()
        got = ind.calc_ao(high, low)
        pd.testing.assert_series_equal(got, expected, check_names=False)


class TestAoSetup(unittest.TestCase):
    def test_buy_needs_base_below_zero_and_confirm_above(self):
        base = _df_from_mid([100.0] * 40 + list(np.linspace(100, 40, 20)))
        confirm = _df_from_mid([40.0] * 40 + list(np.linspace(40, 100, 20)))
        self.assertTrue(ind.check_ao_setup(base, confirm, direction="long"))
        self.assertFalse(ind.check_ao_setup(base, confirm, direction="short"))

    def test_sell_needs_base_above_zero_and_confirm_below(self):
        base = _df_from_mid([40.0] * 40 + list(np.linspace(40, 100, 20)))
        confirm = _df_from_mid([100.0] * 40 + list(np.linspace(100, 40, 20)))
        self.assertTrue(ind.check_ao_setup(base, confirm, direction="short"))
        self.assertFalse(ind.check_ao_setup(base, confirm, direction="long"))

    def test_buy_fails_when_confirm_also_below_zero(self):
        falling = _df_from_mid([100.0] * 40 + list(np.linspace(100, 40, 20)))
        self.assertFalse(ind.check_ao_setup(falling, falling, direction="long"))

    def test_step6_uses_ao_gate(self):
        falling = _df_from_mid([100.0] * 40 + list(np.linspace(100, 40, 20)))
        candidate = {
            "sym": "ETHUSDT",
            "base_frame": 60,
            "confirm_frame": 180,
            "triple_frame": 20,
            "df_base": falling,
            "df_confirm": falling,
            "ready_since": falling["ts"].iloc[10],
        }
        ok, reason = steps.ao_setup(candidate)
        self.assertFalse(ok)
        self.assertEqual(reason, "ao_setup")


if __name__ == "__main__":
    unittest.main()
