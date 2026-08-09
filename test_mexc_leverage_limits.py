"""Tests for MEXC risk-tier resolution at a requested leverage."""

import unittest

import mexc_leverage_limits as mll

# شرائح BTC_USDT و DOT_USDT كما ترجعها MEXC في contract/detail.
BTC = {
    "symbol": "BTC_USDT",
    "contractSize": 0.0001,
    "initialMarginRate": 0.002,
    "riskLimitMode": "CUSTOM",
    "riskLimitCustom": [
        {"level": 1, "maxVol": 50000, "maxLeverage": 500},
        {"level": 2, "maxVol": 310000, "maxLeverage": 200},
        {"level": 3, "maxVol": 480000, "maxLeverage": 100},
        {"level": 4, "maxVol": 2800000, "maxLeverage": 50},
        {"level": 5, "maxVol": 17500000, "maxLeverage": 20},
        {"level": 6, "maxVol": 19000000, "maxLeverage": 10},
    ],
}

DOT = {
    "symbol": "DOT_USDT",
    "contractSize": 0.1,
    "initialMarginRate": 0.00333333,
    "riskLimitMode": "CUSTOM",
    "riskLimitCustom": [
        {"level": 1, "maxVol": 500000, "maxLeverage": 300},
        {"level": 2, "maxVol": 13000000, "maxLeverage": 200},
    ],
}

INCREASING = {
    "symbol": "FAKE_USDT",
    "contractSize": 1,
    "initialMarginRate": 0.005,
    "riskIncrImr": 0.005,
    "riskBaseVol": 1000,
    "riskIncrVol": 1000,
    "riskLevelLimit": 4,
    "riskLimitMode": "INCREASE",
}


class TestCustomTierTable(unittest.TestCase):
    def test_picks_widest_tier_allowing_the_leverage(self):
        self.assertEqual(mll.max_volume_at_leverage(BTC, 100), 480000)

    def test_higher_leverage_narrows_the_cap(self):
        self.assertEqual(mll.max_volume_at_leverage(BTC, 500), 50000)
        self.assertEqual(mll.max_volume_at_leverage(BTC, 200), 310000)

    def test_lower_leverage_widens_the_cap(self):
        self.assertEqual(mll.max_volume_at_leverage(BTC, 10), 19000000)

    def test_leverage_above_every_tier_is_unavailable(self):
        self.assertIsNone(mll.max_volume_at_leverage(BTC, 501))

    def test_tier_ceiling_may_sit_below_the_top_leverage_tier(self):
        # DOT allows 300x, but 100x unlocks the far wider 200x tier.
        self.assertEqual(mll.max_volume_at_leverage(DOT, 300), 500000)
        self.assertEqual(mll.max_volume_at_leverage(DOT, 100), 13000000)


class TestIncreasingTiers(unittest.TestCase):
    def test_tier_one_only_at_top_leverage(self):
        self.assertEqual(mll.max_volume_at_leverage(INCREASING, 200), 1000)

    def test_lower_leverage_accumulates_tiers(self):
        self.assertEqual(mll.max_volume_at_leverage(INCREASING, 100), 2000)
        self.assertEqual(mll.max_volume_at_leverage(INCREASING, 50), 4000)

    def test_leverage_beyond_tier_one_is_unavailable(self):
        self.assertIsNone(mll.max_volume_at_leverage(INCREASING, 201))


class TestNotionalMatchesExchangeDisplay(unittest.TestCase):
    """الأرقام المرجعية مأخوذة من حاسبة MEXC نفسها."""

    def test_btc_at_100x(self):
        volume = mll.max_volume_at_leverage(BTC, 100)
        self.assertAlmostEqual(volume * BTC["contractSize"] * 65174.0, 3128352.0, places=2)

    def test_dot_at_100x(self):
        volume = mll.max_volume_at_leverage(DOT, 100)
        self.assertAlmostEqual(volume * DOT["contractSize"] * 0.8060, 1047800.0, places=2)


if __name__ == "__main__":
    unittest.main()
