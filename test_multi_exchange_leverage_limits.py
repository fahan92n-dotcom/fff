"""Offline tests for OKX / Bybit / Hyperliquid leverage resolvers."""

import math
import unittest

import okx_leverage_limits as okx
import bybit_leverage_limits as bybit
import hyperliquid_leverage_limits as hl
import kucoin_leverage_limits as kucoin


class TestOkxTiers(unittest.TestCase):
    def test_picks_widest_tier_still_allowing_leverage(self):
        tiers = [
            {"maxLever": "100", "maxSz": "1000"},
            {"maxLever": "50", "maxSz": "20000"},
        ]
        self.assertEqual(okx.max_contracts_at_leverage(tiers, 100), (1000.0, 100.0))
        self.assertEqual(okx.max_contracts_at_leverage(tiers, 50), (20000.0, 50.0))


class TestBybitTiers(unittest.TestCase):
    def test_picks_widest_risk_limit(self):
        tiers = [
            {"maxLeverage": "100", "riskLimitValue": "100000"},
            {"maxLeverage": "50", "riskLimitValue": "1000000"},
        ]
        self.assertEqual(bybit.max_notional_at_leverage(tiers, 100), (100000.0, 100.0))
        self.assertEqual(bybit.max_notional_at_leverage(tiers, 50), (1000000.0, 50.0))


class TestHyperliquidTiers(unittest.TestCase):
    def test_next_lower_tier_sets_the_cap(self):
        tiers = [
            {"lowerBound": "0.0", "maxLeverage": 40},
            {"lowerBound": "150000000.0", "maxLeverage": 20},
        ]
        self.assertEqual(hl.max_notional_at_leverage(tiers, 40), (150000000.0, 40.0))
        amount, tier = hl.max_notional_at_leverage(tiers, 20)
        self.assertTrue(math.isinf(amount))
        self.assertEqual(tier, 20.0)

    def test_leverage_above_platform_max_is_unavailable(self):
        tiers = [{"lowerBound": "0.0", "maxLeverage": 40}]
        self.assertEqual(hl.max_notional_at_leverage(tiers, 100), (None, None))


class TestKucoinTiers(unittest.TestCase):
    def test_100x_takes_the_wider_tier_not_the_125x_tier(self):
        tiers = [
            {"maxLeverage": 125, "maxRiskLimit": 250000},
            {"maxLeverage": 100, "maxRiskLimit": 600000},
            {"maxLeverage": 50, "maxRiskLimit": 10000000},
        ]
        self.assertEqual(kucoin.max_notional_at_leverage(tiers, 100), (600000.0, 100.0))
        self.assertEqual(kucoin.max_notional_at_leverage(tiers, 125), (250000.0, 125.0))


if __name__ == "__main__":
    unittest.main()
