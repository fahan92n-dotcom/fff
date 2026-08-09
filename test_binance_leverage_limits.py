"""Tests for Binance notional-bracket resolution at a requested leverage."""

import unittest

import binance_leverage_limits as bll

# (أقصى رافعة، سقف القيمة الاسمية) بترتيب الشرائح كما ترسلها Binance.
BTC_TIERS = [(125, 50_000), (100, 600_000), (50, 3_000_000), (20, 15_000_000)]
ALT_TIERS = [(25, 5_000), (10, 25_000), (5, 100_000)]

SIGNED_PAYLOAD = [{
    "symbol": "BTCUSDT",
    "brackets": [
        {"bracket": 1, "initialLeverage": 125, "notionalCap": 50_000},
        {"bracket": 2, "initialLeverage": 100, "notionalCap": 600_000},
    ],
}]

PUBLIC_PAYLOAD = {"data": [{
    "symbol": "BTCUSDT",
    "riskBrackets": [
        {"bracketSeq": 1, "maxOpenPosLeverage": 125, "bracketNotionalCap": 50_000},
        {"bracketSeq": 2, "maxOpenPosLeverage": 100, "bracketNotionalCap": 600_000},
    ],
}]}


class TestMaxNotional(unittest.TestCase):
    def test_takes_the_widest_bracket_allowing_the_leverage(self):
        self.assertEqual(bll.max_notional_at_leverage(BTC_TIERS, 100), 600_000)

    def test_top_leverage_is_confined_to_the_first_bracket(self):
        self.assertEqual(bll.max_notional_at_leverage(BTC_TIERS, 125), 50_000)

    def test_lower_leverage_unlocks_larger_positions(self):
        self.assertEqual(bll.max_notional_at_leverage(BTC_TIERS, 20), 15_000_000)

    def test_leverage_above_every_bracket_is_unavailable(self):
        self.assertIsNone(bll.max_notional_at_leverage(BTC_TIERS, 126))
        self.assertIsNone(bll.max_notional_at_leverage(ALT_TIERS, 100))


class TestPayloadNormalisation(unittest.TestCase):
    def test_signed_and_public_shapes_agree(self):
        self.assertEqual(
            bll._normalise_signed(SIGNED_PAYLOAD),
            bll._normalise_public(PUBLIC_PAYLOAD),
        )

    def test_symbol_without_brackets_is_skipped(self):
        self.assertEqual(bll._normalise_public({"data": [{"symbol": "X", "riskBrackets": []}]}), {})

    def test_alternate_field_names_are_accepted(self):
        payload = {"data": [{"symbol": "BTCUSDT", "brackets": [
            {"maxLeverage": 100, "maxNotionalValue": 600_000},
        ]}]}
        self.assertEqual(bll._normalise_public(payload), {"BTCUSDT": [(100, 600_000)]})

    def test_bare_list_payload_is_accepted(self):
        payload = [{"symbol": "BTCUSDT", "brackets": [
            {"initialLeverage": 125, "notionalCap": 50_000},
        ]}]
        self.assertEqual(bll._normalise_public(payload), {"BTCUSDT": [(125, 50_000)]})


class TestBuildRows(unittest.TestCase):
    def setUp(self):
        self.brackets = {"BTCUSDT": BTC_TIERS, "ALTUSDT": ALT_TIERS, "ETHBUSD": BTC_TIERS}

    def test_ranks_by_amount_and_drops_symbols_below_the_leverage(self):
        rows = bll.build_rows(100, brackets=self.brackets)
        self.assertEqual([row["symbol"] for row in rows], ["BTCUSDT"])
        self.assertEqual(rows[0]["max_amount_usdt"], 600_000)
        self.assertEqual(rows[0]["margin_needed_usdt"], 6_000)

    def test_quote_filter_can_be_disabled(self):
        rows = bll.build_rows(100, brackets=self.brackets, quote="")
        self.assertEqual(sorted(row["symbol"] for row in rows), ["BTCUSDT", "ETHBUSD"])

    def test_lower_leverage_admits_more_symbols(self):
        rows = bll.build_rows(5, brackets=self.brackets)
        self.assertEqual([row["symbol"] for row in rows], ["BTCUSDT", "ALTUSDT"])


if __name__ == "__main__":
    unittest.main()
