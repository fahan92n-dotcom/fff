"""Guardrails for the TradingView Cascade 8 results table."""

import unittest
from pathlib import Path


PINE = Path(__file__).resolve().parent / "pine" / "cascade_8steps.pine"


class TestCascadePineResultsTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = PINE.read_text(encoding="utf-8")

    def test_pine_file_exists(self):
        self.assertTrue(PINE.is_file(), msg="pine/cascade_8steps.pine is missing")

    def test_score_table_is_not_gated_on_last_bar_only(self):
        self.assertNotIn("if barstate.islast", self.src)
        self.assertIn("var table scoreBox", self.src)
        self.assertIn("Total PnL", self.src)
        self.assertIn("Max Drawdown", self.src)
        self.assertIn("table.set_position(scoreBox", self.src)

    def test_log_is_a_fixed_table_not_a_last_bar_label(self):
        self.assertNotIn("var label logLbl", self.src)
        self.assertIn("var table logTab", self.src)
        self.assertIn('position.top_left', self.src)

    def test_loaded_bars_warning_and_position_choices(self):
        self.assertIn("شارت محمّل", self.src)
        self.assertIn("منتصف اليمين", self.src)
        self.assertIn("position.middle_right", self.src)
        self.assertIn("f_window_pnl", self.src)


if __name__ == "__main__":
    unittest.main()
