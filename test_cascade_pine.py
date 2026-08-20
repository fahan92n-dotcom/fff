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

    def test_per_frame_win_loss_dates_next_to_steps(self):
        self.assertIn("var array<string> winDates", self.src)
        self.assertIn("var array<string> lossDates", self.src)
        self.assertIn("f_append_date", self.src)
        self.assertIn('str.format_time(et, "dd/MM HH:mm")', self.src)
        self.assertIn('side == 1 ? "شراء" : "بيع"', self.src)
        self.assertIn("input.int(73,", self.src)
        self.assertNotIn("input.int(90,", self.src)
        self.assertIn("table.new(position.top_left, 5, 16", self.src)
        self.assertIn('table.cell(logTab, 3, 0, "ناجحة"', self.src)
        self.assertIn('table.cell(logTab, 4, 0, "فاشلة"', self.src)
        self.assertIn("array.get(winDates, pairIdx)", self.src)
        self.assertIn("array.get(lossDates, pairIdx)", self.src)


if __name__ == "__main__":
    unittest.main()
