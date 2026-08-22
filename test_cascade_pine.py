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
        self.assertIn("f_levels_txt", self.src)
        self.assertIn("input.float(1.00,", self.src)
        self.assertIn("input.float(0.75,", self.src)
        self.assertIn("baseMin <= 30 ? shortWin : longWin", self.src)
        self.assertIn("baseMin <= 30 ? shortLoss : longLoss", self.src)
        self.assertNotIn("input.float(0.80,", self.src)
        self.assertNotIn("input.float(0.67,", self.src)
        self.assertNotIn("input.float(0.78,", self.src)

    def test_pine_does_not_stop_on_smi_signal_exit(self):
        self.assertNotIn("f_sat_ended_long", self.src)
        self.assertNotIn("f_sat_ended_short", self.src)
        self.assertNotIn("smiSig", self.src)
        self.assertNotIn("satEnded", self.src)
        self.assertIn("input.int(50, \"EMA\"", self.src)
        self.assertNotIn("input.int(60, \"EMA\"", self.src)
        self.assertIn("bool s2 = mOk", self.src)
        self.assertNotIn("bool s2 = true", self.src)
        self.assertIn("bool s6 = emaOk and rsiCf", self.src)
        self.assertNotIn("bool s6 = rsiCf", self.src)
        self.assertIn("not na(histCf) and histCf > 0", self.src)
        self.assertIn("not na(histCf) and histCf < 0", self.src)
        self.assertIn("input.int(5,  \"AO سريع\"", self.src)
        self.assertIn("input.int(34, \"AO بطيء\"", self.src)
        self.assertIn("f_ao()", self.src)
        self.assertIn("aoB > 0 and not na(aoCf) and aoCf > 0", self.src)
        self.assertIn("aoB < 0 and not na(aoCf) and aoCf < 0", self.src)
        self.assertIn("bool aoOk = isLong", self.src)
        self.assertIn("s1 and s2 and s3 and s4 and s5 and s6 and s7 and aoOk", self.src)
        self.assertIn("ao60, ao180", self.src)


if __name__ == "__main__":
    unittest.main()
