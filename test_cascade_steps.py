"""Tests for strategy predicates extracted from cascade orchestration."""

import unittest
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pandas as pd

import cascade_pipeline as pipeline
import cascade_steps as strategy
import fahadal92 as bot


class TestStepLabelsHtmlSafe(unittest.TestCase):
    def test_step_labels_do_not_embed_raw_less_than(self):
        """Telegram parse_mode=HTML rejects plain '<' outside tags (e.g. Stoch<80)."""
        for labels in (strategy.STEP_LABELS, strategy.SHORT_STEP_LABELS):
            for name, label in labels.items():
                self.assertNotIn(
                    "<",
                    label.replace("&lt;", ""),
                    msg=f"{name} label embeds raw '<' which breaks Telegram HTML: {label!r}",
                )

    def test_sell_step8_label_escapes_less_than(self):
        label = strategy.SHORT_STEP_LABELS["rsi_stoch_short"]
        self.assertIn("Stoch&lt;80", label)
        self.assertNotIn("Stoch<80", label)


class TestStepOwnership(unittest.TestCase):
    def test_pipeline_and_compatibility_module_reexport_strategy_steps(self):
        self.assertIs(pipeline.step1, strategy.step1)
        self.assertIs(bot.step1, strategy.step1)
        self.assertEqual(strategy.step1.__module__, "cascade_steps")

    def test_long_and_short_steps_share_directional_implementation(self):
        candidate = {"df_base": pd.DataFrame({"close": [1.0] * 1000})}
        long_rules = replace(
            strategy.LONG_RULES,
            base_histogram_check=Mock(return_value=False),
        )
        short_rules = replace(
            strategy.SHORT_RULES,
            base_histogram_check=Mock(return_value=False),
        )

        self.assertEqual(strategy._step2(candidate, long_rules), (True, "passed"))
        self.assertEqual(strategy._step2(candidate, short_rules), (True, "passed"))


class TestStep2BaseMacdUngated(unittest.TestCase):
    """MACD الفريم الأساسي غير مشروط — الخطوة ② تمر دائماً."""

    def test_passes_even_when_histogram_and_line_fail(self):
        histogram_check = Mock(return_value=False)
        line_check = Mock(return_value=False)
        pinned_rules = replace(
            strategy.LONG_RULES,
            base_histogram_check=histogram_check,
            macd_line_check=line_check,
        )
        candidate = {
            "df_base": pd.DataFrame({"close": [1.0] * 10}),
            "base_frame": 60,
        }
        self.assertEqual(strategy._step2(candidate, pinned_rules), (True, "passed"))
        histogram_check.assert_not_called()
        line_check.assert_not_called()

    def test_short_also_passes_without_base_macd(self):
        candidate = {"df_base": pd.DataFrame({"close": [1.0] * 10})}
        self.assertEqual(
            strategy._step2(candidate, strategy.SHORT_RULES),
            (True, "passed"),
        )


class TestImmediateHigherFrame(unittest.TestCase):
    def test_saturation_uses_only_next_timeframe(self):
        candidate = {
            "sym": "BTCUSDT",
            "base_frame": 60,
            "base_api": "60m",
        }
        n = 120
        higher = pd.DataFrame(
            {
                "ts": pd.date_range("2024-01-01", periods=n, freq="90min", tz="UTC"),
                "open": [1.0] * n,
                "high": [1.1] * n,
                "low": [0.9] * n,
                "close": [1.0] * n,
                "vol": [1.0] * n,
            }
        )
        native = higher.copy()
        get_resampled = Mock(return_value=higher)
        smi = pd.Series([-50.0] * n)

        with patch.object(
            strategy,
            "get_cached",
            return_value=native,
        ) as get_cached, patch.object(
            strategy,
            "calc_smi",
            return_value=(smi, smi, smi),
        ):
            result = strategy._has_higher_tf_saturation(
                candidate,
                "buy",
                get_resampled,
            )

        self.assertTrue(result)
        get_cached.assert_called_once_with("BTCUSDT", "30m")

    def test_higher_tf_uses_live_tradingview_bar_when_asof_set(self):
        """كل سقف أكبر (مو 180 فقط) يُقرأ على الشمعة الجارية مثل TradingView."""
        asof = datetime(2026, 8, 12, 1, 0, tzinfo=timezone.utc)
        n = 120
        live = pd.DataFrame(
            {
                "ts": pd.date_range("2026-07-01", periods=n, freq="180min", tz="UTC"),
                "open": [1.0] * n,
                "high": [1.1] * n,
                "low": [0.9] * n,
                "close": [1.0] * n,
                "vol": [1.0] * n,
            }
        )
        smi = pd.Series([-10.0] * n)
        for base in strategy.TIMEFRAME_CHAIN:
            higher = strategy.NEXT_TF[base]
            native_api = strategy.TF_TO_API[higher]
            candidate = {
                "sym": "BTCUSDT",
                "base_frame": base,
                "base_api": strategy.TF_TO_API[base],
                "asof": asof,
                "get_raw": Mock(return_value=live.copy()),
            }
            get_resampled = Mock(return_value=pd.DataFrame())
            with self.subTest(base=base, higher=higher, source=native_api):
                with patch.object(
                    strategy,
                    "confirm_macd_frame",
                    return_value=live,
                ) as live_fn, patch.object(
                    strategy,
                    "calc_smi",
                    return_value=(smi, smi, smi),
                ):
                    result = strategy._has_higher_tf_saturation(
                        candidate,
                        "buy",
                        get_resampled,
                    )
                self.assertFalse(result)
                live_fn.assert_called_once()
                self.assertEqual(live_fn.call_args.args[2], higher)
                self.assertEqual(live_fn.call_args.kwargs["now"], asof)
                get_resampled.assert_not_called()


if __name__ == "__main__":
    unittest.main()
