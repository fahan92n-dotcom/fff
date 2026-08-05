"""Tests for strategy predicates extracted from cascade orchestration."""

import unittest
from dataclasses import replace
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

        self.assertEqual(
            strategy._step2(candidate, long_rules),
            (False, "macd_histogram_not_red"),
        )
        self.assertEqual(
            strategy._step2(candidate, short_rules),
            (False, "macd_histogram_not_green"),
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
        get_resampled.assert_called_once_with(
            native,
            "BTCUSDT",
            "30m",
            90,
        )


if __name__ == "__main__":
    unittest.main()
