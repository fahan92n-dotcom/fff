"""Tests for strategy experiment variants and ranking."""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

import cascade_steps as steps
import fahadal92 as bot
import indicators as ind
import strategy_variants as experiments


def _ohlcv(start, closes, minutes=15):
    rows = []
    for index, close in enumerate(closes):
        px = float(close)
        rows.append(
            {
                "ts": start + timedelta(minutes=minutes * index),
                "open": px,
                "high": px * 1.01,
                "low": px * 0.99,
                "close": px,
                "vol": 1.0,
            }
        )
    return pd.DataFrame(rows)


class TestBtcCorrelation(unittest.TestCase):
    def test_high_correlation_passes(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        rng = np.random.default_rng(0)
        btc_rets = rng.normal(0, 0.01, size=59)
        alt_rets = btc_rets + rng.normal(0, 0.001, size=59)
        btc_closes = 100 * np.cumprod(1.0 + np.concatenate([[0.0], btc_rets]))
        alt_closes = 100 * np.cumprod(1.0 + np.concatenate([[0.0], alt_rets]))
        alt = _ohlcv(start, alt_closes)
        btc = _ohlcv(start, btc_closes)
        corr = ind.calc_close_correlation(alt, btc, lookback=50)
        self.assertIsNotNone(corr)
        self.assertGreater(corr, 0.8)
        self.assertTrue(
            ind.check_btc_correlation(alt, btc, lookback=50, min_corr=0.5)
        )

    def test_uncorrelated_fails(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        rng = np.random.default_rng(1)
        alt_rets = rng.normal(0, 0.01, size=59)
        btc_rets = rng.normal(0, 0.01, size=59)
        alt = _ohlcv(
            start,
            100 * np.cumprod(1.0 + np.concatenate([[0.0], alt_rets])),
        )
        btc = _ohlcv(
            start,
            100 * np.cumprod(1.0 + np.concatenate([[0.0], btc_rets])),
        )
        self.assertFalse(
            ind.check_btc_correlation(alt, btc, lookback=50, min_corr=0.9)
        )


class TestVariantStepOverrides(unittest.TestCase):
    def test_step4_skips_donchian_confirm(self):
        candidate = {
            "sym": "ETHUSDT",
            "base_api": "1m",
            "confirm_frame": 27,
            "df_confirm": pd.DataFrame(),
            "disable_ribbon_cache": True,
            "variant": {"skip_donchian_confirm": True},
        }
        ok, reason = steps.step4(candidate)
        self.assertTrue(ok)
        self.assertEqual(reason, "passed")

    def test_step6_can_disable_confirm_rsi(self):
        # Build a frame where base EMA condition fails → still false.
        # Here we only assert the RSI lookback None path does not crash and
        # that missing ready_since fails on ema first.
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        closes = [100 + i * 0.1 for i in range(220)]
        df = _ohlcv(start, closes, minutes=60)
        candidate = {
            "sym": "ETHUSDT",
            "base_frame": 60,
            "confirm_frame": 180,
            "triple_frame": 20,
            "df_base": df,
            "df_confirm": df,
            "ready_since": start,
            "variant": {"confirm_rsi_lookback": None},
        }
        # Without a true saturation+EMA episode this should fail ema, not RSI.
        ok, reason = steps.step6(candidate)
        self.assertFalse(ok)
        self.assertEqual(reason, "ema50")

    def test_baseline_variant_dict_omits_rsi_override(self):
        baseline = experiments.EXPERIMENT_VARIANTS[0]
        payload = baseline.to_dict()
        self.assertNotIn("confirm_rsi_lookback", payload)
        self.assertFalse(payload["ema_on_confirm"])
        self.assertFalse(payload["skip_donchian_confirm"])
        self.assertIsNone(payload["btc_corr_min"])


class TestExperimentRanking(unittest.TestCase):
    def test_score_prefers_higher_expectancy(self):
        weak = experiments.score_scan_result(
            {
                "wins": [{}] * 2,
                "losses": [{}] * 2,
                "opens": [],
                "total": 4,
            }
        )
        strong = experiments.score_scan_result(
            {
                "wins": [{}] * 5,
                "losses": [{}] * 1,
                "opens": [],
                "total": 6,
            }
        )
        # 2*1 - 2*0.7 = 0.6  vs  5*1 - 1*0.7 = 4.3
        self.assertGreater(strong["expectancy"], weak["expectancy"])
        self.assertGreater(
            experiments.rank_key(strong),
            experiments.rank_key(weak),
        )

    def test_format_names_winner(self):
        variant = experiments.EXPERIMENT_VARIANTS[2]
        bundle = {
            "ready": True,
            "symbols_scanned": 10,
            "winner": {
                "variant": variant,
                "summary": {
                    "wins": 3,
                    "losses": 1,
                    "opens": 0,
                    "closed": 4,
                    "win_rate": 75.0,
                    "expectancy": 2.3,
                },
            },
            "rows": [
                {
                    "variant": variant,
                    "summary": {
                        "wins": 3,
                        "losses": 1,
                        "opens": 0,
                        "closed": 4,
                        "win_rate": 75.0,
                        "expectancy": 2.3,
                    },
                }
            ],
        }
        text = "\n".join(experiments.format_experiments_report(bundle))
        self.assertIn("الأفضل الآن", text)
        self.assertIn("إلغاء Donchian", text)
        self.assertIn("مقارنة النسخ", text)

    def test_experiments_command_routes(self):
        called = {}

        def fake_handle(chat_id, send_fn):
            called["chat_id"] = chat_id

        with patch.object(
            bot,
            "handle_experiments_command",
            side_effect=fake_handle,
        ):
            bot._dispatch_command("/experiments", "7")
            bot._dispatch_command("/تجارب", "7")

        self.assertEqual(called["chat_id"], "7")


if __name__ == "__main__":
    unittest.main()
