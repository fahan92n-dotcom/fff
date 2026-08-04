"""Focused tests for the extracted cascade pipeline orchestration."""

import unittest
from unittest.mock import Mock, patch

import pandas as pd

import cascade_pipeline as pipeline
import fahadal92 as bot
import state_manager as state


def _candidate(symbol="BTCUSDT", base_frame=9):
    frame = pd.DataFrame(
        {
            "close": [100.0],
            "high": [101.0],
            "low": [99.0],
        }
    )
    return {
        "sym": symbol,
        "base_api": "1m",
        "triple_api": "1m",
        "base_frame": base_frame,
        "confirm_frame": base_frame * 3,
        "triple_frame": max(1, base_frame // 3),
        "df_base": frame,
        "df_confirm": frame,
        "df_triple": frame,
        "raw_base": frame,
        "get_resampled": Mock(),
    }


class CascadePipelineTestCase(unittest.TestCase):
    def setUp(self):
        for survivors, lock in (
            (state.last_complete_survivors, state.last_complete_lock),
            (
                state.last_complete_short_survivors,
                state.last_complete_short_lock,
            ),
        ):
            with lock:
                survivors.clear()
        for stats, results, stats_lock, results_lock in (
            (
                state.cascade_stats,
                state.cascade_results,
                state.cascade_stats_lock,
                state.cascade_results_lock,
            ),
            (
                state.short_cascade_stats,
                state.short_cascade_results,
                state.short_cascade_stats_lock,
                state.short_cascade_results_lock,
            ),
        ):
            with stats_lock, results_lock:
                for step_num in range(1, 9):
                    stats[step_num] = {"total": 0, "passed": 0}
                    results[step_num].clear()


class TestCompatibilityExports(CascadePipelineTestCase):
    def test_bot_reexports_pipeline_functions(self):
        self.assertIs(bot.run_cascade_scan, pipeline.run_cascade_scan)
        self.assertIs(
            bot.run_short_cascade_scan,
            pipeline.run_short_cascade_scan,
        )
        self.assertIs(bot.quick_check_watcher, pipeline.quick_check_watcher)
        self.assertIs(bot.step1, pipeline.step1)


class TestStepBatch(CascadePipelineTestCase):
    def test_candidate_exception_is_isolated(self):
        good = {"id": "good"}
        bad = {"id": "bad"}

        def evaluate(candidate):
            if candidate is bad:
                raise ValueError("bad candidate")
            return True, "passed"

        results = pipeline._run_step_batch(
            [good, bad],
            evaluate,
            2,
            "TEST",
            max_workers=2,
        )
        by_id = {
            candidate["id"]: (passed, reason)
            for candidate, passed, reason in results
        }

        self.assertEqual(by_id["good"], (True, "passed"))
        self.assertEqual(by_id["bad"], (False, "bad candidate"))


class TestFullScan(CascadePipelineTestCase):
    def test_shared_scan_runner_records_all_five_stages_for_both_sides(self):
        for signal_type in ("buy", "sell"):
            candidate = _candidate(
                symbol=f"{signal_type.upper()}USDT",
                base_frame=12,
            )
            side_steps = [Mock(return_value=(True, "passed")) for _ in range(5)]

            with patch.object(
                pipeline,
                "_build_tripling_candidates",
                return_value=[candidate],
            ), patch.object(
                pipeline,
                "steps" if signal_type == "buy" else "short_steps",
                side_steps,
            ), patch.object(
                pipeline,
                "symbols_cache",
                [candidate["sym"]],
            ):
                pipeline._run_cascade_scan(
                    signal_type,
                    require_cache=False,
                )

            if signal_type == "buy":
                stats = state.cascade_stats
            else:
                stats = state.short_cascade_stats
            self.assertEqual(stats[5], {"total": 1, "passed": 1})
            self.assertEqual(
                state.get_stage_candidates(signal_type, 5),
                [candidate],
            )


class TestQuickStageAdvancement(CascadePipelineTestCase):
    def test_one_side_advances_from_stage5_to_signal(self):
        candidate = _candidate()
        with state.last_complete_lock:
            state.last_complete_survivors[5] = [candidate]
        handler = Mock()
        original_handler = pipeline._signal_handler
        pipeline.set_signal_handler(handler)

        def refresh_stage(signal_type, stage_num, _get_resampled):
            return state.get_stage_candidates(signal_type, stage_num)

        try:
            with patch.object(
                pipeline,
                "_refresh_and_validate_step5",
                return_value=candidate,
            ), patch.object(
                pipeline,
                "_refresh_stage",
                side_effect=refresh_stage,
            ), patch.object(
                pipeline,
                "_filter_higher_saturation",
                side_effect=lambda _side, _stage, candidates, _resample: candidates,
            ), patch.object(
                pipeline,
                "_run_step_batch",
                side_effect=lambda candidates, *_args, **_kwargs: [
                    (item, True, "passed") for item in candidates
                ],
            ):
                pipeline._advance_pipeline("buy", [candidate], Mock())
        finally:
            pipeline.set_signal_handler(original_handler)

        self.assertEqual(state.get_stage_candidates("buy", 5), [])
        self.assertEqual(state.get_stage_candidates("buy", 6), [])
        self.assertEqual(state.get_stage_candidates("buy", 7), [candidate])
        self.assertEqual(state.get_stage_candidates("buy", 8), [candidate])
        handler.assert_called_once()


if __name__ == "__main__":
    unittest.main()
