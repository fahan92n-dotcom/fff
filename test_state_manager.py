"""Tests for shared cascade state and its compatibility exports."""

import threading
import unittest
from datetime import datetime, timedelta, timezone

import fahadal92 as bot
import state_manager as state


def _candidate(symbol, base_frame, confirm_frame=None, triple_frame=None):
    return {
        "sym": symbol,
        "base_frame": base_frame,
        "confirm_frame": confirm_frame or base_frame * 3,
        "triple_frame": triple_frame or max(1, base_frame // 3),
    }


class StateManagerTestCase(unittest.TestCase):
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
        for store, lock in (
            (state.step1_ready_since, state.step1_ready_since_lock),
            (state.step6_ready_since, state.step6_ready_since_lock),
            (state.step7_ready_since, state.step7_ready_since_lock),
            (state.step5_entry_time, state.step5_entry_time_lock),
            (state.alerted_keys, state.alerted_keys_lock),
        ):
            with lock:
                store.clear()
        state.STEP5_MAX_WAIT_SECONDS = None

    def tearDown(self):
        self.setUp()


class TestSharedStateIdentity(StateManagerTestCase):
    def test_bot_reexports_same_state_objects(self):
        self.assertIs(bot.alerted_keys, state.alerted_keys)
        self.assertIs(bot.trades_history, state.trades_history)
        self.assertIs(
            bot.last_complete_survivors,
            state.last_complete_survivors,
        )
        self.assertIs(
            bot.last_complete_short_survivors,
            state.last_complete_short_survivors,
        )
        self.assertIs(bot.step5_entry_time, state.step5_entry_time)
        self.assertIs(bot.last_complete_lock, state.last_complete_lock)


class TestReadyTimestamps(StateManagerTestCase):
    def test_first_ready_timestamp_wins(self):
        store = {}
        lock = threading.Lock()
        first = datetime(2024, 1, 1, tzinfo=timezone.utc)
        second = first + timedelta(minutes=1)

        state._set_ready_since(store, lock, "key", first)
        state._set_ready_since(store, lock, "key", second)

        self.assertEqual(store["key"], first)

    def test_purge_removes_step1_timestamp_when_not_waiting(self):
        candidate = _candidate("BTCUSDT", 9)
        state.mark_stage_ready("buy", 1, [candidate])
        key = state.get_signal_key(
            "BTCUSDT",
            9,
            candidate["confirm_frame"],
            candidate["triple_frame"],
            "buy",
        )
        self.assertIn(key, state.step1_ready_since)

        state._purge_orphaned_ready_timestamps("buy")

        self.assertNotIn(key, state.step1_ready_since)

    def test_abandon_clears_timestamp_and_stages(self):
        candidate = _candidate("ETHUSDT", 12)
        with state.last_complete_lock:
            state.last_complete_survivors[5] = [candidate]
        state.mark_stage_ready("buy", 1, [candidate])

        state.abandon_waiting_candidate("buy", candidate)

        self.assertEqual(state.get_stage_candidates("buy", 5), [])
        self.assertNotIn(
            state.get_signal_key(
                "ETHUSDT",
                12,
                candidate["confirm_frame"],
                candidate["triple_frame"],
                "buy",
            ),
            state.step1_ready_since,
        )

    def test_waiting_candidate_keeps_timestamp_after_purge(self):
        candidate = _candidate("SOLUSDT", 15)
        with state.last_complete_lock:
            state.last_complete_survivors[6] = [candidate]
        state.mark_stage_ready("buy", 1, [candidate])
        key = state.get_signal_key(
            "SOLUSDT",
            15,
            candidate["confirm_frame"],
            candidate["triple_frame"],
            "buy",
        )

        state._purge_orphaned_ready_timestamps("buy")

        self.assertIn(key, state.step1_ready_since)


class TestStep5Storage(StateManagerTestCase):
    def test_far_frames_coexist(self):
        candidates = [
            _candidate("BTCUSDT", 12),
            _candidate("BTCUSDT", 240),
        ]
        state._store_step5_waiters("buy", candidates)

        stored = state.get_stage_candidates("buy", 5)
        self.assertEqual({item["base_frame"] for item in stored}, {12, 240})

    def test_near_frames_keep_larger(self):
        candidates = [
            _candidate("ETHUSDT", 60),
            _candidate("ETHUSDT", 90),
        ]
        state._store_step5_waiters("buy", candidates)

        stored = state.get_stage_candidates("buy", 5)
        self.assertEqual([item["base_frame"] for item in stored], [90])
        self.assertNotIn(("buy", "ETHUSDT", 60), state.step5_entry_time)

    def test_candidate_already_in_later_stage_is_blocked(self):
        candidate = _candidate("SOLUSDT", 30)
        with state.last_complete_lock:
            state.last_complete_survivors[6] = [candidate]

        state._store_step5_waiters("buy", [candidate])

        self.assertEqual(state.get_stage_candidates("buy", 5), [])

    def test_expired_waiter_is_removed_when_limit_enabled(self):
        candidate = _candidate("XRPUSDT", 45)
        with state.step5_entry_time_lock:
            state.step5_entry_time[("buy", "XRPUSDT", 45)] = (
                datetime.now(timezone.utc) - timedelta(seconds=10)
            )
        state.STEP5_MAX_WAIT_SECONDS = 1

        state._store_step5_waiters("buy", [candidate])

        self.assertEqual(state.get_stage_candidates("buy", 5), [])


class TestStageTransitions(StateManagerTestCase):
    def test_promote_and_clear_candidate(self):
        candidate = _candidate("ADAUSDT", 24)
        signal_key = state.get_signal_key(
            "ADAUSDT",
            candidate["base_frame"],
            candidate["confirm_frame"],
            candidate["triple_frame"],
            "buy",
        )
        with state.last_complete_lock:
            state.last_complete_survivors[5] = [candidate]
        state.mark_stage_ready("buy", 1, [candidate])
        state.mark_stage_ready("buy", 6, [candidate])

        state._promote_candidates("buy", 5, 6, [candidate])

        self.assertEqual(state.get_stage_candidates("buy", 5), [])
        self.assertEqual(state.get_stage_candidates("buy", 6), [candidate])
        self.assertIn(signal_key, state.step1_ready_since)
        self.assertIn(signal_key, state.step6_ready_since)

        state._clear_waiting_candidate(
            "ADAUSDT",
            candidate["base_frame"],
            candidate["confirm_frame"],
            candidate["triple_frame"],
            "buy",
        )

        self.assertEqual(state.get_stage_candidates("buy", 6), [])
        self.assertNotIn(signal_key, state.step1_ready_since)
        self.assertNotIn(signal_key, state.step6_ready_since)


class TestAlertDeduplication(StateManagerTestCase):
    def test_recent_signal_can_only_be_claimed_once(self):
        key = ("BTCUSDT", 9, 27, 3, "buy")

        first = state.claim_signal(key)
        second = state.claim_signal(key)

        self.assertIsNotNone(first)
        self.assertIsNone(second)

    def test_cleanup_removes_only_expired_keys(self):
        now = datetime.now(timezone.utc)
        with state.alerted_keys_lock:
            state.alerted_keys["old"] = now - timedelta(hours=5)
            state.alerted_keys["fresh"] = now

        state.cleanup_alerted_keys(expiry_hours=4)

        self.assertNotIn("old", state.alerted_keys)
        self.assertIn("fresh", state.alerted_keys)


if __name__ == "__main__":
    unittest.main()
