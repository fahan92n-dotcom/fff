"""Tests for ownership of mutable cascade state."""

import unittest

import state_manager
from state_store import STATE, CascadeStateStore


class TestStateOwnership(unittest.TestCase):
    def test_manager_aliases_single_owned_store(self):
        self.assertIs(state_manager.last_complete_survivors, STATE.last_complete_survivors)
        self.assertIs(state_manager.last_complete_lock, STATE.last_complete_lock)
        self.assertIs(state_manager.step5_entry_time, STATE.step5_entry_time)
        self.assertIs(state_manager.step5_entry_time_lock, STATE.step5_entry_time_lock)

    def test_store_instances_do_not_share_mutable_state(self):
        first = CascadeStateStore()
        second = CascadeStateStore()
        first.last_complete_survivors[5] = [{"sym": "BTCUSDT"}]

        self.assertEqual(second.last_complete_survivors, {})
        self.assertIsNot(first.last_complete_lock, second.last_complete_lock)


if __name__ == "__main__":
    unittest.main()
