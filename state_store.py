"""Owned mutable storage for cascade scans.

The singleton in this module is the only place where cascade collections and
their matching locks are created.  ``state_manager`` exposes operations over
the store and compatibility aliases for callers that still inspect snapshots.
"""

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field


def _step_stats():
    return {step: {"total": 0, "passed": 0} for step in range(1, 9)}


@dataclass
class CascadeStateStore:
    """All mutable cascade state, grouped with the locks that protect it."""

    alerted_keys: dict = field(default_factory=dict)
    alerted_keys_lock: threading.Lock = field(default_factory=threading.Lock)
    trades_history: deque = field(default_factory=lambda: deque(maxlen=2000))
    trades_lock: threading.Lock = field(default_factory=threading.Lock)

    cascade_results: defaultdict = field(
        default_factory=lambda: defaultdict(dict)
    )
    cascade_results_lock: threading.Lock = field(default_factory=threading.Lock)
    cascade_stats: dict = field(default_factory=_step_stats)
    cascade_stats_lock: threading.Lock = field(default_factory=threading.Lock)

    last_complete_stats: dict = field(default_factory=_step_stats)
    last_complete_results: defaultdict = field(
        default_factory=lambda: defaultdict(dict)
    )
    last_complete_survivors: dict = field(default_factory=dict)
    last_complete_lock: threading.Lock = field(default_factory=threading.Lock)

    short_cascade_results: defaultdict = field(
        default_factory=lambda: defaultdict(dict)
    )
    short_cascade_results_lock: threading.Lock = field(
        default_factory=threading.Lock
    )
    short_cascade_stats: dict = field(default_factory=_step_stats)
    short_cascade_stats_lock: threading.Lock = field(
        default_factory=threading.Lock
    )

    last_complete_short_stats: dict = field(default_factory=_step_stats)
    last_complete_short_results: defaultdict = field(
        default_factory=lambda: defaultdict(dict)
    )
    last_complete_short_survivors: dict = field(default_factory=dict)
    last_complete_short_lock: threading.Lock = field(
        default_factory=threading.Lock
    )

    last_complete_scan_time: dict = field(
        default_factory=lambda: {"buy": None, "sell": None}
    )
    last_complete_scan_time_lock: threading.Lock = field(
        default_factory=threading.Lock
    )

    step1_ready_since: dict = field(default_factory=dict)
    step1_ready_since_lock: threading.Lock = field(default_factory=threading.Lock)
    step6_ready_since: dict = field(default_factory=dict)
    step6_ready_since_lock: threading.Lock = field(default_factory=threading.Lock)
    step7_ready_since: dict = field(default_factory=dict)
    step7_ready_since_lock: threading.Lock = field(default_factory=threading.Lock)

    step5_entry_time: dict = field(default_factory=dict)
    step5_entry_time_lock: threading.Lock = field(default_factory=threading.Lock)


STATE = CascadeStateStore()
