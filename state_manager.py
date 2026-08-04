"""Thread-safe shared state for cascade scans, stage transitions, and alerts."""

from datetime import datetime, timedelta, timezone

from state_store import STATE


ALERT_EXPIRY_HOURS = 4
STEP5_MAX_WAIT_SECONDS = None
STAGE5_COEXIST_MIN_RATIO = 4.0

# Compatibility aliases.  Storage ownership remains in ``state_store.STATE``.
alerted_keys = STATE.alerted_keys
alerted_keys_lock = STATE.alerted_keys_lock
trades_history = STATE.trades_history
trades_lock = STATE.trades_lock
cascade_results = STATE.cascade_results
cascade_results_lock = STATE.cascade_results_lock
cascade_stats = STATE.cascade_stats
cascade_stats_lock = STATE.cascade_stats_lock
last_complete_stats = STATE.last_complete_stats
last_complete_results = STATE.last_complete_results
last_complete_survivors = STATE.last_complete_survivors
last_complete_lock = STATE.last_complete_lock
short_cascade_results = STATE.short_cascade_results
short_cascade_results_lock = STATE.short_cascade_results_lock
short_cascade_stats = STATE.short_cascade_stats
short_cascade_stats_lock = STATE.short_cascade_stats_lock
last_complete_short_stats = STATE.last_complete_short_stats
last_complete_short_results = STATE.last_complete_short_results
last_complete_short_survivors = STATE.last_complete_short_survivors
last_complete_short_lock = STATE.last_complete_short_lock
last_complete_scan_time = STATE.last_complete_scan_time
last_complete_scan_time_lock = STATE.last_complete_scan_time_lock
step1_ready_since = STATE.step1_ready_since
step1_ready_since_lock = STATE.step1_ready_since_lock
step6_ready_since = STATE.step6_ready_since
step6_ready_since_lock = STATE.step6_ready_since_lock
step7_ready_since = STATE.step7_ready_since
step7_ready_since_lock = STATE.step7_ready_since_lock
step5_entry_time = STATE.step5_entry_time
step5_entry_time_lock = STATE.step5_entry_time_lock


def cleanup_alerted_keys(expiry_hours=ALERT_EXPIRY_HOURS):
    """Remove alert deduplication keys older than the configured expiry."""
    now = datetime.now(timezone.utc)
    with alerted_keys_lock:
        expired = [
            key
            for key, alerted_at in alerted_keys.items()
            if now - alerted_at > timedelta(hours=expiry_hours)
        ]
        for key in expired:
            del alerted_keys[key]


def claim_signal(signal_key, expiry_hours=ALERT_EXPIRY_HOURS):
    """Atomically reserve a signal key unless a recent alert already owns it."""
    now = datetime.now(timezone.utc)
    with alerted_keys_lock:
        last_alert = alerted_keys.get(signal_key)
        if last_alert and now - last_alert < timedelta(hours=expiry_hours):
            return None
        alerted_keys[signal_key] = now
    return now


def save_signal(symbol, price, base_frame, confirm_frame, triple_frame, signal_type="buy"):
    """Append one emitted signal to the bounded in-memory history."""
    with trades_lock:
        trades_history.append(
            {
                "time": datetime.now(timezone.utc),
                "symbol": symbol,
                "price": price,
                "timeframe": f"{base_frame}m/{confirm_frame}m/{triple_frame}m",
                "type": signal_type,
            }
        )


def get_candidate_key(candidate):
    return (
        candidate["sym"],
        candidate["base_frame"],
        candidate["confirm_frame"],
        candidate["triple_frame"],
    )


def get_signal_key(symbol, base_frame, confirm_frame, triple_frame, signal_type="buy"):
    return (symbol, base_frame, confirm_frame, triple_frame, signal_type)


def _set_ready_since(store, lock, key, ready_ts=None):
    """Record the first ready timestamp for a stage key."""
    timestamp = ready_ts or datetime.now(timezone.utc)
    with lock:
        store.setdefault(key, timestamp)
    return timestamp


def get_ready_since(symbol, base_frame, confirm_frame, triple_frame, signal_type="buy"):
    """Return the timestamp at which a candidate first passed stage 7."""
    key = get_signal_key(symbol, base_frame, confirm_frame, triple_frame, signal_type)
    with step7_ready_since_lock:
        return step7_ready_since.get(key)


def get_step1_ready_since(
    symbol,
    base_frame,
    confirm_frame,
    triple_frame,
    signal_type="buy",
):
    """Return the initial saturation timestamp recorded after stage 1."""
    key = get_signal_key(symbol, base_frame, confirm_frame, triple_frame, signal_type)
    with step1_ready_since_lock:
        return step1_ready_since.get(key)


def _get_stage_maps(signal_type):
    if signal_type == "buy":
        return last_complete_survivors, last_complete_lock
    if signal_type == "sell":
        return last_complete_short_survivors, last_complete_short_lock
    raise ValueError(f"Unsupported signal type: {signal_type}")


def _get_scan_maps(signal_type):
    if signal_type == "buy":
        return (
            cascade_stats,
            cascade_stats_lock,
            cascade_results,
            cascade_results_lock,
            last_complete_stats,
            last_complete_results,
            last_complete_survivors,
            last_complete_lock,
        )
    if signal_type == "sell":
        return (
            short_cascade_stats,
            short_cascade_stats_lock,
            short_cascade_results,
            short_cascade_results_lock,
            last_complete_short_stats,
            last_complete_short_results,
            last_complete_short_survivors,
            last_complete_short_lock,
        )
    raise ValueError(f"Unsupported signal type: {signal_type}")


def _upsert_stage_candidate(survivors_dict, stage_num, candidate):
    key = get_candidate_key(candidate)
    items = list(survivors_dict.get(stage_num, []))
    for index, existing in enumerate(items):
        if get_candidate_key(existing) == key:
            items[index] = candidate
            survivors_dict[stage_num] = items
            return
    items.append(candidate)
    survivors_dict[stage_num] = items


def _remove_stage_candidate(survivors_dict, stage_num, candidate_key):
    survivors_dict[stage_num] = [
        candidate
        for candidate in survivors_dict.get(stage_num, [])
        if get_candidate_key(candidate) != candidate_key
    ]


def _candidate_keys_in_stages(survivors_dict, stages):
    return {
        get_candidate_key(candidate)
        for stage_num in stages
        for candidate in survivors_dict.get(stage_num, [])
    }


def get_stage_candidates(signal_type, stage_num):
    """Return a snapshot of candidates waiting at one stage."""
    survivors_dict, survivors_lock = _get_stage_maps(signal_type)
    with survivors_lock:
        return list(survivors_dict.get(stage_num, []))


def remove_stage_candidate(signal_type, stage_num, candidate):
    """Remove one candidate from a stage under the side's survivor lock."""
    survivors_dict, survivors_lock = _get_stage_maps(signal_type)
    with survivors_lock:
        _remove_stage_candidate(
            survivors_dict,
            stage_num,
            get_candidate_key(candidate),
        )


def _clear_ready_timestamps(signal_type, symbol, base_frame, confirm_frame, triple_frame):
    """Drop saturation/ready timestamps for one candidate key."""
    ready_key = get_signal_key(
        symbol,
        base_frame,
        confirm_frame,
        triple_frame,
        signal_type,
    )
    for ready_store, ready_lock in (
        (step1_ready_since, step1_ready_since_lock),
        (step6_ready_since, step6_ready_since_lock),
        (step7_ready_since, step7_ready_since_lock),
    ):
        with ready_lock:
            ready_store.pop(ready_key, None)
    _forget_step5_entry(signal_type, symbol, base_frame)


def abandon_waiting_candidate(signal_type, candidate):
    """Remove a waiting candidate and forget its saturation episode."""
    _clear_waiting_candidate(
        candidate["sym"],
        candidate["base_frame"],
        candidate["confirm_frame"],
        candidate["triple_frame"],
        signal_type=signal_type,
    )


def _frames_far_apart(frame1, frame2):
    """Return whether two base frames may coexist in stage 5."""
    larger, smaller = max(frame1, frame2), min(frame1, frame2)
    return larger / smaller >= STAGE5_COEXIST_MIN_RATIO


def _remember_step5_entry(signal_type, symbol, base_frame, timestamp):
    with step5_entry_time_lock:
        step5_entry_time.setdefault((signal_type, symbol, base_frame), timestamp)


def _forget_step5_entry(signal_type, symbol, base_frame):
    with step5_entry_time_lock:
        step5_entry_time.pop((signal_type, symbol, base_frame), None)


def _store_step5_waiters(signal_type, candidates):
    """Replace stage-5 waiters, coalescing nearby frames per symbol."""
    survivors_dict, survivors_lock = _get_stage_maps(signal_type)
    now = datetime.now(timezone.utc)

    with survivors_lock:
        blocked = _candidate_keys_in_stages(survivors_dict, (6, 7))
        stage5_by_symbol = {}

        for candidate in candidates:
            if get_candidate_key(candidate) in blocked:
                continue

            symbol = candidate["sym"]
            base_frame = candidate["base_frame"]
            symbol_frames = stage5_by_symbol.setdefault(symbol, {})
            close_frame = next(
                (
                    existing_frame
                    for existing_frame in symbol_frames
                    if not _frames_far_apart(base_frame, existing_frame)
                ),
                None,
            )

            if close_frame is None:
                symbol_frames[base_frame] = candidate
                _remember_step5_entry(signal_type, symbol, base_frame, now)
            elif base_frame > close_frame:
                dropped = symbol_frames[close_frame]
                _clear_ready_timestamps(
                    signal_type,
                    dropped["sym"],
                    dropped["base_frame"],
                    dropped["confirm_frame"],
                    dropped["triple_frame"],
                )
                del symbol_frames[close_frame]
                symbol_frames[base_frame] = candidate
                _remember_step5_entry(signal_type, symbol, base_frame, now)

        if STEP5_MAX_WAIT_SECONDS is not None:
            with step5_entry_time_lock:
                expired = [
                    (symbol, base_frame)
                    for (stored_type, symbol, base_frame), entry_time
                    in step5_entry_time.items()
                    if stored_type == signal_type
                    and (now - entry_time).total_seconds() > STEP5_MAX_WAIT_SECONDS
                ]
            for symbol, base_frame in expired:
                symbol_frames = stage5_by_symbol.get(symbol)
                if symbol_frames is not None:
                    symbol_frames.pop(base_frame, None)
                    if not symbol_frames:
                        stage5_by_symbol.pop(symbol, None)
                _forget_step5_entry(signal_type, symbol, base_frame)

        survivors_dict[5] = [
            candidate
            for symbol_frames in stage5_by_symbol.values()
            for candidate in symbol_frames.values()
        ]


def _promote_candidates(signal_type, from_stage, to_stage, candidates):
    if not candidates:
        return
    survivors_dict, survivors_lock = _get_stage_maps(signal_type)
    with survivors_lock:
        for candidate in candidates:
            key = get_candidate_key(candidate)
            _remove_stage_candidate(survivors_dict, from_stage, key)
            _upsert_stage_candidate(survivors_dict, to_stage, candidate)


def _set_step8_survivors(signal_type, candidates):
    survivors_dict, survivors_lock = _get_stage_maps(signal_type)
    with survivors_lock:
        stage8 = {
            get_candidate_key(candidate): candidate
            for candidate in survivors_dict.get(8, [])
        }
        for candidate in candidates:
            stage8[get_candidate_key(candidate)] = candidate
        survivors_dict[8] = list(stage8.values())


def _clear_waiting_candidate(
    symbol,
    base_frame,
    confirm_frame,
    triple_frame,
    signal_type="buy",
):
    candidate_key = (symbol, base_frame, confirm_frame, triple_frame)
    survivors_dict, survivors_lock = _get_stage_maps(signal_type)
    with survivors_lock:
        for stage_num in (5, 6, 7, 8):
            _remove_stage_candidate(survivors_dict, stage_num, candidate_key)
    _clear_ready_timestamps(
        signal_type,
        symbol,
        base_frame,
        confirm_frame,
        triple_frame,
    )


def _active_ready_keys(signal_type):
    """Ready-key set for candidates currently waiting in stages 5–8."""
    survivors_dict, survivors_lock = _get_stage_maps(signal_type)
    with survivors_lock:
        return {
            get_signal_key(
                candidate["sym"],
                candidate["base_frame"],
                candidate["confirm_frame"],
                candidate["triple_frame"],
                signal_type,
            )
            for stage_num in (5, 6, 7, 8)
            for candidate in survivors_dict.get(stage_num, [])
        }


def _purge_orphaned_ready_timestamps(signal_type):
    """
    امسح timestamps التشبع للمرشحين الذين فشلوا بعد Step 1
    ولم يعودوا في طابور الانتظار (5–8).
    """
    active = _active_ready_keys(signal_type)
    for ready_store, ready_lock in (
        (step1_ready_since, step1_ready_since_lock),
        (step6_ready_since, step6_ready_since_lock),
        (step7_ready_since, step7_ready_since_lock),
    ):
        with ready_lock:
            stale = [
                key
                for key in ready_store
                if key[4] == signal_type and key not in active
            ]
            for key in stale:
                del ready_store[key]


def _candle_ready_timestamp(candidate, stage_num):
    """
    وقت جاهزية المرحلة من شمعة الإغلاق، لا من ساعة الجدار.

    Step1 كان يُسجَّل بـ datetime.now() بعد قفل الشمعة، فيصبح since_ts
    أحدث من ts كل الشموع المغلقة → نافذة EMA/RSI فارغة وStep6 يفشل للجميع.
    """
    frame = None
    if stage_num == 1:
        frame = candidate.get("df_base")
    elif stage_num in (6, 7):
        frame = candidate.get("df_triple")
        if frame is None or getattr(frame, "empty", True):
            frame = candidate.get("df_base")

    if frame is None or getattr(frame, "empty", True):
        return datetime.now(timezone.utc)
    if "ts" not in getattr(frame, "columns", []):
        return datetime.now(timezone.utc)

    candle_ts = frame["ts"].iloc[-1]
    if getattr(candle_ts, "to_pydatetime", None) is not None:
        candle_ts = candle_ts.to_pydatetime()
    if getattr(candle_ts, "tzinfo", None) is None:
        candle_ts = candle_ts.replace(tzinfo=timezone.utc)
    return candle_ts


def mark_stage_ready(signal_type, stage_num, candidates):
    """Set the first ready timestamp for candidates promoted to a stage."""
    if stage_num == 1:
        ready_store, ready_lock = step1_ready_since, step1_ready_since_lock
    elif stage_num == 6:
        ready_store, ready_lock = step6_ready_since, step6_ready_since_lock
    elif stage_num == 7:
        ready_store, ready_lock = step7_ready_since, step7_ready_since_lock
    else:
        raise ValueError(f"Unsupported ready stage: {stage_num}")

    for candidate in candidates:
        ready_key = get_signal_key(
            candidate["sym"],
            candidate["base_frame"],
            candidate["confirm_frame"],
            candidate["triple_frame"],
            signal_type,
        )
        _set_ready_since(
            ready_store,
            ready_lock,
            ready_key,
            _candle_ready_timestamp(candidate, stage_num),
        )


def reset_scan_state(signal_type):
    """Reset mutable stats/results for stages evaluated by a full scan."""
    stats, stats_lock, results, results_lock, *_ = _get_scan_maps(signal_type)
    with stats_lock, results_lock:
        for step_num in range(1, 6):
            stats[step_num] = {"total": 0, "passed": 0}
            results[step_num].clear()


def record_scan_step(signal_type, step_num, evaluations):
    """Store full-scan evaluations and return candidates that passed."""
    stats, stats_lock, results, results_lock, *_ = _get_scan_maps(signal_type)
    now = datetime.now(timezone.utc)
    passed = [candidate for candidate, ok, _ in evaluations if ok]
    step_results = {
        get_candidate_key(candidate): {
            "passed": ok,
            "reason": reason,
            "time": now,
        }
        for candidate, ok, reason in evaluations
    }
    with stats_lock, results_lock:
        stats[step_num] = {"total": len(evaluations), "passed": len(passed)}
        results[step_num] = step_results

    if step_num == 1:
        mark_stage_ready(signal_type, 1, passed)
    return passed


def complete_scan(signal_type, step_survivors):
    """Publish one full stages 1–5 scan using a canonical lock order."""
    (
        stats,
        stats_lock,
        results,
        results_lock,
        complete_stats,
        complete_results,
        complete_survivors,
        complete_lock,
    ) = _get_scan_maps(signal_type)

    with stats_lock:
        has_input = stats.get(1, {}).get("total", 0) > 0
    if not has_input:
        return False

    _store_step5_waiters(signal_type, step_survivors.get(5, []))
    # مرشحو Step1 الذين فشلوا لاحقًا لا يبقون بـ timestamp تشبع قديم.
    _purge_orphaned_ready_timestamps(signal_type)
    with complete_lock, stats_lock, results_lock:
        for step_num in range(1, 5):
            complete_stats[step_num] = dict(stats.get(step_num, {}))
            complete_results[step_num] = dict(results.get(step_num, {}))
            complete_survivors[step_num] = list(
                step_survivors.get(step_num, [])
            )
        complete_stats[5] = dict(stats.get(5, {}))
        complete_results[5] = dict(results.get(5, {}))

    with last_complete_scan_time_lock:
        last_complete_scan_time[signal_type] = datetime.now(timezone.utc)
    return True


def _update_last_complete_step(signal_type, step_num, evaluations):
    """Publish quick-check evaluations for a single stage."""
    if signal_type == "buy":
        stats = last_complete_stats
        results = last_complete_results
        complete_lock = last_complete_lock
    elif signal_type == "sell":
        stats = last_complete_short_stats
        results = last_complete_short_results
        complete_lock = last_complete_short_lock
    else:
        raise ValueError(f"Unsupported signal type: {signal_type}")

    now = datetime.now(timezone.utc)
    step_results = {
        get_candidate_key(candidate): {
            "passed": ok,
            "reason": reason,
            "time": now,
        }
        for candidate, ok, reason in evaluations
    }
    with complete_lock:
        stats[step_num] = {
            "total": len(evaluations),
            "passed": sum(1 for _, ok, _ in evaluations if ok),
        }
        results[step_num] = step_results


def touch_scan_times():
    """Mark both sides as refreshed by the quick-check cycle."""
    now = datetime.now(timezone.utc)
    with last_complete_scan_time_lock:
        last_complete_scan_time["buy"] = now
        last_complete_scan_time["sell"] = now
