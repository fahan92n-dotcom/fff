"""Cascade signal pipeline, including full scans and quick stage advancement."""

import concurrent.futures
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from binance_data import (
    MARKET_MODE,
    cache_merge,
    fast_prefetch_done,
    get_cached,
    get_ohlcv,
    get_ohlcv_futures,
    ohlcv_cache,
    ohlcv_cache_lock,
    symbols_cache,
    symbols_cache_lock,
)
from indicators import (
    MIN_CANDLES,
    _ribbon_cache,
    _ribbon_cache_lock,
    calc_smi,
    check_donchian_trend_ribbon,
    check_macd_green,
    check_macd_line_long,
    check_macd_line_short,
    check_macd_red,
    resample_ohlcv,
)
from cascade_steps import (
    NEXT_TF,
    SHORT_STEP_LABELS,
    SHORT_STEP_NAMES,
    STEP_LABELS,
    STEP_NAMES,
    TF_TO_API,
    TIMEFRAME_CHAIN,
    TRIPLING_PAIRS,
    _has_higher_tf_saturation,
    short_step1,
    short_step2,
    short_step3,
    short_step4,
    short_step5,
    short_step6,
    short_step7,
    short_step8,
    short_steps,
    step1,
    step2,
    step3,
    step4,
    step5,
    step6,
    step7,
    step8,
    steps,
)
from state_manager import (
    _promote_candidates,
    _set_step8_survivors,
    _update_last_complete_step,
    complete_scan,
    get_stage_candidates,
    mark_stage_ready,
    record_scan_step,
    remove_stage_candidate,
    reset_scan_state,
    touch_scan_times,
)


log = logging.getLogger(__name__)

QUICK_CHECK_INTERVAL_SECONDS = 3

_signal_handler = None


def set_signal_handler(handler):
    """Register the application callback used after a candidate passes stage 8."""
    global _signal_handler
    _signal_handler = handler


def _new_resampler():
    cache = {}

    def get_resampled(raw_df, symbol, source_tf, minutes):
        key = (symbol, source_tf, minutes)
        if key not in cache:
            cache[key] = resample_ohlcv(raw_df, minutes)
        return cache[key]

    return cache, get_resampled


def _build_tripling_candidates(symbols, get_resampled):
    candidates = []
    for symbol in symbols:
        raw_by_tf = {
            "1m": get_cached(symbol, "1m"),
            "30m": get_cached(symbol, "30m"),
            "60m": get_cached(symbol, "60m"),
        }
        for (
            base_frame,
            confirm_frame,
            triple_frame,
            base_api,
            triple_api,
        ) in TRIPLING_PAIRS:
            raw_base = raw_by_tf.get(base_api, pd.DataFrame())
            raw_triple = raw_by_tf.get(triple_api, pd.DataFrame())
            if raw_base.empty or raw_triple.empty:
                continue

            df_base = get_resampled(
                raw_base,
                symbol,
                base_api,
                base_frame,
            )
            df_confirm = get_resampled(
                raw_base,
                symbol,
                base_api,
                confirm_frame,
            )
            df_triple = get_resampled(
                raw_triple,
                symbol,
                triple_api,
                triple_frame,
            )
            if df_base.empty or df_confirm.empty or df_triple.empty:
                continue
            if len(df_base) < MIN_CANDLES:
                continue

            candidates.append(
                {
                    "sym": symbol,
                    "base_api": base_api,
                    "triple_api": triple_api,
                    "base_frame": base_frame,
                    "confirm_frame": confirm_frame,
                    "triple_frame": triple_frame,
                    "df_base": df_base,
                    "df_confirm": df_confirm,
                    "df_triple": df_triple,
                    "raw_base": raw_base,
                    "get_resampled": get_resampled,
                }
            )
    return candidates


def _run_step_batch(
    candidates,
    step_fn,
    step_num,
    signal_label,
    *,
    max_workers=20,
    timeout=120,
):
    """Evaluate a stage concurrently while isolating candidate-level failures."""
    if not candidates:
        return []

    def run_one(candidate):
        try:
            return candidate, *step_fn(candidate)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Intentional plugin boundary: one symbol must not abort the batch.
            log.exception(
                "❌ خطأ في الخطوة %d (%s): %s",
                step_num,
                signal_label,
                exc,
            )
            return candidate, False, str(exc)

    executor = None
    results = []
    try:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = [executor.submit(run_one, candidate) for candidate in candidates]
        try:
            for future in concurrent.futures.as_completed(
                futures,
                timeout=timeout,
            ):
                results.append(future.result())
        except concurrent.futures.TimeoutError:
            log.warning(
                "⚠️ بعض المهام لم تكتمل خلال المهلة المحددة في الخطوة %d (%s)",
                step_num,
                signal_label,
            )
    except (OSError, RuntimeError) as exc:
        log.exception(
            "❌ تعذر تشغيل الخطوة %d (%s): %s",
            step_num,
            signal_label,
            exc,
        )
        return []
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
    return results


def _run_cascade_scan(signal_type, require_cache):
    label = "LONG" if signal_type == "buy" else "SHORT"
    side_steps = steps if signal_type == "buy" else short_steps

    with symbols_cache_lock:
        symbols = list(symbols_cache)
    if not symbols:
        log.warning("⚠️ لا توجد symbols في الكاش (%s)", label)
        return

    if require_cache:
        with ohlcv_cache_lock:
            cache_size = len(ohlcv_cache)
        if cache_size < len(symbols) * 0.8:
            log.info(
                "⏳ الكاش غير كافٍ بعد (%d مفتاح)، تخطي المسح",
                cache_size,
            )
            return
        log.info("✅ الكاش كافٍ (%d مفتاح)", cache_size)

    reset_scan_state(signal_type)
    resample_cache, get_resampled = _new_resampler()
    candidates = _build_tripling_candidates(symbols, get_resampled)
    step_survivors = {}
    log.info(
        "🔄 Cascade Scan (%s): %d مرشح قبل الخطوات",
        label,
        len(candidates),
    )

    for step_num, step_fn in enumerate(side_steps[:5], start=1):
        if not candidates:
            log.info(
                "⏸️ انقطعت المعالجة في الخطوة %d (%s)",
                step_num,
                label,
            )
            break

        evaluations = _run_step_batch(
            candidates,
            step_fn,
            step_num,
            label,
            max_workers=15,
        )
        candidates = record_scan_step(
            signal_type,
            step_num,
            evaluations,
        )
        step_survivors[step_num] = candidates
        if evaluations:
            log.info(
                "📍 خطوة %d (%s): %d/%d نجحوا",
                step_num,
                label,
                len(candidates),
                len(evaluations),
            )
        else:
            log.warning("⚠️ لا توجد نتائج في الخطوة %d (%s)", step_num, label)

    complete_scan(signal_type, step_survivors)
    log.info(
        "🪜 مرشحو %s المحفوظون بعد الخطوة 5: %d",
        label,
        len(step_survivors.get(5, [])),
    )
    resample_cache.clear()
    with _ribbon_cache_lock:
        _ribbon_cache.clear()


def run_cascade_scan():
    _run_cascade_scan("buy", require_cache=True)


def run_short_cascade_scan():
    _run_cascade_scan("sell", require_cache=False)


def _refresh_waiting_candidate(
    candidate,
    get_resampled,
    *,
    need_base=False,
    need_triple=False,
):
    refreshed = dict(candidate)
    symbol = candidate["sym"]

    if need_base:
        raw_base = get_cached(symbol, candidate["base_api"])
        if raw_base.empty:
            return None
        df_base = get_resampled(
            raw_base,
            symbol,
            candidate["base_api"],
            candidate["base_frame"],
        )
        df_confirm = get_resampled(
            raw_base,
            symbol,
            candidate["base_api"],
            candidate["confirm_frame"],
        )
        if df_base.empty or df_confirm.empty or len(df_base) < MIN_CANDLES:
            return None
        refreshed.update(
            {
                "raw_base": raw_base,
                "df_base": df_base,
                "df_confirm": df_confirm,
            }
        )

    if need_triple:
        raw_triple = get_cached(symbol, candidate["triple_api"])
        if raw_triple.empty:
            return None
        df_triple = get_resampled(
            raw_triple,
            symbol,
            candidate["triple_api"],
            candidate["triple_frame"],
        )
        if df_triple.empty or len(df_triple) < MIN_CANDLES:
            return None
        refreshed["df_triple"] = df_triple

    refreshed["get_resampled"] = get_resampled
    return refreshed


def _refresh_and_validate_step5_side(
    candidate,
    get_resampled,
    signal_type,
):
    symbol = candidate["sym"]
    raw_base = get_cached(symbol, candidate["base_api"])
    if raw_base.empty:
        return None

    df_base = get_resampled(
        raw_base,
        symbol,
        candidate["base_api"],
        candidate["base_frame"],
    )
    df_confirm = get_resampled(
        raw_base,
        symbol,
        candidate["base_api"],
        candidate["confirm_frame"],
    )
    if df_base.empty or df_confirm.empty or len(df_base) < MIN_CANDLES:
        return None

    smi, _, _ = calc_smi(
        df_base["high"],
        df_base["low"],
        df_base["close"],
    )
    current_smi = float(smi.iloc[-1])
    if signal_type == "buy":
        if current_smi > -40:
            return None
        base_macd_ok = check_macd_red(df_base) and check_macd_line_long(
            df_base,
            base_frame=candidate["base_frame"],
        )
        ribbon_direction = "green"
        confirm_macd_ok = check_macd_green(df_confirm)
    elif signal_type == "sell":
        if current_smi < 40:
            return None
        base_macd_ok = check_macd_green(df_base) and check_macd_line_short(
            df_base,
            base_frame=candidate["base_frame"],
        )
        ribbon_direction = "red"
        confirm_macd_ok = check_macd_red(df_confirm)
    else:
        raise ValueError(f"Unsupported signal type: {signal_type}")

    candidate["raw_base"] = raw_base
    if _has_higher_tf_saturation(
        candidate,
        signal_type,
        get_resampled,
    ):
        return None
    if not base_macd_ok:
        return None

    base_key = (
        symbol,
        candidate["base_api"],
        candidate["base_frame"],
    )
    if not check_donchian_trend_ribbon(
        df_base,
        ribbon_direction,
        cache_key=base_key,
    ):
        return None

    confirm_key = (
        symbol,
        candidate["base_api"],
        candidate["confirm_frame"],
    )
    if not check_donchian_trend_ribbon(
        df_confirm,
        ribbon_direction,
        cache_key=confirm_key,
    ):
        return None
    if not confirm_macd_ok:
        return None

    candidate.update(
        {
            "df_base": df_base,
            "df_confirm": df_confirm,
            "get_resampled": get_resampled,
        }
    )
    return candidate


def _refresh_and_validate_step5(candidate, get_resampled):
    return _refresh_and_validate_step5_side(
        candidate,
        get_resampled,
        "buy",
    )


def _refresh_and_validate_step5_short(candidate, get_resampled):
    return _refresh_and_validate_step5_side(
        candidate,
        get_resampled,
        "sell",
    )


def _refresh_stage(signal_type, stage_num, get_resampled):
    refreshed = []
    for candidate in get_stage_candidates(signal_type, stage_num):
        candidate2 = _refresh_waiting_candidate(
            candidate,
            get_resampled,
            need_triple=True,
        )
        if candidate2 is None:
            remove_stage_candidate(signal_type, stage_num, candidate)
        else:
            refreshed.append(candidate2)
    return refreshed


def _filter_higher_saturation(
    signal_type,
    stage_num,
    candidates,
    get_resampled,
):
    filtered = []
    for candidate in candidates:
        if _has_higher_tf_saturation(
            candidate,
            signal_type,
            get_resampled,
        ):
            remove_stage_candidate(signal_type, stage_num, candidate)
        else:
            filtered.append(candidate)
    return filtered


def _evaluate_transition(
    signal_type,
    from_stage,
    candidates,
    step_fn,
    label,
):
    """Evaluate one quick-check transition and publish its state changes."""
    to_stage = from_stage + 1
    evaluations = _run_step_batch(
        candidates,
        step_fn,
        to_stage,
        label,
    )
    _update_last_complete_step(signal_type, to_stage, evaluations)
    passed = [candidate for candidate, ok, _ in evaluations if ok]
    if to_stage < 8:
        mark_stage_ready(signal_type, to_stage, passed)
        _promote_candidates(signal_type, from_stage, to_stage, passed)
    else:
        _set_step8_survivors(signal_type, passed)
    return passed


def _waiting_transition_candidates(signal_type, stage_num, get_resampled):
    refreshed = _refresh_stage(signal_type, stage_num, get_resampled)
    return _filter_higher_saturation(
        signal_type,
        stage_num,
        refreshed,
        get_resampled,
    )


def _emit_signals(signal_type, label, passed, evaluated_count):
    if not passed:
        return
    if _signal_handler is None:
        raise RuntimeError("Cascade signal handler is not configured")
    for candidate in passed:
        _signal_handler(
            candidate["sym"],
            candidate["base_frame"],
            candidate["confirm_frame"],
            candidate["triple_frame"],
            candidate["df_base"],
            signal_type=signal_type,
        )
    log.info(
        "⚡ Quick check (%s): %d إشارة من %d مرشح محفوظ",
        label,
        len(passed),
        evaluated_count,
    )


def _advance_pipeline(signal_type, stage5_candidates, get_resampled):
    label = "LONG" if signal_type == "buy" else "SHORT"
    if signal_type == "buy":
        validate_step5 = _refresh_and_validate_step5
        stage6_fn, stage7_fn, stage8_fn = step6, step7, step8
    elif signal_type == "sell":
        validate_step5 = _refresh_and_validate_step5_short
        stage6_fn, stage7_fn, stage8_fn = (
            short_step6,
            short_step7,
            short_step8,
        )
    else:
        raise ValueError(f"Unsupported signal type: {signal_type}")

    validated_stage5 = []
    for candidate in stage5_candidates:
        refreshed = validate_step5(candidate, get_resampled)
        if refreshed is None:
            remove_stage_candidate(signal_type, 5, candidate)
        else:
            validated_stage5.append(refreshed)

    if validated_stage5:
        _evaluate_transition(
            signal_type,
            5,
            validated_stage5,
            stage6_fn,
            label,
        )

    stage6_candidates = _waiting_transition_candidates(
        signal_type,
        6,
        get_resampled,
    )
    if stage6_candidates:
        _evaluate_transition(
            signal_type,
            6,
            stage6_candidates,
            stage7_fn,
            label,
        )

    stage7_candidates = _waiting_transition_candidates(
        signal_type,
        7,
        get_resampled,
    )
    if not stage7_candidates:
        return
    passed = _evaluate_transition(
        signal_type,
        7,
        stage7_candidates,
        stage8_fn,
        label,
    )
    _emit_signals(
        signal_type,
        label,
        passed,
        len(stage7_candidates),
    )


def _snapshot_quick_candidates():
    return {
        signal_type: {
            stage_num: get_stage_candidates(signal_type, stage_num)
            for stage_num in (5, 6, 7)
        }
        for signal_type in ("buy", "sell")
    }


def _refresh_quick_data(snapshot):
    all_candidates = [
        candidate
        for side_stages in snapshot.values()
        for candidates in side_stages.values()
        for candidate in candidates
    ]
    refresh_items = {
        (candidate["sym"], candidate["base_api"])
        for candidate in all_candidates
    }
    for candidate in all_candidates:
        higher_tf = NEXT_TF.get(candidate["base_frame"])
        if higher_tf is not None:
            refresh_items.add(
                (
                    candidate["sym"],
                    TF_TO_API.get(higher_tf, candidate["base_api"]),
                )
            )

    def fetch_tf(item):
        symbol, source_tf = item
        fetch_fn = (
            get_ohlcv_futures
            if MARKET_MODE == "futures"
            else get_ohlcv
        )
        fresh = fetch_fn(symbol, source_tf, limit=3)
        if not fresh.empty:
            cache_merge(symbol, source_tf, fresh)

    if refresh_items:
        with ThreadPoolExecutor(max_workers=20) as executor:
            list(executor.map(fetch_tf, refresh_items))


def quick_check_once():
    """Run one quick-check iteration; separated from the daemon loop for tests."""
    if not fast_prefetch_done.is_set():
        return

    with _ribbon_cache_lock:
        _ribbon_cache.clear()
    snapshot = _snapshot_quick_candidates()
    _refresh_quick_data(snapshot)
    _, get_resampled = _new_resampler()
    for signal_type in ("buy", "sell"):
        _advance_pipeline(
            signal_type,
            snapshot[signal_type][5],
            get_resampled,
        )
    touch_scan_times()


def quick_check_watcher():
    """Continuously advance saved stage-5/6/7 candidates."""
    while True:
        time.sleep(QUICK_CHECK_INTERVAL_SECONDS)
        try:
            quick_check_once()
        except Exception:  # pylint: disable=broad-exception-caught
            # Intentional daemon boundary: a later cycle must still run.
            log.exception("❌ خطأ في quick_check_watcher")
