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
    WARMUP_SMI,
    _ribbon_cache,
    _ribbon_cache_lock,
    calc_smi,
    check_donchian_trend_ribbon,
    check_macd_green,
    check_macd_line_long,
    check_macd_line_short,
    check_macd_red,
    find_step8_entry_index,
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
    abandon_waiting_candidate,
    complete_scan,
    get_stage_candidates,
    get_step1_ready_since,
    mark_stage_ready,
    record_scan_step,
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


def _classify_broken_frame(
    raw_by_tf,
    symbol,
    base_frame,
    confirm_frame,
    triple_frame,
    base_api,
    triple_api,
    get_resampled,
):
    """Return a broken-frame record when a tripling pair cannot build, else None."""
    raw_base = raw_by_tf.get(base_api, pd.DataFrame())
    raw_triple = raw_by_tf.get(triple_api, pd.DataFrame())

    if raw_base is None or getattr(raw_base, "empty", True):
        return {
            "base_frame": base_frame,
            "confirm_frame": confirm_frame,
            "triple_frame": triple_frame,
            "base_api": base_api,
            "triple_api": triple_api,
            "reason": "missing_raw_base",
            "detail": f"بيانات المصدر {base_api} ناقصة",
            "candle_count": 0,
        }

    if raw_triple is None or getattr(raw_triple, "empty", True):
        return {
            "base_frame": base_frame,
            "confirm_frame": confirm_frame,
            "triple_frame": triple_frame,
            "base_api": base_api,
            "triple_api": triple_api,
            "reason": "missing_raw_triple",
            "detail": f"بيانات مصدر التثليث {triple_api} ناقصة",
            "candle_count": 0,
        }

    df_base = get_resampled(raw_base, symbol, base_api, base_frame)
    if df_base is None or getattr(df_base, "empty", True):
        return {
            "base_frame": base_frame,
            "confirm_frame": confirm_frame,
            "triple_frame": triple_frame,
            "base_api": base_api,
            "triple_api": triple_api,
            "reason": "empty_resample_base",
            "detail": f"إعادة العينة للفريم الأساسي {base_frame}m فارغة",
            "candle_count": 0,
        }

    df_confirm = get_resampled(raw_base, symbol, base_api, confirm_frame)
    if df_confirm is None or getattr(df_confirm, "empty", True):
        return {
            "base_frame": base_frame,
            "confirm_frame": confirm_frame,
            "triple_frame": triple_frame,
            "base_api": base_api,
            "triple_api": triple_api,
            "reason": "empty_resample_confirm",
            "detail": f"إعادة العينة لفريم التأكيد {confirm_frame}m فارغة",
            "candle_count": len(df_base),
        }

    df_triple = get_resampled(raw_triple, symbol, triple_api, triple_frame)
    if df_triple is None or getattr(df_triple, "empty", True):
        return {
            "base_frame": base_frame,
            "confirm_frame": confirm_frame,
            "triple_frame": triple_frame,
            "base_api": base_api,
            "triple_api": triple_api,
            "reason": "empty_resample_triple",
            "detail": f"إعادة العينة لفريم التثليث {triple_frame}m فارغة",
            "candle_count": len(df_base),
        }

    candle_count = len(df_base)
    if candle_count < MIN_CANDLES:
        return {
            "base_frame": base_frame,
            "confirm_frame": confirm_frame,
            "triple_frame": triple_frame,
            "base_api": base_api,
            "triple_api": triple_api,
            "reason": "min_candles",
            "detail": f"شموع غير كافية على الأساسي ({candle_count}/{MIN_CANDLES})",
            "candle_count": candle_count,
        }

    return None


def audit_broken_frames(symbols=None):
    """
    Inspect every symbol × TRIPLING_PAIRS entry and classify frames as ok/broken.

    Mirrors the silent skips in `_build_tripling_candidates` so the report matches
    what the scanner actually drops. Returns both healthy and broken frames per
    symbol so a full ~100-coin audit can show صالح vs معطوب clearly.
    """
    if not fast_prefetch_done.is_set():
        return {
            "ready": False,
            "symbols_checked": 0,
            "total_pairs": len(TRIPLING_PAIRS),
            "broken_by_symbol": {},
            "ok_frames_by_symbol": {},
            "ok_symbols": [],
            "broken_frame_count": 0,
            "ok_frame_count": 0,
        }

    if symbols is None:
        with symbols_cache_lock:
            symbols = list(symbols_cache)
    else:
        symbols = list(symbols)

    _, get_resampled = _new_resampler()
    broken_by_symbol = {}
    ok_frames_by_symbol = {}
    ok_symbols = []

    for symbol in symbols:
        raw_by_tf = {
            "1m": get_cached(symbol, "1m"),
            "30m": get_cached(symbol, "30m"),
            "60m": get_cached(symbol, "60m"),
        }
        broken = []
        ok_frames = []
        for (
            base_frame,
            confirm_frame,
            triple_frame,
            base_api,
            triple_api,
        ) in TRIPLING_PAIRS:
            issue = _classify_broken_frame(
                raw_by_tf,
                symbol,
                base_frame,
                confirm_frame,
                triple_frame,
                base_api,
                triple_api,
                get_resampled,
            )
            if issue is not None:
                broken.append(issue)
            else:
                ok_frames.append(
                    {
                        "base_frame": base_frame,
                        "confirm_frame": confirm_frame,
                        "triple_frame": triple_frame,
                        "base_api": base_api,
                        "triple_api": triple_api,
                    }
                )

        ok_frames_by_symbol[symbol] = ok_frames
        if broken:
            broken_by_symbol[symbol] = broken
        else:
            ok_symbols.append(symbol)

    return {
        "ready": True,
        "symbols_checked": len(symbols),
        "total_pairs": len(TRIPLING_PAIRS),
        "broken_by_symbol": broken_by_symbol,
        "ok_frames_by_symbol": ok_frames_by_symbol,
        "ok_symbols": ok_symbols,
        "broken_frame_count": sum(
            len(items) for items in broken_by_symbol.values()
        ),
        "ok_frame_count": sum(
            len(items) for items in ok_frames_by_symbol.values()
        ),
    }


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
            abandon_waiting_candidate(signal_type, candidate)
        else:
            refreshed.append(candidate2)
    return refreshed


def _base_smi_still_saturated(candidate, signal_type):
    """True only while the main-TF SMI episode is still active."""
    df_base = candidate.get("df_base")
    if df_base is None or getattr(df_base, "empty", True):
        return False
    if len(df_base) < WARMUP_SMI:
        return False
    smi, _, _ = calc_smi(df_base["high"], df_base["low"], df_base["close"])
    current_smi = float(smi.iloc[-1])
    if signal_type == "buy":
        return current_smi <= -40
    if signal_type == "sell":
        return current_smi >= 40
    raise ValueError(f"Unsupported signal type: {signal_type}")


def _filter_base_saturation(signal_type, candidates):
    """Drop waiters when main-TF SMI saturation ends — all stages, not only 5."""
    filtered = []
    for candidate in candidates:
        if not _base_smi_still_saturated(candidate, signal_type):
            abandon_waiting_candidate(signal_type, candidate)
            log.info(
                "⛔ %s %s/%s/%s: انتهى تشبع الفريم الرئيسي — أُلغي من الانتظار",
                candidate.get("sym"),
                candidate.get("base_frame"),
                candidate.get("confirm_frame"),
                candidate.get("triple_frame"),
            )
        else:
            filtered.append(candidate)
    return filtered


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
            abandon_waiting_candidate(signal_type, candidate)
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
        # أخرج الناجحين من المرحلة 7 حتى لا يُعاد تقييمهم بشمعة لاحقة.
        _promote_candidates(signal_type, from_stage, to_stage, passed)
        _set_step8_survivors(signal_type, passed)
    return passed


def _waiting_transition_candidates(signal_type, stage_num, get_resampled):
    refreshed = _refresh_stage(signal_type, stage_num, get_resampled)
    still_saturated = _filter_base_saturation(signal_type, refreshed)
    return _filter_higher_saturation(
        signal_type,
        stage_num,
        still_saturated,
        get_resampled,
    )


def _resolve_entry_signal_candle(candidate, signal_type):
    """
    شمعة التحقق على فريم الدخول (الثُلث): سعر إغلاقها ووقت فتحها.

    الترتيب: تشبع SMI أولًا ثم RSI/Stoch. لا نستخدم آخر شمعة ولا df_base.
    """
    entry_frame = candidate.get("df_triple")
    if entry_frame is None or entry_frame.empty:
        raise ValueError("Candidate is missing entry timeframe data")

    since_ts = get_step1_ready_since(
        candidate["sym"],
        candidate["base_frame"],
        candidate["confirm_frame"],
        candidate["triple_frame"],
        signal_type,
    )
    if signal_type == "buy":
        smi_threshold, rsi_threshold, direction = -40, 35, "long"
    elif signal_type == "sell":
        smi_threshold, rsi_threshold, direction = 40, 65, "short"
    else:
        raise ValueError(f"Unsupported signal type: {signal_type}")

    entry_index = find_step8_entry_index(
        entry_frame,
        since_ts,
        smi_threshold=smi_threshold,
        rsi_threshold=rsi_threshold,
        direction=direction,
        max_gap=3,
    )
    if entry_index is None:
        entry_index = len(entry_frame) - 1

    candle = entry_frame.iloc[entry_index]
    candle_ts = candle["ts"]
    if getattr(candle_ts, "to_pydatetime", None) is not None:
        candle_ts = candle_ts.to_pydatetime()
    return entry_frame, float(candle["close"]), candle_ts


def _emit_signals(signal_type, label, passed, evaluated_count):
    if not passed:
        return
    if _signal_handler is None:
        raise RuntimeError("Cascade signal handler is not configured")
    for candidate in passed:
        entry_frame, price, candle_ts = _resolve_entry_signal_candle(
            candidate,
            signal_type,
        )
        _signal_handler(
            candidate["sym"],
            candidate["base_frame"],
            candidate["confirm_frame"],
            candidate["triple_frame"],
            entry_frame,
            signal_type=signal_type,
            price=price,
            candle_ts=candle_ts,
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
            abandon_waiting_candidate(signal_type, candidate)
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
        refresh_items.add((candidate["sym"], candidate["triple_api"]))
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
