"""Historical week scan: fetch strategy trades from market data (not storage).

`/week` replays the cascade on the last 7 days of OHLCV, then classifies each
entry by base-frame outcome levels:
  - 9m..27m   → win +0.67% / loss 0.51% against
  - 30m..240m → win +1.00% / loss 0.80% against
  - open      → neither level hit yet by the end of available data
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from html import escape as html_escape

import pandas as pd

from binance_data import (
    API_FETCH_CANDLES,
    CUSTOM_SYMBOLS,
    MARKET_MODE,
    cache_merge,
    fast_prefetch_done,
    get_cached,
    get_ohlcv_full,
    get_ohlcv_full_futures,
    symbols_cache,
    symbols_cache_lock,
)
from cascade_steps import (
    TRIPLING_PAIRS,
    short_step1,
    short_step2,
    short_step3,
    short_step4,
    short_step5,
    short_step6,
    short_step7,
    short_step8,
    step1,
    step2,
    step3,
    step4,
    step5,
    step6,
    step7,
    step8,
)
from indicators import (
    MIN_CANDLES,
    WARMUP_SMI,
    calc_smi,
    check_donchian_trend_ribbon,
    check_macd_at_saturation_start,
    check_macd_green,
    check_macd_red,
    find_step8_entry_index,
    resample_ohlcv_closed,
)
from state_manager import ALERT_EXPIRY_HOURS

log = logging.getLogger(__name__)

# Outcome levels by base timeframe (minutes).
SHORT_TF_MAX = 27
SHORT_WIN_PCT = 0.67
SHORT_LOSS_PCT = 0.51
LONG_WIN_PCT = 1.0
LONG_LOSS_PCT = 0.80

# Back-compat defaults = long-frame bucket (30m..240m).
WIN_PCT = LONG_WIN_PCT
LOSS_PCT = LONG_LOSS_PCT
WEEK_DAYS = 7
DEDUPE_HOURS = ALERT_EXPIRY_HOURS
MIN_1M_BARS = 20_000


def outcome_levels(base_frame):
    """Return (win_pct, loss_pct) for a base timeframe in minutes."""
    if base_frame is None:
        return LONG_WIN_PCT, LONG_LOSS_PCT
    try:
        frame = int(base_frame)
    except (TypeError, ValueError):
        return LONG_WIN_PCT, LONG_LOSS_PCT
    if frame <= SHORT_TF_MAX:
        return SHORT_WIN_PCT, SHORT_LOSS_PCT
    return LONG_WIN_PCT, LONG_LOSS_PCT

_week_scan_lock = threading.Lock()
_week_scan_running = False

_LONG_STEPS_1_5 = (step1, step2, step3, step4, step5)
_SHORT_STEPS_1_5 = (
    short_step1,
    short_step2,
    short_step3,
    short_step4,
    short_step5,
)


def _utc(ts):
    if ts is None:
        return None
    if getattr(ts, "to_pydatetime", None) is not None:
        ts = ts.to_pydatetime()
    if getattr(ts, "tzinfo", None) is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _candle_end(ts, minutes):
    return _utc(ts) + timedelta(minutes=int(minutes))


def _slice_closed(full_df, minutes, asof):
    """Return only candles whose period has fully closed by ``asof``."""
    if full_df is None or full_df.empty:
        return pd.DataFrame()
    asof_ts = pd.Timestamp(_utc(asof))
    ends = full_df["ts"] + pd.Timedelta(minutes=int(minutes))
    sliced = full_df.loc[ends <= asof_ts]
    if sliced.empty:
        return pd.DataFrame()
    return sliced.reset_index(drop=True)


def _ensure_symbol_raw(symbol):
    """Return raw 1m/30m/60m frames, fetching from Binance when cache is short."""
    raw = {
        "1m": get_cached(symbol, "1m"),
        "30m": get_cached(symbol, "30m"),
        "60m": get_cached(symbol, "60m"),
    }
    fetch_fn = (
        get_ohlcv_full_futures if MARKET_MODE == "futures" else get_ohlcv_full
    )
    needs_fetch = (
        raw["1m"].empty
        or len(raw["1m"]) < MIN_1M_BARS
        or raw["30m"].empty
        or raw["60m"].empty
    )
    if not needs_fetch:
        return raw

    for tf, target in API_FETCH_CANDLES.items():
        current = raw.get(tf)
        if current is not None and not current.empty and (
            tf != "1m" or len(current) >= MIN_1M_BARS
        ):
            continue
        df = fetch_fn(symbol, tf, target=target)
        if not df.empty:
            cache_merge(symbol, tf, df)
            raw[tf] = df
    return raw


def evaluate_outcome(
    signal_type,
    entry_price,
    future_1m,
    *,
    win_pct=WIN_PCT,
    loss_pct=LOSS_PCT,
):
    """
    Walk 1m bars after entry.

    Buy:  win at +win_pct, loss at -loss_pct.
    Sell: win at -win_pct, loss at +loss_pct.
    If both levels trade in the same bar, count loss (conservative).
    """
    if future_1m is None or future_1m.empty or entry_price <= 0:
        return "open", None, None

    if signal_type == "buy":
        tp = entry_price * (1.0 + win_pct / 100.0)
        sl = entry_price * (1.0 - loss_pct / 100.0)
        for row in future_1m.itertuples(index=False):
            hit_sl = float(row.low) <= sl
            hit_tp = float(row.high) >= tp
            if hit_sl:
                return "loss", sl, _utc(row.ts)
            if hit_tp:
                return "win", tp, _utc(row.ts)
    elif signal_type == "sell":
        tp = entry_price * (1.0 - win_pct / 100.0)
        sl = entry_price * (1.0 + loss_pct / 100.0)
        for row in future_1m.itertuples(index=False):
            hit_sl = float(row.high) >= sl
            hit_tp = float(row.low) <= tp
            if hit_sl:
                return "loss", sl, _utc(row.ts)
            if hit_tp:
                return "win", tp, _utc(row.ts)
    else:
        raise ValueError(f"Unsupported signal type: {signal_type}")

    return "open", None, None


def _stage5_still_valid(candidate, signal_type):
    """Mirror live stage-5 refresh using already-sliced frames."""
    df_base = candidate["df_base"]
    df_confirm = candidate["df_confirm"]
    if df_base.empty or df_confirm.empty or len(df_base) < MIN_CANDLES:
        return False

    smi, _, _ = calc_smi(df_base["high"], df_base["low"], df_base["close"])
    current_smi = float(smi.iloc[-1])
    variant = candidate.get("variant") if isinstance(candidate.get("variant"), dict) else {}

    # فحص MACD الأساسي مثبّت على أول شمعة إغلاق متشبعة (مطابق للمسار الحي).
    if signal_type == "buy":
        if current_smi > -40:
            return False
        base_macd_ok = check_macd_at_saturation_start(
            df_base,
            candidate["base_frame"],
            direction="long",
        )
        ribbon_direction = "green"
        confirm_macd_ok = check_macd_green(df_confirm)
        steps_1_5 = _LONG_STEPS_1_5
    else:
        if current_smi < 40:
            return False
        base_macd_ok = check_macd_at_saturation_start(
            df_base,
            candidate["base_frame"],
            direction="short",
        )
        ribbon_direction = "red"
        confirm_macd_ok = check_macd_red(df_confirm)
        steps_1_5 = _SHORT_STEPS_1_5

    # Higher-TF skip via step1 (uses candidate.get_raw + asof resampler).
    ok, _reason = steps_1_5[0](candidate)
    if not ok:
        return False
    if not base_macd_ok:
        return False
    if not check_donchian_trend_ribbon(
        df_base, ribbon_direction, cache_key=None
    ):
        return False
    if not variant.get("skip_donchian_confirm"):
        if not check_donchian_trend_ribbon(
            df_confirm, ribbon_direction, cache_key=None
        ):
            return False
    if not confirm_macd_ok:
        return False
    return True


def _passes_steps_1_5(candidate, signal_type):
    steps = _LONG_STEPS_1_5 if signal_type == "buy" else _SHORT_STEPS_1_5
    for step_fn in steps:
        ok, _reason = step_fn(candidate)
        if not ok:
            return False
    return True


def _try_step8_entry(candidate, signal_type):
    if signal_type == "buy":
        ok6, _ = step6(candidate)
        if not ok6:
            return None
        ok7, _ = step7(candidate)
        if not ok7:
            return None
        ok8, _ = step8(candidate)
        if not ok8:
            return None
        smi_threshold, rsi_threshold, direction = -40, 35, "long"
    else:
        ok6, _ = short_step6(candidate)
        if not ok6:
            return None
        ok7, _ = short_step7(candidate)
        if not ok7:
            return None
        ok8, _ = short_step8(candidate)
        if not ok8:
            return None
        smi_threshold, rsi_threshold, direction = 40, 65, "short"

    entry_index = find_step8_entry_index(
        candidate["df_triple"],
        candidate["ready_since"],
        smi_threshold=smi_threshold,
        rsi_threshold=rsi_threshold,
        direction=direction,
        max_gap=3,
    )
    if entry_index is None:
        return None

    candle = candidate["df_triple"].iloc[entry_index]
    return {
        "entry_ts": _utc(candle["ts"]),
        "price": float(candle["close"]),
        "triple_frame": candidate["triple_frame"],
    }


def _build_candidate(
    symbol,
    base_frame,
    confirm_frame,
    triple_frame,
    base_api,
    triple_api,
    df_base,
    df_confirm,
    df_triple,
    get_resampled,
    get_raw,
    ready_since=None,
    variant=None,
    df_btc_base=None,
):
    return {
        "sym": symbol,
        "base_api": base_api,
        "triple_api": triple_api,
        "base_frame": base_frame,
        "confirm_frame": confirm_frame,
        "triple_frame": triple_frame,
        "df_base": df_base,
        "df_confirm": df_confirm,
        "df_triple": df_triple,
        "get_resampled": get_resampled,
        "get_raw": get_raw,
        "ready_since": ready_since,
        "disable_ribbon_cache": True,
        "variant": variant or {},
        "df_btc_base": df_btc_base,
    }


def _scan_pair_side(
    symbol,
    pair,
    signal_type,
    raw_by_tf,
    start,
    end,
    raw_1m,
    variant=None,
    btc_raw_by_tf=None,
):
    """Replay one symbol × tripling pair × side; return signal dicts."""
    variant = variant or {}
    base_frame, confirm_frame, triple_frame, base_api, triple_api = pair
    raw_base = raw_by_tf.get(base_api, pd.DataFrame())
    raw_triple = raw_by_tf.get(triple_api, pd.DataFrame())
    if raw_base.empty or raw_triple.empty:
        return []

    df_base_full = resample_ohlcv_closed(raw_base, base_frame)
    df_confirm_full = resample_ohlcv_closed(raw_base, confirm_frame)
    df_triple_full = resample_ohlcv_closed(raw_triple, triple_frame)
    if (
        df_base_full.empty
        or df_confirm_full.empty
        or df_triple_full.empty
        or len(df_base_full) < MIN_CANDLES
    ):
        return []

    btc_base_full = None
    if variant.get("btc_corr_min") is not None and btc_raw_by_tf:
        btc_raw = btc_raw_by_tf.get(base_api, pd.DataFrame())
        if not btc_raw.empty:
            btc_base_full = resample_ohlcv_closed(btc_raw, base_frame)

    # asof-aware resampler closed over mutable tip time
    asof_box = [None]
    full_cache = {
        (base_api, base_frame): df_base_full,
        (base_api, confirm_frame): df_confirm_full,
        (triple_api, triple_frame): df_triple_full,
    }

    def get_raw(sym, tf):
        if sym != symbol:
            return pd.DataFrame()
        return raw_by_tf.get(tf, pd.DataFrame())

    def get_resampled(raw_df, sym, source_tf, minutes):
        key = (source_tf, int(minutes))
        if key not in full_cache:
            full_cache[key] = resample_ohlcv_closed(raw_df, minutes)
        return _slice_closed(full_cache[key], minutes, asof_box[0])

    def _btc_base_at(asof):
        if btc_base_full is None:
            return None
        return _slice_closed(btc_base_full, base_frame, asof)

    signals = []
    waiting = False
    ready_since = None
    last_emitted_entry_end = None
    start_i = max(MIN_CANDLES - 1, WARMUP_SMI)

    smi_full, _, _ = calc_smi(
        df_base_full["high"],
        df_base_full["low"],
        df_base_full["close"],
    )
    if signal_type == "buy":
        saturated = smi_full <= -40
    else:
        saturated = smi_full >= 40

    walk_start = start - timedelta(days=2)
    walk_end = end + timedelta(minutes=base_frame)

    def record_entry(entry, tip_asof):
        nonlocal waiting, ready_since, last_emitted_entry_end
        entry_end = _candle_end(entry["entry_ts"], triple_frame)
        if last_emitted_entry_end is not None and entry_end <= last_emitted_entry_end:
            return False
        if not (start <= entry["entry_ts"] < end and entry_end <= tip_asof):
            return False
        future = raw_1m.loc[raw_1m["ts"] > entry["entry_ts"]]
        win_pct, loss_pct = outcome_levels(base_frame)
        outcome, exit_price, exit_ts = evaluate_outcome(
            signal_type,
            entry["price"],
            future,
            win_pct=win_pct,
            loss_pct=loss_pct,
        )
        signals.append(
            {
                "symbol": symbol,
                "type": signal_type,
                "base_frame": base_frame,
                "confirm_frame": confirm_frame,
                "triple_frame": triple_frame,
                "time": entry["entry_ts"],
                "price": entry["price"],
                "outcome": outcome,
                "exit_price": exit_price,
                "exit_ts": exit_ts,
                "win_pct": win_pct,
                "loss_pct": loss_pct,
            }
        )
        last_emitted_entry_end = entry_end
        waiting = False
        ready_since = None
        return True

    i = start_i
    while i < len(df_base_full):
        tip_ts = _utc(df_base_full["ts"].iloc[i])
        asof = tip_ts + timedelta(minutes=base_frame)
        if asof > walk_end:
            break
        if asof < walk_start and not waiting:
            i += 1
            continue

        # Fast path: ignore non-saturated tips until a waiter exists.
        if not waiting and not bool(saturated.iloc[i]):
            i += 1
            continue

        asof_box[0] = asof
        df_base = _slice_closed(df_base_full, base_frame, asof)
        df_confirm = _slice_closed(df_confirm_full, confirm_frame, asof)
        df_triple = _slice_closed(df_triple_full, triple_frame, asof)
        if len(df_base) < MIN_CANDLES or df_confirm.empty or df_triple.empty:
            i += 1
            continue

        candidate = _build_candidate(
            symbol,
            base_frame,
            confirm_frame,
            triple_frame,
            base_api,
            triple_api,
            df_base,
            df_confirm,
            df_triple,
            get_resampled,
            get_raw,
            ready_since=ready_since,
            variant=variant,
            df_btc_base=_btc_base_at(asof),
        )

        if waiting:
            if not _stage5_still_valid(candidate, signal_type):
                waiting = False
                ready_since = None
                i += 1
                continue

            prev_asof = (
                _utc(df_base_full["ts"].iloc[i - 1]) + timedelta(minutes=base_frame)
                if i > 0
                else asof - timedelta(minutes=base_frame)
            )
            triple_close_ends = df_triple_full["ts"] + pd.Timedelta(
                minutes=triple_frame
            )
            mid_mask = (triple_close_ends > prev_asof) & (triple_close_ends <= asof)
            eval_asofs = sorted(
                {_utc(ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts)
                 for ts in triple_close_ends.loc[mid_mask]}
                | {asof}
            )

            for tip_asof in eval_asofs:
                asof_box[0] = tip_asof
                tip_candidate = _build_candidate(
                    symbol,
                    base_frame,
                    confirm_frame,
                    triple_frame,
                    base_api,
                    triple_api,
                    _slice_closed(df_base_full, base_frame, tip_asof),
                    _slice_closed(df_confirm_full, confirm_frame, tip_asof),
                    _slice_closed(df_triple_full, triple_frame, tip_asof),
                    get_resampled,
                    get_raw,
                    ready_since=ready_since,
                    variant=variant,
                    df_btc_base=_btc_base_at(tip_asof),
                )
                if (
                    tip_candidate["df_base"].empty
                    or tip_candidate["df_triple"].empty
                ):
                    continue
                if not _stage5_still_valid(tip_candidate, signal_type):
                    waiting = False
                    ready_since = None
                    break
                entry = _try_step8_entry(tip_candidate, signal_type)
                if entry is not None and record_entry(entry, tip_asof):
                    break
            i += 1
            continue

        if not _passes_steps_1_5(candidate, signal_type):
            i += 1
            continue

        ready_since = _utc(df_base["ts"].iloc[-1])
        waiting = True
        candidate["ready_since"] = ready_since
        entry = _try_step8_entry(candidate, signal_type)
        if entry is not None:
            record_entry(entry, asof)
        i += 1

    return signals


def _dedupe_signals(signals, hours=DEDUPE_HOURS):
    """Keep earliest signal per (symbol, frames, side) within ``hours``."""
    signals = sorted(signals, key=lambda item: item["time"])
    kept = []
    last_by_key = {}
    window = timedelta(hours=hours)
    for sig in signals:
        key = (
            sig["symbol"],
            sig["base_frame"],
            sig["confirm_frame"],
            sig["triple_frame"],
            sig["type"],
        )
        prev = last_by_key.get(key)
        if prev is not None and sig["time"] - prev < window:
            continue
        last_by_key[key] = sig["time"]
        kept.append(sig)
    return kept


def scan_week_trades(
    symbols=None,
    *,
    days=WEEK_DAYS,
    now=None,
    progress_callback=None,
    variant=None,
    preloaded_raw=None,
    btc_raw_by_tf=None,
):
    """
    Fetch/replay strategy trades for the last ``days`` and classify outcomes.

    ``variant`` is an optional experiment override dict (see strategy_variants).
    ``preloaded_raw`` maps symbol -> raw_by_tf to avoid re-fetching across experiments.
    """
    now = _utc(now) or datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    end = now
    variant = variant or {}

    if symbols is None:
        with symbols_cache_lock:
            symbols = list(symbols_cache) or list(CUSTOM_SYMBOLS)
    else:
        symbols = list(symbols)

    if not symbols:
        return {
            "ready": False,
            "reason": "no_symbols",
            "start": start,
            "end": end,
            "symbols_scanned": 0,
            "wins": [],
            "losses": [],
            "opens": [],
            "total": 0,
            "variant": variant,
        }

    needs_btc = variant.get("btc_corr_min") is not None
    if needs_btc and btc_raw_by_tf is None:
        btc_raw_by_tf = _ensure_symbol_raw("BTCUSDT")

    all_signals = []
    total = len(symbols)
    for index, symbol in enumerate(symbols, start=1):
        if progress_callback is not None and (
            index == 1 or index == total or index % 10 == 0
        ):
            try:
                progress_callback(index, total, symbol)
            except Exception:  # pragma: no cover - progress must not abort scan
                log.exception("week scan progress callback failed")

        try:
            if preloaded_raw is not None and symbol in preloaded_raw:
                raw_by_tf = preloaded_raw[symbol]
            else:
                raw_by_tf = _ensure_symbol_raw(symbol)
            raw_1m = raw_by_tf.get("1m", pd.DataFrame())
            if raw_1m.empty:
                continue
            for pair in TRIPLING_PAIRS:
                for signal_type in ("buy", "sell"):
                    all_signals.extend(
                        _scan_pair_side(
                            symbol,
                            pair,
                            signal_type,
                            raw_by_tf,
                            start,
                            end,
                            raw_1m,
                            variant=variant,
                            btc_raw_by_tf=btc_raw_by_tf,
                        )
                    )
        except Exception:
            log.exception("week scan failed for %s", symbol)

    deduped = _dedupe_signals(all_signals)
    wins = [s for s in deduped if s["outcome"] == "win"]
    losses = [s for s in deduped if s["outcome"] == "loss"]
    opens = [s for s in deduped if s["outcome"] == "open"]
    wins.sort(key=lambda item: item["time"])
    losses.sort(key=lambda item: item["time"])
    opens.sort(key=lambda item: item["time"])

    return {
        "ready": True,
        "start": start,
        "end": end,
        "symbols_scanned": total,
        "wins": wins,
        "losses": losses,
        "opens": opens,
        "total": len(deduped),
        "variant": variant,
    }


def _format_trade_line(trade):
    icon = "🟢" if trade["type"] == "buy" else "🔴"
    side = "شراء" if trade["type"] == "buy" else "بيع"
    frames = (
        f"{trade['base_frame']}m/"
        f"{trade['confirm_frame']}m/"
        f"{trade['triple_frame']}m"
    )
    when = trade["time"].strftime("%m-%d %H:%M")
    return (
        f"{icon} <code>{html_escape(trade['symbol'])}</code> | {side} | "
        f"{frames} | {trade['price']:.4g} | <code>{when}</code> UTC"
    )


def format_week_trades_report(result):
    """Format scan result into Telegram HTML chunks."""
    if not result.get("ready"):
        reason = result.get("reason")
        if reason == "no_symbols":
            return ["⚠️ لا توجد عملات للفحص."]
        if reason == "busy":
            return ["⏳ فحص الأسبوع الماضي يعمل الآن — انتظر انتهاءه."]
        if reason == "not_ready":
            return [
                "⏳ بيانات السوق لسه تتحمل.\n"
                "بعد اكتمال التحميل السريع أعد <code>/week</code>."
            ]
        return ["⚠️ تعذر جلب صفقات الأسبوع الماضي."]

    wins = result.get("wins") or []
    losses = result.get("losses") or []
    opens = result.get("opens") or []
    start = result["start"].strftime("%Y-%m-%d %H:%M")
    end = result["end"].strftime("%Y-%m-%d %H:%M")
    total = int(result.get("total") or 0)
    levels_note = (
        f"• 9–{SHORT_TF_MAX}م: ربح +{SHORT_WIN_PCT:g}% | "
        f"خسارة {SHORT_LOSS_PCT:g}%\n"
        f"• 30–240م: ربح +{LONG_WIN_PCT:g}% | "
        f"خسارة {LONG_LOSS_PCT:g}%"
    )

    header = (
        "🗓️ <b>صفقات الاستراتيجية — آخر 7 أيام</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"الفترة: <code>{start}</code> → <code>{end}</code> UTC\n"
        f"عملات مفحوصة: <b>{result.get('symbols_scanned', 0)}</b>\n"
        f"إجمالي الصفقات: <b>{total}</b>\n"
        f"✅ نجاح: <b>{len(wins)}</b>\n"
        f"❌ خسارة: <b>{len(losses)}</b>\n"
        f"⏳ مفتوحة: <b>{len(opens)}</b>\n"
        f"معايير الخروج:\n{levels_note}\n"
    )

    if total == 0:
        return [
            header
            + "\nلا توجد صفقات مطابقة للاستراتيجية خلال الأسبوع الماضي."
        ]

    chunks = [header]

    win_block = ["✅ <b>الناجحون</b>:"]
    if wins:
        win_block.extend(_format_trade_line(t) for t in wins)
    else:
        win_block.append("— لا يوجد")
    chunks.append("\n".join(win_block))

    loss_block = ["❌ <b>الخاسرون</b>:"]
    if losses:
        loss_block.extend(_format_trade_line(t) for t in losses)
    else:
        loss_block.append("— لا يوجد")
    chunks.append("\n".join(loss_block))

    if opens:
        open_block = ["⏳ <b>مفتوحة</b> (لم يصل هدف الربح ولا وقف الخسارة بعد):"]
        open_block.extend(_format_trade_line(t) for t in opens)
        chunks.append("\n".join(open_block))

    # Pack to Telegram-safe sizes (~3500).
    packed = []
    current = ""
    for block in chunks:
        if not current:
            current = block
            continue
        if len(current) + 2 + len(block) > 3500:
            packed.append(current)
            current = block
        else:
            current = current + "\n\n" + block
    if current:
        packed.append(current)
    return packed


def handle_week_command(chat_id, send_telegram):
    """Telegram entry: scan last week from market data and report win/loss."""
    global _week_scan_running

    if not _week_scan_lock.acquire(blocking=False):
        send_telegram(
            "⏳ فحص الأسبوع الماضي يعمل الآن — انتظر انتهاءه.",
            chat_id,
        )
        return
    if _week_scan_running:
        _week_scan_lock.release()
        send_telegram(
            "⏳ فحص الأسبوع الماضي يعمل الآن — انتظر انتهاءه.",
            chat_id,
        )
        return

    _week_scan_running = True
    try:
        if not fast_prefetch_done.is_set():
            # Still allow fetch-on-demand, but warn that first run may be slow.
            send_telegram(
                "📡 جاري جلب بيانات الأسبوع الماضي من Binance "
                "(التحميل الأولي لم يكتمل بعد — قد يأخذ وقت أطول)...",
                chat_id,
            )
        else:
            send_telegram(
                "📡 جاري جلب صفقات الاستراتيجية الأساسية لآخر 7 أيام...\n"
                f"• 9–{SHORT_TF_MAX}م: ربح <b>+{SHORT_WIN_PCT:g}%</b> | "
                f"خسارة <b>{SHORT_LOSS_PCT:g}%</b>\n"
                f"• 30–240م: ربح <b>+{LONG_WIN_PCT:g}%</b> | "
                f"خسارة <b>{LONG_LOSS_PCT:g}%</b>",
                chat_id,
            )

        def on_progress(index, total, symbol):
            if index in (1, total) or index % 25 == 0:
                send_telegram(
                    f"⏳ فحص الأسبوع: {index}/{total} "
                    f"(<code>{html_escape(symbol)}</code>)...",
                    chat_id,
                )

        result = scan_week_trades(progress_callback=on_progress)
        for chunk in format_week_trades_report(result):
            send_telegram(chunk, chat_id)
    except Exception as exc:
        log.exception("week command failed")
        send_telegram(
            f"❌ فشل جلب صفقات الأسبوع الماضي: "
            f"<code>{html_escape(str(exc))}</code>",
            chat_id,
        )
    finally:
        _week_scan_running = False
        _week_scan_lock.release()
