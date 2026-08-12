"""SMI reverse-sat + Donchian confirm (no EMA60, no RSI).

Main SMI sat owns the side (largest main cancels smaller).
Reverse SMI sat must print on the raised confirm range; the stop
frame blocks. Donchian on the 3× main TF must be green (buy) / red
(sell). After reverse sat forms, the entry TF waits until Donchian
is unmet, then enters on the first later candle where it holds.

Raised reverse-sat floors (old 45m floor was 8 → now 10, etc.).
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from html import escape as html_escape
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from indicators import (
    DONCHIAN_DLEN,
    WARMUP_SMI,
    calc_donchian_trend_series,
    calc_smi,
    resample_ohlcv_closed,
)
from pullback_bot.strategy import (
    DEDUPE_HOURS,
    MONTH_DAYS,
    SMI_BUY,
    SMI_SELL,
    _bool_step,
    _utc,
    evaluate_outcome,
    fetch_1m_vision,
)

log = logging.getLogger(__name__)

SYMBOLS = ("BTCUSDT", "ETHUSDT", "XRPUSDT")
# main, reverse_min, reverse_stop, entry, don_tf, win_pct, loss_pct, group
# group "a" = +0.50/−0.37 ; group "b" = +0.67/−0.54
LEVELS = (
    (45, 10, 18, 6, 135, 0.50, 0.37, "a"),
    (60, 12, 24, 5, 180, 0.50, 0.37, "a"),
    (90, 18, 36, 8, 270, 0.50, 0.37, "a"),
    (120, 25, 48, 10, 360, 0.50, 0.37, "a"),
    (150, 30, 60, 13, 450, 0.50, 0.37, "a"),
    (180, 35, 72, 15, 540, 0.50, 0.37, "a"),
    (90, 18, 36, 9, 270, 0.67, 0.54, "b"),
    (120, 25, 48, 10, 360, 0.67, 0.54, "b"),
    (150, 30, 60, 11, 450, 0.67, 0.54, "b"),
)
MAINS = (45, 60, 90, 120, 150, 180)
# 100 SMI bars on 540m ≈ 37d warmup + 30d scan
MIN_1M_BARS = 110_000


def _all_needed_frames():
    frames = set(MAINS)
    for main, reverse_min, reverse_stop, entry, don_tf, _win, _loss, _group in LEVELS:
        frames.add(main)
        frames.add(entry)
        frames.add(don_tf)
        frames.add(reverse_stop)
        for minutes in range(reverse_min, reverse_stop):
            frames.add(minutes)
    return sorted(frames)


def _smi_don_features(df_1m, minutes):
    """Resample and compute SMI saturation + Donchian ribbon (no EMA/RSI)."""
    df = resample_ohlcv_closed(df_1m, minutes)
    if df.empty or len(df) < max(WARMUP_SMI, DONCHIAN_DLEN + 2):
        return None

    smi, _, _ = calc_smi(df["high"], df["low"], df["close"])
    don = calc_donchian_trend_series(
        df["close"].to_numpy(),
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        DONCHIAN_DLEN,
    )
    end_ts = df["ts"] + pd.Timedelta(minutes=minutes)
    don_arr = don.to_numpy()
    return pd.DataFrame(
        {
            "ts": df["ts"].to_numpy(),
            "end_ts": end_ts.to_numpy(),
            "close": df["close"].to_numpy(),
            "smi": smi.to_numpy(),
            "sell_sat": (smi <= SMI_SELL).to_numpy(),
            "buy_sat": (smi >= SMI_BUY).to_numpy(),
            "don_green": don_arr == 1,
            "don_red": don_arr == -1,
        }
    )


def _precompute_stepped(frame_data, grid):
    stepped = {}
    for minutes, feat in frame_data.items():
        if feat is None:
            continue
        ends = feat["end_ts"].to_numpy()
        stepped[minutes] = {
            "sell_sat": _bool_step(ends, feat["sell_sat"].to_numpy(), grid),
            "buy_sat": _bool_step(ends, feat["buy_sat"].to_numpy(), grid),
            "don_green": _bool_step(ends, feat["don_green"].to_numpy(), grid),
            "don_red": _bool_step(ends, feat["don_red"].to_numpy(), grid),
        }
    return stepped


def _dedupe_signals(signals, hours=DEDUPE_HOURS):
    """Dedupe per symbol/side/main/entry/TP-SL so group A and B stay distinct."""
    if not signals:
        return []
    ordered = sorted(signals, key=lambda item: item["time"])
    window = timedelta(hours=hours)
    last_by_key = {}
    kept = []
    for sig in ordered:
        key = (
            sig["symbol"],
            sig["type"],
            sig["base_frame"],
            sig["triple_frame"],
            sig["win_pct"],
            sig["loss_pct"],
        )
        prev = last_by_key.get(key)
        if prev is not None and sig["time"] - prev < window:
            continue
        last_by_key[key] = sig["time"]
        kept.append(sig)
    return kept


def _or_sat(stepped, minutes_iter, key, n_grid):
    out = np.zeros(n_grid, dtype=bool)
    for minutes in minutes_iter:
        feat = stepped.get(minutes)
        if feat is None:
            continue
        out |= feat[key]
    return out


def _scan_side(side, stepped, entry_data, grid, start, end, raw_1m, symbol):
    """Main SMI sat + raised reverse sat + 3× Donchian + entry Donchian flip."""
    is_sell = side == "sell"
    sat_key = "sell_sat" if is_sell else "buy_sat"
    reverse_key = "buy_sat" if is_sell else "sell_sat"
    don_key = "don_red" if is_sell else "don_green"
    n_grid = len(grid)

    active = np.full(n_grid, -1, dtype=int)
    main_masks = []
    for main in MAINS:
        feat = stepped.get(main)
        main_masks.append(
            feat[sat_key] if feat is not None else np.zeros(n_grid, dtype=bool)
        )
    stacked = np.vstack(main_masks)
    for i in range(n_grid):
        chosen = -1
        for idx in range(len(MAINS)):
            if stacked[idx, i]:
                chosen = idx
        active[i] = chosen

    variants = defaultdict(list)
    for level in LEVELS:
        variants[level[0]].append(level)

    signals = []
    start_ts = pd.Timestamp(start)
    end_ts_limit = pd.Timestamp(end)
    idle, wait_clear, armed = 0, 1, 2
    for main_idx, main in enumerate(MAINS):
        for (
            _main,
            reverse_min,
            reverse_stop,
            entry,
            don_tf,
            win_pct,
            loss_pct,
            group,
        ) in variants[main]:
            entry_df = entry_data.get(entry)
            if entry_df is None:
                continue

            reverse_any = _or_sat(
                stepped, range(reverse_min, reverse_stop), reverse_key, n_grid
            )
            stop_feat = stepped.get(reverse_stop)
            stop_mask = (
                stop_feat[reverse_key]
                if stop_feat is not None
                else np.zeros(n_grid, dtype=bool)
            )
            don_feat = stepped.get(don_tf)
            don_ok = (
                don_feat[don_key]
                if don_feat is not None
                else np.zeros(n_grid, dtype=bool)
            )
            owned = (active == main_idx) & ~stop_mask & don_ok
            confirm_window = owned & reverse_any
            hold_window = owned

            ends = pd.DatetimeIndex(pd.to_datetime(entry_df["end_ts"], utc=True))
            state = idle
            for row_i, candle_end in enumerate(ends):
                if candle_end < start_ts or candle_end > end_ts_limit:
                    state = idle
                    continue
                pos = int(grid.searchsorted(candle_end, side="right") - 1)
                if pos < 0 or not hold_window[pos]:
                    state = idle
                    continue

                green = bool(entry_df["don_green"].iloc[row_i])
                red = bool(entry_df["don_red"].iloc[row_i])
                price = float(entry_df["close"].iloc[row_i])
                holds = red if is_sell else green
                cleared = (not red) if is_sell else (not green)
                counter_now = bool(confirm_window[pos])

                if state == idle and counter_now:
                    state = armed if cleared else wait_clear
                elif state == wait_clear and cleared:
                    state = armed

                if state != armed or not holds:
                    continue

                future = raw_1m.loc[raw_1m["ts"] > candle_end]
                outcome, exit_price, exit_ts = evaluate_outcome(
                    "sell" if is_sell else "buy",
                    price,
                    future,
                    win_pct=win_pct,
                    loss_pct=loss_pct,
                )
                signals.append(
                    {
                        "symbol": symbol,
                        "type": "sell" if is_sell else "buy",
                        "time": _utc(candle_end),
                        "price": price,
                        "base_frame": main,
                        "confirm_frame": don_tf,
                        "reverse_min": reverse_min,
                        "reverse_stop": reverse_stop,
                        "triple_frame": entry,
                        "win_pct": win_pct,
                        "loss_pct": loss_pct,
                        "group": group,
                        "outcome": outcome,
                        "exit_price": exit_price,
                        "exit_ts": exit_ts,
                    }
                )
                state = idle
    return signals


def scan_symbol(symbol, *, days=MONTH_DAYS, now=None, raw_1m=None):
    """Scan one symbol over the last ``days`` with raised reverse-sat floors."""
    now = _utc(now) or datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    end = now
    empty = {
        "ready": False,
        "reason": "no_data",
        "start": start,
        "end": end,
        "days": int(days),
        "wins": [],
        "losses": [],
        "opens": [],
        "total": 0,
        "market": "spot-vision",
        "symbol": symbol,
    }

    if raw_1m is None:
        target = max(MIN_1M_BARS, int(days) * 1440 + 55_000)
        log.info("Fetching %s 1m bars for %s...", target, symbol)
        raw_1m = fetch_1m_vision(symbol, target=target)
        log.info("%s bars: %s", symbol, 0 if raw_1m is None else len(raw_1m))
    if raw_1m is None or raw_1m.empty:
        return empty

    raw_1m = raw_1m.sort_values("ts").reset_index(drop=True)
    closes_1m = raw_1m["ts"] + pd.Timedelta(minutes=1)
    grid_mask = (closes_1m >= pd.Timestamp(start)) & (closes_1m <= pd.Timestamp(end))
    grid = pd.DatetimeIndex(closes_1m.loc[grid_mask].to_numpy())
    if len(grid) == 0:
        empty["ready"] = True
        empty["reason"] = None
        return empty

    needed = _all_needed_frames()
    entry_frames = {lvl[3] for lvl in LEVELS}
    log.info("%s: computing %s frames...", symbol, len(needed))
    frame_data = {}
    entry_data = {}
    for minutes in needed:
        feat = _smi_don_features(raw_1m, minutes)
        frame_data[minutes] = feat
        if minutes in entry_frames and feat is not None:
            entry_data[minutes] = feat

    stepped = _precompute_stepped(frame_data, grid)
    all_signals = []
    all_signals.extend(
        _scan_side("sell", stepped, entry_data, grid, start, end, raw_1m, symbol)
    )
    all_signals.extend(
        _scan_side("buy", stepped, entry_data, grid, start, end, raw_1m, symbol)
    )

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
        "days": int(days),
        "wins": wins,
        "losses": losses,
        "opens": opens,
        "total": len(deduped),
        "market": "spot-vision",
        "symbol": symbol,
        "symbols_scanned": 1,
    }


def scan_all(symbols=SYMBOLS, *, days=MONTH_DAYS, now=None, raw_by_symbol=None):
    """Scan BTC/ETH/XRP (or ``symbols``) and merge trades."""
    now = _utc(now) or datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    end = now
    raw_by_symbol = raw_by_symbol or {}
    merged = []
    per_symbol = {}
    failed = []
    for symbol in symbols:
        result = scan_symbol(
            symbol,
            days=days,
            now=now,
            raw_1m=raw_by_symbol.get(symbol),
        )
        per_symbol[symbol] = result
        if not result.get("ready"):
            failed.append(symbol)
            continue
        merged.extend(result["wins"])
        merged.extend(result["losses"])
        merged.extend(result["opens"])

    deduped = _dedupe_signals(merged, hours=DEDUPE_HOURS)
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
        "days": int(days),
        "wins": wins,
        "losses": losses,
        "opens": opens,
        "total": len(deduped),
        "market": "spot-vision",
        "symbols": list(symbols),
        "per_symbol": per_symbol,
        "failed": failed,
        "symbols_scanned": len(symbols) - len(failed),
    }


def _pnl_r(trade):
    if trade["outcome"] == "win":
        return float(trade["win_pct"])
    if trade["outcome"] == "loss":
        return -float(trade["loss_pct"])
    return 0.0


def _summarize(trades):
    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    opens = [t for t in trades if t["outcome"] == "open"]
    closed = len(wins) + len(losses)
    pnl = sum(_pnl_r(t) for t in trades)
    win_rate = (100.0 * len(wins) / closed) if closed else 0.0
    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "opens": opens,
        "total": len(trades),
        "closed": closed,
        "win_rate": win_rate,
        "pnl": pnl,
    }


def _level_key(level):
    main, _rmin, _rstop, entry, _don, win_pct, loss_pct, group = level
    return (main, entry, win_pct, loss_pct, group)


def group_results(result):
    """Split merged trades by TP/SL group and by symbol/level."""
    all_trades = list(result.get("wins") or [])
    all_trades.extend(result.get("losses") or [])
    all_trades.extend(result.get("opens") or [])
    all_trades.sort(key=lambda item: item["time"])

    group_a = [t for t in all_trades if t.get("group") == "a"]
    group_b = [t for t in all_trades if t.get("group") == "b"]
    by_level = {}
    for level in LEVELS:
        key = _level_key(level)
        main, reverse_min, reverse_stop, entry, don_tf, win_pct, loss_pct, group = level
        by_level[key] = _summarize(
            [
                t
                for t in all_trades
                if t["base_frame"] == main
                and t["triple_frame"] == entry
                and t.get("group") == group
            ]
        )
        by_level[key]["reverse_min"] = reverse_min
        by_level[key]["reverse_stop"] = reverse_stop
        by_level[key]["don_tf"] = don_tf
        by_level[key]["win_pct"] = win_pct
        by_level[key]["loss_pct"] = loss_pct
    by_symbol = {}
    for symbol in result.get("symbols") or SYMBOLS:
        by_symbol[symbol] = _summarize(
            [t for t in all_trades if t["symbol"] == symbol]
        )
    return {
        "all": _summarize(all_trades),
        "group_a": _summarize(group_a),
        "group_b": _summarize(group_b),
        "by_level": by_level,
        "by_symbol": by_symbol,
    }


def _format_trade_line(trade):
    icon = "🟢" if trade["type"] == "buy" else "🔴"
    side = "شراء" if trade["type"] == "buy" else "بيع"
    frames = (
        f"{trade['base_frame']}m/"
        f"{trade['reverse_min']}-{trade['reverse_stop'] - 1}m/"
        f"D{trade['confirm_frame']}m/"
        f"{trade['triple_frame']}m"
    )
    when = trade["time"].strftime("%m-%d %H:%M")
    out = {"win": "✅", "loss": "❌", "open": "⏳"}.get(trade["outcome"], trade["outcome"])
    return (
        f"{out} {icon} <code>{html_escape(trade['symbol'])}</code> | {side} | "
        f"{frames} | {trade['price']:.4g} | <code>{when}</code> UTC"
    )


def _format_summary_line(title, summary):
    return (
        f"{title}: صفقات <b>{summary['total']}</b> | "
        f"✅ {len(summary['wins'])} | ❌ {len(summary['losses'])} | "
        f"⏳ {len(summary['opens'])} | "
        f"نجاح {summary['win_rate']:.0f}% | "
        f"صافي {summary['pnl']:+.2f}%"
    )


def _level_title(level):
    main, reverse_min, reverse_stop, entry, don_tf, win_pct, loss_pct, _group = level
    return (
        f"{main}م | عكس {reverse_min}–{reverse_stop - 1} يتوقف {reverse_stop} | "
        f"Don {don_tf}م | دخول {entry}م (+{win_pct:g}/−{loss_pct:g})"
    )


def format_report(result):
    if not result.get("ready"):
        return ["⚠️ تعذر فحص تشبع SMI + Donchian."]

    grouped = group_results(result)
    start = result["start"].strftime("%Y-%m-%d %H:%M")
    end = result["end"].strftime("%Y-%m-%d %H:%M")
    days = int(result.get("days") or MONTH_DAYS)
    symbols = ", ".join(result.get("symbols") or SYMBOLS)
    failed = result.get("failed") or []

    header = (
        f"🗓️ <b>تشبع عكسي مرفوع + Donchian 3× (بدون EMA60/RSI) — آخر {days} يومًا</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"العملات: <code>{html_escape(symbols)}</code>\n"
        f"الفترة: <code>{start}</code> → <code>{end}</code> UTC\n"
        "الرئيسي: تشبع SMI فقط. الأكبر يلغي الأصغر.\n"
        "العكس: تشبع SMI معاكس على النطاق المرفوع؛ فريم التوقف يلغي.\n"
        "التأكيد 3×: Donchian أخضر للشراء / أحمر للبيع.\n"
        "الدخول: بعد العكس، ننتظر دونشيان غير متحقق ثم ندخل عند تحققه.\n"
    )
    if failed:
        header += f"⚠️ بلا بيانات: <code>{html_escape(', '.join(failed))}</code>\n"
    header += _format_summary_line("الإجمالي", grouped["all"]) + "\n"
    header += (
        _format_summary_line("مجموعة أ (+0.50/−0.37)", grouped["group_a"]) + "\n"
    )
    header += (
        _format_summary_line("مجموعة ب (+0.67/−0.54)", grouped["group_b"]) + "\n"
    )

    chunks = [header]
    level_lines = ["📊 <b>حسب المستوى</b>"]
    for level in LEVELS:
        summary = grouped["by_level"][_level_key(level)]
        level_lines.append(_format_summary_line(_level_title(level), summary))
    chunks.append("\n".join(level_lines))

    symbol_lines = ["💱 <b>حسب العملة</b>"]
    for symbol in result.get("symbols") or SYMBOLS:
        symbol_lines.append(
            _format_summary_line(
                f"<code>{html_escape(symbol)}</code>", grouped["by_symbol"][symbol]
            )
        )
    chunks.append("\n".join(symbol_lines))

    all_trades = grouped["all"]["trades"]
    if all_trades:
        trade_lines = ["📋 <b>الصفقات</b>"]
        trade_lines.extend(_format_trade_line(t) for t in all_trades)
        chunks.append("\n".join(trade_lines))
    else:
        chunks.append("لا توجد صفقات مطابقة خلال الفترة.")

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


def format_plain_report(result):
    """Plain-text report for CLI / agent summary."""
    texts = []
    for chunk in format_report(result):
        texts.append(
            chunk.replace("<b>", "")
            .replace("</b>", "")
            .replace("<code>", "")
            .replace("</code>", "")
        )
    return "\n\n".join(texts)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("Scanning %s for %s days...", ",".join(SYMBOLS), MONTH_DAYS)
    result = scan_all(days=MONTH_DAYS)
    print(format_plain_report(result))


if __name__ == "__main__":
    main()
