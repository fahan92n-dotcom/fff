"""SMI sat + 3× Donchian confirm, with Signal/K zone-cross halt.

No EMA60, no RSI.

  Main SMI sat owns the side (largest main cancels smaller).
  Confirm TF = 3× main: Donchian green on buys, red on sells.
  Entry TF: after confirm is aligned, wait until Donchian is unmet,
  then enter on the first later candle where it holds.

  Extra halt: when Signal Length crosses K Length and that cross is
  above +40 (buy) or below −40 (sell), the close confirms it and that
  main TF stops entering for the rest of the sat episode.
"""

from __future__ import annotations

import logging
import sys
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
# main, don_confirm, entry, win_pct, loss_pct, group
LEVELS = (
    (45, 135, 5, 0.50, 0.37, "a"),
    (60, 180, 5, 0.50, 0.37, "a"),
    (90, 270, 9, 0.67, 0.54, "b"),
    (120, 360, 10, 0.67, 0.54, "b"),
    (150, 450, 11, 0.67, 0.54, "b"),
)
MAINS = tuple(lvl[0] for lvl in LEVELS)
MIN_1M_BARS = 100_000


def _all_needed_frames():
    frames = set()
    for main, confirm, entry, _win, _loss, _group in LEVELS:
        frames.add(main)
        frames.add(confirm)
        frames.add(entry)
    return sorted(frames)


def signal_k_zone_cross(smi, signal):
    """Detect Signal crossing K with the cross above +40 or below −40.

    Returns (any_zone_cross, cross_above_40, cross_below_minus_40).
    The confirming close is the bar where the cross is first visible.
    """
    smi_arr = np.asarray(smi, dtype=float)
    sig_arr = np.asarray(signal, dtype=float)
    n = len(smi_arr)
    crossed = np.zeros(n, dtype=bool)
    if n >= 2:
        prev_above = sig_arr[:-1] > smi_arr[:-1]
        now_above = sig_arr[1:] > smi_arr[1:]
        crossed[1:] = prev_above != now_above
    high = crossed & (sig_arr > 40.0) & (smi_arr > 40.0)
    low = crossed & (sig_arr < -40.0) & (smi_arr < -40.0)
    return high | low, high, low


def halt_after_event(active, event):
    """Halt from the confirming close while ``active`` (sat) remains true."""
    active_arr = np.asarray(active, dtype=bool)
    event_arr = np.asarray(event, dtype=bool)
    out = np.zeros(len(active_arr), dtype=bool)
    halted = False
    for i, is_active in enumerate(active_arr):
        if not is_active:
            halted = False
            continue
        if event_arr[i]:
            halted = True
        out[i] = halted
    return out


def _smi_don_features(df_1m, minutes):
    """Resample and compute SMI sat, Signal/K halt, and Donchian (no EMA/RSI)."""
    df = resample_ohlcv_closed(df_1m, minutes)
    if df.empty or len(df) < max(WARMUP_SMI, DONCHIAN_DLEN + 2):
        return None

    smi, signal, _ = calc_smi(df["high"], df["low"], df["close"])
    don = calc_donchian_trend_series(
        df["close"].to_numpy(),
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        DONCHIAN_DLEN,
    )
    sell_sat = (smi <= SMI_SELL).to_numpy()
    buy_sat = (smi >= SMI_BUY).to_numpy()
    _any, cross_high, cross_low = signal_k_zone_cross(smi, signal)
    end_ts = df["ts"] + pd.Timedelta(minutes=minutes)
    don_arr = don.to_numpy()
    return pd.DataFrame(
        {
            "ts": df["ts"].to_numpy(),
            "end_ts": end_ts.to_numpy(),
            "close": df["close"].to_numpy(),
            "smi": smi.to_numpy(),
            "signal": signal.to_numpy(),
            "sell_sat": sell_sat,
            "buy_sat": buy_sat,
            "don_green": don_arr == 1,
            "don_red": don_arr == -1,
            "halt_buy": halt_after_event(buy_sat, cross_high),
            "halt_sell": halt_after_event(sell_sat, cross_low),
        }
    )


def _entry_features(df_1m, minutes):
    df = resample_ohlcv_closed(df_1m, minutes)
    if df.empty or len(df) < DONCHIAN_DLEN + 2:
        return None
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
        row = {
            "sell_sat": _bool_step(ends, feat["sell_sat"].to_numpy(), grid),
            "buy_sat": _bool_step(ends, feat["buy_sat"].to_numpy(), grid),
            "don_green": _bool_step(ends, feat["don_green"].to_numpy(), grid),
            "don_red": _bool_step(ends, feat["don_red"].to_numpy(), grid),
        }
        if "halt_buy" in feat.columns:
            row["halt_buy"] = _bool_step(ends, feat["halt_buy"].to_numpy(), grid)
            row["halt_sell"] = _bool_step(ends, feat["halt_sell"].to_numpy(), grid)
        else:
            row["halt_buy"] = np.zeros(len(grid), dtype=bool)
            row["halt_sell"] = np.zeros(len(grid), dtype=bool)
        stepped[minutes] = row
    return stepped


def _dedupe_signals(signals, hours=DEDUPE_HOURS):
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


def _scan_side(
    side, stepped, entry_data, grid, start, end, raw_1m, symbol, *, use_halt=True
):
    """Main SMI sat + 3× Donchian confirm + entry Donchian flip.

    ``use_halt`` applies the optional Signal/K zone-cross stop on the main.
    """
    is_sell = side == "sell"
    sat_key = "sell_sat" if is_sell else "buy_sat"
    don_key = "don_red" if is_sell else "don_green"
    halt_key = "halt_sell" if is_sell else "halt_buy"
    n_grid = len(grid)

    active = np.full(n_grid, -1, dtype=int)
    main_masks = []
    for main, _confirm, _entry, _win, _loss, _group in LEVELS:
        feat = stepped.get(main)
        main_masks.append(
            feat[sat_key] if feat is not None else np.zeros(n_grid, dtype=bool)
        )
    stacked = np.vstack(main_masks) if main_masks else np.zeros((0, n_grid), dtype=bool)
    for i in range(n_grid):
        chosen = -1
        for idx in range(len(LEVELS)):
            if stacked[idx, i]:
                chosen = idx
        active[i] = chosen

    signals = []
    start_ts = pd.Timestamp(start)
    end_ts_limit = pd.Timestamp(end)
    idle, wait_clear, armed = 0, 1, 2
    for level_idx, (main, confirm, entry, win_pct, loss_pct, group) in enumerate(LEVELS):
        entry_df = entry_data.get(entry)
        if entry_df is None:
            continue

        confirm_feat = stepped.get(confirm)
        confirm_don = (
            confirm_feat[don_key]
            if confirm_feat is not None
            else np.zeros(n_grid, dtype=bool)
        )
        main_feat = stepped.get(main)
        halted = (
            main_feat[halt_key]
            if use_halt and main_feat is not None
            else np.zeros(n_grid, dtype=bool)
        )
        owned = (active == level_idx) & confirm_don & ~halted

        ends = pd.DatetimeIndex(pd.to_datetime(entry_df["end_ts"], utc=True))
        state = idle
        for row_i, candle_end in enumerate(ends):
            if candle_end < start_ts or candle_end > end_ts_limit:
                state = idle
                continue
            pos = int(grid.searchsorted(candle_end, side="right") - 1)
            if pos < 0 or not owned[pos]:
                state = idle
                continue

            green = bool(entry_df["don_green"].iloc[row_i])
            red = bool(entry_df["don_red"].iloc[row_i])
            price = float(entry_df["close"].iloc[row_i])
            holds = red if is_sell else green
            cleared = (not red) if is_sell else (not green)

            if state == idle:
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
                    "confirm_frame": confirm,
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


def scan_symbol(symbol, *, days=MONTH_DAYS, now=None, raw_1m=None, use_halt=True):
    """Scan one symbol over the last ``days``. No EMA60/RSI."""
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
        target = max(MIN_1M_BARS, int(days) * 1440 + 50_000)
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
    entry_frames = {lvl[2] for lvl in LEVELS}
    log.info("%s: computing %s frames...", symbol, len(needed))
    frame_data = {}
    entry_data = {}
    for minutes in needed:
        if minutes in entry_frames:
            entry_data[minutes] = _entry_features(raw_1m, minutes)
        else:
            frame_data[minutes] = _smi_don_features(raw_1m, minutes)

    stepped = _precompute_stepped(frame_data, grid)
    all_signals = []
    all_signals.extend(
        _scan_side(
            "sell",
            stepped,
            entry_data,
            grid,
            start,
            end,
            raw_1m,
            symbol,
            use_halt=use_halt,
        )
    )
    all_signals.extend(
        _scan_side(
            "buy",
            stepped,
            entry_data,
            grid,
            start,
            end,
            raw_1m,
            symbol,
            use_halt=use_halt,
        )
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
        "use_halt": bool(use_halt),
    }


def scan_all(
    symbols=SYMBOLS, *, days=MONTH_DAYS, now=None, raw_by_symbol=None, use_halt=True
):
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
            use_halt=use_halt,
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
        "use_halt": bool(use_halt),
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
    main, _confirm, entry, win_pct, loss_pct, group = level
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
        main, confirm, entry, win_pct, loss_pct, group = level
        by_level[key] = _summarize(
            [
                t
                for t in all_trades
                if t["base_frame"] == main
                and t["triple_frame"] == entry
                and t.get("group") == group
            ]
        )
        by_level[key]["confirm"] = confirm
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
        f"{trade['base_frame']}m/{trade['confirm_frame']}m/{trade['triple_frame']}m"
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


def format_report(result):
    if not result.get("ready"):
        return ["⚠️ تعذر فحص تشبع SMI + Donchian."]

    grouped = group_results(result)
    start = result["start"].strftime("%Y-%m-%d %H:%M")
    end = result["end"].strftime("%Y-%m-%d %H:%M")
    days = int(result.get("days") or MONTH_DAYS)
    symbols = ", ".join(result.get("symbols") or SYMBOLS)
    failed = result.get("failed") or []

    use_halt = bool(result.get("use_halt", True))
    halt_note = (
        "إيقاف: إذا Signal تجاوز K فوق +40 أو تحت −40، بعد الإغلاق يُوقف الفريم.\n"
        if use_halt
        else "بدون إيقاف Signal/K.\n"
    )
    title = (
        "SMI + Donchian 3× + إيقاف Signal/K (بدون EMA60/RSI)"
        if use_halt
        else "SMI + Donchian 3× (بدون EMA60/RSI وبدون Signal/K)"
    )
    header = (
        f"🗓️ <b>{title} — آخر {days} يومًا</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"العملات: <code>{html_escape(symbols)}</code>\n"
        f"الفترة: <code>{start}</code> → <code>{end}</code> UTC\n"
        "الرئيسي: تشبع SMI فقط. الأكبر يلغي الأصغر.\n"
        "التأكيد 3×: Donchian أخضر للشراء / أحمر للبيع.\n"
        + halt_note
        + "الدخول: بعد التأكيد، ننتظر دونشيان غير متحقق ثم ندخل عند تحققه.\n"
    )
    if failed:
        header += f"⚠️ بلا بيانات: <code>{html_escape(', '.join(failed))}</code>\n"
    header += _format_summary_line("الإجمالي", grouped["all"]) + "\n"
    header += (
        _format_summary_line("45م+60م (+0.50/−0.37)", grouped["group_a"]) + "\n"
    )
    header += (
        _format_summary_line("90–150م (+0.67/−0.54)", grouped["group_b"]) + "\n"
    )

    chunks = [header]
    level_lines = ["📊 <b>حسب المستوى</b>"]
    for level in LEVELS:
        main, confirm, entry, win_pct, loss_pct, _group = level
        summary = grouped["by_level"][_level_key(level)]
        level_lines.append(
            _format_summary_line(
                f"{main}م | تأكيد {confirm}م | دخول {entry}م "
                f"(+{win_pct:g}/−{loss_pct:g})",
                summary,
            )
        )
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


def main(use_halt=True):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("Scanning %s for %s days (halt=%s)...", ",".join(SYMBOLS), MONTH_DAYS, use_halt)
    result = scan_all(days=MONTH_DAYS, use_halt=use_halt)
    print(format_plain_report(result))


if __name__ == "__main__":
    main(use_halt="--no-halt" not in sys.argv)
