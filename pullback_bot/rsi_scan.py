"""RSI-only scan: main RSI 50 + entry RSI 45/55. No EMA, Donchian, SMI, MACD.

  Buy:  main RSI close > 50, and entry RSI close > 45.
  Sell: main RSI close < 50, and entry RSI close < 55.
  After the main side is live, wait until the entry RSI is unmet,
  then enter on the first later candle where it holds.
  Each level is independent (no larger-main cancel).
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

from indicators import WARMUP_RSI, calc_rsi_tv, resample_ohlcv_closed
from pullback_bot.strategy import (
    DEDUPE_HOURS,
    MONTH_DAYS,
    _bool_step,
    _utc,
    evaluate_outcome,
    fetch_1m_vision,
)

log = logging.getLogger(__name__)

SYMBOLS = ("BTCUSDT", "ETHUSDT", "XRPUSDT")
# main, entry, win_pct, loss_pct, group
LEVELS = (
    (45, 5, 0.50, 0.37, "a"),
    (60, 5, 0.50, 0.37, "a"),
    (90, 9, 0.67, 0.54, "b"),
    (120, 10, 0.67, 0.54, "b"),
    (150, 11, 0.67, 0.54, "b"),
)
MAIN_BUY = 50.0
MAIN_SELL = 50.0
ENTRY_BUY = 45.0
ENTRY_SELL = 55.0
MIN_1M_BARS = 100_000


def _all_needed_frames():
    frames = set()
    for main, entry, _win, _loss, _group in LEVELS:
        frames.add(main)
        frames.add(entry)
    return sorted(frames)


def _rsi_features(df_1m, minutes, buy_level, sell_level):
    df = resample_ohlcv_closed(df_1m, minutes)
    if df.empty or len(df) < WARMUP_RSI:
        return None
    rsi = calc_rsi_tv(df["close"], period=14)
    rsi_arr = rsi.to_numpy(dtype=float)
    end_ts = df["ts"] + pd.Timedelta(minutes=minutes)
    return pd.DataFrame(
        {
            "ts": df["ts"].to_numpy(),
            "end_ts": end_ts.to_numpy(),
            "close": df["close"].to_numpy(),
            "rsi": rsi_arr,
            "rsi_buy": rsi_arr > buy_level,
            "rsi_sell": rsi_arr < sell_level,
        }
    )


def _precompute_stepped(frame_data, grid):
    stepped = {}
    n = len(grid)
    for minutes, feat in frame_data.items():
        if feat is None:
            continue
        ends = feat["end_ts"].to_numpy()
        stepped[minutes] = {
            "rsi_buy": _bool_step(ends, feat["rsi_buy"].to_numpy(), grid),
            "rsi_sell": _bool_step(ends, feat["rsi_sell"].to_numpy(), grid),
        }
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


def _scan_side(side, stepped, entry_data, grid, start, end, raw_1m, symbol):
    is_sell = side == "sell"
    main_key = "rsi_sell" if is_sell else "rsi_buy"
    n_grid = len(grid)
    signals = []
    start_ts = pd.Timestamp(start)
    end_ts_limit = pd.Timestamp(end)
    idle, wait_clear, armed = 0, 1, 2

    for main, entry, win_pct, loss_pct, group in LEVELS:
        entry_df = entry_data.get(entry)
        if entry_df is None:
            continue
        main_feat = stepped.get(main)
        owned = (
            main_feat[main_key]
            if main_feat is not None
            else np.zeros(n_grid, dtype=bool)
        )

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

            buy_ok = bool(entry_df["rsi_buy"].iloc[row_i])
            sell_ok = bool(entry_df["rsi_sell"].iloc[row_i])
            price = float(entry_df["close"].iloc[row_i])
            holds = sell_ok if is_sell else buy_ok
            cleared = (not sell_ok) if is_sell else (not buy_ok)

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
                    "confirm_frame": entry,
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
    entry_frames = {lvl[1] for lvl in LEVELS}
    main_frames = {lvl[0] for lvl in LEVELS}
    log.info("%s: computing %s frames...", symbol, len(needed))
    frame_data = {}
    entry_data = {}
    for minutes in needed:
        if minutes in entry_frames:
            entry_data[minutes] = _rsi_features(
                raw_1m, minutes, ENTRY_BUY, ENTRY_SELL
            )
        if minutes in main_frames:
            frame_data[minutes] = _rsi_features(
                raw_1m, minutes, MAIN_BUY, MAIN_SELL
            )

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
    main, entry, win_pct, loss_pct, group = level
    return (main, entry, win_pct, loss_pct, group)


def group_results(result):
    all_trades = list(result.get("wins") or [])
    all_trades.extend(result.get("losses") or [])
    all_trades.extend(result.get("opens") or [])
    all_trades.sort(key=lambda item: item["time"])
    group_a = [t for t in all_trades if t.get("group") == "a"]
    group_b = [t for t in all_trades if t.get("group") == "b"]
    by_level = {}
    for level in LEVELS:
        key = _level_key(level)
        main, entry, win_pct, loss_pct, group = key
        by_level[key] = _summarize(
            [
                t
                for t in all_trades
                if t["base_frame"] == main
                and t["triple_frame"] == entry
                and t.get("group") == group
                and float(t["win_pct"]) == float(win_pct)
                and float(t["loss_pct"]) == float(loss_pct)
            ]
        )
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
    frames = f"{trade['base_frame']}m/{trade['triple_frame']}m"
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
        return ["⚠️ تعذر فحص RSI."]

    grouped = group_results(result)
    start = result["start"].strftime("%Y-%m-%d %H:%M")
    end = result["end"].strftime("%Y-%m-%d %H:%M")
    days = int(result.get("days") or MONTH_DAYS)
    symbols = ", ".join(result.get("symbols") or SYMBOLS)
    failed = result.get("failed") or []

    header = (
        f"🗓️ <b>RSI فقط (بدون EMA/Donchian/SMI) — آخر {days} يومًا</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"العملات: <code>{html_escape(symbols)}</code>\n"
        f"الفترة: <code>{start}</code> → <code>{end}</code> UTC\n"
        "شراء: إغلاق RSI الرئيسي &gt; 50، وإغلاق RSI الدخول &gt; 45.\n"
        "بيع: إغلاق RSI الرئيسي &lt; 50، وإغلاق RSI الدخول &lt; 55.\n"
        "الدخول: بعد تحقق الرئيسي، ننتظر شرط الدخول ينطفئ ثم ندخل عند تحققه.\n"
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
        main, entry, win_pct, loss_pct, _group = level
        summary = grouped["by_level"][_level_key(level)]
        level_lines.append(
            _format_summary_line(
                f"{main}م | دخول {entry}م (+{win_pct:g}/−{loss_pct:g})",
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
    texts = []
    for chunk in format_report(result):
        texts.append(
            chunk.replace("<b>", "")
            .replace("</b>", "")
            .replace("<code>", "")
            .replace("</code>", "")
            .replace("&gt;", ">")
            .replace("&lt;", "<")
        )
    return "\n\n".join(texts)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("Scanning %s for %s days (RSI only)...", ",".join(SYMBOLS), MONTH_DAYS)
    result = scan_all(days=MONTH_DAYS)
    print(format_plain_report(result))


if __name__ == "__main__":
    main()
