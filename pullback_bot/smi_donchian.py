"""SMI sat + reverse-sat + 3× Donchian confirm + MACD zero line.

Flow:
  1) Largest main TF with SMI sat owns the side and cancels smaller mains.
     5h SMI sat halts that side. Main is SMI sat only (no EMA). MACD line
     on the main TF must agree at the close: buy only if MACD > 0, sell
     only if MACD < 0.
  2) Wait for reverse/counter SMI sat inside the owned main.
  3) Donchian on the 3× confirm TF must be green (buy) / red (sell)
     at entry. If it flips, the path dies.
  4) After reverse sat, watch the entry TF: wait until Donchian and EMA50
     are both unmet, then enter when both hold
     (buy: green AND close above EMA50; sell: red AND close below EMA50).
  5) If Signal Length crosses K Length above +40 (buy) or below −40
     (sell) before entry, that main stops for the rest of the sat episode.
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
    _calc_macd_full,
    calc_donchian_trend_series,
    calc_ema,
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

SYMBOLS = (
    "ADAUSDT",
    "SUIUSDT",
    "HYPEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "AAVEUSDT",
    "TAOUSDT",
    "XLMUSDT",
    "HBARUSDT",
    "DOTUSDT",
)

WIN_PCT = 1.0
LOSS_PCT = 0.77
EMA_SPAN = 50
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
HALT_MAIN_MINUTES = 5 * 60  # 5h sat stops this experiment on that side

# main, reverse_min, reverse_last, reverse_abort, don_confirm (3×), entry
# Reverse sat accepted on reverse_min..reverse_last inclusive; abort TF
# kills the level. 2h accepts 20–46 and aborts at 48 (47 is not used).
LEVELS = (
    (60, 10, 23, 24, 180, 5),
    (90, 15, 35, 36, 270, 8),
    (120, 20, 46, 48, 360, 10),
    (150, 25, 59, 60, 450, 13),
    (180, 30, 71, 72, 540, 15),
    (210, 35, 83, 84, 630, 17),
    (240, 40, 95, 96, 720, 20),
)
MIN_1M_BARS = 130_000


def _all_needed_frames():
    frames = {HALT_MAIN_MINUTES}
    for main, reverse_min, reverse_last, reverse_abort, don_confirm, entry in LEVELS:
        frames.add(main)
        frames.add(don_confirm)
        frames.add(entry)
        frames.add(reverse_abort)
        for minutes in range(reverse_min, reverse_last + 1):
            frames.add(minutes)
    return sorted(frames)


def _don_frames():
    frames = set()
    for _main, _rmin, _rlast, _abort, don_confirm, entry in LEVELS:
        frames.add(don_confirm)
        frames.add(entry)
    return frames


def _entry_frames():
    return {lvl[5] for lvl in LEVELS}


def _main_frames():
    return {lvl[0] for lvl in LEVELS}


def signal_k_zone_cross(smi, signal):
    """Signal crossing K above +40 (buy-sat over) or below −40 (sell-sat over).

    Buy rollover: Signal was at/below K and closes above K, both > +40.
    Sell rollover: Signal was at/above K and closes below K, both < −40.
    """
    smi_arr = np.asarray(smi, dtype=float)
    sig_arr = np.asarray(signal, dtype=float)
    n = len(smi_arr)
    cross_up = np.zeros(n, dtype=bool)
    cross_down = np.zeros(n, dtype=bool)
    if n >= 2:
        cross_up[1:] = (sig_arr[:-1] <= smi_arr[:-1]) & (sig_arr[1:] > smi_arr[1:])
        cross_down[1:] = (sig_arr[:-1] >= smi_arr[:-1]) & (sig_arr[1:] < smi_arr[1:])
    high = cross_up & (sig_arr > 40.0) & (smi_arr > 40.0)
    low = cross_down & (sig_arr < -40.0) & (smi_arr < -40.0)
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


def _smi_don_features(df_1m, minutes, *, need_don=False, need_ema=False, need_macd=False):
    """Resample SMI sat; Donchian on confirm/entry; EMA50 on entry; MACD on main."""
    df = resample_ohlcv_closed(df_1m, minutes)
    min_bars = max(
        WARMUP_SMI,
        DONCHIAN_DLEN + 2 if need_don else 0,
        EMA_SPAN + 5 if need_ema else 0,
        MACD_SLOW + MACD_SIGNAL + 2 if need_macd else 0,
    )
    if df.empty or len(df) < min_bars:
        return None

    smi, ema_signal, _ = calc_smi(df["high"], df["low"], df["close"])
    sell_sat = (smi <= SMI_SELL).to_numpy()
    buy_sat = (smi >= SMI_BUY).to_numpy()
    _any, cross_high, cross_low = signal_k_zone_cross(smi, ema_signal)
    n = len(df)
    end_ts = df["ts"] + pd.Timedelta(minutes=minutes)
    payload = {
        "ts": df["ts"].to_numpy(),
        "end_ts": end_ts.to_numpy(),
        "close": df["close"].to_numpy(),
        "smi": smi.to_numpy(),
        "sell_sat": sell_sat,
        "buy_sat": buy_sat,
        "don_green": np.zeros(n, dtype=bool),
        "don_red": np.zeros(n, dtype=bool),
        "above_ema": np.zeros(n, dtype=bool),
        "below_ema": np.zeros(n, dtype=bool),
        "buy_macd": np.ones(n, dtype=bool),
        "sell_macd": np.ones(n, dtype=bool),
        "halt_buy": halt_after_event(buy_sat, cross_high),
        "halt_sell": halt_after_event(sell_sat, cross_low),
    }
    if need_don:
        don = calc_donchian_trend_series(
            df["close"].to_numpy(),
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            DONCHIAN_DLEN,
        ).to_numpy()
        payload["don_green"] = don == 1
        payload["don_red"] = don == -1
    if need_ema:
        ema = calc_ema(df["close"], span=EMA_SPAN)
        payload["above_ema"] = (df["close"] > ema).to_numpy()
        payload["below_ema"] = (df["close"] < ema).to_numpy()
    if need_macd:
        macd_line, _signal, _hist = _calc_macd_full(df["close"])
        payload["buy_macd"] = (macd_line > 0).fillna(False).to_numpy()
        payload["sell_macd"] = (macd_line < 0).fillna(False).to_numpy()
    return pd.DataFrame(payload)


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
            "halt_buy": _bool_step(ends, feat["halt_buy"].to_numpy(), grid)
            if "halt_buy" in feat.columns
            else np.zeros(len(grid), dtype=bool),
            "halt_sell": _bool_step(ends, feat["halt_sell"].to_numpy(), grid)
            if "halt_sell" in feat.columns
            else np.zeros(len(grid), dtype=bool),
            "buy_macd": _bool_step(ends, feat["buy_macd"].to_numpy(), grid)
            if "buy_macd" in feat.columns
            else np.ones(len(grid), dtype=bool),
            "sell_macd": _bool_step(ends, feat["sell_macd"].to_numpy(), grid)
            if "sell_macd" in feat.columns
            else np.ones(len(grid), dtype=bool),
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
        key = (sig["symbol"], sig["type"], sig["base_frame"], sig["triple_frame"])
        prev = last_by_key.get(key)
        if prev is not None and sig["time"] - prev < window:
            continue
        last_by_key[key] = sig["time"]
        kept.append(sig)
    return kept


def _scan_side(side, stepped, entry_data, grid, start, end, raw_1m, symbol):
    """Largest SMI sat owns the side; reverse sat then Donchian entry."""
    is_sell = side == "sell"
    sat_key = "sell_sat" if is_sell else "buy_sat"
    reverse_key = "buy_sat" if is_sell else "sell_sat"
    don_key = "don_red" if is_sell else "don_green"
    halt_key = "halt_sell" if is_sell else "halt_buy"
    macd_key = "sell_macd" if is_sell else "buy_macd"
    n_grid = len(grid)

    halt = stepped.get(HALT_MAIN_MINUTES)
    halt_sat = (
        halt[sat_key] if halt is not None else np.zeros(n_grid, dtype=bool)
    )

    main_masks = []
    for main, _rmin, _rlast, _abort, _don, _entry in LEVELS:
        feat = stepped.get(main)
        main_masks.append(
            feat[sat_key] if feat is not None else np.zeros(n_grid, dtype=bool)
        )
    stacked = np.vstack(main_masks) if main_masks else np.zeros((0, n_grid), dtype=bool)
    active = np.full(n_grid, -1, dtype=int)
    for i in range(n_grid):
        if halt_sat[i]:
            active[i] = -2
            continue
        chosen = -1
        for idx in range(len(LEVELS)):
            if stacked[idx, i]:
                chosen = idx
        active[i] = chosen

    signals = []
    start_ts = pd.Timestamp(start)
    end_ts_limit = pd.Timestamp(end)
    idle, wait_clear, armed = 0, 1, 2
    for level_idx, (main, reverse_min, reverse_last, reverse_abort, don_tf, entry) in enumerate(LEVELS):
        entry_df = entry_data.get(entry)
        if entry_df is None:
            continue

        confirm_any = np.zeros(n_grid, dtype=bool)
        for minutes in range(reverse_min, reverse_last + 1):
            feat = stepped.get(minutes)
            if feat is None:
                continue
            confirm_any |= feat[reverse_key]
        stop_feat = stepped.get(reverse_abort)
        confirm_stop_mask = (
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
        main_feat = stepped.get(main)
        halted = (
            main_feat[halt_key]
            if main_feat is not None and halt_key in main_feat
            else np.zeros(n_grid, dtype=bool)
        )
        macd_ok = (
            main_feat[macd_key]
            if main_feat is not None and macd_key in main_feat
            else np.ones(n_grid, dtype=bool)
        )

        owned = (active == level_idx) & macd_ok & ~confirm_stop_mask & don_ok & ~halted
        confirm_window = owned & confirm_any
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
            above = bool(entry_df["above_ema"].iloc[row_i]) if "above_ema" in entry_df.columns else False
            below = bool(entry_df["below_ema"].iloc[row_i]) if "below_ema" in entry_df.columns else False
            price = float(entry_df["close"].iloc[row_i])
            if is_sell:
                holds = red and below
                cleared = (not red) and (not below)
            else:
                holds = green and above
                cleared = (not green) and (not above)
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
                win_pct=WIN_PCT,
                loss_pct=LOSS_PCT,
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
                    "reverse_stop": reverse_abort,
                    "triple_frame": entry,
                    "win_pct": WIN_PCT,
                    "loss_pct": LOSS_PCT,
                    "outcome": outcome,
                    "exit_price": exit_price,
                    "exit_ts": exit_ts,
                }
            )
            state = idle
    return signals


def scan_symbol(symbol, *, days=MONTH_DAYS, now=None, raw_1m=None):
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
        target = max(MIN_1M_BARS, int(days) * 1440 + 80_000)
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
    don_needed = _don_frames()
    entry_frames = _entry_frames()
    main_frames = _main_frames()
    log.info("%s: computing %s frames...", symbol, len(needed))
    frame_data = {}
    entry_data = {}
    for index, minutes in enumerate(needed, start=1):
        if index == 1 or index % 20 == 0 or index == len(needed):
            log.info("%s: frame %s/%s (%sm)", symbol, index, len(needed), minutes)
        feat = _smi_don_features(
            df_1m=raw_1m,
            minutes=minutes,
            need_don=minutes in don_needed,
            need_ema=minutes in entry_frames,
            need_macd=minutes in main_frames,
        )
        frame_data[minutes] = feat
        if minutes in entry_frames:
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
    """Scan the requested symbols and merge trades."""
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
    main, _rmin, _rlast, _abort, _don, entry = level
    return (main, entry)


def group_results(result):
    all_trades = list(result.get("wins") or [])
    all_trades.extend(result.get("losses") or [])
    all_trades.extend(result.get("opens") or [])
    all_trades.sort(key=lambda item: item["time"])

    by_level = {}
    for level in LEVELS:
        main, _rmin, _rlast, _abort, don_tf, entry = level
        by_level[_level_key(level)] = _summarize(
            [
                t
                for t in all_trades
                if t["base_frame"] == main and t["triple_frame"] == entry
            ]
        )
        by_level[_level_key(level)]["confirm"] = don_tf
    by_symbol = {}
    for symbol in result.get("symbols") or SYMBOLS:
        by_symbol[symbol] = _summarize(
            [t for t in all_trades if t["symbol"] == symbol]
        )
    return {
        "all": _summarize(all_trades),
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
        return ["⚠️ تعذر فحص SMI + Donchian 3× + MACD."]

    grouped = group_results(result)
    start = result["start"].strftime("%Y-%m-%d %H:%M")
    end = result["end"].strftime("%Y-%m-%d %H:%M")
    days = int(result.get("days") or MONTH_DAYS)
    symbols = ", ".join(result.get("symbols") or SYMBOLS)
    failed = result.get("failed") or []

    header = (
        f"🗓️ <b>SMI + تشبع عكسي + Donchian 3× + MACD — آخر {days} يومًا</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"العملات: <code>{html_escape(symbols)}</code>\n"
        f"الفترة: <code>{start}</code> → <code>{end}</code> UTC\n"
        "بدون EMA على الرئيس وبدون RSI.\n"
        "الرئيسي: تشبع SMI. الأكبر يلغي الأصغر. تشبع 5س يوقف الجانب.\n"
        "MACD على الرئيس: شراء فقط إذا MACD>0 عند الإغلاق، بيع فقط إذا MACD<0.\n"
        "إذا Signal قطع K فوق +40 أو تحت −40 قبل الدخول، يُوقف الفريم.\n"
        "بعد التشبع العكسي: Donchian التأكيد 3× أخضر شراء / أحمر بيع.\n"
        "الدخول: Donchian + تجاوز EMA50 على فريم الدخول فقط.\n"
        f"الربح {WIN_PCT:g}% / الخسارة {LOSS_PCT:g}%.\n"
    )
    if failed:
        header += f"⚠️ بلا بيانات: <code>{html_escape(', '.join(failed))}</code>\n"
    header += _format_summary_line("الإجمالي", grouped["all"])

    chunks = [header]
    level_lines = ["📊 <b>حسب المستوى</b>"]
    for level in LEVELS:
        main, _rmin, _rlast, _abort, don_tf, entry = level
        summary = grouped["by_level"][_level_key(level)]
        level_lines.append(
            _format_summary_line(
                f"{main}م | تأكيد {don_tf}م | دخول {entry}م",
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
        )
    return "\n\n".join(texts)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("Scanning %s for %s days...", ",".join(SYMBOLS), MONTH_DAYS)
    result = scan_all(days=MONTH_DAYS)
    print(format_plain_report(result))


if __name__ == "__main__":
    main()
