"""Pullback saturation strategy (SMI + EMA60 + Donchian + RSI) and week scan.

Flow:
  1) Main sat on closed candle: SMI <= -40 (sell) / >= +40 (buy),
     with RSI < 50 (sell) / > 50 (buy) on that main-TF candle.
  2) Wait for reverse/counter sat on confirm TFs (inside the main sat).
  3) Once a reverse-sat candle closes, start watching the entry TF.
  4) On the FIRST watched entry-TF candle:
     - If BOTH entry conditions already hold (buy: Donchian green AND
       close above EMA60; sell: red AND below EMA60) → reject the
       episode: no entry until a fresh counter sat starts a new one.
     - Otherwise (one or both missing) wait: each condition may become
       true at its own time, partial states in between are fine, and
       enter on the first candle where both hold together.
     (Reverse sat may already have cleared by the entry candle.)

Sell path example (30 → 5..11 → 2):
  Main 30m sell-sat → counter buy-sat on 5..11 (12m stops) → on 2m after
  counter confirms (not already red+below EMA60 on the first candle),
  enter on the first candle red AND below EMA60.

Buy path is the exact mirror. Main frames stop at 6h; 7h+ halts that side.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from html import escape as html_escape
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from indicators import (
    DONCHIAN_DLEN,
    WARMUP_SMI,
    calc_donchian_trend_series,
    calc_ema,
    calc_rsi_tv,
    calc_smi,
    resample_ohlcv_closed,
)

log = logging.getLogger(__name__)

SYMBOL = "BTCUSDT"
EMA_SPAN = 60
SMI_SELL = -40
SMI_BUY = 40
RSI_MID = 50
HALT_MAIN_MINUTES = 7 * 60  # 7h — stop, no level beyond 6h
VISION_KLINES = "https://data-api.binance.vision/api/v3/klines"
MIN_1M_BARS = 45_000  # need ~100+ bars even on 6h for SMI warmup
WIN_PCT = 1.0
LOSS_PCT = 0.70
WEEK_DAYS = 7
DEDUPE_HOURS = 4
_SESSION = requests.Session()

# main, confirm_min, confirm_stop, entry
LEVELS = (
    (30, 5, 12, 2),
    (45, 8, 18, 3),
    (60, 10, 24, 4),
    (90, 15, 36, 6),
    (120, 20, 48, 8),
    (150, 25, 60, 10),
    (180, 30, 72, 12),
    (210, 35, 84, 14),
    (240, 40, 96, 16),
    (270, 45, 84, 18),
    (300, 50, 108, 20),
    (330, 55, 132, 22),
    (360, 60, 144, 24),
)

_scan_lock = threading.Lock()
_scan_running = False


def _utc(ts):
    if ts is None:
        return None
    if getattr(ts, "to_pydatetime", None) is not None:
        ts = ts.to_pydatetime()
    if getattr(ts, "tzinfo", None) is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def evaluate_outcome(
    signal_type,
    entry_price,
    future_1m,
    *,
    win_pct=WIN_PCT,
    loss_pct=LOSS_PCT,
):
    """Buy/sell win at ±win_pct; loss at adverse loss_pct. Same-bar both → loss."""
    if future_1m is None or future_1m.empty or entry_price <= 0:
        return "open", None, None

    if signal_type == "buy":
        tp = entry_price * (1.0 + win_pct / 100.0)
        sl = entry_price * (1.0 - loss_pct / 100.0)
        for row in future_1m.itertuples(index=False):
            if float(row.low) <= sl:
                return "loss", sl, _utc(row.ts)
            if float(row.high) >= tp:
                return "win", tp, _utc(row.ts)
    elif signal_type == "sell":
        tp = entry_price * (1.0 - win_pct / 100.0)
        sl = entry_price * (1.0 + loss_pct / 100.0)
        for row in future_1m.itertuples(index=False):
            if float(row.high) >= sl:
                return "loss", sl, _utc(row.ts)
            if float(row.low) <= tp:
                return "win", tp, _utc(row.ts)
    else:
        raise ValueError(f"Unsupported signal type: {signal_type}")
    return "open", None, None


def _all_needed_frames():
    frames = {HALT_MAIN_MINUTES}
    for main, confirm_min, confirm_stop, entry in LEVELS:
        frames.add(main)
        frames.add(entry)
        frames.add(confirm_stop)
        for minutes in range(confirm_min, confirm_stop):
            frames.add(minutes)
    return sorted(frames)


def fetch_btc_1m_vision(target=MIN_1M_BARS):
    """Fetch BTCUSDT 1m spot OHLCV via Binance Vision (geo-friendly mirror)."""
    tf_ms = 60_000
    bin_max = 1000
    all_dfs = []
    end_ms = int(time.time() * 1000)
    fetched = 0
    retries = 0
    while fetched < target:
        batch = min(bin_max, target - fetched)
        start_ms = end_ms - batch * tf_ms
        try:
            resp = _SESSION.get(
                VISION_KLINES,
                params={
                    "symbol": SYMBOL,
                    "interval": "1m",
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": batch,
                },
                timeout=20,
            )
            if resp.status_code in (429, 418):
                time.sleep(int(resp.headers.get("Retry-After", 30)))
                continue
            data = resp.json()
            if not isinstance(data, list) or not data:
                retries += 1
                if retries >= 3:
                    break
                time.sleep(2 ** retries)
                continue
            df = pd.DataFrame(
                data,
                columns=[
                    "ts",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vol",
                    "close_time",
                    "quote_vol",
                    "trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )
            for col in ("open", "high", "low", "close", "vol"):
                df[col] = df[col].astype(float)
            df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
            df = df[["ts", "open", "high", "low", "close", "vol"]]
            all_dfs.insert(0, df)
            fetched += len(df)
            retries = 0
            end_ms = int(df["ts"].iloc[0].timestamp() * 1000) - 1
            if len(df) < batch:
                break
        except requests.RequestException:
            retries += 1
            if retries >= 3:
                break
            time.sleep(2)

    if not all_dfs:
        return pd.DataFrame()
    return (
        pd.concat(all_dfs)
        .drop_duplicates(subset="ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )


def _frame_features(df_1m, minutes):
    """Resample and compute SMI/RSI (and entry extras when needed)."""
    df = resample_ohlcv_closed(df_1m, minutes)
    if df.empty or len(df) < max(WARMUP_SMI, EMA_SPAN + 5, DONCHIAN_DLEN + 2):
        return None

    smi, _, _ = calc_smi(df["high"], df["low"], df["close"])
    rsi = calc_rsi_tv(df["close"], period=14)
    end_ts = df["ts"] + pd.Timedelta(minutes=minutes)
    out = pd.DataFrame(
        {
            "ts": df["ts"].to_numpy(),
            "end_ts": end_ts.to_numpy(),
            "close": df["close"].to_numpy(),
            "smi": smi.to_numpy(),
            "rsi": rsi.to_numpy(),
            "sell_main": ((smi <= SMI_SELL) & (rsi < RSI_MID)).to_numpy(),
            "buy_main": ((smi >= SMI_BUY) & (rsi > RSI_MID)).to_numpy(),
            "sell_sat": (smi <= SMI_SELL).to_numpy(),
            "buy_sat": (smi >= SMI_BUY).to_numpy(),
        }
    )
    return out


def _entry_features(df_1m, minutes):
    df = resample_ohlcv_closed(df_1m, minutes)
    if df.empty or len(df) < max(EMA_SPAN + 5, DONCHIAN_DLEN + 2, WARMUP_SMI):
        return None
    ema = calc_ema(df["close"], span=EMA_SPAN)
    don = calc_donchian_trend_series(
        df["close"].to_numpy(),
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        DONCHIAN_DLEN,
    )
    end_ts = df["ts"] + pd.Timedelta(minutes=minutes)
    return pd.DataFrame(
        {
            "ts": df["ts"].to_numpy(),
            "end_ts": end_ts.to_numpy(),
            "close": df["close"].to_numpy(),
            "ema": ema.to_numpy(),
            "don": don.to_numpy(),
            "above_ema": (df["close"] > ema).to_numpy(),
            "below_ema": (df["close"] < ema).to_numpy(),
            "don_green": (don.to_numpy() == 1),
            "don_red": (don.to_numpy() == -1),
        }
    )


def _bool_step(end_ts, values, grid):
    if end_ts is None or len(end_ts) == 0:
        return np.zeros(len(grid), dtype=bool)
    order = np.argsort(end_ts)
    ends = pd.DatetimeIndex(pd.to_datetime(end_ts[order], utc=True))
    vals = np.asarray(values, dtype=bool)[order]
    series = pd.Series(vals, index=ends)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        return np.zeros(len(grid), dtype=bool)
    filled = series.reindex(grid, method="ffill")
    first_valid = series.index[0]
    out = np.zeros(len(grid), dtype=bool)
    mask = grid >= first_valid
    out[mask] = filled.fillna(False).to_numpy()[mask].astype(bool)
    return out


def _precompute_stepped(frame_data, grid):
    """Cache stepped boolean series for main/confirm keys on the 1m grid."""
    stepped = {}
    for minutes, feat in frame_data.items():
        if feat is None:
            continue
        ends = feat["end_ts"].to_numpy()
        stepped[minutes] = {
            "sell_main": _bool_step(ends, feat["sell_main"].to_numpy(), grid),
            "buy_main": _bool_step(ends, feat["buy_main"].to_numpy(), grid),
            "sell_sat": _bool_step(ends, feat["sell_sat"].to_numpy(), grid),
            "buy_sat": _bool_step(ends, feat["buy_sat"].to_numpy(), grid),
        }
    return stepped


def _scan_side(side, stepped, entry_data, grid, start, end, raw_1m):
    """Replay one side (sell/buy) across the hierarchy; return signal dicts."""
    is_sell = side == "sell"
    main_key = "sell_main" if is_sell else "buy_main"
    confirm_key = "buy_sat" if is_sell else "sell_sat"

    active = np.full(len(grid), -1, dtype=int)
    halt = stepped.get(HALT_MAIN_MINUTES)
    halt_sat = halt[main_key] if halt is not None else np.zeros(len(grid), dtype=bool)

    main_masks = []
    for main, _cmin, _cstop, _entry in LEVELS:
        feat = stepped.get(main)
        main_masks.append(
            feat[main_key] if feat is not None else np.zeros(len(grid), dtype=bool)
        )

    stacked = np.vstack(main_masks) if main_masks else np.zeros((0, len(grid)), dtype=bool)
    for i in range(len(grid)):
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
    for level_idx, (main, confirm_min, confirm_stop, entry) in enumerate(LEVELS):
        entry_df = entry_data.get(entry)
        if entry_df is None:
            continue

        confirm_any = np.zeros(len(grid), dtype=bool)
        for minutes in range(confirm_min, confirm_stop):
            feat = stepped.get(minutes)
            if feat is None:
                continue
            confirm_any |= feat[confirm_key]

        stop_feat = stepped.get(confirm_stop)
        confirm_stop_mask = (
            stop_feat[confirm_key]
            if stop_feat is not None
            else np.zeros(len(grid), dtype=bool)
        )
        # Reverse-sat confirm window: main active + counter sat closed + not stop.
        # After the first confirmed reverse-sat close we keep watching (even if
        # counter clears) until main dies, confirm-stop hits, or we enter.
        confirm_window = (active == level_idx) & confirm_any & ~confirm_stop_mask
        hold_window = (active == level_idx) & ~confirm_stop_mask

        ends = pd.DatetimeIndex(pd.to_datetime(entry_df["end_ts"], utc=True))
        # Episode state: IDLE = no reverse sat seen; WATCH = waiting for the
        # two entry conditions to complete; REJECT = both conditions already
        # held on the first watched candle — dead until a fresh counter sat.
        IDLE, WATCH, REJECT = 0, 1, 2
        state = IDLE
        for row_i, candle_end in enumerate(ends):
            if candle_end < start_ts or candle_end > end_ts_limit:
                state = IDLE
                continue
            pos = int(grid.searchsorted(candle_end, side="right") - 1)
            if pos < 0 or not hold_window[pos]:
                state = IDLE
                continue

            above = bool(entry_df["above_ema"].iloc[row_i])
            below = bool(entry_df["below_ema"].iloc[row_i])
            green = bool(entry_df["don_green"].iloc[row_i])
            red = bool(entry_df["don_red"].iloc[row_i])
            price = float(entry_df["close"].iloc[row_i])

            counter_now = bool(confirm_window[pos])
            both_hold = (red and below) if is_sell else (green and above)

            if state == IDLE and counter_now:
                # First watched candle of a new episode: if both entry
                # conditions already hold, reject — nothing flipped after
                # the reverse sat. Otherwise start waiting for them.
                state = REJECT if both_hold else WATCH
            elif state == REJECT and not counter_now:
                # Counter sat ended; a fresh one may start a new episode.
                state = IDLE

            # WATCH survives the counter clearing — the entry may complete
            # after the reverse sat is over.
            hit = state == WATCH and both_hold

            if hit:
                future = raw_1m.loc[raw_1m["ts"] > candle_end]
                outcome, exit_price, exit_ts = evaluate_outcome(
                    "sell" if is_sell else "buy", price, future
                )
                signals.append(
                    {
                        "symbol": SYMBOL,
                        "type": "sell" if is_sell else "buy",
                        "time": _utc(candle_end),
                        "price": price,
                        "base_frame": main,
                        "confirm_frame": confirm_min,
                        "triple_frame": entry,
                        "confirm_stop": confirm_stop,
                        "outcome": outcome,
                        "exit_price": exit_price,
                        "exit_ts": exit_ts,
                    }
                )
                # One entry per reverse-sat episode; wait for a fresh counter.
                state = IDLE
    return signals


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


def scan_pullback_week(
    *,
    days=WEEK_DAYS,
    now=None,
    raw_1m=None,
):
    """Scan BTCUSDT pullback strategy over the last ``days``."""
    now = _utc(now) or datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    end = now

    if raw_1m is None:
        raw_1m = fetch_btc_1m_vision()
    if raw_1m is None or raw_1m.empty:
        return {
            "ready": False,
            "reason": "no_data",
            "start": start,
            "end": end,
            "wins": [],
            "losses": [],
            "opens": [],
            "total": 0,
            "market": "spot-vision",
            "symbol": SYMBOL,
        }

    raw_1m = raw_1m.sort_values("ts").reset_index(drop=True)
    # Build 1m evaluation grid on candle close times inside [start, end]
    # Use all 1m bars for indicator warmup, grid only for the week window.
    closes_1m = raw_1m["ts"] + pd.Timedelta(minutes=1)
    grid_mask = (closes_1m >= pd.Timestamp(start)) & (closes_1m <= pd.Timestamp(end))
    grid = pd.DatetimeIndex(closes_1m.loc[grid_mask].to_numpy())
    if len(grid) == 0:
        return {
            "ready": True,
            "start": start,
            "end": end,
            "wins": [],
            "losses": [],
            "opens": [],
            "total": 0,
            "market": "spot-vision",
            "symbol": SYMBOL,
            "symbols_scanned": 1,
        }

    needed = _all_needed_frames()
    frame_data = {}
    entry_frames = {lvl[3] for lvl in LEVELS}
    entry_data = {}
    for minutes in needed:
        if minutes in entry_frames:
            entry_data[minutes] = _entry_features(raw_1m, minutes)
        frame_data[minutes] = _frame_features(raw_1m, minutes)

    stepped = _precompute_stepped(frame_data, grid)
    all_signals = []
    all_signals.extend(
        _scan_side("sell", stepped, entry_data, grid, start, end, raw_1m)
    )
    all_signals.extend(
        _scan_side("buy", stepped, entry_data, grid, start, end, raw_1m)
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
        "wins": wins,
        "losses": losses,
        "opens": opens,
        "total": len(deduped),
        "market": "spot-vision",
        "symbol": SYMBOL,
        "symbols_scanned": 1,
    }


def _format_trade_line(trade):
    icon = "🟢" if trade["type"] == "buy" else "🔴"
    side = "شراء" if trade["type"] == "buy" else "بيع"
    frames = (
        f"{trade['base_frame']}m/"
        f"{trade['confirm_frame']}-{trade['confirm_stop'] - 1}m/"
        f"{trade['triple_frame']}m"
    )
    when = trade["time"].strftime("%m-%d %H:%M")
    return (
        f"{icon} <code>{html_escape(trade['symbol'])}</code> | {side} | "
        f"{frames} | {trade['price']:.4g} | <code>{when}</code> UTC"
    )


def format_pullback_week_report(result):
    if not result.get("ready"):
        reason = result.get("reason")
        if reason == "busy":
            return ["⏳ فحص استراتيجية الـ Pullback يعمل الآن — انتظر."]
        if reason == "no_data":
            return ["⚠️ تعذر جلب بيانات BTC من Binance Vision."]
        return ["⚠️ تعذر فحص استراتيجية الـ Pullback."]

    wins = result.get("wins") or []
    losses = result.get("losses") or []
    opens = result.get("opens") or []
    start = result["start"].strftime("%Y-%m-%d %H:%M")
    end = result["end"].strftime("%Y-%m-%d %H:%M")
    total = int(result.get("total") or 0)

    header = (
        "🗓️ <b>صفقات Pullback (SMI/EMA60/Donchian/RSI) — آخر 7 أيام</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"العملة: <code>{html_escape(result.get('symbol', SYMBOL))}</code> "
        f"(سوق: {html_escape(result.get('market', 'spot-vision'))})\n"
        f"الفترة: <code>{start}</code> → <code>{end}</code> UTC\n"
        f"إجمالي الصفقات: <b>{total}</b>\n"
        f"✅ نجاح (+{WIN_PCT:g}%): <b>{len(wins)}</b>\n"
        f"❌ خسارة (−{LOSS_PCT:g}% ضد الاتجاه): <b>{len(losses)}</b>\n"
        f"⏳ مفتوحة: <b>{len(opens)}</b>\n"
    )

    if total == 0:
        return [
            header
            + "\nلا توجد صفقات مطابقة لهذه الاستراتيجية خلال الأسبوع الماضي."
        ]

    chunks = [header]
    win_block = ["✅ <b>الناجحون</b> (تحقق +1%):"]
    win_block.extend(_format_trade_line(t) for t in wins) if wins else win_block.append("— لا يوجد")
    chunks.append("\n".join(win_block))

    loss_block = ["❌ <b>الخاسرون</b> (ارتداد 0.70% ضد الصفقة):"]
    loss_block.extend(_format_trade_line(t) for t in losses) if losses else loss_block.append("— لا يوجد")
    chunks.append("\n".join(loss_block))

    if opens:
        open_block = ["⏳ <b>مفتوحة</b>:"]
        open_block.extend(_format_trade_line(t) for t in opens)
        chunks.append("\n".join(open_block))

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


def handle_pullback_week_command(chat_id, send_telegram):
    """Telegram entry for pullback strategy week scan (BTC only)."""
    global _scan_running
    if not _scan_lock.acquire(blocking=False):
        send_telegram("⏳ فحص Pullback للأسبوع يعمل الآن — انتظر.", chat_id)
        return
    if _scan_running:
        _scan_lock.release()
        send_telegram("⏳ فحص Pullback للأسبوع يعمل الآن — انتظر.", chat_id)
        return

    _scan_running = True
    try:
        send_telegram(
            "📡 جاري فحص استراتيجية Pullback على <code>BTCUSDT</code> "
            f"لآخر 7 أيام...\nمعيار النجاح: <b>+{WIN_PCT:g}%</b> | "
            f"الخسارة: <b>{LOSS_PCT:g}%</b> ضد الصفقة.",
            chat_id,
        )
        result = scan_pullback_week()
        for chunk in format_pullback_week_report(result):
            send_telegram(chunk, chat_id)
    except Exception as exc:
        log.exception("pullback week command failed")
        send_telegram(
            f"❌ فشل فحص Pullback: <code>{html_escape(str(exc))}</code>",
            chat_id,
        )
    finally:
        _scan_running = False
        _scan_lock.release()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("Fetching BTC 1m via Binance Vision...")
    raw = fetch_btc_1m_vision()
    log.info("Bars: %s", 0 if raw is None else len(raw))
    result = scan_pullback_week(raw_1m=raw)
    for chunk in format_pullback_week_report(result):
        # Print plain-ish text for the agent summary
        text = (
            chunk.replace("<b>", "")
            .replace("</b>", "")
            .replace("<code>", "")
            .replace("</code>", "")
        )
        print(text)
        print()


if __name__ == "__main__":
    main()
