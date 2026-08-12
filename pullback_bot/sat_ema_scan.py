"""Main-TF SMI sat + EMA60 close only. No Donchian, reverse sat, or extra gates.

Buy: during SMI buy-sat (>= +40), enter on the first closed main candle
above EMA60. Sell: during SMI sell-sat (<= -40), enter on the first
closed main candle below EMA60. One entry per sat episode.
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

from indicators import WARMUP_SMI, calc_ema, calc_smi, resample_ohlcv_closed
from pullback_bot.strategy import (
    DEDUPE_HOURS,
    EMA_SPAN,
    LEVELS,
    LOSS_PCT,
    MONTH_DAYS,
    SMI_BUY,
    SMI_SELL,
    WIN_PCT,
    _utc,
    evaluate_outcome,
    fetch_1m_vision,
)

log = logging.getLogger(__name__)

SYMBOLS = ("BTCUSDT", "ETHUSDT", "XRPUSDT")
MAINS = tuple(lvl[0] for lvl in LEVELS)
MIN_1M_BARS = 90_000


def _main_features(df_1m, minutes):
    df = resample_ohlcv_closed(df_1m, minutes)
    if df.empty or len(df) < max(WARMUP_SMI, EMA_SPAN + 5):
        return None
    smi, _, _ = calc_smi(df["high"], df["low"], df["close"])
    ema = calc_ema(df["close"], span=EMA_SPAN)
    end_ts = df["ts"] + pd.Timedelta(minutes=minutes)
    return pd.DataFrame(
        {
            "ts": df["ts"].to_numpy(),
            "end_ts": end_ts.to_numpy(),
            "close": df["close"].to_numpy(),
            "ema": ema.to_numpy(),
            "buy_sat": (smi >= SMI_BUY).to_numpy(),
            "sell_sat": (smi <= SMI_SELL).to_numpy(),
            "above_ema": (df["close"] > ema).to_numpy(),
            "below_ema": (df["close"] < ema).to_numpy(),
        }
    )


def _scan_side(side, feat, start, end, raw_1m, symbol, minutes):
    """One entry per SMI-sat episode on the first correct EMA60 close."""
    is_sell = side == "sell"
    sat = feat["sell_sat"].to_numpy() if is_sell else feat["buy_sat"].to_numpy()
    side_ok = feat["below_ema"].to_numpy() if is_sell else feat["above_ema"].to_numpy()
    closes = feat["close"].to_numpy()
    ends = pd.DatetimeIndex(pd.to_datetime(feat["end_ts"], utc=True))
    start_ts = pd.Timestamp(start)
    end_ts_limit = pd.Timestamp(end)

    signals = []
    in_ep = False
    entered = False
    for i, candle_end in enumerate(ends):
        if not sat[i]:
            in_ep = False
            entered = False
            continue
        if not in_ep:
            in_ep = True
            entered = False
        if entered:
            continue
        if candle_end < start_ts or candle_end > end_ts_limit:
            continue
        if not side_ok[i]:
            continue
        price = float(closes[i])
        future = raw_1m.loc[raw_1m["ts"] > candle_end]
        outcome, exit_price, exit_ts = evaluate_outcome(
            "sell" if is_sell else "buy", price, future
        )
        signals.append(
            {
                "symbol": symbol,
                "type": "sell" if is_sell else "buy",
                "time": _utc(candle_end),
                "price": price,
                "base_frame": minutes,
                "confirm_frame": minutes,
                "triple_frame": minutes,
                "confirm_stop": minutes,
                "outcome": outcome,
                "exit_price": exit_price,
                "exit_ts": exit_ts,
            }
        )
        entered = True
    return signals


def _dedupe(signals, hours=DEDUPE_HOURS):
    if not signals:
        return []
    ordered = sorted(signals, key=lambda item: item["time"])
    window = timedelta(hours=hours)
    last_by_key = {}
    kept = []
    for sig in ordered:
        key = (sig["symbol"], sig["type"], sig["base_frame"])
        prev = last_by_key.get(key)
        if prev is not None and sig["time"] - prev < window:
            continue
        last_by_key[key] = sig["time"]
        kept.append(sig)
    return kept


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
        target = max(MIN_1M_BARS, int(days) * 1440 + 45_000)
        log.info("Fetching %s 1m bars for %s...", target, symbol)
        raw_1m = fetch_1m_vision(symbol, target=target)
        log.info("%s bars: %s", symbol, 0 if raw_1m is None else len(raw_1m))
    if raw_1m is None or raw_1m.empty:
        return empty

    raw_1m = raw_1m.sort_values("ts").reset_index(drop=True)
    all_signals = []
    for minutes in MAINS:
        feat = _main_features(raw_1m, minutes)
        if feat is None:
            log.warning("%s %sm: not enough bars", symbol, minutes)
            continue
        all_signals.extend(
            _scan_side("buy", feat, start, end, raw_1m, symbol, minutes)
        )
        all_signals.extend(
            _scan_side("sell", feat, start, end, raw_1m, symbol, minutes)
        )

    deduped = _dedupe(all_signals)
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
        "raw_signals": len(all_signals),
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
            symbol, days=days, now=now, raw_1m=raw_by_symbol.get(symbol)
        )
        per_symbol[symbol] = result
        if not result.get("ready"):
            failed.append(symbol)
            continue
        merged.extend(result["wins"])
        merged.extend(result["losses"])
        merged.extend(result["opens"])
    deduped = _dedupe(merged)
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
    }


def _pnl(trade):
    if trade["outcome"] == "win":
        return WIN_PCT
    if trade["outcome"] == "loss":
        return -LOSS_PCT
    return 0.0


def _summarize(trades):
    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    opens = [t for t in trades if t["outcome"] == "open"]
    closed = len(wins) + len(losses)
    return {
        "total": len(trades),
        "wins": wins,
        "losses": losses,
        "opens": opens,
        "win_rate": (100.0 * len(wins) / closed) if closed else 0.0,
        "pnl": sum(_pnl(t) for t in trades),
    }


def format_report(result):
    if not result.get("ready"):
        return ["⚠️ تعذر الفحص."]
    trades = list(result.get("wins") or [])
    trades.extend(result.get("losses") or [])
    trades.extend(result.get("opens") or [])
    trades.sort(key=lambda item: item["time"])
    summary = _summarize(trades)
    start = result["start"].strftime("%Y-%m-%d %H:%M")
    end = result["end"].strftime("%Y-%m-%d %H:%M")
    days = int(result.get("days") or MONTH_DAYS)
    symbols = ", ".join(result.get("symbols") or SYMBOLS)
    header = (
        f"🗓️ <b>تشبع SMI + إغلاق EMA60 فقط — آخر {days} يومًا</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"العملات: <code>{html_escape(symbols)}</code>\n"
        f"الفترة: <code>{start}</code> → <code>{end}</code> UTC\n"
        "بدون Donchian، بدون عكس، بدون هرمية.\n"
        "شراء: تشبع SMI وشموع الرئيسي تقفل فوق EMA60. "
        "بيع: تشبع SMI وتقفل تحت EMA60. صفقة واحدة لكل حلقة تشبع.\n"
        f"ربح +{WIN_PCT:g}% | خسارة {LOSS_PCT:g}%\n"
        f"الإجمالي: صفقات <b>{summary['total']}</b> | "
        f"✅ {len(summary['wins'])} | ❌ {len(summary['losses'])} | "
        f"⏳ {len(summary['opens'])} | "
        f"نجاح {summary['win_rate']:.0f}% | "
        f"صافي {summary['pnl']:+.2f}%\n"
    )
    chunks = [header]
    symbol_lines = ["💱 <b>حسب العملة</b>"]
    for symbol in result.get("symbols") or SYMBOLS:
        sub = _summarize([t for t in trades if t["symbol"] == symbol])
        symbol_lines.append(
            f"<code>{html_escape(symbol)}</code>: صفقات <b>{sub['total']}</b> | "
            f"✅ {len(sub['wins'])} | ❌ {len(sub['losses'])} | "
            f"نجاح {sub['win_rate']:.0f}% | صافي {sub['pnl']:+.2f}%"
        )
    chunks.append("\n".join(symbol_lines))
    level_lines = ["📊 <b>حسب الفريم الرئيسي</b>"]
    for minutes in MAINS:
        sub = _summarize([t for t in trades if t["base_frame"] == minutes])
        if sub["total"] == 0:
            continue
        level_lines.append(
            f"{minutes}م: صفقات <b>{sub['total']}</b> | "
            f"✅ {len(sub['wins'])} | ❌ {len(sub['losses'])} | "
            f"نجاح {sub['win_rate']:.0f}% | صافي {sub['pnl']:+.2f}%"
        )
    chunks.append("\n".join(level_lines))
    if trades:
        lines = ["📋 <b>الصفقات</b>"]
        for trade in trades:
            icon = "🟢" if trade["type"] == "buy" else "🔴"
            side = "شراء" if trade["type"] == "buy" else "بيع"
            mark = {"win": "✅", "loss": "❌", "open": "⏳"}[trade["outcome"]]
            when = trade["time"].strftime("%m-%d %H:%M")
            lines.append(
                f"{mark} {icon} <code>{html_escape(trade['symbol'])}</code> | "
                f"{side} | {trade['base_frame']}م | {trade['price']:.4g} | "
                f"<code>{when}</code> UTC"
            )
        chunks.append("\n".join(lines))
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
    result = scan_all(days=MONTH_DAYS)
    print(format_plain_report(result))


if __name__ == "__main__":
    main()
