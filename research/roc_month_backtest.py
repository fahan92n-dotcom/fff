"""Last-month ROC research on BTCUSDT 5m (Binance).

Two ROC readings of the user's wording:
  A) maroc > 0 buy / maroc not > 0 sell  (indicator color vs zero)
  B) roc > maroc buy / roc < maroc sell  (ROC above/below its EMA)

Also the full sequential strategy with current confirm-TF ROC step A.
Same brackets as the Pine: TP 1.00% / SL 0.75%. Fill at next 5m close.
"""
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import ema_tv
from six_indicators_backtest import (
    PAIRS, SYMBOL, TP, SL,
    fetch_klines, tf_frame, indicators_for, align, at, fat,
    run_engine, outcome, wr,
)

DAYS = 120
MONTH_DAYS = 30
ROC_LEN = 48
COMMISSION = 0.08  # percent round-trip used in earlier research notes


def roc_pack(tfd, length=ROC_LEN):
    c = tfd["close"]
    prev = c.shift(length).to_numpy()
    cv = c.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        roc = np.where(prev != 0, (cv - prev) / prev * 100.0, 0.0)
    maroc = ema_tv(pd.Series(roc, index=c.index), length).to_numpy()
    ok = np.isfinite(roc) & np.isfinite(maroc)
    maroc_up = ok & (maroc > 0)
    above_ma = ok & (roc > maroc)
    return roc, maroc, ok, maroc_up, above_ma


def cross_up(flag):
    prev = np.roll(flag, 1)
    prev[0] = False
    return flag & ~prev


def cross_dn(flag):
    prev = np.roll(flag, 1)
    prev[0] = False
    return ~flag & prev


def summarize(results):
    w, l, amb, op, rate = wr(results)
    closed = w + l
    gross = w * TP - l * SL
    net = w * (TP - COMMISSION) - l * (SL + COMMISSION)
    return {
        "n": len(results), "win": w, "loss": l, "amb": amb, "open": op,
        "wr": rate, "gross": gross, "net": net, "closed": closed,
    }


def fmt(s):
    wr_s = "n/a" if not np.isfinite(s["wr"]) else f"{s['wr']:.1f}%"
    return (f"n={s['n']}  win={s['win']}  loss={s['loss']}  "
            f"(amb={s['amb']} open={s['open']})  WR={wr_s}  "
            f"gross={s['gross']:+.2f}%  net≈{s['net']:+.2f}%")


def take_month(signals, ts, start):
    return [(t, side, tag) for t, side, tag in signals if ts[t] >= start]


def one_position(signals, close, high, low):
    """Keep first signal; ignore later ones until TP/SL/open-horizon exits."""
    ordered = sorted(signals)
    taken = []
    busy_until = -1
    for t, side, tag in ordered:
        if t <= busy_until:
            continue
        r = outcome(t, side, close, high, low, TP, SL)
        taken.append((t, side, tag, r))
        e = close[t]
        if side == "L":
            tp_px, sl_px = e * (1 + TP / 100), e * (1 - SL / 100)
        else:
            tp_px, sl_px = e * (1 - TP / 100), e * (1 + SL / 100)
        exit_t = min(t + 2016, len(close) - 1)
        for u in range(t + 1, min(t + 2016, len(close))):
            if side == "L":
                hit = high[u] >= tp_px or low[u] <= sl_px
            else:
                hit = low[u] <= tp_px or high[u] >= sl_px
            if hit:
                exit_t = u
                break
        busy_until = exit_t
    return taken


def roc_cross_signals(maps, packs, n, kind):
    """kind: 'zero' (maroc vs 0) or 'ma' (roc vs maroc). Confirm TFs only."""
    out = []
    confs = sorted({p[1] for p in PAIRS})
    for cf in confs:
        ci, cnew = maps[cf]
        _roc, _maroc, ok, maroc_up, above_ma = packs[cf]
        flag = maroc_up if kind == "zero" else above_ma
        # Crosses on the HTF series, then seen on the 5m bar that closes that HTF candle.
        up = at(cross_up(flag) & ok, ci) & cnew
        dn = at(cross_dn(flag) & ok, ci) & cnew
        tag = f"conf{cf}"
        out += [(i, "L", tag) for i in np.flatnonzero(up)]
        out += [(i, "S", tag) for i in np.flatnonzero(dn)]
    return out


def full_strategy_signals(n, data, maps, roc_key_up, roc_key_dn):
    signals = []
    for (mn, cf, en, cx) in PAIRS:
        mi, mnew = maps[mn]
        ci, _ = maps[cf]
        ei, enew = maps[en]
        xi, xnew = maps[cx]
        D, C, E, X = data[mn], data[cf], data[en], data[cx]
        m_rsi = fat(D["rsi"], mi)
        c_rsi = fat(C["rsi"], ci)
        with np.errstate(invalid="ignore"):
            gL = ((c_rsi >= 50) & (c_rsi <= 60)
                  & (m_rsi <= c_rsi - 3) & (m_rsi >= c_rsi - 10))
            gS = ((c_rsi >= 40) & (c_rsi <= 50)
                  & (m_rsi >= c_rsi + 3) & (m_rsi <= c_rsi + 10))
        fL, _, _ = run_engine(
            n, xnew & at(X["satL"], xi), mnew,
            at(D["satL"], mi), at(D["macdL"], mi), at(D["donG"], mi), at(D["emaL"], mi),
            at(C["histG"], ci), at(C[roc_key_up], ci), at(E["donR"], ei), enew,
            at(E["satL"], ei), at(E["touchL"], ei), at(E["crossUp"], ei),
            at(E["stUp"], ei), gL)
        fS, _, _ = run_engine(
            n, xnew & at(X["satS"], xi), mnew,
            at(D["satS"], mi), at(D["macdS"], mi), at(D["donR"], mi), at(D["emaS"], mi),
            at(C["histR"], ci), at(C[roc_key_dn], ci), at(E["donG"], ei), enew,
            at(E["satS"], ei), at(E["touchS"], ei), at(E["crossDn"], ei),
            at(E["stDn"], ei), gS)
        tag = f"{mn}/{cf}/{en}"
        signals += [(t, "L", tag) for t in fL] + [(t, "S", tag) for t in fS]
    signals.sort()
    return signals


def report_block(title, signals, ts, start, close, high, low):
    month = take_month(signals, ts, start)
    res = [outcome(t, s, close, high, low, TP, SL) for t, s, _ in month]
    print(f"\n=== {title} ===")
    print("  all overlapping signals:", fmt(summarize(res)))
    taken = one_position(month, close, high, low)
    res1 = [r for *_, r in taken]
    print("  one position at a time: ", fmt(summarize(res1)))
    per = {}
    for t, side, tag, r in taken:
        per.setdefault(tag, []).append(r)
    for tag in sorted(per):
        st = summarize(per[tag])
        wr_s = "n/a" if not np.isfinite(st["wr"]) else f"{st['wr']:.1f}%"
        print(f"     {tag:>12}: n={st['n']:3d}  WR={wr_s:>6}  "
              f"(win {st['win']} / loss {st['loss']})")
    print("  trades:")
    for t, side, tag, r in taken:
        print(f"     {ts.iloc[t]}  {side}  {tag:12}  {r}")
    return taken


def main():
    print(f"fetching {SYMBOL} 5m x {DAYS}d ...", flush=True)
    cache = f"/tmp/{SYMBOL}_5m_{DAYS}d.pkl"
    if os.path.exists(cache):
        df5 = pd.read_pickle(cache)
        print("loaded cache", cache, flush=True)
    else:
        df5 = fetch_klines(SYMBOL, DAYS)
        df5.to_pickle(cache)
    print("bars:", len(df5), df5["ts"].iloc[0], "->", df5["ts"].iloc[-1], flush=True)
    ts = df5["ts"]
    start = ts.iloc[-1] - timedelta(days=MONTH_DAYS)
    print(f"month window: {start} -> {ts.iloc[-1]}", flush=True)
    base_end = (df5["ts"] + pd.Timedelta(minutes=5)).to_numpy()
    close = df5["close"].to_numpy()
    high = df5["high"].to_numpy()
    low = df5["low"].to_numpy()
    n = len(df5)

    tfs = sorted({m for p in PAIRS for m in p})
    data, maps, frames, packs = {}, {}, {}, {}
    for m in tfs:
        tfd = tf_frame(df5, m)
        frames[m] = tfd
        data[m] = indicators_for(tfd)
        # extra ROC vs its EMA (فوق/تحت المتوسط)
        _roc, _maroc, ok, _up, above = roc_pack(tfd)
        data[m]["rocAboveMa"] = ok & above
        data[m]["rocBelowMa"] = ok & ~above
        maps[m] = align(tfd, base_end)
        packs[m] = roc_pack(tfd)
        print(f"tf {m}m candles {len(tfd)}", flush=True)

    print("\n########## LAST 30 DAYS — BTCUSDT ##########")
    print("brackets TP 1.00% / SL 0.75%, fill = next 5m close")

    full_a = full_strategy_signals(n, data, maps, "rocU", "rocD")
    report_block(
        "1) Full strategy + ROC step (maroc>0 buy / not>0 sell) on confirm TF",
        full_a, ts, start, close, high, low)

    full_b = full_strategy_signals(n, data, maps, "rocAboveMa", "rocBelowMa")
    report_block(
        "2) Full strategy + ROC step (roc>maroc buy / roc<maroc sell) on confirm TF",
        full_b, ts, start, close, high, low)

    zero_sigs = roc_cross_signals(maps, packs, n, "zero")
    report_block(
        "3) ROC ONLY: maroc crosses above 0 = buy, below 0 = sell (confirm TFs)",
        zero_sigs, ts, start, close, high, low)

    ma_sigs = roc_cross_signals(maps, packs, n, "ma")
    report_block(
        "4) ROC ONLY: roc crosses above maroc = buy, below maroc = sell (confirm TFs)",
        ma_sigs, ts, start, close, high, low)


if __name__ == "__main__":
    main()
