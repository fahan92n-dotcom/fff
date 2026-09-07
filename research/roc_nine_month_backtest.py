"""9-month ROC research on BTCUSDT 5m (Binance).

Same ROC rule as the 30-day 62.5% study:
  confirm-TF ROC length 48, maroc = EMA(roc, 48)
  A) maroc > 0 buy / not > 0 sell  (indicator color vs zero)
  B) roc > maroc buy / roc < maroc sell

Full sequential strategy with that ROC as step 6 (confirm TF),
TP 1.00% / SL 0.75%, fill at next 5m close. RSI gate on.
Also reports the no-ROC baseline on the same window.
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
    PAIRS, SYMBOL, TP, SL, DAYS,
    fetch_klines, tf_frame, indicators_for, align, at, fat,
    run_engine, outcome, wr,
)

ROC_LEN = 48
COMMISSION = 0.08
MONTH_DAYS = 30


def roc_pack(tfd, length=ROC_LEN):
    c = tfd["close"]
    prev = c.shift(length).to_numpy()
    cv = c.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        roc = np.where(prev != 0, (cv - prev) / prev * 100.0, np.nan)
    maroc = ema_tv(pd.Series(roc, index=c.index), length).to_numpy()
    ok = np.isfinite(roc) & np.isfinite(maroc)
    maroc_up = ok & (maroc > 0)
    above_ma = ok & (roc > maroc)
    return roc, maroc, ok, maroc_up, above_ma


def attach_roc(z, tfd):
    _roc, _maroc, ok, maroc_up, above_ma = roc_pack(tfd)
    z["rocU"] = maroc_up
    z["rocD"] = ok & ~maroc_up
    z["rocAboveMa"] = ok & above_ma
    z["rocBelowMa"] = ok & ~above_ma
    return z


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


def take_window(signals, ts, start):
    return [(t, side, tag) for t, side, tag in signals if ts[t] >= start]


def one_position(signals, close, high, low):
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


def full_strategy_signals(n, data, maps, s6_up, s6_dn):
    signals = []
    always = np.ones(n, dtype=bool)
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
        if s6_up is None:
            up, dn = always, always
        else:
            up = at(C[s6_up], ci)
            dn = at(C[s6_dn], ci)
        fL, _, _ = run_engine(
            n, xnew & at(X["satL"], xi), mnew,
            at(D["satL"], mi), at(D["macdL"], mi), at(D["donG"], mi), at(D["emaL"], mi),
            at(C["histG"], ci), up, at(E["donR"], ei), enew,
            at(E["satL"], ei), at(E["touchL"], ei), at(E["crossUp"], ei),
            at(E["stUp"], ei), gL)
        fS, _, _ = run_engine(
            n, xnew & at(X["satS"], xi), mnew,
            at(D["satS"], mi), at(D["macdS"], mi), at(D["donR"], mi), at(D["emaS"], mi),
            at(C["histR"], ci), dn, at(E["donG"], ei), enew,
            at(E["satS"], ei), at(E["touchS"], ei), at(E["crossDn"], ei),
            at(E["stDn"], ei), gS)
        tag = f"{mn}/{cf}/{en}"
        signals += [(t, "L", tag) for t in fL] + [(t, "S", tag) for t in fS]
    signals.sort()
    return signals


def report_block(title, signals, ts, start, close, high, low, list_trades=False):
    window = take_window(signals, ts, start) if start is not None else list(signals)
    res = [outcome(t, s, close, high, low, TP, SL) for t, s, _ in window]
    print(f"\n=== {title} ===")
    print("  overlapping signals:   ", fmt(summarize(res)))
    taken = one_position(window, close, high, low)
    res1 = [r for *_, r in taken]
    print("  one position at a time:", fmt(summarize(res1)))
    per = {}
    for t, side, tag, r in taken:
        per.setdefault(tag, []).append(r)
    for tag in sorted(per):
        st = summarize(per[tag])
        wr_s = "n/a" if not np.isfinite(st["wr"]) else f"{st['wr']:.1f}%"
        print(f"     {tag:>12}: n={st['n']:3d}  WR={wr_s:>6}  "
              f"(win {st['win']} / loss {st['loss']})")
    if list_trades:
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
    month_start = ts.iloc[-1] - timedelta(days=MONTH_DAYS)
    print(f"full window: {ts.iloc[0]} -> {ts.iloc[-1]}", flush=True)
    print(f"last-30d check: {month_start} -> {ts.iloc[-1]}", flush=True)
    close = df5["close"].to_numpy()
    high = df5["high"].to_numpy()
    low = df5["low"].to_numpy()
    n = len(df5)
    base_end = (df5["ts"] + pd.Timedelta(minutes=5)).to_numpy()

    tfs = sorted({m for p in PAIRS for m in p})
    data, maps = {}, {}
    for m in tfs:
        tfd = tf_frame(df5, m)
        z = indicators_for(tfd)
        data[m] = attach_roc(z, tfd)
        maps[m] = align(tfd, base_end)
        print(f"tf {m}m candles {len(tfd)}", flush=True)

    print("\n########## BTCUSDT — RSI gate ON, TP 1.00% / SL 0.75% ##########")
    variants = [
        ("1) Full + ROC maroc>0 (confirm TF) — same rule as 62.5% month study",
         "rocU", "rocD"),
        ("2) Full + ROC roc>maroc (confirm TF)",
         "rocAboveMa", "rocBelowMa"),
        ("3) Full WITHOUT ROC (step 6 always true)",
         None, None),
    ]
    built = {}
    for title, up, dn in variants:
        built[title] = full_strategy_signals(n, data, maps, up, dn)

    print("\n======== FULL ~9 MONTHS ========")
    for title, sigs in built.items():
        report_block(title, sigs, ts, None, close, high, low, list_trades=False)

    print("\n======== LAST 30 DAYS (sanity vs earlier 8-trade sample) ========")
    for title, sigs in built.items():
        report_block(title, sigs, ts, month_start, close, high, low, list_trades=True)


if __name__ == "__main__":
    main()
