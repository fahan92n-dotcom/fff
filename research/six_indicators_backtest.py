"""
Research: replicate pine/six_indicators_strategy.pine logic in Python and
backtest on Binance klines to diagnose the sub-50% win rate.

Faithful to the Pine implementation:
- all conditions read from the LAST CLOSED candle of each timeframe
- engine advances on chart (5m) bars; fills at the close of the rollover bar
- cancellation: same-direction SMI-saturated close on the next-larger main TF
- TP 1.00% / SL 0.75% brackets
Variants measured: with/without RSI gate, no-delay entry, inverted TP/SL,
random-entry benchmark.
"""
import os, sys, time
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indicators import (
    ema_tv, calc_rsi_tv, calc_smi, calc_stoch_tv,
    calc_donchian_trend_series, _resample_ohlcv_frame, candle_period_ends,
)

BASE_URL = "https://data-api.binance.vision/api/v3/klines"
SYMBOL = "BTCUSDT"
DAYS = 270
TP, SL = 1.00, 0.75
GAP_MAX = 3
HORIZON = 2016  # 5m bars = 1 week

PAIRS = [  # (main, confirm, entry, cancel_main) minutes; cancel for last = 300
    (15, 45, 5, 30), (30, 90, 10, 45), (45, 135, 15, 60), (60, 180, 20, 90),
    (90, 270, 30, 120), (120, 360, 40, 150), (150, 450, 50, 180),
    (180, 540, 60, 210), (210, 630, 70, 240), (240, 720, 80, 300),
]


def fetch_klines(symbol, days):
    end = int(time.time() * 1000)
    start = end - days * 86400_000
    rows = []
    cur = start
    while cur < end:
        r = requests.get(BASE_URL, params={
            "symbol": symbol, "interval": "5m", "startTime": cur, "limit": 1000,
        }, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][0] + 300_000
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows, columns=[
        "ts", "open", "high", "low", "close", "vol", "close_time", "qv",
        "n", "tbv", "tbq", "ig"])
    df = df[["ts", "open", "high", "low", "close", "vol"]].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms", utc=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    # drop the in-progress last 5m candle
    return df.iloc[:-1].reset_index(drop=True)


def tf_frame(df5, minutes):
    if minutes == 5:
        out = df5.copy()
    else:
        out = _resample_ohlcv_frame(df5, minutes)
    ends = candle_period_ends(out["ts"], minutes)
    # keep only candles fully closed within our 5m data span
    last_end = df5["ts"].iloc[-1] + pd.Timedelta(minutes=5)
    keep = ends <= last_end
    out = out.loc[keep].reset_index(drop=True)
    out["end"] = np.asarray(ends)[keep.to_numpy()]
    return out


def indicators_for(tfd):
    c, h, l = tfd["close"], tfd["high"], tfd["low"]
    smi, _, _ = calc_smi(h, l, c, k=10, smooth_period=1, d=3, c=10)
    macd = ema_tv(c, 12) - ema_tv(c, 26)
    hist = macd - ema_tv(macd, 9)
    don = calc_donchian_trend_series(c.values, h.values, l.values, 20)
    ema50 = ema_tv(c, 50)
    rsi = calc_rsi_tv(c, 14)
    rsi_ma = rsi.rolling(14, min_periods=14).mean()
    k, _ = calc_stoch_tv(c, h, l, 15, 3, 3)
    med = (h + l) / 2.0
    ao = med.rolling(5, min_periods=5).mean() - med.rolling(34, min_periods=34).mean()
    z = {
        "aoUp": (ao > 0).to_numpy(), "aoDn": (ao < 0).to_numpy(),
        "satL": (smi <= -40).to_numpy(), "satS": (smi >= 40).to_numpy(),
        "macdL": ((hist < 0) & (macd >= hist)).to_numpy(),
        "macdS": ((hist > 0) & (macd <= hist)).to_numpy(),
        "donG": (don == 1).to_numpy(), "donR": (don == -1).to_numpy(),
        "emaL": (c < ema50).to_numpy(), "emaS": (c > ema50).to_numpy(),
        "histG": (hist > 0).to_numpy(), "histR": (hist < 0).to_numpy(),
        "rsi": rsi.to_numpy(),
        "touchL": (rsi <= 35).to_numpy(), "touchS": (rsi >= 65).to_numpy(),
        "stUp": (k > 20).to_numpy(), "stDn": (k < 80).to_numpy(),
    }
    ru = rsi.to_numpy(); ma = rsi_ma.to_numpy()
    with np.errstate(invalid="ignore"):
        above = ru > ma
        below = ru < ma
    prev_above = np.roll(above, 1); prev_above[0] = False
    prev_below = np.roll(below, 1); prev_below[0] = False
    valid = np.isfinite(ru) & np.isfinite(ma)
    pvalid = np.roll(valid, 1); pvalid[0] = False
    z["crossUp"] = above & ~prev_above & valid & pvalid
    z["crossDn"] = below & ~prev_below & valid & pvalid
    return z


def align(tfd, base_end):
    ends = tfd["end"].to_numpy()
    idx = np.searchsorted(ends, base_end, side="right") - 1
    new = np.zeros(len(base_end), dtype=bool)
    new[1:] = idx[1:] > idx[:-1]
    new[0] = idx[0] >= 0
    return idx, new


def at(cond, idx):
    out = np.zeros(len(idx), dtype=bool)
    ok = idx >= 0
    out[ok] = np.nan_to_num(cond[idx[ok]]).astype(bool)
    return out


def fat(vals, idx):
    out = np.full(len(idx), np.nan)
    ok = idx >= 0
    out[ok] = vals[idx[ok]]
    return out


def run_engine(n, cancel, mainNew, s1, s2, s3, s4, s5, s5b, s6, entryNew,
               s7, s8t, s8c, s9st, s9gate):
    st = 0; gap = 0
    fires = []
    reach = np.zeros(11, dtype=int)
    gate_blocks = 0
    for t in range(n):
        just = False
        if cancel[t]:
            st = 0
        if st == 0 and mainNew[t] and s1[t]:
            st = 1; reach[1] += 1
        if st == 1 and mainNew[t] and s2[t]:
            st = 2; reach[2] += 1
        if st == 2 and s3[t]:
            st = 3; reach[3] += 1
        if st == 3 and s4[t]:
            st = 4; reach[4] += 1
        if st == 4 and s5[t]:
            st = 5; reach[5] += 1
        if st == 5 and s5b[t]:
            st = 6; reach[6] += 1
        if st == 6 and s6[t]:
            st = 7; reach[7] += 1
        if st == 7 and entryNew[t] and s7[t]:
            st = 8; reach[8] += 1
        if st == 8 and entryNew[t] and s8t[t]:
            st = 9; reach[9] += 1
        if st in (9, 10) and entryNew[t] and s8c[t]:
            st = 10; gap = 0; just = True; reach[10] += 1
        if st == 10 and entryNew[t]:
            if not just:
                gap += 1
            if gap > GAP_MAX:
                st = 9
            elif s9st[t]:
                if s9gate[t]:
                    fires.append(t)
                    st = 0
                else:
                    gate_blocks += 1
    return fires, reach, gate_blocks


def outcome(t0, side, close, high, low, tp, sl, entry_price=None):
    e = close[t0] if entry_price is None else entry_price
    if side == "L":
        tp_px, sl_px = e * (1 + tp / 100), e * (1 - sl / 100)
    else:
        tp_px, sl_px = e * (1 - tp / 100), e * (1 + sl / 100)
    for t in range(t0 + 1, min(t0 + HORIZON, len(close))):
        if side == "L":
            hit_tp, hit_sl = high[t] >= tp_px, low[t] <= sl_px
        else:
            hit_tp, hit_sl = low[t] <= tp_px, high[t] >= sl_px
        if hit_tp and hit_sl:
            return "amb"
        if hit_sl:
            return "loss"
        if hit_tp:
            return "win"
    return "open"


def wr(results):
    w = results.count("win"); l = results.count("loss") + results.count("amb")
    tot = w + l
    return (w, l, results.count("amb"), results.count("open"),
            100.0 * w / tot if tot else float("nan"))


def main():
    print(f"fetching {SYMBOL} 5m x {DAYS}d ...", flush=True)
    df5 = fetch_klines(SYMBOL, DAYS)
    print("bars:", len(df5), df5['ts'].iloc[0], "->", df5['ts'].iloc[-1], flush=True)
    base_end = (df5["ts"] + pd.Timedelta(minutes=5)).to_numpy()
    close = df5["close"].to_numpy(); high = df5["high"].to_numpy(); low = df5["low"].to_numpy()
    n = len(df5)

    tfs = sorted({m for p in PAIRS for m in p})
    data, maps = {}, {}
    for m in tfs:
        tfd = tf_frame(df5, m)
        data[m] = indicators_for(tfd)
        maps[m] = align(tfd, base_end)
        print(f"tf {m}m candles {len(tfd)}", flush=True)

    all_signals = {}
    for gate_on in (True, False):
        signals = []
        funnel = {}
        blocks = 0
        for (mn, cf, en, cx) in PAIRS:
            mi, mnew = maps[mn]; ci, _ = maps[cf]; ei, enew = maps[en]; xi, xnew = maps[cx]
            D, C, E, X = data[mn], data[cf], data[en], data[cx]
            m_rsi = fat(D["rsi"], mi); c_rsi = fat(C["rsi"], ci)
            with np.errstate(invalid="ignore"):
                gL = (c_rsi >= 50) & (c_rsi <= 60) & (m_rsi <= c_rsi - 3) & (m_rsi >= c_rsi - 10)
                gS = (c_rsi >= 40) & (c_rsi <= 50) & (m_rsi >= c_rsi + 3) & (m_rsi <= c_rsi + 10)
            if not gate_on:
                gL = np.ones(n, dtype=bool); gS = np.ones(n, dtype=bool)
            fL, rL, bL = run_engine(n, xnew & at(X["satL"], xi), mnew,
                at(D["satL"], mi), at(D["macdL"], mi), at(D["donG"], mi), at(D["emaL"], mi),
                at(C["histG"], ci), at(C["aoUp"], ci), at(E["donR"], ei), enew,
                at(E["satL"], ei), at(E["touchL"], ei), at(E["crossUp"], ei),
                at(E["stUp"], ei), gL)
            fS, rS, bS = run_engine(n, xnew & at(X["satS"], xi), mnew,
                at(D["satS"], mi), at(D["macdS"], mi), at(D["donR"], mi), at(D["emaS"], mi),
                at(C["histR"], ci), at(C["aoDn"], ci), at(E["donG"], ei), enew,
                at(E["satS"], ei), at(E["touchS"], ei), at(E["crossDn"], ei),
                at(E["stDn"], ei), gS)
            tag = f"{mn}/{cf}/{en}"
            funnel[tag] = (rL + rS).tolist()
            blocks += bL + bS
            signals += [(t, "L", tag) for t in fL] + [(t, "S", tag) for t in fS]
        signals.sort()
        all_signals[gate_on] = signals
        label = "WITH gate" if gate_on else "NO gate"
        res = [outcome(t, s, close, high, low, TP, SL) for t, s, _ in signals]
        w, l, amb, op, rate = wr(res)
        print(f"\n=== {label}: signals={len(signals)} win={w} loss={l} (amb={amb}) open={op} WR={rate:.1f}%")
        if gate_on:
            print("gate blocked entries (stoch ok, gate false):", blocks)
        per = {}
        for (t, s, tag), r in zip(signals, res):
            per.setdefault(tag, []).append(r)
        for tag, rs in per.items():
            w2, l2, a2, o2, rt = wr(rs)
            print(f"   {tag:>12}: n={len(rs):3d} WR={rt:5.1f}% (win {w2} / loss {l2})")
        if gate_on:
            print("funnel (times each step reached, L+S):")
            for tag, f in funnel.items():
                print(f"   {tag:>12}: {f[1:]}")

    # variants on the no-gate signal set (bigger sample)
    sigs = all_signals[False]
    inv = [outcome(t, s, close, high, low, 0.75, 1.00) for t, s, _ in sigs]
    w, l, amb, op, rate = wr(inv)
    print(f"\n=== inverted TP0.75/SL1.00 (no-gate signals): WR={rate:.1f}% (win {w} / loss {l})")

    nod = [outcome(t, s, close, high, low, TP, SL, entry_price=close[t - 1]) for t, s, _ in sigs if t > 0]
    w, l, amb, op, rate = wr(nod)
    print(f"=== no-delay entry (fill at signal-candle close): WR={rate:.1f}% (win {w} / loss {l})")

    rng = np.random.default_rng(7)
    rt_idx = rng.integers(300, n - HORIZON, 4000)
    rl = [outcome(int(t), "L", close, high, low, TP, SL) for t in rt_idx[:2000]]
    rs = [outcome(int(t), "S", close, high, low, TP, SL) for t in rt_idx[2000:]]
    w, l, amb, op, rate = wr(rl + rs)
    print(f"=== RANDOM entries benchmark TP1.0/SL0.75: WR={rate:.1f}% (n={w + l})")
    be = (SL + 0.08) / (TP + SL) * 100
    print(f"=== math: breakeven WR with TP{TP}/SL{SL} + 0.08% commission = {(SL + 0.08) / (TP - 0.08 + SL + 0.08) * 100:.1f}%")


if __name__ == "__main__":
    main()
