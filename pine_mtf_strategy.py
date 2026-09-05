"""Replay of the TradingView sequential MTF Pine strategy on Binance OHLCV.

Matches the user script defaults:
  all 13 cascade triples (main ×3 confirm, main ÷3 entry)
  SMI = Stoch_MTM (single EMA D=3, then SMA 5; not double EMA)
  After step 1 (only while waiting for step 2), lookahead main SMI
  leaving saturation (−40 / +40) kills the path unless MACD is already
  true on that bar. Close at ±40 is still OK.
  Donchian is LonesomeTheBlue Trend Ribbon: maintrend = dchannel(20).
  At entry the forming ribbon must be green for a buy and red for a
  sell (close vs hh[1]/ll[1]). Mid-path flips after step 3 are allowed.
  After the fill the color is ignored.
  Confirm-TF Awesome Oscillator (SMA 5 − SMA 34 of median price):
  AO must be strictly above 0 for a buy and strictly below 0 for a sell.
  Main MACD step 2: line stays above a red histogram (buy) or below a
  green histogram (sell), with no other hist bound. The line must also
  be inside 20% of its lookback extreme: last 24h on main TFs ≤ 60m,
  last 72h on main TFs 90–240. Buy ceiling = 20% of the highest
  positive MACD; sell floor = 20% of the lowest negative MACD.
  Step triggers use lookahead=barmerge.lookahead_on (containing HTF bar).

  Persist must use the same HTF series as the triggers. Mapping persist to
  the previous closed main bar false-kills every path: step 1 fires on the
  new bar’s leaked close, then persist reads the old bar still outside ±40.
  TP 1.00% / SL 0.75% on both sides.

This is a market-data simulation, not live Binance account fills.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from cascade_steps import TRIPLING_PAIRS
from indicators import (
    DONCHIAN_DLEN,
    _calc_macd_full,
    calc_donchian_trend_series,
    calc_ema,
    calc_rsi_tv,
    calc_stoch_tv,
    candle_period_ends,
    ema_tv,
    resample_ohlcv,
)
from pullback_bot.strategy import SYMBOL, fetch_btc_1m_vision

log = logging.getLogger(__name__)

MAIN_TF = 15
CONFIRM_TF = 45
ENTRY_TF = 5
CHART_TF = 5
WARMUP_BARS = 500
MAX_BARS_GAP = 3
TP_PCT = 1.00
SL_PCT = 0.75
SMI_THRESH = 40.0
RSI_BUY_TOUCH = 35.0
RSI_SELL_TOUCH = 65.0
STOCH_LEVEL = 80.0
RSI_MA_LEN = 14
WEEK_DAYS = 7
MONTH_DAYS = 30
WARMUP_1M_BARS = 25_000
SMI_K = 10
SMI_D = 3
SMI_EMA_SIGNAL = 10
SMI_SMOOTH = 5

BUY_CONFIRM_RSI = (50.0, 60.0)
BUY_MAIN_DIFF = (3.0, 10.0)
SELL_CONFIRM_RSI = (40.0, 50.0)
SELL_MAIN_DIFF = (3.0, 10.0)
OHLCV_1M_BARS = 45_000
TRIPLE_WARMUP_BARS = 250
DONCHIAN_HOLD_DEFAULT = "at_entry"
AO_FAST = 5
AO_SLOW = 34
MACD_PULLBACK_PCT = 0.20
MACD_FAST_LOOKBACK_HOURS = 24
MACD_SLOW_LOOKBACK_HOURS = 72
MACD_FAST_TF_MAX = 60

PINE_TRIPLES = tuple((main, confirm, entry) for main, confirm, entry, *_ in TRIPLING_PAIRS)
ALT_SYMBOLS = (
    "BTCUSDT",
    "DOGEUSDT",
    "XRPUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "ADAUSDT",
    "SUIUSDT",
    "LINKUSDT",
    "TAOUSDT",
    "BCHUSDT",
    "AVAXUSDT",
    "AAVEUSDT",
    "HBARUSDT",
    "XLMUSDT",
    "UNIUSDT",
    "TIAUSDT",
    "DOTUSDT",
)


def _utc(ts):
    if ts is None:
        return None
    if getattr(ts, "to_pydatetime", None) is not None:
        ts = ts.to_pydatetime()
    if getattr(ts, "tzinfo", None) is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _crossunder_level(series, level):
    prev = series.shift(1)
    return (prev >= level) & (series < level)


def _crossover_level(series, level):
    prev = series.shift(1)
    return (prev <= level) & (series > level)


def _crossunder_series(left, right):
    return (left.shift(1) >= right.shift(1)) & (left < right)


def _crossover_series(left, right):
    return (left.shift(1) <= right.shift(1)) & (left > right)


def calc_smi_stoch_mtm(
    high,
    low,
    close,
    k=SMI_K,
    d=SMI_D,
    ema_len=SMI_EMA_SIGNAL,
    smooth_period=SMI_SMOOTH,
):
    """Surjith Stoch_MTM: one EMA on the range, then SMA, then signal EMA.

    avgrel/avgdiff use a single ``ta.ema(..., D)``, not ``ta.ema(ta.ema(...))``.
    The plotted black line is ``SMA(SMI, 5)``.
    """
    lowest = low.rolling(k, min_periods=k).min()
    highest = high.rolling(k, min_periods=k).max()
    diff = highest - lowest
    rdiff = close - (highest + lowest) / 2
    avgrel = ema_tv(rdiff, d)
    avgdiff = ema_tv(diff, d)
    smi = np.where(
        (avgdiff != 0) & np.isfinite(avgdiff) & np.isfinite(avgrel),
        (avgrel / (avgdiff / 2)) * 100,
        np.nan,
    )
    smi = pd.Series(smi, index=close.index)
    smoothed = smi.rolling(smooth_period, min_periods=smooth_period).mean()
    signal = ema_tv(smoothed, ema_len)
    return smoothed, signal


def calc_awesome_oscillator(high, low, fast=AO_FAST, slow=AO_SLOW):
    """Bill Williams AO: sma(median, 5) - sma(median, 34)."""
    mid = (
        pd.to_numeric(high, errors="coerce") + pd.to_numeric(low, errors="coerce")
    ) / 2.0
    fast_sma = mid.rolling(int(fast), min_periods=int(fast)).mean()
    slow_sma = mid.rolling(int(slow), min_periods=int(slow)).mean()
    return fast_sma - slow_sma


def ao_buy_gate(ao):
    return bool(np.isfinite(ao) and ao > 0)


def ao_sell_gate(ao):
    return bool(np.isfinite(ao) and ao < 0)


def macd_lookback_hours(main_tf):
    return (
        MACD_FAST_LOOKBACK_HOURS
        if int(main_tf) <= MACD_FAST_TF_MAX
        else MACD_SLOW_LOOKBACK_HOURS
    )


def macd_window_extremes(ts, macd, hours):
    """Highest positive and lowest negative MACD in a trailing time window."""
    frame = pd.DataFrame(
        {
            "ts": pd.to_datetime(ts, utc=True),
            "macd": pd.to_numeric(macd, errors="coerce"),
        }
    )
    frame["_i"] = np.arange(len(frame))
    ordered = frame.sort_values("ts")
    series = pd.Series(
        ordered["macd"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(ordered["ts"]),
    )
    window = f"{int(hours)}h"
    peak = series.where(series > 0).rolling(window, min_periods=1).max()
    trough = series.where(series < 0).rolling(window, min_periods=1).min()
    ordered = ordered.copy()
    ordered["peak"] = peak.to_numpy()
    ordered["trough"] = trough.to_numpy()
    restored = ordered.sort_values("_i")
    return restored["peak"].to_numpy(), restored["trough"].to_numpy()


def macd_buy_pullback(macd, peak, pct=MACD_PULLBACK_PCT):
    if not np.isfinite(macd):
        return False
    threshold = (float(peak) * pct) if np.isfinite(peak) and peak > 0 else 0.0
    return macd <= threshold


def macd_sell_pullback(macd, trough, pct=MACD_PULLBACK_PCT):
    if not np.isfinite(macd):
        return False
    threshold = (float(trough) * pct) if np.isfinite(trough) and trough < 0 else 0.0
    return macd >= threshold


def buy_rsi_gate(rsi_confirm, rsi_main):
    if not np.isfinite(rsi_confirm) or not np.isfinite(rsi_main):
        return False
    confirm_ok = BUY_CONFIRM_RSI[0] <= rsi_confirm <= BUY_CONFIRM_RSI[1]
    main_ok = (rsi_confirm - BUY_MAIN_DIFF[1]) <= rsi_main <= (
        rsi_confirm - BUY_MAIN_DIFF[0]
    )
    return confirm_ok and main_ok


def sell_rsi_gate(rsi_confirm, rsi_main):
    if not np.isfinite(rsi_confirm) or not np.isfinite(rsi_main):
        return False
    confirm_ok = SELL_CONFIRM_RSI[0] <= rsi_confirm <= SELL_CONFIRM_RSI[1]
    main_ok = (rsi_confirm + SELL_MAIN_DIFF[0]) <= rsi_main <= (
        rsi_confirm + SELL_MAIN_DIFF[1]
    )
    return confirm_ok and main_ok


def evaluate_outcome(signal_type, entry_price, tp_price, sl_price, future_1m):
    """Walk 1m bars after fill. Same-bar both TP and SL → loss."""
    if future_1m is None or future_1m.empty or entry_price <= 0:
        return "open", None, None
    for row in future_1m.itertuples(index=False):
        high = float(row.high)
        low = float(row.low)
        if signal_type == "buy":
            hit_sl = low <= sl_price
            hit_tp = high >= tp_price
        else:
            hit_sl = high >= sl_price
            hit_tp = low <= tp_price
        if hit_sl and hit_tp:
            return "loss", sl_price, _utc(row.ts)
        if hit_sl:
            return "loss", sl_price, _utc(row.ts)
        if hit_tp:
            return "win", tp_price, _utc(row.ts)
    return "open", None, None


def _indicator_frame(raw_1m, minutes):
    bars = resample_ohlcv(raw_1m, minutes)
    if bars.empty:
        return bars
    close = bars["close"]
    macd, _signal, hist = _calc_macd_full(close)
    smi, _ = calc_smi_stoch_mtm(bars["high"], bars["low"], close)
    # LonesomeTheBlue Donchian Trend Ribbon: maintrend = dchannel(dlen=20)
    trend = calc_donchian_trend_series(
        close.to_numpy(),
        bars["high"].to_numpy(),
        bars["low"].to_numpy(),
        DONCHIAN_DLEN,
    )
    trend.index = bars.index
    rsi = calc_rsi_tv(close, 14)
    bars = bars.copy()
    don_hh = bars["high"].rolling(DONCHIAN_DLEN, min_periods=DONCHIAN_DLEN).max().shift(1)
    don_ll = bars["low"].rolling(DONCHIAN_DLEN, min_periods=DONCHIAN_DLEN).min().shift(1)
    bars["smi"] = smi
    bars["macd"] = macd
    bars["hist"] = hist
    bars["trend"] = trend
    bars["don_hh"] = don_hh
    bars["don_ll"] = don_ll
    bars["trend_prev"] = trend.shift(1)
    bars["ema50"] = calc_ema(close, 50)
    bars["rsi"] = rsi
    bars["rsi_ma"] = rsi.rolling(RSI_MA_LEN, min_periods=RSI_MA_LEN).mean()
    k_stoch, _ = calc_stoch_tv(close, bars["high"], bars["low"])
    bars["stoch_k"] = k_stoch
    return bars


def _map_htf(chart, htf, columns):
    """lookahead_on: every chart bar sees the containing HTF bar's final values.

    Bars are left-labeled (open time), so ``merge_asof(backward)`` is the
    containing period — the same leak as ``request.security(..., lookahead_on)``.
    """
    src = htf[["ts", *columns]].rename(columns={"ts": "htf_ts"}).sort_values("htf_ts")
    mapped = pd.merge_asof(
        chart[["ts"]].sort_values("ts"),
        src,
        left_on="ts",
        right_on="htf_ts",
        direction="backward",
    )
    return mapped


def _live_donchian(close, hh, ll, trend_prev):
    """dchannel at this chart close: forming HTF close vs previous channel.

    Pine security(..., lookahead_on) sees the *current* HTF bar, not the
    future final close applied to every intra bar.
    """
    close = np.asarray(close, dtype=float)
    hh = np.asarray(hh, dtype=float)
    ll = np.asarray(ll, dtype=float)
    prev = np.asarray(trend_prev, dtype=float)
    green = np.isfinite(close) & np.isfinite(hh) & (close > hh)
    red = np.isfinite(close) & np.isfinite(ll) & (close < ll)
    return np.where(green, 1.0, np.where(red, -1.0, prev))


def _map_closed_htf(chart, htf, columns, chart_minutes, htf_minutes):
    """Last HTF bar that has already closed by the chart bar's close.

    Persist rules watch the main SMI/Donchian *close*, not the forming bar
    that lookahead already leaked into step triggers.
    """
    left = pd.DataFrame(
        {
            "ts": chart["ts"].to_numpy(),
            "close_at": candle_period_ends(chart["ts"], chart_minutes),
        }
    ).sort_values("close_at")
    right = htf[["ts", *columns]].copy()
    right["close_at"] = candle_period_ends(right["ts"], htf_minutes)
    right = right.drop(columns=["ts"]).sort_values("close_at")
    mapped = pd.merge_asof(left, right, on="close_at", direction="backward")
    return mapped.sort_values("ts")


def _cached_frame(raw_1m, minutes, cache):
    frame = cache.get(minutes)
    if frame is None:
        frame = _indicator_frame(raw_1m, minutes)
        cache[minutes] = frame
    return frame


def build_chart(
    raw_1m,
    chart_minutes=CHART_TF,
    main_tf=MAIN_TF,
    confirm_tf=CONFIRM_TF,
    frame_cache=None,
):
    """Attach lookahead HTF indicators onto closed chart bars."""
    cache = {} if frame_cache is None else frame_cache
    chart = _cached_frame(raw_1m, chart_minutes, cache)
    main = _cached_frame(raw_1m, main_tf, cache)
    confirm = _cached_frame(raw_1m, confirm_tf, cache)
    if chart.empty or main.empty or confirm.empty:
        return pd.DataFrame()
    if "ao" not in confirm.columns:
        confirm = confirm.copy()
        confirm["ao"] = calc_awesome_oscillator(confirm["high"], confirm["low"])
        cache[confirm_tf] = confirm
    if "macd_peak" not in main.columns:
        main = main.copy()
        peak, trough = macd_window_extremes(
            main["ts"], main["macd"], macd_lookback_hours(main_tf)
        )
        main["macd_peak"] = peak
        main["macd_trough"] = trough
        cache[main_tf] = main

    main_map = _map_htf(
        chart,
        main,
        [
            "smi",
            "macd",
            "hist",
            "trend",
            "ema50",
            "close",
            "rsi",
            "don_hh",
            "don_ll",
            "trend_prev",
            "macd_peak",
            "macd_trough",
        ],
    )
    confirm_map = _map_htf(chart, confirm, ["macd", "hist", "rsi", "ao"])
    out = chart.copy()
    out["smi_main"] = main_map["smi"].to_numpy()
    out["macd_main"] = main_map["macd"].to_numpy()
    out["hist_main"] = main_map["hist"].to_numpy()
    out["trend_main"] = _live_donchian(
        out["close"],
        main_map["don_hh"],
        main_map["don_ll"],
        main_map["trend_prev"],
    )
    out["main_open_ts"] = main_map["htf_ts"].to_numpy()
    out["ema50_main"] = main_map["ema50"].to_numpy()
    out["close_main"] = main_map["close"].to_numpy()
    out["rsi_main"] = main_map["rsi"].to_numpy()
    out["macd_peak_main"] = main_map["macd_peak"].to_numpy()
    out["macd_trough_main"] = main_map["macd_trough"].to_numpy()
    out["macd_confirm"] = confirm_map["macd"].to_numpy()
    out["hist_confirm"] = confirm_map["hist"].to_numpy()
    out["rsi_confirm"] = confirm_map["rsi"].to_numpy()
    out["ao_confirm"] = confirm_map["ao"].to_numpy()
    return out


def _entry_levels(signal_type, signal_close):
    if signal_type == "buy":
        return (
            signal_close * (1.0 + TP_PCT / 100.0),
            signal_close * (1.0 - SL_PCT / 100.0),
        )
    return (
        signal_close * (1.0 - TP_PCT / 100.0),
        signal_close * (1.0 + SL_PCT / 100.0),
    )


def replay_signals(
    chart,
    raw_1m,
    *,
    main_tf=MAIN_TF,
    confirm_tf=CONFIRM_TF,
    entry_tf=ENTRY_TF,
    warmup_bars=None,
    donchian_hold=DONCHIAN_HOLD_DEFAULT,
):
    """Walk chart bars with the Pine sequential state machine."""
    if chart is None or chart.empty:
        return []

    smi_main = chart["smi_main"]
    close_main = chart["close_main"]
    ema50_main = chart["ema50_main"]
    smi_entry = chart["smi"]
    stoch_k = chart["stoch_k"]
    rsi_entry = chart["rsi"].to_numpy(dtype=float)
    smi_persist = smi_main.to_numpy(dtype=float)
    trend_persist = chart["trend_main"].to_numpy(dtype=float)

    c1_buy = _crossunder_level(smi_main, -SMI_THRESH).to_numpy()
    c1_sell = _crossover_level(smi_main, SMI_THRESH).to_numpy()
    c2_buy = ((chart["hist_main"] < 0) & (chart["macd_main"] > chart["hist_main"])).to_numpy()
    c2_sell = ((chart["hist_main"] > 0) & (chart["macd_main"] < chart["hist_main"])).to_numpy()
    c3_buy = (chart["trend_main"] == 1).to_numpy()
    c3_sell = (chart["trend_main"] == -1).to_numpy()
    c4_buy = _crossunder_series(close_main, ema50_main).to_numpy()
    c4_sell = _crossover_series(close_main, ema50_main).to_numpy()
    c5_buy = ((chart["macd_main"] < 0) & (chart["hist_confirm"] > 0)).to_numpy()
    c5_sell = ((chart["macd_main"] > 0) & (chart["hist_confirm"] < 0)).to_numpy()
    c6_buy = (chart["trend"] == -1).to_numpy()
    c6_sell = (chart["trend"] == 1).to_numpy()
    c7_buy = _crossunder_level(smi_entry, -SMI_THRESH).to_numpy()
    c7_sell = _crossover_level(smi_entry, SMI_THRESH).to_numpy()
    stoch_buy_cross = _crossover_level(stoch_k, 100.0 - STOCH_LEVEL).to_numpy()
    stoch_sell_cross = _crossunder_level(stoch_k, STOCH_LEVEL).to_numpy()
    rsi_buy_cross = _crossover_series(chart["rsi"], chart["rsi_ma"]).to_numpy()
    rsi_sell_cross = _crossunder_series(chart["rsi"], chart["rsi_ma"]).to_numpy()

    n = len(chart)
    long_step = 0
    short_step = 0
    long_rsi_touched = False
    short_rsi_touched = False
    long_rsi_cross_bar = None
    short_rsi_cross_bar = None
    signals = []

    start_i = min(WARMUP_BARS if warmup_bars is None else warmup_bars, n)
    ts_values = chart["ts"].tolist()
    close_values = chart["close"].to_numpy(dtype=float)
    rsi_confirm = chart["rsi_confirm"].to_numpy(dtype=float)
    rsi_main = chart["rsi_main"].to_numpy(dtype=float)
    ao_confirm = chart["ao_confirm"].to_numpy(dtype=float)
    macd_main = chart["macd_main"].to_numpy(dtype=float)
    macd_peak = chart["macd_peak_main"].to_numpy(dtype=float)
    macd_trough = chart["macd_trough_main"].to_numpy(dtype=float)
    if "main_open_ts" in chart.columns:
        main_closes = candle_period_ends(chart["main_open_ts"], main_tf)
        chart_closes = candle_period_ends(chart["ts"], entry_tf)
        main_just_closed = (main_closes == chart_closes).to_numpy()
    else:
        main_just_closed = np.ones(n, dtype=bool)

    def _reset_long():
        nonlocal long_step, long_rsi_touched, long_rsi_cross_bar
        long_step = 0
        long_rsi_touched = False
        long_rsi_cross_bar = None

    def _reset_short():
        nonlocal short_step, short_rsi_touched, short_rsi_cross_bar
        short_step = 0
        short_rsi_touched = False
        short_rsi_cross_bar = None

    for i in range(start_i, n):
        smi_now = smi_persist[i]
        trend_now = trend_persist[i]
        long_c8 = False
        short_c8 = False
        if long_step == 7:
            if not long_rsi_touched and rsi_entry[i] <= RSI_BUY_TOUCH:
                long_rsi_touched = True
            if long_rsi_touched and bool(rsi_buy_cross[i]):
                long_c8 = True
                long_rsi_cross_bar = i
        if short_step == 7:
            if not short_rsi_touched and rsi_entry[i] >= RSI_SELL_TOUCH:
                short_rsi_touched = True
            if short_rsi_touched and bool(rsi_sell_cross[i]):
                short_c8 = True
                short_rsi_cross_bar = i

        # through: stay the step-3 color until a main bar closes wrong.
        # at_entry: only the signal bar must be green (buy) / red (sell).
        if donchian_hold == "through":
            if (
                long_step >= 3
                and main_just_closed[i]
                and np.isfinite(trend_now)
                and trend_now != 1
            ):
                _reset_long()
            if (
                short_step >= 3
                and main_just_closed[i]
                and np.isfinite(trend_now)
                and trend_now != -1
            ):
                _reset_short()

        # SMI persist only while waiting for MACD. Take step 2 first if
        # both fire on the same lookahead bar.
        if long_step == 0 and bool(c1_buy[i]):
            long_step = 1
        elif long_step == 1 and bool(c2_buy[i]) and macd_buy_pullback(
            macd_main[i], macd_peak[i]
        ):
            long_step = 2
        elif long_step == 1 and np.isfinite(smi_now) and smi_now > -SMI_THRESH:
            _reset_long()
        elif long_step == 2 and bool(c3_buy[i]):
            long_step = 3
        elif long_step == 3 and bool(c4_buy[i]):
            long_step = 4
        elif long_step == 4 and bool(c5_buy[i]):
            long_step = 5
        elif long_step == 5 and bool(c6_buy[i]):
            long_step = 6
        elif long_step == 6 and bool(c7_buy[i]):
            long_step = 7
        elif long_step == 7 and long_c8:
            long_step = 8
            long_rsi_touched = False

        if short_step == 0 and bool(c1_sell[i]):
            short_step = 1
        elif short_step == 1 and bool(c2_sell[i]) and macd_sell_pullback(
            macd_main[i], macd_trough[i]
        ):
            short_step = 2
        elif short_step == 1 and np.isfinite(smi_now) and smi_now < SMI_THRESH:
            _reset_short()
        elif short_step == 2 and bool(c3_sell[i]):
            short_step = 3
        elif short_step == 3 and bool(c4_sell[i]):
            short_step = 4
        elif short_step == 4 and bool(c5_sell[i]):
            short_step = 5
        elif short_step == 5 and bool(c6_sell[i]):
            short_step = 6
        elif short_step == 6 and bool(c7_sell[i]):
            short_step = 7
        elif short_step == 7 and short_c8:
            short_step = 8
            short_rsi_touched = False

        long_entry = False
        short_entry = False
        if long_step == 8 and long_rsi_cross_bar is not None:
            bars_since = i - long_rsi_cross_bar
            if (
                bool(stoch_buy_cross[i])
                and bars_since <= MAX_BARS_GAP
                and buy_rsi_gate(rsi_confirm[i], rsi_main[i])
                and ao_buy_gate(ao_confirm[i])
                and macd_buy_pullback(macd_main[i], macd_peak[i])
                and np.isfinite(trend_now)
                and trend_now == 1
            ):
                long_entry = True
                long_step = 0
            elif bars_since > MAX_BARS_GAP:
                long_step = 0
        if short_step == 8 and short_rsi_cross_bar is not None:
            bars_since = i - short_rsi_cross_bar
            if (
                bool(stoch_sell_cross[i])
                and bars_since <= MAX_BARS_GAP
                and sell_rsi_gate(rsi_confirm[i], rsi_main[i])
                and ao_sell_gate(ao_confirm[i])
                and macd_sell_pullback(macd_main[i], macd_trough[i])
                and np.isfinite(trend_now)
                and trend_now == -1
            ):
                short_entry = True
                short_step = 0
            elif bars_since > MAX_BARS_GAP:
                short_step = 0

        if not long_entry and not short_entry:
            continue
        if long_entry and short_entry:
            # Pine would reverse; skip the dual-fire bar.
            continue
        signal_type = "buy" if long_entry else "sell"
        if i + 1 >= n:
            continue
        signal_ts = _utc(ts_values[i])
        signal_close = float(close_values[i])
        fill_ts = _utc(ts_values[i + 1])
        entry_price = float(chart["open"].iloc[i + 1])
        tp_price, sl_price = _entry_levels(signal_type, signal_close)
        future = raw_1m.loc[raw_1m["ts"] >= fill_ts]
        outcome, exit_price, exit_ts = evaluate_outcome(
            signal_type, entry_price, tp_price, sl_price, future
        )
        signals.append(
            {
                "type": signal_type,
                "signal_ts": signal_ts,
                "fill_ts": fill_ts,
                "price": entry_price,
                "signal_close": signal_close,
                "tp": tp_price,
                "sl": sl_price,
                "outcome": outcome,
                "exit_price": exit_price,
                "exit_ts": exit_ts,
                "rsi_confirm": float(rsi_confirm[i]),
                "rsi_main": float(rsi_main[i]),
                "ao_confirm": float(ao_confirm[i]),
                "macd_main": float(macd_main[i]),
                "macd_peak_main": float(macd_peak[i]),
                "macd_trough_main": float(macd_trough[i]),
                "main_tf": main_tf,
                "confirm_tf": confirm_tf,
                "entry_tf": entry_tf,
                "symbol": chart.attrs.get("symbol") if hasattr(chart, "attrs") else None,
            }
        )
        if signal_type == "buy":
            long_rsi_cross_bar = None
        else:
            short_rsi_cross_bar = None
    return signals


def period_bounds(now=None, days=WEEK_DAYS):
    end = now or datetime.now(timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    else:
        end = end.astimezone(timezone.utc)
    return end - timedelta(days=days), end


def filter_week(signals, start, end):
    week = []
    for trade in signals:
        ts = trade["fill_ts"]
        if start <= ts < end:
            week.append(trade)
    return week


def summarize(trades):
    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    opens = [t for t in trades if t["outcome"] == "open"]
    net_pct = 0.0
    for trade in trades:
        if trade["outcome"] == "win":
            net_pct += TP_PCT
        elif trade["outcome"] == "loss":
            net_pct -= SL_PCT
    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "opens": opens,
        "net_pct": net_pct,
    }


def scan_triple(
    raw_1m,
    main_tf,
    confirm_tf,
    entry_tf,
    start,
    end,
    frame_cache=None,
    donchian_hold=DONCHIAN_HOLD_DEFAULT,
    symbol=None,
):
    chart = build_chart(
        raw_1m,
        chart_minutes=entry_tf,
        main_tf=main_tf,
        confirm_tf=confirm_tf,
        frame_cache=frame_cache,
    )
    if chart.empty:
        return summarize([]), 0
    signals = replay_signals(
        chart,
        raw_1m,
        main_tf=main_tf,
        confirm_tf=confirm_tf,
        entry_tf=entry_tf,
        warmup_bars=TRIPLE_WARMUP_BARS,
        donchian_hold=donchian_hold,
    )
    if symbol:
        for trade in signals:
            trade["symbol"] = symbol
    week = filter_week(signals, start, end)
    return summarize(week), len(signals)


def _ohlcv_target(days):
    return max(OHLCV_1M_BARS, int(days) * 1440 + WARMUP_1M_BARS)


def scan_week(
    raw_1m=None,
    now=None,
    days=WEEK_DAYS,
    triples=None,
    symbol=SYMBOL,
    donchian_hold=DONCHIAN_HOLD_DEFAULT,
):
    if raw_1m is None:
        target = _ohlcv_target(days)
        log.info("Fetching %s 1m from Binance Vision (%s bars)...", symbol, target)
        raw_1m = fetch_btc_1m_vision(target=target, symbol=symbol)
    if raw_1m is None or raw_1m.empty:
        raise RuntimeError(f"No OHLCV from Binance Vision for {symbol}")
    start, end = period_bounds(now=now, days=days)
    wanted = list(PINE_TRIPLES if triples is None else triples)
    frame_cache = {}
    by_triple = []
    all_trades = []
    for main_tf, confirm_tf, entry_tf in wanted:
        log.info("Scanning %s %sm / %sm / %sm ...", symbol, main_tf, confirm_tf, entry_tf)
        summary, all_count = scan_triple(
            raw_1m,
            main_tf,
            confirm_tf,
            entry_tf,
            start,
            end,
            frame_cache,
            donchian_hold=donchian_hold,
            symbol=symbol,
        )
        by_triple.append(
            {
                "main_tf": main_tf,
                "confirm_tf": confirm_tf,
                "entry_tf": entry_tf,
                "all_signals": all_count,
                **summary,
            }
        )
        all_trades.extend(summary["trades"])
    result = summarize(all_trades)
    result["start"] = start
    result["end"] = end
    result["symbol"] = symbol
    result["donchian_hold"] = donchian_hold
    result["bars_1m"] = len(raw_1m)
    result["by_triple"] = by_triple
    result["all_signals"] = sum(item["all_signals"] for item in by_triple)
    return result


def format_report(result):
    start = result["start"].strftime("%Y-%m-%d %H:%M")
    end = result["end"].strftime("%Y-%m-%d %H:%M")
    wins = result["wins"]
    losses = result["losses"]
    opens = result["opens"]
    lines = [
        f"{result.get('symbol', SYMBOL)} | 13 ثلاثي | SMI Stoch_MTM | "
        f"دونشيان {'حتى الدخول' if result.get('donchian_hold') == 'through' else 'عند الدخول'} | "
        f"AO{AO_FAST}/{AO_SLOW} تأكيد | "
        f"MACD{int(MACD_PULLBACK_PCT * 100)}% | "
        f"TP {TP_PCT:.2f}% / SL {SL_PCT:.2f}%",
        f"الفترة: {start} → {end} UTC ({(result['end'] - result['start']).days} يوم)",
        f"إجمالي الصفقات: {len(result['trades'])}",
        f"ناجحة: {len(wins)}",
        f"فاشلة: {len(losses)}",
        f"مفتوحة: {len(opens)}",
        f"صافي تقديري (بدون رسوم): {result['net_pct']:+.2f}%",
        "",
        "حسب الثلاثي:",
    ]
    for item in result.get("by_triple") or []:
        lines.append(
            f"{item['main_tf']}/{item['confirm_tf']}/{item['entry_tf']} | "
            f"صفقات {len(item['trades'])} | "
            f"ناجحة {len(item['wins'])} | "
            f"فاشلة {len(item['losses'])} | "
            f"مفتوحة {len(item['opens'])}"
        )
    lines.extend(["", "الصفقات:"])
    if not result["trades"]:
        lines.append("— لا توجد صفقات في هذه الفترة")
        return "\n".join(lines)
    ordered = sorted(result["trades"], key=lambda trade: trade["fill_ts"])
    for trade in ordered:
        side = "شراء" if trade["type"] == "buy" else "بيع"
        if trade["outcome"] == "win":
            mark = "رابحة"
        elif trade["outcome"] == "loss":
            mark = "خاسرة"
        else:
            mark = "مفتوحة"
        when = trade["fill_ts"].strftime("%m-%d %H:%M")
        frames = (
            f"{trade.get('main_tf', MAIN_TF)}/"
            f"{trade.get('confirm_tf', CONFIRM_TF)}/"
            f"{trade.get('entry_tf', ENTRY_TF)}"
        )
        coin = trade.get("symbol") or result.get("symbol") or ""
        prefix = f"{coin} | " if coin else ""
        lines.append(
            f"{prefix}{when} UTC | {frames} | {side} @ {trade['price']:.6g} | {mark} | "
            f"RSI تأكيد {trade['rsi_confirm']:.2f} / رئيسي {trade['rsi_main']:.2f}"
        )
    return "\n".join(lines)


def scan_alts(days=MONTH_DAYS, symbols=None, donchian_hold=DONCHIAN_HOLD_DEFAULT):
    wanted = list(ALT_SYMBOLS if symbols is None else symbols)
    start, end = period_bounds(days=days)
    by_symbol = []
    all_trades = []
    for symbol in wanted:
        try:
            result = scan_week(
                days=days, symbol=symbol, donchian_hold=donchian_hold
            )
        except RuntimeError as exc:
            log.warning("%s", exc)
            result = summarize([])
            result["start"] = start
            result["end"] = end
            result["symbol"] = symbol
            result["donchian_hold"] = donchian_hold
            result["by_triple"] = []
            result["bars_1m"] = 0
            result["all_signals"] = 0
        by_symbol.append(result)
        all_trades.extend(result["trades"])
    combined = summarize(all_trades)
    combined["start"] = start
    combined["end"] = end
    combined["symbol"] = ",".join(wanted)
    combined["donchian_hold"] = donchian_hold
    combined["by_symbol"] = by_symbol
    return combined


def format_alts_report(combined):
    lines = [
        format_report(combined),
        "",
        "حسب العملة:",
    ]
    for item in combined.get("by_symbol") or []:
        lines.append(
            f"{item.get('symbol')} | "
            f"صفقات {len(item['trades'])} | "
            f"ناجحة {len(item['wins'])} | "
            f"فاشلة {len(item['losses'])} | "
            f"مفتوحة {len(item['opens'])} | "
            f"صافي {item['net_pct']:+.2f}%"
        )
    return "\n".join(lines)


def main(days=WEEK_DAYS, alts=False):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if alts:
        result = scan_alts(days=days, donchian_hold=DONCHIAN_HOLD_DEFAULT)
        print(format_alts_report(result))
        return result
    result = scan_week(days=days)
    print(format_report(result))
    return result


if __name__ == "__main__":
    import sys

    days = WEEK_DAYS
    alts = False
    if len(sys.argv) > 1:
        if sys.argv[1] in {"alts", "alt"}:
            alts = True
            days = MONTH_DAYS
            if len(sys.argv) > 2:
                days = int(sys.argv[2])
        else:
            days = int(sys.argv[1])
            alts = len(sys.argv) > 2 and sys.argv[2] in {"alts", "alt"}
    main(days=days, alts=alts)
