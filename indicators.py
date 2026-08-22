"""
indicators.py — الحسابات التقنية (RSI, MACD, SMI, Donchian, Resampling) المستخدمة في بايبلاين الفحص Cascade.

⚠️ أي تعديل على هذا الملف يؤثر مباشرة على منطق التداول في fahadal92.py.
"""
import threading
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

# ------------------------------------------
# WARMUP Constants
# ------------------------------------------

WARMUP_EMA = 200
WARMUP_MACD = 200
WARMUP_SMI = 100
WARMUP_RSI = 200
WARMUP_STOCH = 100
WARMUP_DON = 50
WARMUP_AO = 34
AO_FAST = 5
AO_SLOW = 34
MIN_CANDLES = 300

# ------------------------------------------
# Donchian constant
# ------------------------------------------

DONCHIAN_DLEN = 20  # Pine's default dlen (matches dlen = input(defval = 20, ...))

# ------------------------------------------
# Ribbon cache (shared state for check_donchian_trend_ribbon)
# ------------------------------------------

_ribbon_cache = {}
_ribbon_cache_lock = threading.Lock()

# ------------------------------------------
# Resampling
# ------------------------------------------

_RESAMPLE_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "vol": "sum",
}
_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MINUTES_PER_DAY = 24 * 60


def _resample_origin_mode(minutes):
    """Epoch when the TF tiles a UTC day; else midnight of each UTC day.

    Applies to every cascade frame (base, confirm, entry), not a subset.
    If 1440 % minutes == 0 (3/4/5/6/8/9/10/12/15/18/20/24/30/36/40/45/60/72/80/
    90/120/180/240/360/720) Unix epoch already matches TradingView.
    Otherwise (7/21/27/50/54/63/70/81/135/150/210/270/300/450/540/630, …) epoch
    drifts and bars restart at 00:00 UTC each day like the chart.
    """
    return "epoch" if _MINUTES_PER_DAY % int(minutes) == 0 else "utc_day"


def candle_period_end(ts, minutes):
    """UTC close time of the candle that opens at ``ts``.

    TFs that tile a UTC day last exactly ``minutes``.
    TFs that restart at midnight are clipped at the next 00:00 UTC so a
    remainder bar (27m at 23:51, 135m/150m/270m at 22:30) closes when TradingView
    closes it — at the session end — not ``minutes`` later into the next day.
    """
    minutes = int(minutes)
    start = pd.Timestamp(ts)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    nominal = start + pd.Timedelta(minutes=minutes)
    if _resample_origin_mode(minutes) == "epoch":
        return nominal
    next_midnight = start.floor("D") + pd.Timedelta(days=1)
    return min(nominal, next_midnight)


def candle_period_ends(ts_series, minutes):
    """Vectorized ``candle_period_end`` for a Series of bar opens."""
    minutes = int(minutes)
    start = pd.to_datetime(ts_series, utc=True)
    nominal = start + pd.Timedelta(minutes=minutes)
    if _resample_origin_mode(minutes) == "epoch":
        return nominal
    next_midnight = start.dt.floor("D") + pd.Timedelta(days=1)
    return nominal.clip(upper=next_midnight)


def _resample_ohlcv_frame(df, minutes):
    """Resample OHLCV on ``ts`` to ``minutes`` using the matching origin."""
    minutes = int(minutes)
    frame = df.copy()
    if "ts" in frame.columns:
        frame = frame.set_index("ts")
    if frame.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "vol"])
    if frame.index.tz is None:
        frame = frame.tz_localize("UTC")
    else:
        frame = frame.tz_convert("UTC")

    def _agg(part, origin):
        return (
            part.resample(
                f"{minutes}min",
                closed="left",
                label="left",
                origin=origin,
            )
            .agg(_RESAMPLE_AGG)
            .dropna()
        )

    if _resample_origin_mode(minutes) == "epoch":
        resampled = _agg(frame, _EPOCH_UTC)
    else:
        parts = [
            _agg(group, day_start)
            for day_start, group in frame.groupby(frame.index.floor("D"))
            if not group.empty
        ]
        resampled = (
            pd.concat(parts)
            if parts
            else pd.DataFrame(columns=["open", "high", "low", "close", "vol"])
        )
    out = resampled.reset_index()
    if "ts" not in out.columns:
        out = out.rename(columns={out.columns[0]: "ts"})
    return out


def resample_ohlcv(df, minutes):
    """
    يُعيد تجميع (resample) بيانات OHLCV إلى فريم زمني محدد بالدقائق.

    الفريمات التي تنقسم على 1440 دقيقة تستخدم Unix epoch (مطابقة Binance).
    الفريمات التي لا تنقسم (21/27/135/150/210...) تُحاذى على منتصف الليل UTC كل يوم
    حتى تطابق شبكة TradingView. شمعة آخر اليوم القصيرة تُقفل عند 00:00 UTC.

    هذه الدالة تحذف الشمعة الأخيرة إذا لم تُغلق بعد (شمعة جارية).
    استخدم هذه الدالة حصرًا في مسارات تقييم الإشارات (step1-step8 وما شابه).
    """
    if df.empty:
        return pd.DataFrame()
    now = datetime.now(timezone.utc)
    resampled = _resample_ohlcv_frame(df, minutes)
    if resampled.empty:
        return resampled
    last_candle_end = candle_period_end(resampled["ts"].iloc[-1], minutes)
    now_ts = pd.Timestamp(now)
    if now_ts.tzinfo is None:
        now_ts = now_ts.tz_localize("UTC")
    else:
        now_ts = now_ts.tz_convert("UTC")
    if now_ts < last_candle_end:
        resampled = resampled.iloc[:-1]
    return resampled


def resample_ohlcv_closed(df, minutes):
    """
    يُعيد تجميع OHLCV دون حذف الشمعة الأخيرة. نفس محاذاة ``resample_ohlcv``.

    ⚠️ لا تستخدمها مباشرة في خطوات الإشارة إلا عبر ``confirm_macd_frame``
    (MACD ودونشيان فريم التأكيد يطابقان شمعة TradingView الجارية من مصدر مغلق).
    """
    if df.empty:
        return pd.DataFrame()
    return _resample_ohlcv_frame(df, minutes)


_SOURCE_TF_MINUTES = {"1m": 1, "30m": 30, "60m": 60}


def _as_utc_timestamp(now):
    ts = pd.Timestamp(now if now is not None else datetime.now(timezone.utc))
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def confirm_macd_frame(raw_df, source_tf, confirm_minutes, now=None):
    """Confirm-TF OHLCV including the in-progress confirm bar.

    TradingView paints MACD and Donchian on the current confirm bar (every
    cascade TF: 45/54/63/…/135/180/270/360/450) as soon as closed 1m/30m/60m
    source candles land in that bucket. Live cascade used to drop that
    incomplete confirm bar, so a LONG could pass on a *previous* closed green
    confirm bar while the chart's current bar was already red.

    Only source candles whose period has fully closed by ``now`` are used, so
    this does not read an unclosed 1m/30m/60m candle.
    """
    if raw_df is None or raw_df.empty or "ts" not in raw_df.columns:
        return pd.DataFrame()
    asof = _as_utc_timestamp(now)
    source_minutes = int(_SOURCE_TF_MINUTES.get(source_tf, 1))
    ends = pd.to_datetime(raw_df["ts"], utc=True) + pd.Timedelta(
        minutes=source_minutes
    )
    src = raw_df.loc[ends <= asof]
    if src.empty:
        return pd.DataFrame()
    return resample_ohlcv_closed(src, int(confirm_minutes))

# ------------------------------------------
# MACD
# ------------------------------------------

# Live / baseline step-② band vs the window peak/trough above/below zero.
# None disables that 40% cap/floor and keeps only the histogram-side check.
DEFAULT_MACD_LINE_PCT = 0.40


def resolve_macd_line_pct(variant=None):
    """Return the MACD line band pct, or None to leave the far side open."""
    if isinstance(variant, dict) and "macd_line_pct" in variant:
        return variant["macd_line_pct"]
    return DEFAULT_MACD_LINE_PCT


def ema_tv(series, length):
    """TradingView ``ta.ema``: SMA of the first ``length`` bars, then recursive.

    pandas ``ewm(adjust=False)`` seeds from the first tick instead of that SMA,
    so MACD/EMA/SMI drift off the Binance Futures chart.
    """
    length = int(length)
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=True)
    n = len(values)
    out = np.full(n, np.nan)
    if n < length or length <= 0:
        return pd.Series(out, index=series.index)
    start = None
    for index in range(length - 1, n):
        window = values[index - length + 1 : index + 1]
        if np.isfinite(window).all():
            start = index
            out[index] = float(window.mean())
            break
    if start is None:
        return pd.Series(out, index=series.index)
    alpha = 2.0 / (length + 1.0)
    prev = out[start]
    for index in range(start + 1, n):
        value = values[index]
        if not np.isfinite(value):
            continue
        prev = alpha * value + (1.0 - alpha) * prev
        out[index] = prev
    return pd.Series(out, index=series.index)


def rma_tv(series, length):
    """TradingView ``ta.rma`` / Wilder: SMA seed, then alpha = 1/length."""
    length = int(length)
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=True)
    n = len(values)
    out = np.full(n, np.nan)
    if n < length or length <= 0:
        return pd.Series(out, index=series.index)
    start = None
    for index in range(length - 1, n):
        window = values[index - length + 1 : index + 1]
        if np.isfinite(window).all():
            start = index
            out[index] = float(window.mean())
            break
    if start is None:
        return pd.Series(out, index=series.index)
    alpha = 1.0 / length
    prev = out[start]
    for index in range(start + 1, n):
        value = values[index]
        if not np.isfinite(value):
            continue
        prev = alpha * value + (1.0 - alpha) * prev
        out[index] = prev
    return pd.Series(out, index=series.index)


def wilder_rma(series, period):
    return rma_tv(series, period)


def _calc_macd_full(close):
    macd_line = ema_tv(close, 12) - ema_tv(close, 26)
    signal_line = ema_tv(macd_line, 9)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _calc_macd_hist(close):
    _, _, histogram = _calc_macd_full(close)
    return histogram

def check_macd_red(df):
    if len(df) < WARMUP_MACD:
        return False
    return bool(_calc_macd_hist(df["close"]).iloc[-1] < 0)

def check_macd_green(df):
    if len(df) < WARMUP_MACD:
        return False
    return bool(_calc_macd_hist(df["close"]).iloc[-1] > 0)


def calc_ao(high, low, fast=AO_FAST, slow=AO_SLOW):
    """Awesome Oscillator: SMA(mid, 5) − SMA(mid, 34)."""
    mid = (pd.to_numeric(high, errors="coerce") + pd.to_numeric(low, errors="coerce")) / 2.0
    fast_sma = mid.rolling(int(fast), min_periods=int(fast)).mean()
    slow_sma = mid.rolling(int(slow), min_periods=int(slow)).mean()
    return fast_sma - slow_sma


def check_ao_setup(df_base, df_confirm, direction="long"):
    """
    شراء: AO الفريم الرئيس تحت 0 وAO التأكيد فوق 0.
    بيع: AO الفريم الرئيس فوق 0 وAO التأكيد تحت 0.
    مثال: رئيس 60م / تأكيد 180م / دخول 20م.
    """
    if df_base is None or df_confirm is None:
        return False
    if len(df_base) < WARMUP_AO or len(df_confirm) < WARMUP_AO:
        return False
    ao_base = calc_ao(df_base["high"], df_base["low"])
    ao_confirm = calc_ao(df_confirm["high"], df_confirm["low"])
    base_val = float(ao_base.iloc[-1])
    confirm_val = float(ao_confirm.iloc[-1])
    if not np.isfinite(base_val) or not np.isfinite(confirm_val):
        return False
    if direction == "long":
        return base_val < 0 and confirm_val > 0
    return base_val > 0 and confirm_val < 0

def _get_macd_window_hours(base_frame_minutes):
    """نافذة قياس الـ 40٪ بمقياس يومي حسب حجم الفريم:
    - فريم ≤ 60 دقيقة → يوم واحد (24 ساعة)
    - فريم > 60 دقيقة → 3 أيام (72 ساعة)
      (اليوم الواحد على الفريمات الكبيرة يعطي شموعًا قليلة فيفسد القياس)
    """
    return 24 if base_frame_minutes <= 60 else 72


def _macd_window_series(macd_line, ts, base_frame):
    """يقطع سلسلة MACD على النافذة اليومية المناسبة للفريم."""
    window_hours = _get_macd_window_hours(base_frame)
    last_ts = pd.Timestamp(ts.iloc[-1])
    cutoff_ts = last_ts - pd.Timedelta(hours=window_hours)
    window = macd_line[pd.to_datetime(ts) >= cutoff_ts]
    return window if not window.empty else macd_line


def check_macd_line_long(df, pct=DEFAULT_MACD_LINE_PCT, base_frame=60):
    """
    شرط MACD Line للشراء (مع هيستوجرام أحمر متوقع من المستدعي):
    - الحد السفلي: الخط الأزرق فوق الهوستقرام أو يلامسه (macd >= hist) — ممنوع تحته
    - الحد العلوي: ≤ pct من أقصى ارتفاع فوق خط الصفر خلال النافذة اليومية
      مثال: أعلى قيمة موجبة = 100 → السقف = 40
    - pct=None يلغي السقف ويُبقي شرط الهوستقرام فقط (الحد العلوي مفتوح)
    """
    if len(df) < WARMUP_MACD:
        return False
    macd_line, _, histogram = _calc_macd_full(df["close"])
    current_macd = float(macd_line.iloc[-1])
    current_hist = float(histogram.iloc[-1])

    # الحد السفلي: فوق الهوستقرام الأحمر أو يلامسه
    if current_macd < current_hist:
        return False
    if pct is None:
        return True

    window = _macd_window_series(macd_line, df["ts"], base_frame)
    # أقصى ارتفاع فوق خط الصفر فقط
    positive = window[window > 0]
    if positive.empty:
        # لا يوجد ارتفاع فوق الصفر في النافذة → السقف = 0
        threshold = 0.0
    else:
        threshold = float(positive.max()) * pct

    if current_macd > threshold:
        return False
    return True


def check_macd_line_short(df, pct=DEFAULT_MACD_LINE_PCT, base_frame=60):
    """
    شرط MACD Line للبيع (مع هيستوجرام أخضر متوقع من المستدعي):
    - الحد العلوي: الخط الأزرق تحت الهوستقرام أو يلامسه (macd <= hist) — ممنوع فوقه
    - الحد السفلي: ≥ pct من أقصى نزول تحت خط الصفر خلال النافذة اليومية
      مثال: أدنى قيمة = -100 → الأرضية = -40 (ولا ينزل أعمق منها)
    - pct=None يلغي الأرضية ويُبقي شرط الهوستقرام فقط (الحد السفلي مفتوح)
    """
    if len(df) < WARMUP_MACD:
        return False
    macd_line, _, histogram = _calc_macd_full(df["close"])
    current_macd = float(macd_line.iloc[-1])
    current_hist = float(histogram.iloc[-1])

    # الحد العلوي: تحت الهوستقرام الأخضر أو يلامسه
    if current_macd > current_hist:
        return False
    if pct is None:
        return True

    window = _macd_window_series(macd_line, df["ts"], base_frame)
    # أقصى نزول تحت خط الصفر فقط
    negative = window[window < 0]
    if negative.empty:
        # لا يوجد نزول تحت الصفر في النافذة → الأرضية = 0
        threshold = 0.0
    else:
        threshold = float(negative.min()) * pct

    if current_macd < threshold:
        return False
    return True

# ------------------------------------------
# Donchian
# ------------------------------------------

def calc_donchian_trend_series(close_arr, high_arr, low_arr, length):
    """
    Pine-exact Donchian trend series replicating dchannel(len):

        hh = highest(len)
        ll = lowest(len)
        trend := close > hh[1] ? 1 : close < ll[1] ? -1 : nz(trend[1])

    Returns a float Series of 1 (bullish), -1 (bearish), or 0 (unknown/warmup).
    """
    n = len(close_arr)
    if n == 0:
        return pd.Series(dtype=float)

    high_s = pd.Series(high_arr, dtype=float)
    low_s = pd.Series(low_arr, dtype=float)
    close_s = pd.Series(close_arr, dtype=float)

    if n < length + 1:
        return pd.Series(0.0, index=close_s.index, dtype=float)

    hh = high_s.rolling(length, min_periods=length).max().shift(1)
    ll = low_s.rolling(length, min_periods=length).min().shift(1)

    raw = pd.Series(np.nan, index=close_s.index, dtype=float)
    raw[close_s.gt(hh).fillna(False)] = 1.0
    raw[close_s.lt(ll).fillna(False)] = -1.0
    return raw.ffill().fillna(0.0)


def calc_donchian_trend_pine(close_arr, high_arr, low_arr, length):
    """
    Pine-exact Donchian trend replicating dchannel(len) from the Pine Script.

    Returns 1 (bullish), -1 (bearish), or 0 (insufficient data).
    """
    trend = calc_donchian_trend_series(close_arr, high_arr, low_arr, length)
    if trend.empty:
        return 0
    try:
        return int(trend.iloc[-1])
    except (IndexError, TypeError, ValueError):
        return 0


def _calc_donchian_ribbon_result(close, high, low):
    """
    Return the visible green/red direction of TradingView's Donchian Trend Ribbon.

    In the original Pine script the hue of all ten plotted bands is selected by:

        maintrend = dchannel(dlen)

    With dlen=20, maintrend=1 makes every band green and maintrend=-1 makes
    every band red.  The auxiliary lengths 19..11 only change alpha
    (#00FF00ff vs #00FF009f, or the red equivalents); they do not change hue.

    Therefore the bot's binary green/red condition must use dchannel(20) only.
    Requiring all auxiliary trends to agree creates a third state that the
    TradingView indicator does not display.
    """
    return calc_donchian_trend_pine(
        close,
        high,
        low,
        DONCHIAN_DLEN,
    )


# ------------------ Donchian check + cache ------------------
def check_donchian_trend_ribbon(df, direction="green", cache_key=None):
    """
    Evaluates and caches TradingView's visible Donchian ribbon hue.

    direction="green" → dchannel(20) maintrend == 1
    direction="red"   → dchannel(20) maintrend == -1

    The 19..11 sub-trends only control opacity in the Pine script and do not
    affect this binary green/red decision.

    Thread-safe via _ribbon_cache_lock.
    """
    if df.empty or len(df) < DONCHIAN_DLEN + 1:
        return False

    if cache_key is not None:
        with _ribbon_cache_lock:
            cached = _ribbon_cache.get(cache_key)
        if cached is None:
            close = df["close"].values
            high = df["high"].values
            low = df["low"].values
            result = _calc_donchian_ribbon_result(close, high, low)
            # double-checked locking: another thread may have stored it already
            with _ribbon_cache_lock:
                cached = _ribbon_cache.get(cache_key)
                if cached is None:
                    _ribbon_cache[cache_key] = result
                    cached = result
    else:
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        cached = _calc_donchian_ribbon_result(close, high, low)

    return (cached == 1) if direction == "green" else (cached == -1)

# ------------------------------------------
# EMA
# ------------------------------------------

def calc_ema(close, span=60):
    """EMA على الإغلاق — ``ta.ema`` في TradingView (بذرة SMA ثم تكرار)."""
    return ema_tv(close, int(span))


BASE_EMA_LEN = 50


def check_ema50_below(df):
    """آخر شمعة تقفل تحت EMA50 (للتوافق؛ الخطوة 6 تستخدم النسخة since)."""
    ema = calc_ema(df["close"], span=BASE_EMA_LEN)
    return bool(df["close"].iloc[-1] < ema.iloc[-1])

def check_ema50_above(df):
    """آخر شمعة تقفل فوق EMA50 (للتوافق؛ الخطوة 6 تستخدم النسخة since)."""
    ema = calc_ema(df["close"], span=BASE_EMA_LEN)
    return bool(df["close"].iloc[-1] > ema.iloc[-1])


def check_ema50_closed_below_since(df, since_ts, smi_threshold=-40):
    """
    شراء: هل أقفلت أي شمعة تحت EMA50 أثناء تشبع SMI منذ since_ts؟
    يُحسب EMA/SMI على السلسلة كاملة، ويُقبل فقط إغلاق على شمعة متشبعة.
    اللمس بالفتيل لا يكفي — الإغلاق (close) فقط.
    """
    if df.empty or since_ts is None or len(df) < max(BASE_EMA_LEN, WARMUP_SMI):
        return False
    time_mask = df["ts"] >= since_ts
    if not time_mask.any():
        return False
    ema = calc_ema(df["close"], span=BASE_EMA_LEN)
    smi, _, _ = calc_smi(df["high"], df["low"], df["close"])
    sat_mask = time_mask & (smi <= smi_threshold)
    if not sat_mask.any():
        return False
    return bool((df.loc[sat_mask, "close"] < ema.loc[sat_mask]).any())


def check_ema50_closed_above_since(df, since_ts, smi_threshold=40):
    """
    بيع: هل أقفلت أي شمعة فوق EMA50 أثناء تشبع SMI منذ since_ts؟
    يُحسب EMA/SMI على السلسلة كاملة، ويُقبل فقط إغلاق على شمعة متشبعة.
    اللمس بالفتيل لا يكفي — الإغلاق (close) فقط.
    """
    if df.empty or since_ts is None or len(df) < max(BASE_EMA_LEN, WARMUP_SMI):
        return False
    time_mask = df["ts"] >= since_ts
    if not time_mask.any():
        return False
    ema = calc_ema(df["close"], span=BASE_EMA_LEN)
    smi, _, _ = calc_smi(df["high"], df["low"], df["close"])
    sat_mask = time_mask & (smi >= smi_threshold)
    if not sat_mask.any():
        return False
    return bool((df.loc[sat_mask, "close"] > ema.loc[sat_mask]).any())

# ------------------------------------------
# SMI
# ------------------------------------------

def calc_smi(high, low, close, k=10, smooth_period=1, d=3, c=10):
    """
    Stochastic Momentum Index — Pine ``ta.ema(ta.ema(...))`` (Stoch_MTM / TV SMI).

    K=10, double EMA D=3, signal EMA C=10. Levels ±40 are SMI saturation,
    not the MACD 40% band.
    """
    ll = low.rolling(k, min_periods=k).min()
    hh = high.rolling(k, min_periods=k).max()
    diff = hh - ll
    rdiff = close - (hh + ll) / 2

    avgrel = ema_tv(ema_tv(rdiff, d), d)
    avgdiff = ema_tv(ema_tv(diff, d), d)

    smi = np.where(
        (avgdiff != 0) & np.isfinite(avgdiff) & np.isfinite(avgrel),
        (avgrel / (avgdiff / 2)) * 100,
        0.0,
    )
    smi = pd.Series(smi, index=close.index)

    smi_smoothed = smi.rolling(smooth_period, min_periods=smooth_period).mean()
    smi_signal = ema_tv(smi_smoothed, d)
    ema_signal = ema_tv(smi_smoothed, c)
    return smi_smoothed, ema_signal, smi_signal

def check_smi_oversold(df, threshold=-40):
    if len(df) < WARMUP_SMI:
        return False
    smi, _, _ = calc_smi(df["high"], df["low"], df["close"])
    return bool(smi.iloc[-1] <= threshold)


def smi_signal_cycle_ended_from_series(smi, signal, direction="long", os_lvl=-40, ob_lvl=40):
    """True after EMA Signal entered ±40, exited the other side, and crossed K.

    That sequence means saturation is over for this frame until Signal
    re-enters the zone. ``smi`` is the K line; ``signal`` is EMA Signal.
    """
    if smi is None or signal is None or len(smi) < 2 or len(signal) < 2:
        return False
    if direction not in ("long", "short"):
        raise ValueError(f"Unsupported direction: {direction}")
    been_in = False
    ended = False
    prev_sig = None
    prev_k = None
    for idx in range(len(smi)):
        sig = signal.iloc[idx]
        k_val = smi.iloc[idx]
        if pd.isna(sig) or pd.isna(k_val):
            prev_sig, prev_k = sig, k_val
            continue
        in_zone = sig <= os_lvl if direction == "long" else sig >= ob_lvl
        if in_zone:
            been_in = True
            ended = False
        elif been_in and prev_sig is not None and not pd.isna(prev_sig) and not pd.isna(prev_k):
            prev_diff = float(prev_sig) - float(prev_k)
            diff = float(sig) - float(k_val)
            crossed = (prev_diff > 0 and diff <= 0) or (prev_diff < 0 and diff >= 0)
            if crossed:
                ended = True
                been_in = False
        prev_sig, prev_k = sig, k_val
    return bool(ended)


def check_smi_signal_cycle_ended(df, signal_type="buy"):
    """Frame should stop: SMI EMA Signal left ±40 and crossed K."""
    if df is None or getattr(df, "empty", True) or len(df) < WARMUP_SMI:
        return False
    smi, ema_signal, _ = calc_smi(df["high"], df["low"], df["close"])
    direction = "long" if signal_type == "buy" else "short"
    return smi_signal_cycle_ended_from_series(smi, ema_signal, direction=direction)


def find_saturation_start_index(df, threshold=-40, direction="long"):
    """
    أول شمعة مغلقة في نوبة تشبع SMI الحالية (المتصلة بآخر شمعة مغلقة).

    direction:
      - "long"  => SMI <= threshold (افتراضي -40)
      - "short" => SMI >= threshold (افتراضي +40)

    يرجع None إذا لم تكن آخر شمعة مغلقة متشبعة.
    """
    if df.empty or len(df) < WARMUP_SMI:
        return None
    if direction not in ("long", "short"):
        raise ValueError(f"Unsupported direction: {direction}")
    smi, _, _ = calc_smi(df["high"], df["low"], df["close"])
    if direction == "long":
        saturated = smi <= threshold
    else:
        saturated = smi >= threshold
    if not bool(saturated.iloc[-1]):
        return None
    index = len(df) - 1
    while index > 0 and bool(saturated.iloc[index - 1]):
        index -= 1
    return index


def check_macd_at_saturation_start(
    df, base_frame, direction="long", pct=DEFAULT_MACD_LINE_PCT
):
    """
    فحص MACD مرة واحدة فقط: على أول شمعة إغلاق متشبعة في نوبة التشبع
    الحالية، لا على آخر شمعة. تُقصّ السلسلة عند تلك الشمعة ثم يُقيَّم
    الهيستوجرام وشرط الخط هناك، فتبقى النتيجة ثابتة طوال النوبة.
    """
    threshold = -40 if direction == "long" else 40
    start_index = find_saturation_start_index(
        df,
        threshold=threshold,
        direction=direction,
    )
    if start_index is None:
        return False
    df_eval = df.iloc[: start_index + 1]
    if len(df_eval) < WARMUP_MACD:
        return False
    if direction == "long":
        return check_macd_red(df_eval) and check_macd_line_long(
            df_eval,
            pct=pct,
            base_frame=base_frame,
        )
    return check_macd_green(df_eval) and check_macd_line_short(
        df_eval,
        pct=pct,
        base_frame=base_frame,
    )

def check_smi_overbought(df, threshold=40):
    if len(df) < WARMUP_SMI:
        return False
    smi, _, _ = calc_smi(df["high"], df["low"], df["close"])
    return bool(smi.iloc[-1] >= threshold)

def check_ema50_above_since_overbought(df, smi_threshold=40):
    if len(df) < WARMUP_SMI:
        return False
    smi, _, _ = calc_smi(df["high"], df["low"], df["close"])
    ema = calc_ema(df["close"], span=BASE_EMA_LEN)
    overbought_mask = smi >= smi_threshold
    if not overbought_mask.any():
        return False
    last_idx = overbought_mask[::-1].idxmax()
    return bool((df["close"].loc[last_idx:] > ema.loc[last_idx:]).any())

# ------------------------------------------
# RSI / Stochastic
# ------------------------------------------

def calc_rsi_tv(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    up = wilder_rma(gain, period)
    down = wilder_rma(loss, period)
    return 100.0 - (100.0 / (1.0 + up / (down + 1e-10)))

def calc_stoch_tv(close, high, low, k_len=15, k_smooth=3, d_smooth=3):
    lo = low.rolling(k_len, min_periods=k_len).min()
    hi = high.rolling(k_len, min_periods=k_len).max()
    raw = 100.0 * (close - lo) / (hi - lo + 1e-10)
    k = raw.rolling(k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(d_smooth, min_periods=d_smooth).mean()
    return k, d

def check_rsi_touched_oversold(df, lookback=10, threshold=35):
    if len(df) < WARMUP_RSI + lookback:
        return False
    rsi = calc_rsi_tv(df["close"], period=14)
    return bool((rsi.iloc[-lookback:] <= threshold).any())

def check_rsi_overbought_short(df, lookback=10, threshold=65):
    if len(df) < WARMUP_RSI + lookback:
        return False
    rsi = calc_rsi_tv(df["close"], period=14)
    return bool((rsi.iloc[-lookback:] >= threshold).any())

def check_rsi_not_oversold_recently(df, lookback=50, threshold=30):
    if len(df) < WARMUP_RSI + lookback:
        return True
    rsi = calc_rsi_tv(df["close"], period=14)
    return not bool((rsi.iloc[-lookback:] <= threshold).any())

def check_rsi_not_overbought_recently(df, lookback=50, threshold=70):
    if len(df) < WARMUP_RSI + lookback:
        return True
    rsi = calc_rsi_tv(df["close"], period=14)
    return not bool((rsi.iloc[-lookback:] >= threshold).any())

def check_confirm_rsi_not_oversold(df, lookback=30, threshold=30):
    if len(df) < WARMUP_RSI + lookback:
        return True
    rsi = calc_rsi_tv(df["close"], period=14)
    return not bool((rsi.iloc[-lookback:] <= threshold).any())

def check_confirm_rsi_not_overbought(df, lookback=30, threshold=70):
    if len(df) < WARMUP_RSI + lookback:
        return True
    rsi = calc_rsi_tv(df["close"], period=14)
    return not bool((rsi.iloc[-lookback:] >= threshold).any())


def calc_close_correlation(df_a, df_b, lookback=50):
    """
    Pearson correlation of log-returns between two OHLCV frames aligned on ``ts``.

    Returns None when there is not enough overlapping history.
    """
    if (
        df_a is None
        or df_b is None
        or getattr(df_a, "empty", True)
        or getattr(df_b, "empty", True)
        or lookback < 2
    ):
        return None
    left = df_a[["ts", "close"]].rename(columns={"close": "close_a"})
    right = df_b[["ts", "close"]].rename(columns={"close": "close_b"})
    merged = left.merge(right, on="ts", how="inner")
    if len(merged) < lookback + 1:
        return None
    window = merged.tail(lookback + 1)
    ret_a = np.log(window["close_a"].astype(float)).diff().dropna()
    ret_b = np.log(window["close_b"].astype(float)).diff().dropna()
    if len(ret_a) < lookback or len(ret_b) < lookback:
        return None
    corr = ret_a.corr(ret_b)
    if pd.isna(corr):
        return None
    return float(corr)


def check_btc_correlation(df_alt, df_btc, lookback=50, min_corr=0.5):
    """True when alt/BTC close-return correlation is at least ``min_corr``."""
    corr = calc_close_correlation(df_alt, df_btc, lookback=lookback)
    if corr is None:
        return False
    return corr >= float(min_corr)

def check_rsi_closed_oversold(df, threshold=35):
    if len(df) < WARMUP_RSI:
        return False
    rsi = calc_rsi_tv(df["close"], period=14)
    return bool(rsi.iloc[-1] <= threshold)

def check_rsi_closed_overbought(df, threshold=65):
    if len(df) < WARMUP_RSI:
        return False
    rsi = calc_rsi_tv(df["close"], period=14)
    return bool(rsi.iloc[-1] >= threshold)

def check_rsi_touched_since(df, since_ts, threshold=35, direction="long"):
    """
    يفحص هل RSI لمس المستوى المطلوب منذ وقت since_ts وحتى الآن
    direction:
      - "long"  => لمس <= threshold
      - "short" => لمس >= threshold

    يُحسب RSI على السلسلة الكاملة ثم تُفلتر النافذة الزمنية — حتى لا يفسد الـ warmup
    إذا كانت النافذة قصيرة.
    """
    if df.empty or since_ts is None or len(df) < WARMUP_RSI:
        return False

    mask = df["ts"] >= since_ts
    if not mask.any():
        return False

    rsi = calc_rsi_tv(df["close"], period=14)
    rsi_window = rsi[mask]

    if direction == "long":
        return bool((rsi_window <= threshold).any())
    return bool((rsi_window >= threshold).any())


def find_smi_touch_index(df, since_ts, threshold=-40, direction="long"):
    """
    أول شمعة لمس فيها SMI مستوى التشبع منذ since_ts.

    direction:
      - "long"  => SMI <= threshold (افتراضي -40)
      - "short" => SMI >= threshold (افتراضي +40)
    """
    if df.empty or since_ts is None or len(df) < WARMUP_SMI:
        return None
    if direction not in ("long", "short"):
        raise ValueError(f"Unsupported direction: {direction}")

    mask = df["ts"] >= since_ts
    if not mask.any():
        return None

    smi, _, _ = calc_smi(df["high"], df["low"], df["close"])
    for index in df.index[mask]:
        value = float(smi.iloc[index])
        if direction == "long" and value <= threshold:
            return int(index)
        if direction == "short" and value >= threshold:
            return int(index)
    return None


def check_smi_touched_since(df, since_ts, threshold=-40, direction="long"):
    """
    يفحص هل SMI لمس المستوى المطلوب منذ since_ts.
    يُحسب SMI على السلسلة الكاملة ثم تُفلتر النافذة الزمنية للحفاظ على الـ warmup.
    """
    return find_smi_touch_index(
        df,
        since_ts,
        threshold=threshold,
        direction=direction,
    ) is not None


def find_rsi_touch_index(df, since_ts, threshold=35, direction="long"):
    """
    أول شمعة مغلقة لمس فيها RSI المستوى منذ since_ts.

    يُستخدم سعر RSI الخام فقط — بدون متوسط RSI.
    LONG: RSI <= 35 | SHORT: RSI >= 65
    """
    if df.empty or since_ts is None or len(df) < WARMUP_RSI:
        return None
    if direction not in ("long", "short"):
        raise ValueError(f"Unsupported direction: {direction}")

    mask = df["ts"] >= since_ts
    if not mask.any():
        return None

    rsi = calc_rsi_tv(df["close"], period=14)
    for index in df.index[mask]:
        value = float(rsi.iloc[index])
        if direction == "long" and value <= threshold:
            return int(index)
        if direction == "short" and value >= threshold:
            return int(index)
    return None


def find_rsi_ma_cross_index(df, since_ts, side="long", at_or_after=None):
    """
    أول تقاطع RSI مع متوسطه SMA(14) بعد since_ts.
    LONG: تقاطع لفوق المتوسط | SHORT: تقاطع لتحت المتوسط

    at_or_after: يُستخدم لفرض أن التقاطع لا يُحتسب قبل لمس RSI للمستوى.
    """
    if df.empty or since_ts is None or len(df) < WARMUP_RSI:
        return None
    if side not in ("long", "short"):
        raise ValueError(f"Unsupported side: {side}")

    rsi = calc_rsi_tv(df["close"], period=14)
    rsi_ma = rsi.rolling(14).mean()
    start_positions = list(df.index[df["ts"] >= since_ts])
    if not start_positions:
        return None
    start_pos = max(int(start_positions[0]), 1)
    if at_or_after is not None:
        start_pos = max(start_pos, int(at_or_after))

    for i in range(start_pos, len(df)):
        try:
            prev_rsi = float(rsi.iloc[i - 1])
            prev_ma = float(rsi_ma.iloc[i - 1])
            curr_rsi = float(rsi.iloc[i])
            curr_ma = float(rsi_ma.iloc[i])
            if side == "long" and prev_rsi < prev_ma and curr_rsi >= curr_ma:
                return i
            if side == "short" and prev_rsi > prev_ma and curr_rsi <= curr_ma:
                return i
        except (ValueError, IndexError, TypeError):
            continue
    return None


def find_stoch_level_after_index(
    df,
    cross_index,
    side="long",
    stoch_level=None,
    max_gap=3,
):
    """
    بعد تقاطع RSI: أول شمعة يكون فيها Stochastic بالاتجاه خلال max_gap شموع.
    LONG: %K > stoch_level (20) | SHORT: %K < stoch_level (80)
    """
    if df.empty or cross_index is None or len(df) < WARMUP_STOCH:
        return None
    if side not in ("long", "short"):
        raise ValueError(f"Unsupported side: {side}")
    if stoch_level is None:
        stoch_level = 20 if side == "long" else 80

    k, _ = calc_stoch_tv(df["close"], df["high"], df["low"])
    start = int(cross_index)
    end = min(len(df) - 1, start + int(max_gap))
    for i in range(start, end + 1):
        try:
            value = float(k.iloc[i])
            if side == "long" and value > stoch_level:
                return i
            if side == "short" and value < stoch_level:
                return i
        except (ValueError, IndexError, TypeError):
            continue
    return None


def find_rsi_stoch_entry_index(
    df,
    since_ts,
    max_gap=3,
    side="long",
    rsi_threshold=None,
    stoch_level=None,
):
    """
    بعد تشبع SMI:
    1) لمس RSI الخام للمستوى على شمعة مغلقة (35/65) — بدون متوسط
    2) بعدها تقاطع RSI مع متوسطه (هنا فقط يُستخدم المتوسط)
    3) بعده خلال max_gap شموع: Stochastic %K فوق 20 / تحت 80 — بدون %D

    التقاطع قبل لمس 35/65 مرفوض. شمعة الإشارة = تحقق Stoch.
    """
    if df.empty or since_ts is None or len(df) < WARMUP_RSI:
        return None
    if side not in ("long", "short"):
        raise ValueError(f"Unsupported side: {side}")

    if rsi_threshold is None:
        rsi_threshold = 35 if side == "long" else 65
    if stoch_level is None:
        stoch_level = 20 if side == "long" else 80
    if max_gap is None:
        max_gap = 3

    direction = "long" if side == "long" else "short"
    rsi_touch = find_rsi_touch_index(
        df,
        since_ts,
        threshold=rsi_threshold,
        direction=direction,
    )
    if rsi_touch is None:
        return None

    # التقاطع يُبحث من شمعة اللمس فما بعد — لا تقاطع سابق على اللمس.
    rsi_cross = find_rsi_ma_cross_index(
        df,
        since_ts,
        side=side,
        at_or_after=rsi_touch,
    )
    if rsi_cross is None:
        return None

    return find_stoch_level_after_index(
        df,
        rsi_cross,
        side=side,
        stoch_level=stoch_level,
        max_gap=max_gap,
    )


def find_step8_entry_index(
    df,
    since_ts,
    *,
    smi_threshold,
    rsi_threshold,
    direction="long",
    max_gap=3,
    stoch_level=None,
):
    """
    ترتيب Step 8 على فريم الثلث (شموع مغلقة):
    1) إغلاق كامل لتشبع SMI أولًا
    2) لمس RSI (35 تشبع بيعي / 65 تشبع شرائي)
    3) تقاطع RSI مع متوسطه
    4) خلال 3 شموع بعده: Stoch فوق 20 / تحت 80
    """
    smi_index = find_smi_touch_index(
        df,
        since_ts,
        threshold=smi_threshold,
        direction=direction,
    )
    if smi_index is None:
        return None

    after_smi_ts = df["ts"].iloc[smi_index]
    side = "long" if direction == "long" else "short"
    if stoch_level is None:
        stoch_level = 20 if side == "long" else 80
    return find_rsi_stoch_entry_index(
        df,
        after_smi_ts,
        max_gap=max_gap if max_gap is not None else 3,
        side=side,
        rsi_threshold=rsi_threshold,
        stoch_level=stoch_level,
    )


def check_rsi_stoch(df, since_ts, max_gap=3):
    """
    LONG: لمس RSI≤35، تقاطع فوق المتوسط، ثم Stoch>20 خلال max_gap شموع.
    """
    return find_rsi_stoch_entry_index(
        df,
        since_ts,
        max_gap=max_gap,
        side="long",
        rsi_threshold=35,
        stoch_level=20,
    ) is not None


def check_rsi_stoch_short(df, since_ts, max_gap=3):
    """
    SHORT: لمس RSI≥65، تقاطع تحت المتوسط، ثم Stoch<80 خلال max_gap شموع.
    """
    return find_rsi_stoch_entry_index(
        df,
        since_ts,
        max_gap=max_gap,
        side="short",
        rsi_threshold=65,
        stoch_level=80,
    ) is not None
