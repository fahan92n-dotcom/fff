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

def resample_ohlcv(df, minutes):
    """
    يُعيد تجميع (resample) بيانات OHLCV إلى فريم زمني محدد بالدقائق.

    نقطة البداية origin=datetime(1970,1,1,UTC) هي نقطة أصل Unix القياسية (epoch).
    بما أن كل الفريمات في TIMEFRAME_CHAIN (9،12،15،18،21،24،27،30،45،60،90،120،150،180،210،240 دقيقة)
    هي مضاعفات صحيحة تنقسم على 1440 دقيقة (يوم) أو تتوافق مع حدود UTC اليومية،
    فإن epoch ينتج حدود شموع مطابقة تمامًا لما تعرضه Binance وTradingView.
    ⚠️ لا تُغيّر origin أو تُضِف offset دون التحقق من التوافق مع جميع الفريمات أعلاه.

    هذه الدالة تحذف الشمعة الأخيرة إذا لم تُغلق بعد (شمعة جارية).
    استخدم هذه الدالة حصرًا في مسارات تقييم الإشارات (step1-step8 وما شابه).
    """
    if df.empty:
        return pd.DataFrame()
    now = datetime.now(timezone.utc)
    resampled = (df.copy().set_index("ts")
                 .resample(f"{minutes}min", closed="left", label="left", origin=datetime(1970, 1, 1, tzinfo=timezone.utc))
                 .agg({"open": "first", "high": "max", "low": "min", "close": "last", "vol": "sum"})
                 .dropna().reset_index())
    if resampled.empty:
        return resampled
    # احذف فقط إذا الشمعة الأخيرة لم تُغلق بعد
    last_candle_end = resampled["ts"].iloc[-1] + pd.Timedelta(minutes=minutes)
    if now < last_candle_end:
        resampled = resampled.iloc[:-1]
    return resampled

def resample_ohlcv_closed(df, minutes):
    """
    يُعيد تجميع (resample) بيانات OHLCV إلى فريم زمني محدد بالدقائق دون حذف الشمعة الأخيرة.

    نفس origin=datetime(1970,1,1,UTC) المستخدمة في resample_ohlcv — راجع تعليق تلك الدالة
    للتفصيل حول سبب اختيار epoch وتوافقه مع Binance/TradingView.

    ⚠️ هذه الدالة لا تحذف الشمعة الجارية غير المغلقة.
    يجب عدم استخدامها في مسارات تقييم الإشارات (step1-step8 وما شابه) لتجنب الحساب على شمعة غير مكتملة.
    """
    if df.empty:
        return pd.DataFrame()
    return (df.copy().set_index("ts").resample(f"{minutes}min", closed="left", label="left", origin=datetime(1970, 1, 1, tzinfo=timezone.utc))
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "vol": "sum"})
            .dropna().reset_index())

# ------------------------------------------
# MACD
# ------------------------------------------

def wilder_rma(series, period):
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

def _calc_macd_hist(close):
    macd_line = (close.ewm(span=12, min_periods=12, adjust=False).mean()
                 - close.ewm(span=26, min_periods=26, adjust=False).mean())
    signal = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    return macd_line - signal

def _calc_macd_full(close):
    macd_line = (close.ewm(span=12, min_periods=12, adjust=False).mean()
                 - close.ewm(span=26, min_periods=26, adjust=False).mean())
    signal_line = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def check_macd_red(df):
    if len(df) < WARMUP_MACD:
        return False
    return bool(_calc_macd_hist(df["close"]).iloc[-1] < 0)

def check_macd_green(df):
    if len(df) < WARMUP_MACD:
        return False
    return bool(_calc_macd_hist(df["close"]).iloc[-1] > 0)

def _get_macd_window_hours(base_frame_minutes):
    """إرجاع عدد ساعات نافذة المشاهدة حسب حجم الفريم:
    - فريم ≤ 60 دقيقة → 24 ساعة
    - فريم > 60 دقيقة → 72 ساعة (3 أيام)
    """
    return 24 if base_frame_minutes <= 60 else 72

def check_macd_line_long(df, pct=0.40, base_frame=60):
    if len(df) < WARMUP_MACD:
        return False
    macd_line, _, histogram = _calc_macd_full(df["close"])
    current_macd = float(macd_line.iloc[-1])
    current_hist = float(histogram.iloc[-1])
    if current_hist < 0 and current_macd < current_hist:
        return False
    
    # نافذة زمنية ديناميكية: 24 ساعة للفريمات ≤ 60 دقيقة، 72 ساعة للفريمات > 60 دقيقة
    window_hours = _get_macd_window_hours(base_frame)
    last_ts = df["ts"].iloc[-1]
    cutoff_ts = last_ts - timedelta(hours=window_hours)
    macd_today = macd_line[df["ts"].values >= np.datetime64(cutoff_ts)]
    if macd_today.empty:
        macd_today = macd_line
    
    # أعلى قيمة في النافذة الزمنية المحددة
    max_window = float(macd_today.max())
    
    # 40% من أعلى قيمة
    threshold = max_window * pct
    
    if current_macd > threshold:
        return False
    return True

def check_macd_line_short(df, pct=0.40, base_frame=60):
    if len(df) < WARMUP_MACD:
        return False
    macd_line, _, histogram = _calc_macd_full(df["close"])
    current_macd = float(macd_line.iloc[-1])
    current_hist = float(histogram.iloc[-1])
    if current_hist > 0 and current_macd > current_hist:
        return False
    
    # نافذة زمنية ديناميكية: 24 ساعة للفريمات ≤ 60 دقيقة، 72 ساعة للفريمات > 60 دقيقة
    window_hours = _get_macd_window_hours(base_frame)
    last_ts = df["ts"].iloc[-1]
    cutoff_ts = last_ts - timedelta(hours=window_hours)
    macd_today = macd_line[df["ts"].values >= np.datetime64(cutoff_ts)]
    if macd_today.empty:
        macd_today = macd_line
    
    # أدنى قيمة في النافذة الزمنية المحددة
    min_window = float(macd_today.min())
    
    # 40% من أدنى قيمة
    threshold = min_window * pct
    
    if current_macd < threshold:
        return False
    return True

# ------------------------------------------
# Donchian
# ------------------------------------------

def calc_donchian_trend_pine(close_arr, high_arr, low_arr, length):
    """
    Pine-exact Donchian trend replicating dchannel(len) from the Pine Script:

        hh = highest(len)          -- rolling max of high INCLUDING current bar
        ll = lowest(len)           -- rolling min of low  INCLUDING current bar
        trend := close > hh[1] ? 1 : close < ll[1] ? -1 : nz(trend[1])

    hh[1] / ll[1] in Pine = prior bar's rolling max/min, implemented here as
    rolling(length).max/min().shift(1).  Trend is carried forward via ffill
    (equivalent to nz(trend[1])).  Only closed-candle data should be passed.

    Returns 1 (bullish), -1 (bearish), or 0 (insufficient data).
    """
    n = len(close_arr)
    if n < length + 1:
        return 0

    high_s = pd.Series(high_arr, dtype=float)
    low_s = pd.Series(low_arr, dtype=float)
    close_s = pd.Series(close_arr, dtype=float)

    # hh[1] / ll[1]: rolling max/min over `length` bars, shifted back 1 bar
    hh = high_s.rolling(length, min_periods=length).max().shift(1)
    ll = low_s.rolling(length, min_periods=length).min().shift(1)

    raw = pd.Series(np.nan, index=close_s.index, dtype=float)
    raw[close_s.gt(hh).fillna(False)] = 1.0
    raw[close_s.lt(ll).fillna(False)] = -1.0

    # nz(trend[1]): carry the last known trend forward
    trend = raw.ffill().fillna(0)
    try:
        return int(trend.iloc[-1])
    except Exception:
        return 0


def _calc_donchian_ribbon_result(close, high, low):
    """
    Evaluate the full 10-band Donchian Trend Ribbon matching the Pine plot ladder:

        plot( 5, color = dchannelalt(dlen - 0, maintrend), ...)
        plot(10, color = dchannelalt(dlen - 1, maintrend), ...)
        ...
        plot(50, color = dchannelalt(dlen - 9, maintrend), ...)

    Band 0 (length = dlen)     sets maintrend.
    Bands 1-9 (lengths dlen-1 .. dlen-9) are sub-trends that must agree with
    maintrend for the ribbon to be fully solid green or fully solid red.

    Returns:
         1  when maintrend == 1  and all 10 sub-trends == 1  (fully green ribbon)
        -1  when maintrend == -1 and all 10 sub-trends == -1 (fully red ribbon)
         0  otherwise (mixed, neutral, or insufficient data)
    """
    dlen = DONCHIAN_DLEN
    n = len(close)

    high_s = pd.Series(high, dtype=float)
    low_s = pd.Series(low, dtype=float)
    close_s = pd.Series(close, dtype=float)

    maintrend = 0
    for i in range(10):
        length = dlen - i  # 20, 19, 18, ..., 11
        if n < length + 1:
            return 0
        hh = high_s.rolling(length, min_periods=length).max().shift(1)
        ll = low_s.rolling(length, min_periods=length).min().shift(1)
        raw = pd.Series(np.nan, index=close_s.index, dtype=float)
        raw[close_s.gt(hh).fillna(False)] = 1.0
        raw[close_s.lt(ll).fillna(False)] = -1.0
        t = int(raw.ffill().fillna(0).iloc[-1])
        if i == 0:
            maintrend = t
            if maintrend == 0:
                return 0
        elif t != maintrend:
            return 0

    return maintrend  # all 10 bands agree with maintrend


# ------------------ Donchian check + cache ------------------
def check_donchian_trend_ribbon(df, direction="green", cache_key=None):
    """
    Evaluates the full Donchian Trend Ribbon (Pine-exact) and caches the result.

    Replicates the Pine Script's ten dchannelalt() plot calls using the ladder
    of lengths dlen-0 through dlen-9 (20 down to 11).

    direction="green" → True only when all 10 bands agree on bullish  (result == 1)
    direction="red"   → True only when all 10 bands agree on bearish  (result == -1)

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

def check_ema50_below(df):
    ema = df["close"].ewm(span=50, adjust=False).mean()
    return bool(df["close"].iloc[-1] < ema.iloc[-1])

def check_ema50_above(df):
    ema = df["close"].ewm(span=50, adjust=False).mean()
    return bool(df["close"].iloc[-1] > ema.iloc[-1])

# ------------------------------------------
# SMI
# ------------------------------------------

def calc_smi(high, low, close, k=10, smooth_period=1, d=3, c=10):
    """
    Stochastic Momentum Index (SMI) - مطابق تماماً لـ Pine Script v5 (Stoch_MTM)
    """
    ll = low.rolling(k, min_periods=k).min()
    hh = high.rolling(k, min_periods=k).max()
    diff = hh - ll
    rdiff = close - (hh + ll) / 2

    avgrel = rdiff.ewm(span=d, min_periods=d, adjust=False).mean()
    avgdiff = diff.ewm(span=d, min_periods=d, adjust=False).mean()

    smi = np.where(avgdiff != 0, (avgrel / (avgdiff / 2)) * 100, 0.0)
    smi = pd.Series(smi, index=close.index)

    smi_smoothed = smi.rolling(smooth_period, min_periods=smooth_period).mean()
    
    # ✅ التصحيح - D Line (optional)
    smi_signal = smi_smoothed.ewm(span=d, min_periods=d, adjust=False).mean()
    
    # ✅ Signal Line الصحيح (المهم)
    ema_signal = smi_smoothed.ewm(span=c, min_periods=c, adjust=False).mean()

    return smi_smoothed, ema_signal, smi_signal  # ⬅️ لاحظ الترتيب

def check_smi_oversold(df, threshold=-40):
    if len(df) < WARMUP_SMI:
        return False
    smi, _, _ = calc_smi(df["high"], df["low"], df["close"])
    return bool(smi.iloc[-1] <= threshold)

def check_smi_overbought(df, threshold=40):
    if len(df) < WARMUP_SMI:
        return False
    smi, _, _ = calc_smi(df["high"], df["low"], df["close"])
    return bool(smi.iloc[-1] >= threshold)

def check_ema50_above_since_overbought(df, smi_threshold=40):
    if len(df) < WARMUP_SMI:
        return False
    smi, _, _ = calc_smi(df["high"], df["low"], df["close"])
    ema = df["close"].ewm(span=50, adjust=False).mean()
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


def check_smi_touched_since(df, since_ts, threshold=-40, direction="long"):
    """
    يفحص هل SMI لمس المستوى المطلوب منذ since_ts.
    يُحسب SMI على السلسلة الكاملة ثم تُفلتر النافذة الزمنية للحفاظ على الـ warmup.
    """
    if df.empty or since_ts is None or len(df) < WARMUP_SMI:
        return False
    mask = df["ts"] >= since_ts
    if not mask.any():
        return False
    smi, _, _ = calc_smi(df["high"], df["low"], df["close"])
    smi_window = smi[mask]
    if direction == "long":
        return bool((smi_window <= threshold).any())
    return bool((smi_window >= threshold).any())


def check_rsi_stoch(df, since_ts, max_gap=3):
    if df.empty or since_ts is None or len(df) < WARMUP_RSI:
        return False
    rsi = calc_rsi_tv(df["close"], period=14)
    rsi_sig = rsi.rolling(14).mean()
    k, _ = calc_stoch_tv(df["close"], df["high"], df["low"])
    
    # ✅ الحالة الحالية يجب تكون آمنة
    current_k = float(k.iloc[-1])
    current_rsi = float(rsi.iloc[-1])
    
    if current_k <= 20:  # تشبع بيعي
        return False
    if current_k >= 80:  # تشبع شرائي ❌ ارفض!
        return False
    if current_rsi < float(rsi_sig.iloc[-1]):  # ارتد ❌ ارفض!
        return False

    start_positions = df.index[df["ts"] >= since_ts]
    if len(start_positions) < 2:
        return False
    start_pos = max(int(start_positions[0]), 1)
    
    stoch_crosses = []
    rsi_crosses = []
    for i in range(start_pos, len(df)):
        try:
            if float(k.iloc[i - 1]) <= 20 and float(k.iloc[i]) > 20:
                stoch_crosses.append(i)
            if float(rsi.iloc[i - 1]) < float(rsi_sig.iloc[i - 1]) and float(rsi.iloc[i]) >= float(rsi_sig.iloc[i]):
                rsi_crosses.append(i)
        except (ValueError, IndexError):
            continue
    
    for sc in stoch_crosses:
        for rc in rsi_crosses:
            if abs(sc - rc) <= max_gap:
                return True
    return False

def check_rsi_stoch_short(df, since_ts, max_gap=3):
    if df.empty or since_ts is None or len(df) < WARMUP_RSI:
        return False
    rsi = calc_rsi_tv(df["close"], period=14)
    rsi_sig = rsi.rolling(14).mean()
    k, _ = calc_stoch_tv(df["close"], df["high"], df["low"])
    
    # ✅ الحالة الحالية يجب تكون آمنة (بين 20 و 80)
    current_k = float(k.iloc[-1])
    current_rsi = float(rsi.iloc[-1])
    
    if current_k >= 80:  # تشبع شرائي
        return False
    if current_k <= 20:  # تشبع بيعي ❌ ارفض!
        return False
    if current_rsi > float(rsi_sig.iloc[-1]):  # ارتد لفوق ❌ ارفض!
        return False

    start_positions = df.index[df["ts"] >= since_ts]
    if len(start_positions) < 2:
        return False
    start_pos = max(int(start_positions[0]), 1)
    
    stoch_crosses = []
    rsi_crosses = []
    for i in range(start_pos, len(df)):
        try:
            if float(k.iloc[i - 1]) >= 80 and float(k.iloc[i]) < 80:
                stoch_crosses.append(i)
            if float(rsi.iloc[i - 1]) > float(rsi_sig.iloc[i - 1]) and float(rsi.iloc[i]) <= float(rsi_sig.iloc[i]):
                rsi_crosses.append(i)
        except (ValueError, IndexError):
            continue
    
    for sc in stoch_crosses:
        for rc in rsi_crosses:
            if abs(sc - rc) <= max_gap:
                return True
    return False
