"""بوت مسح العملات من Binance مع تنبيهات Telegram.""" 
"""بوت مسح العملات من Binance مع تنبيهات Telegram - نسخة Cascade Pipeline."""
import os
import time
import logging
import threading
import sys
import traceback
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer, BaseHTTPRequestHandler

import requests
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------
# Main Settings
# ------------------------------------------

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8907286779:AAFTn1sfkpOnUgwlChN3RIV9xLqQ9EqAnzk")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "-1003972769219")

BINANCE_BASE = "https://data-api.binance.vision"
TOP_SYMBOLS_LIMIT = 200
PORT = int(os.environ.get("PORT", "8080"))
ALERT_EXPIRY_HOURS = 4

TF_MAP = {"1m": "1m", "5m": "5m", "60m": "1h"}

TRIPLING_PAIRS = [
    (9, 27, 3, "1m", "1m"),
    (12, 36, 4, "1m", "1m"),
    (15, 45, 5, "1m", "1m"),
    (18, 54, 6, "1m", "1m"),
    (21, 63, 7, "1m", "1m"),
    (24, 72, 8, "1m", "1m"),
    (27, 81, 9, "1m", "1m"),
    (30, 90, 10, "1m", "1m"),
    (45, 135, 15, "1m", "1m"),
    (60, 180, 20, "60m", "1m"),
    (90, 270, 30, "60m", "1m"),
    (120, 360, 40, "60m", "1m"),
    (180, 540, 60, "60m", "60m"),
]

TIMEFRAME_CHAIN = [9, 12, 15, 18, 21, 24, 27, 30, 45, 60, 90, 120, 180]
NEXT_TF = {TIMEFRAME_CHAIN[i]: TIMEFRAME_CHAIN[i + 1] for i in range(len(TIMEFRAME_CHAIN) - 1)}

FAST_FETCH_CANDLES = {"1m": 3500, "60m": 250}
API_FETCH_CANDLES = {"1m": 15_000, "60m": 2_000}
CACHE_MAX_CANDLES = {"1m": 16_000, "60m": 2_500}
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

WARMUP_EMA = 200
WARMUP_MACD = 200
WARMUP_SMI = 100
WARMUP_RSI = 200
WARMUP_STOCH = 100
WARMUP_DON = 50
MIN_CANDLES = 250

# ------------------------------------------
# Shared State
# ------------------------------------------

alerted_keys = {}
alerted_keys_lock = threading.Lock()
trades_history = deque(maxlen=2000)
trades_lock = threading.Lock()
symbols_cache = []
symbols_cache_lock = threading.Lock()
ohlcv_cache = {}
ohlcv_cache_lock = threading.Lock()

# ============ CASCADE PIPELINE STATE ============
cascade_results = defaultdict(dict)
cascade_results_lock = threading.Lock()

cascade_stats = {i: {"total": 0, "passed": 0} for i in range(1, 8)}
cascade_stats_lock = threading.Lock()
# ================================================
last_complete_stats = {i: {"total": 0, "passed": 0} for i in range(1, 8)}
last_complete_results = defaultdict(dict)
last_complete_lock = threading.Lock()
fast_prefetch_done = threading.Event()
prefetch_done = threading.Event()

_local = threading.local()

# ------------------------------------------
# Diagnostics Labels
# ------------------------------------------

STEP_NAMES = [
    "smi_oversold",
    "macd_red",
    "donchian_entry",
    "donchian_confirm",
    "macd_confirm",
    "ema50",
    "rsi_stoch",
]

STEP_LABELS = {
    "smi_oversold": "① تشبع بيعي SMI",
    "macd_red": "② MACD أحمر",
    "donchian_entry": "③ Donchian Ribbon أخضر",
    "donchian_confirm": "④ Donchian Ribbon Confirm أخضر",
    "macd_confirm": "⑤ MACD Confirm أخضر",
    "ema50": "⑥ السعر تحت EMA50",
    "rsi_stoch": "⑦ RSI/Stochastic تقاطع",
}


# ------------------------------------------
# Helpers
# ------------------------------------------

def get_session():
    """Return a thread-local requests session."""
    if not hasattr(_local, "s"):
        session = requests.Session()
        session.headers.update({"Accept-Encoding": "gzip", "User-Agent": "Mozilla/5.0"})
        _local.s = session
    return _local.s


def delete_webhook():
    """Delete the Telegram webhook."""
    try:
        r = get_session().post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook",
            json={"drop_pending_updates": True}, timeout=10,
        ).json()
        if r.get("ok"):
            log.info("✅ تم حذف الـ Webhook")
    except requests.RequestException as exc:
        log.error("deleteWebhook error: %s", exc)


def cleanup_alerted_keys():
    """Remove expired alert keys."""
    now = datetime.now(timezone.utc)
    with alerted_keys_lock:
        expired = [
            k for k, t in list(alerted_keys.items())
            if now - t > timedelta(hours=ALERT_EXPIRY_HOURS)
        ]
        for k in expired:
            del alerted_keys[k]


def save_signal(symbol, price, entry_min, confirm_min, third_min):
    """Save a trading signal to history."""
    with trades_lock:
        trades_history.append({
            "time": datetime.now(timezone.utc),
            "symbol": symbol,
            "price": price,
            "timeframe": f"{entry_min}m/{confirm_min}m/{third_min}m",
        })


def send_telegram(msg, chat_id=None):
    """Send a message via Telegram."""
    target = chat_id or TELEGRAM_CHAT_ID
    try:
        r = get_session().post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": target, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        ).json()
        return r.get("ok", False)
    except requests.RequestException as exc:
        log.error("Telegram send error: %s", exc)
        return False


def get_report(period="today"):
    """Generate a report of signals for the given period."""
    now = datetime.now(timezone.utc)

    if period == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end, title = now, "📅 إشارات اليوم"
    elif period == "yesterday":
        end = now.replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=1)
        title = "📅 إشارات أمس"
    else:
        start = now - timedelta(days=7)
        end, title = now, "🗓️ آخر 7 أيام"

    with trades_lock:
        rows = [t for t in trades_history if start <= t["time"] < end]

    if not rows:
        return f"<b>{title}:</b>\nلا توجد إشارات."

    lines = [f"<b>{title} ({len(rows)})</b>\n" + "━" * 15]
    for t in rows:
        lines.append(
            f"✅ {t['symbol']} | {t['timeframe']} | "
            f"{t['price']:.4g} | {t['time'].strftime('%H:%M UTC')}"
        )
    return "\n".join(lines)

# ------------------------------------------
# Binance OHLCV
# ------------------------------------------


def _parse_binance_klines(resp):
    """Parse raw Binance kline response into a DataFrame."""
    df = pd.DataFrame(resp, columns=[
        "ts", "open", "high", "low", "close", "vol",
        "close_time", "quote_vol", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "vol"]:
        df[col] = df[col].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)[
        ["ts", "open", "high", "low", "close", "vol"]
    ]


def get_ohlcv(symbol, tf, limit=500):
    """Fetch OHLCV data from Binance for the given symbol and timeframe."""
    binance_tf = TF_MAP.get(tf, "1m")
    try:
        resp = get_session().get(
            f"{BINANCE_BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": binance_tf, "limit": min(limit, 1000)},
            timeout=10,
        ).json()
        if isinstance(resp, list) and resp:
            return _parse_binance_klines(resp)
    except requests.RequestException as exc:
        log.error("get_ohlcv %s %s: %s", symbol, tf, exc)
    return pd.DataFrame()


def get_ohlcv_full(symbol, tf, target):
    """Fetch a large batch of OHLCV data by paginating backwards."""
    binance_tf = TF_MAP.get(tf, "1m")
    tf_ms = 60_000 if tf == "1m" else 3_600_000
    bin_max = 1000
    all_dfs, end_ms, fetched, retries = [], int(time.time() * 1000), 0, 0

    while fetched < target:
        batch = min(bin_max, target - fetched)
        start_ms = end_ms - batch * tf_ms
        try:
            resp = get_session().get(
                f"{BINANCE_BASE}/api/v3/klines",
                params={
                    "symbol": symbol, "interval": binance_tf,
                    "startTime": start_ms, "endTime": end_ms, "limit": batch,
                },
                timeout=15,
            ).json()
            if not isinstance(resp, list) or not resp:
                retries += 1
                if retries >= 3:
                    break
                time.sleep(2 ** retries)
                continue
            df = _parse_binance_klines(resp)
            all_dfs.insert(0, df)
            fetched += len(df)
            retries = 0
            end_ms = start_ms - 1
            if len(df) < batch:
                break
        except requests.RequestException:
            retries += 1
            if retries >= 3:
                break
            time.sleep(2)

    return (
        pd.concat(all_dfs).drop_duplicates(subset="ts")
        .sort_values("ts").reset_index(drop=True)
        if all_dfs else pd.DataFrame()
    )


def cache_merge(symbol, tf, new_df):
    """Merge new OHLCV data into the cache for the given symbol/timeframe."""
    if new_df.empty:
        return
    key = (symbol, tf)
    maxc = CACHE_MAX_CANDLES.get(tf, 5000)
    with ohlcv_cache_lock:
        old = ohlcv_cache.get(key)
        if old is not None and not old.empty:
            merged = pd.concat([old, new_df]).drop_duplicates(subset="ts").sort_values("ts")
            ohlcv_cache[key] = merged.tail(maxc).reset_index(drop=True)
        else:
            ohlcv_cache[key] = new_df.tail(maxc).reset_index(drop=True)


def get_cached(symbol, tf):
    """Return a copy of the cached OHLCV data for the given symbol/timeframe."""
    with ohlcv_cache_lock:
        df = ohlcv_cache.get((symbol, tf))
    return df.copy() if df is not None else pd.DataFrame()


def prefetch_all(symbols):
    """Prefetch OHLCV data for all symbols in two passes: fast then full."""
    def fetch_sym_fast(sym):
        for tf, n in FAST_FETCH_CANDLES.items():
            df = get_ohlcv_full(sym, tf, target=n)
            cache_merge(sym, tf, df)

    def fetch_sym_full(sym):
        for tf, n in API_FETCH_CANDLES.items():
            df = get_ohlcv_full(sym, tf, target=n)
            cache_merge(sym, tf, df)

    log.info("🚀 بدء التحميل السريع بالـ threads...")
    with ThreadPoolExecutor(max_workers=15) as executor:
        executor.map(fetch_sym_fast, symbols)
    fast_prefetch_done.set()
    send_telegram("⚡ <b>التحميل السريع اكتمل — البوت يعمل الآن!</b>")

    log.info("📦 بدء التحميل الكامل...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(fetch_sym_full, symbols)
    prefetch_done.set()
    send_telegram("✅ <b>التحميل الكامل اكتمل وجاهز للعمل!</b>")


def _update_batch(symbols, tf, limit):
    """Fetch recent candles for a batch of symbols and update cache."""
    def fetch_one(sym):
        df = get_ohlcv(sym, tf, limit=limit)
        if not df.empty:
            cache_merge(sym, tf, df)
    with ThreadPoolExecutor(max_workers=30) as executor:
        executor.map(fetch_one, symbols)


def cache_updater_1m():
    """Background thread: refresh 1m cache every 55 seconds."""
    while True:
        if not fast_prefetch_done.is_set():
            time.sleep(5)
            continue
        with symbols_cache_lock:
            syms = list(symbols_cache)
        if syms:
            _update_batch(syms, "1m", limit=5)
        time.sleep(55)


def cache_updater_60m():
    """Background thread: refresh 60m cache every hour."""
    while True:
        time.sleep(3600)
        if fast_prefetch_done.is_set():
            with symbols_cache_lock:
                syms = list(symbols_cache)
            if syms:
                _update_batch(syms, "60m", limit=5)


# ------------------------------------------
# Technical Indicators
# ------------------------------------------

def resample_ohlcv(df, minutes):
    """Resample OHLCV data to a larger timeframe, excluding the last (open) candle."""
    if df.empty:
        return pd.DataFrame()
    return (
        df.copy().set_index("ts")
        .resample(f"{minutes}min", closed="left", label="left", origin=EPOCH)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "vol": "sum"})
        .dropna().iloc[:-1].reset_index()
    )


def resample_ohlcv_closed(df, minutes):
    """Resample OHLCV data to a larger timeframe, including the last candle."""
    if df.empty:
        return pd.DataFrame()
    return (
        df.copy().set_index("ts")
        .resample(f"{minutes}min", closed="left", label="left", origin=EPOCH)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "vol": "sum"})
        .dropna().reset_index()
    )


def wilder_rma(series, period):
    """Calculate Wilder's smoothed moving average."""
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _calc_macd_hist(close):
    """Calculate MACD histogram."""
    macd_line = (
        close.ewm(span=12, min_periods=12, adjust=False).mean()
        - close.ewm(span=26, min_periods=26, adjust=False).mean()
    )
    signal = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    return macd_line - signal


def _calc_macd_full(close):
    """Calculate full MACD: line, signal, and histogram."""
    macd_line = (
        close.ewm(span=12, min_periods=12, adjust=False).mean()
        - close.ewm(span=26, min_periods=26, adjust=False).mean()
    )
    signal_line = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def check_macd_red(df):
    """Return True if the latest MACD histogram is negative."""
    if len(df) < WARMUP_MACD:
        return False
    return bool(_calc_macd_hist(df["close"]).iloc[-1] < 0)


def check_macd_green(df):
    """Return True if the latest MACD histogram is positive."""
    if len(df) < WARMUP_MACD:
        return False
    return bool(_calc_macd_hist(df["close"]).iloc[-1] > 0)


# ------------------------------------------
# Donchian Trend Ribbon
# ------------------------------------------

def calc_donchian_trend(df, length=20):
    """Calculate the Donchian channel trend direction."""
    if len(df) < length + 2:
        return []
    hh = df["high"].rolling(length).max().shift(1)
    ll = df["low"].rolling(length).min().shift(1)
    trend = [0] * len(df)
    for i in range(1, len(df)):
        if pd.isna(hh.iloc[i]) or pd.isna(ll.iloc[i]):
            trend[i] = trend[i - 1]
            continue
        if df["close"].iloc[i] > hh.iloc[i]:
            trend[i] = 1
        elif df["close"].iloc[i] < ll.iloc[i]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    return trend


def calc_donchian_trend_ribbon_correct(df, length=20):
    """
    Donchian Trend Ribbon - حساب صحيح:
    ✅ main trend بـ length الأساسي (20)
    ✅ 10 طبقات: 20, 19, 18, ..., 11
    ✅ أخضر: كل الـ 10 طبقات صاعدة
    ✅ أحمر: كل الـ 10 طبقات هابطة
    """
    if len(df) < length + 2:
        return 0, False

    main_trend = calc_donchian_trend(df, length=length)
    if not main_trend:
        return 0, False

    current_main = main_trend[-1]

    layers = []
    for offset in range(10):
        layer_len = length - offset
        layer_trends = calc_donchian_trend(df, length=layer_len)
        if not layer_trends:
            return 0, False
        layers.append(layer_trends[-1])

    if len(layers) < 10:
        return 0, False

    all_consistent = all(t == current_main for t in layers)
    return current_main, all_consistent


def check_donchian_trend_ribbon(df, direction="green"):
    """فحص Donchian Trend Ribbon بشكل صحيح"""
    if len(df) < 35:
        return False

    trend, consistent = calc_donchian_trend_ribbon_correct(df, length=20)

    if not consistent:
        return False

    if direction == "green":
        return trend == 1
    return trend == -1


def check_ema50_below(df):
    """Return True if the latest close is below EMA50."""
    ema = df["close"].ewm(span=50, adjust=False).mean()
    return bool(df["close"].iloc[-1] < ema.iloc[-1])


def calc_smi(high, low, close, k=10, d=3, ema_len=10, smooth=1):
    """Calculate the Stochastic Momentum Index (SMI)."""
    hh = high.rolling(k, min_periods=k).max()
    ll = low.rolling(k, min_periods=k).min()
    diff = hh - ll
    rdiff = close - (hh + ll) / 2
    avgrel = rdiff.ewm(span=d, min_periods=d, adjust=False).mean()
    avgdiff = diff.ewm(span=d, min_periods=d, adjust=False).mean()
    smi_arr = np.where(avgdiff != 0, (avgrel / (avgdiff / 2)) * 100, 0.0)
    smi = pd.Series(smi_arr, index=close.index)
    if smooth > 1:
        smi = smi.rolling(smooth, min_periods=smooth).mean()
    sig = smi.ewm(span=ema_len, min_periods=ema_len, adjust=False).mean()
    return smi, sig


def check_smi_oversold(df, threshold=-40):
    """Return True if the latest SMI value is at or below the oversold threshold."""
    if len(df) < WARMUP_SMI:
        return False
    smi, _ = calc_smi(df["high"], df["low"], df["close"])
    return bool(smi.iloc[-1] <= threshold)


def calc_rsi_tv(close, period=14):
    """Calculate RSI using Wilder's smoothing method."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    up = wilder_rma(gain, period)
    down = wilder_rma(loss, period)
    return 100.0 - (100.0 / (1.0 + up / (down + 1e-10)))


def calc_stoch_tv(close, high, low, k_len=15, k_smooth=3, d_smooth=3):
    """Calculate Stochastic oscillator K and D lines."""
    lo = low.rolling(k_len, min_periods=k_len).min()
    hi = high.rolling(k_len, min_periods=k_len).max()
    raw = 100.0 * (close - lo) / (hi - lo + 1e-10)
    k = raw.rolling(k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(d_smooth, min_periods=d_smooth).mean()
    return k, d


def check_rsi_touched_oversold(df, lookback=10, threshold=35):
    """Return True if RSI touched 35 or below in the last 10 candles."""
    if len(df) < WARMUP_RSI + lookback:
        return False
    rsi = calc_rsi_tv(df["close"], period=14)
    return bool((rsi.iloc[-lookback:] <= threshold).any())


def check_rsi_stoch(df, lookback=20, max_gap=5):
    """Return True if RSI and Stochastic both crossed up with max 5 candles between them."""
    if len(df) < WARMUP_RSI + lookback:
        return False

    rsi = calc_rsi_tv(df["close"], period=14)
    rsi_sig = rsi.rolling(14).mean()
    k, _ = calc_stoch_tv(df["close"], df["high"], df["low"])

    stoch_crosses = []
    rsi_crosses = []

    for i in range(-lookback, 0):
        try:
            if float(k.iloc[i - 1]) < 20 <= float(k.iloc[i]):
                stoch_crosses.append(i)
            if float(rsi.iloc[i - 1]) < float(rsi_sig.iloc[i - 1]) and \
               float(rsi.iloc[i]) >= float(rsi_sig.iloc[i]):
                rsi_crosses.append(i)
        except (ValueError, IndexError):
            continue

    for sc in stoch_crosses:
        for rc in rsi_crosses:
            if 1 <= abs(sc - rc) <= max_gap:
                return True

    return False

        
# ------------------------------------------
# CASCADE PIPELINE - محسّن مع thread-safe resample
# ------------------------------------------

def run_cascade_scan():
    """
    Run the cascade pipeline مرة واحدة فقط لكل scan:
    - يحسب الـ DataFrames مرة واحدة فقط ويخزنها
    - جميع الحسابات تتم قبل التشغيل المتوازي (thread-safe بالكامل)
    - يصفّر الإحصاء والنتائج في بداية كل دورة
    - resample_cache آمن تماماً بدون race conditions
    """
    with symbols_cache_lock:
        symbols = list(symbols_cache)

    if not symbols:
        return

    # ── تصفير الإحصاء والنتائج في بداية كل دورة ──
    with cascade_stats_lock, cascade_results_lock:
        for i in range(1, 8):
            cascade_stats[i]["total"] = 0
            cascade_stats[i]["passed"] = 0
            cascade_results[i].clear()  # ← تنظيف البيانات القديمة

    # ── Cache للـ resample (يتم حسابها مرة واحدة فقط) ──
    resample_cache = {}  # {(sym, tf, minutes): DataFrame}

    def get_resampled(raw_df, sym, tf, minutes):
        """احصل على DataFrame المعاد عينته، مع التخزين المؤقت"""
        key = (sym, tf, minutes)
        if key not in resample_cache:
            resample_cache[key] = resample_ohlcv(raw_df, minutes)
        return resample_cache[key]

    # ── بناء الـ candidates مع جميع DataFrames محسوبة مسبقاً ──
    # هذا يتم في single thread، بدون أي race conditions
    candidates = []
    for sym in symbols:
        raw_ec_1m = get_cached(sym, "1m")
        raw_ec_60m = get_cached(sym, "60m")

        for entry_min, confirm_min, third_min, ec_api, t_api in TRIPLING_PAIRS:
            raw_ec = raw_ec_1m if ec_api == "1m" else raw_ec_60m
            raw_t = raw_ec_1m if t_api == "1m" else raw_ec_60m

            if raw_ec.empty or raw_t.empty:
                continue

            df_entry = get_resampled(raw_ec, sym, ec_api, entry_min)
            df_confirm = get_resampled(raw_ec, sym, ec_api, confirm_min)
            df_third = get_resampled(raw_t, sym, t_api, third_min)

            if df_entry.empty or df_confirm.empty or df_third.empty:
                continue
            if len(df_entry) < MIN_CANDLES:
                continue

            # احسب next_tf مسبقاً وضعه في الـ candidate
            next_tf = NEXT_TF.get(entry_min)
            df_next_tf = get_resampled(raw_ec, sym, ec_api, next_tf) if next_tf else None

            candidates.append({
                "sym": sym,
                "ec_api": ec_api,
                "t_api": t_api,
                "entry_min": entry_min,
                "confirm_min": confirm_min,
                "third_min": third_min,
                "df_entry": df_entry,
                "df_confirm": df_confirm,
                "df_third": df_third,
                "df_next_tf": df_next_tf,  # ← جاهز مسبقاً، لا توجد كتابة في threads
                "raw_ec": raw_ec,
            })

    log.info("🔄 Cascade Scan: %d مرشح (resample cache: %d)", len(candidates), len(resample_cache))

    # ── تعريف فحوصات كل خطوة (آمنة تماماً، بدون كتابة) ──
    def step1(c):
        if not check_smi_oversold(c["df_entry"]):
            return False, "smi_oversold"
        df_next = c["df_next_tf"]
        if df_next is not None and not df_next.empty and check_smi_oversold(df_next):
            return False, "active_skip"
        return True, "passed"

    def step2(c):
        if not check_macd_red(c["df_entry"]):
            return False, "macd_red"
        return True, "passed"

    def step3(c):
        if not check_donchian_trend_ribbon(c["df_entry"], "green"):
            return False, "donchian_entry"
        return True, "passed"

    def step4(c):
        if not check_donchian_trend_ribbon(c["df_confirm"], "green"):
            return False, "donchian_confirm"
        return True, "passed"

    def step5(c):
        if not check_macd_green(c["df_confirm"]):
            return False, "macd_confirm"
        return True, "passed"

    def step6(c):
        if not check_ema50_below(c["df_entry"]):
            return False, "ema50"
        return True, "passed"

    def step7(c):
        if not check_rsi_touched_oversold(c["df_entry"]):
            return False, "rsi_stoch"
        if not check_rsi_stoch(c["df_third"]):
            return False, "rsi_stoch"
        return True, "passed"

    steps = [step1, step2, step3, step4, step5, step6, step7]

    # ── تشغيل الخطوات ──
    for step_num, step_fn in enumerate(steps, start=1):
        if not candidates:
            break

        def run_one(c, fn=step_fn):
            """Closure آمن: fn مثبتة بـ default argument"""
            return c, *fn(c)

        with ThreadPoolExecutor(max_workers=50) as executor:
            results = list(executor.map(run_one, candidates))

        passed = []
        now = datetime.now(timezone.utc)

        with cascade_results_lock, cascade_stats_lock:
            cascade_stats[step_num]["total"] = len(results)
            for c, ok, reason in results:
                key = (c["sym"], c["entry_min"], c["confirm_min"], c["third_min"])
                cascade_results[step_num][key] = {
                    "passed": ok, "reason": reason, "time": now
                }
                if ok:
                    cascade_stats[step_num]["passed"] += 1
                    passed.append(c)

        log.info("📍 خطوة %d: %d/%d نجحوا", step_num, len(passed), len(results))
        candidates = passed
    # ── حفظ نسخة مكتملة ──
    with last_complete_lock, cascade_stats_lock, cascade_results_lock:
        for i in range(1, 8):
            last_complete_stats[i] = dict(cascade_stats[i])
            last_complete_results[i] = dict(cascade_results[i])
    # ── إرسال الإشارات النهائية ──
    log.info("🎉 الإشارات النهائية: %d", len(candidates))
    for c in candidates:
        _fire_signal(
            c["sym"], c["entry_min"], c["confirm_min"],
            c["third_min"], c["df_entry"]
        )


def _fire_signal(symbol, entry_min, confirm_min, third_min, df_entry):
    """Send the Telegram alert and record the signal."""
    key = (symbol, entry_min, confirm_min, third_min)
    now = datetime.now(timezone.utc)
    with alerted_keys_lock:
        last_alert = alerted_keys.get(key)
        if last_alert and now - last_alert < timedelta(hours=ALERT_EXPIRY_HOURS):
            return
        alerted_keys[key] = now

    try:
        price = df_entry["close"].iloc[-1]
        candle_close = df_entry["ts"].iloc[-1] + pd.Timedelta(minutes=entry_min)
        entry_time = candle_close.strftime("%Y-%m-%d %H:%M UTC")
        save_signal(symbol, price, entry_min, confirm_min, third_min)
        send_telegram(
            f"🚨 <b>إشارة دخول:</b> {symbol}\n"
            f"🕐 الفريم: {entry_min}m / {confirm_min}m / {third_min}m\n"
            f"💰 سعر الدخول: <b>{price:.6g}</b>\n"
            f"🕐 وقت الدخول: <b>{entry_time}</b>"
        )
    except Exception as exc:
        log.error("❌ خطأ في إرسال الإشارة %s: %s", symbol, exc)


def cascade_watcher():
    """Background thread: run cascade scan every 30 seconds with performance monitoring."""
    while True:
        time.sleep(30)
        if not fast_prefetch_done.is_set():
            continue
        try:
            start = time.time()
            run_cascade_scan()
            elapsed = time.time() - start
            log.info("⏱ Cascade scan اكتمل في %.1f ثانية", elapsed)
            
            # تأكد من وجود فاصل 30 ثانية فعلي بين الـ scans
            remaining = 30 - elapsed
            if remaining > 0:
                time.sleep(remaining)
            else:
                log.warning("⚠️ Cascade scan يأخذ أكثر من 30 ثانية (%.1f ث) — قد تحتاج لزيادة الفاصل", elapsed)
        except Exception as exc:
            log.error("Cascade scan error: %s", exc)
            time.sleep(10)


# ------------------------------------------
# Telegram Commands
# ------------------------------------------

def _cmd_status(chat_id):
    """Send bot status message."""
    with trades_lock:
        cnt = len(trades_history)
    with alerted_keys_lock:
        active = len(alerted_keys)
    with ohlcv_cache_lock:
        keys = len(ohlcv_cache)
    send_telegram(
        f"🤖 البوت يعمل — Binance API\n"
        f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"📊 إجمالي الإشارات: {cnt}\n"
        f"🔑 تنبيهات نشطة: {active}\n"
        f"💾 الكاش: {keys} مفتاح\n"
        f"⚡ تحميل سريع: {'✅' if fast_prefetch_done.is_set() else '⏳'}\n"
        f"📦 تحميل كامل: {'✅' if prefetch_done.is_set() else '⏳'}",
        chat_id,
    )


def _cmd_cascade_diag(chat_id):
    """Show detailed cascade diagnostic report."""
    with last_complete_lock:
        lines = [
            "🔍 <b>تقرير Cascade Pipeline — الـ 2600 فريم</b>",
            "━━━━━━━━━━━━━━━━━━━━━━",
        ]

        prev_total = 0
        for step_num in range(1, 8):
            step_name = STEP_NAMES[step_num - 1]
            step_label = STEP_LABELS[step_name]
            stats = last_complete_stats[step_num]
            total_t = stats["total"]
            total_p = stats["passed"]
            fail_count = total_t - total_p
            pct = int(total_p / total_t * 100) if total_t else 0
            bar = "█" * (pct // 10) + "░" * (10 - pct // 10)

            lines.append(
                f"\n{step_label}\n"
                f"{bar}\n"
                f"✅ نجح: <b>{total_p}</b> | ❌ فشل: <b>{fail_count}</b> "
                f"| دخل: <b>{total_t}</b> ({pct}%)"
            )
            prev_total = total_p  # الناجحون يصبحون input الخطوة التالية


        # آخر وقت فحص
        all_times = [
            v["time"]
            for step_data in cascade_results.values()
            for v in step_data.values()
            if "time" in v
        ]
        if all_times:
            lines.append(f"\n🕐 آخر فحص: {max(all_times).strftime('%H:%M UTC')}")

        msg = "\n".join(lines)

    if len(msg) > 4000:
        for i in range(0, len(msg), 4000):
            send_telegram(msg[i:i + 4000], chat_id)
    else:
        send_telegram(msg, chat_id)


def _dispatch_command(txt, chat_id):
    """Route a Telegram command to its handler."""
    if txt == "/status":
        _cmd_status(chat_id)
    elif txt in ("1", "/today"):
        send_telegram(get_report("today"), chat_id)
    elif txt in ("2", "/yesterday"):
        send_telegram(get_report("yesterday"), chat_id)
    elif txt in ("3", "/week"):
        send_telegram(get_report("week"), chat_id)
    elif txt in ("/سبب", "/diag"):
        _cmd_cascade_diag(chat_id)
    elif txt.startswith("/check5"):
        parts = txt.split()
        symbol = parts[1].upper() if len(parts) > 1 else "BTCUSDT"
        if not symbol.endswith("USDT"):
            symbol += "USDT"
        threading.Thread(
            target=handle_check5, args=(chat_id, symbol), daemon=True
        ).start()
    elif txt == "/help":
        send_telegram(
            "📋 <b>الأوامر المتاحة:</b>\n"
            "1️⃣ <code>1</code> — إشارات اليوم\n"
            "2️⃣ <code>2</code> — إشارات أمس\n"
            "3️⃣ <code>3</code> — آخر 7 أيام\n"
            "🔍 <code>/سبب</code> — تقرير Cascade (جميع الخطوات + الأسباب)\n"
            "📊 <code>/status</code> — حالة البوت\n"
            "📋 <code>/help</code> — قائمة الأوامر",
            chat_id,
        )


def poll_telegram_commands():
    """Long-poll Telegram for commands and dispatch them."""
    last_id = 0
    while True:
        try:
            r = get_session().get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
                params={"offset": last_id + 1, "timeout": 30}, timeout=35,
            ).json()
            for upd in r.get("result", []):
                last_id = upd["update_id"]
                txt = upd.get("message", {}).get("text", "").strip()
                chat_id = str(upd.get("message", {}).get("chat", {}).get("id", ""))
                if txt and chat_id:
                    _dispatch_command(txt, chat_id)
        except Exception:
            time.sleep(10)

def handle_check5(chat_id, symbol="BTCUSDT"):
    send_telegram(f"🔄 جاري جلب بيانات {symbol} — فريم 5 دقايق...", chat_id)
    try:
        df_fresh = get_ohlcv(symbol, "1m", limit=1000)
        if not df_fresh.empty:
            cache_merge(symbol, "1m", df_fresh)

        df_raw = get_cached(symbol, "1m")
        if df_raw.empty:
            send_telegram("❌ فشل جلب البيانات من Binance", chat_id)
            return

        df5 = resample_ohlcv_closed(df_raw, 5)

        if df5.empty or len(df5) < MIN_CANDLES:
            send_telegram(
                f"⚠️ شموع غير كافية: {len(df5)} (المطلوب {MIN_CANDLES})\n"
                f"💡 جرب بعد اكتمال التحميل الكامل", chat_id
            )
            return

        now             = datetime.now(timezone.utc)
        last_candle_end = df5["ts"].iloc[-1] + timedelta(minutes=5)

        if now < last_candle_end:
            df5 = df5.iloc[:-1]

        if df5.empty or len(df5) < MIN_CANDLES:
            send_telegram("⚠️ شموع غير كافية بعد الفلترة", chat_id)
            return

        price     = df5["close"].iloc[-1]
        candle_ts = df5["ts"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")
        fetch_ts  = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        rsi_series = calc_rsi_tv(df5["close"], period=14)
        rsi_val    = round(float(rsi_series.iloc[-1]), 2)

        k_series, d_series = calc_stoch_tv(df5["close"], df5["high"], df5["low"])
        stoch_k = round(float(k_series.iloc[-1]), 2)
        stoch_d = round(float(d_series.iloc[-1]), 2)

        macd_line, signal_line, histogram = _calc_macd_full(df5["close"])
        macd_hist_val   = round(float(histogram.iloc[-1]),   4)
        macd_line_val   = round(float(macd_line.iloc[-1]),   4)
        signal_line_val = round(float(signal_line.iloc[-1]), 4)
        macd_color      = "🟢" if macd_hist_val > 0 else "🔴"

        smi_series, smi_sig_series = calc_smi(df5["high"], df5["low"], df5["close"])
        smi_val = round(float(smi_series.iloc[-1]),     2)
        smi_sig = round(float(smi_sig_series.iloc[-1]), 2)

        don_trend = calc_donchian_trend(df5)
        if don_trend:
            don_val = don_trend[-1]
            if don_val == 1:
                don_color = "🟢 أخضر (صاعد)"
            elif don_val == -1:
                don_color = "🔴 أحمر (هابط)"
            else:
                don_color = "⚪ محايد"
        else:
            don_color = "⚪ محايد"

        rsi_zone   = ("🔴 تشبع بيعي" if rsi_val < 30
                      else ("🟠 تشبع شرائي" if rsi_val > 70 else "🟡 محايد"))
        stoch_zone = ("🔴 تشبع بيعي" if stoch_k < 20
                      else ("🟠 تشبع شرائي" if stoch_k > 80 else "🟡 محايد"))
        smi_zone   = ("🔴 تشبع بيعي" if smi_val <= -40
                      else ("🟠 تشبع شرائي" if smi_val >= 40 else "🟡 محايد"))

        send_telegram(
            f"📊 <b>{symbol} — فريم 5 دقايق</b>\n"
            f"🕯️ الشمعة المغلقة: <b>{candle_ts}</b>\n"
            f"🕐 وقت الجلب: {fetch_ts}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💰 السعر: <b>{price:.2f}$</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"🎀 Donchian Ribbon (20): {don_color}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📈 RSI (14): <b>{rsi_val}</b> {rsi_zone}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📉 Stoch K(15,3): <b>{stoch_k}</b> {stoch_zone}\n"
            f"  Stoch D(3): <b>{stoch_d}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"⚡ MACD Histogram: {macd_color} <b>{macd_hist_val}</b>\n"
            f"  MACD Line: <b>{macd_line_val}</b>\n"
            f"  Signal Line: <b>{signal_line_val}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"🔵 SMI: <b>{smi_val}</b> {smi_zone}\n"
            f"  Signal: <b>{smi_sig}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📦 شموع الـ5m: {len(df5)} | بيانات الـ1m: {len(df_raw)}",
            chat_id,
        )
    except Exception as e:
        log.error(f"check5 error: {e}")
        send_telegram(f"❌ خطأ في /check5: {e}", chat_id)
        
# ------------------------------------------
# Symbols Loop
# ------------------------------------------

def update_symbols_loop():
    """Periodically refresh the top symbols list from Binance."""
    while True:
        try:
            resp = get_session().get(f"{BINANCE_BASE}/api/v3/ticker/24hr").json()
            if isinstance(resp, list):
                tickers = resp
            elif isinstance(resp, dict):
                tickers = resp.get("data", [])
            else:
                tickers = []

            top = sorted(
                [
                    t for t in tickers
                    if isinstance(t, dict) and t.get("symbol", "").endswith("USDT")
                ],
                key=lambda x: float(x.get("quoteVolume", 0)),
                reverse=True
            )[:TOP_SYMBOLS_LIMIT]

            with symbols_cache_lock:
                symbols_cache[:] = [t["symbol"] for t in top]
            log.info("✅ عملات: %s — أول 5: %s", len(symbols_cache), symbols_cache[:5])
            if not fast_prefetch_done.is_set():
                threading.Thread(
                    target=prefetch_all, args=(list(symbols_cache),), daemon=True
                ).start()
        except requests.RequestException as exc:
            log.error("update_symbols_loop: %s", exc)
        time.sleep(3600)


# ------------------------------------------
# Health Server
# ------------------------------------------

class HealthHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler that responds OK to health checks."""

    def do_GET(self):
        """Handle GET requests with a 200 OK response."""
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *_):
        """Suppress default request logging."""


# ------------------------------------------
# Main
# ------------------------------------------

def main():
    """Start all background threads and run the main heartbeat loop."""
    def handle_exception(exc_type, exc_value, exc_tb):
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        log.error("💥 خطأ غير متوقع أوقف البوت:\n%s", msg)
        try:
            send_telegram(f"💥 <b>البوت توقف بسبب خطأ:</b>\n<code>{exc_value}</code>")
        except Exception:
            pass

    sys.excepthook = handle_exception

    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("✅ Health server شغّال على port %s", PORT)

    delete_webhook()

    threading.Thread(target=update_symbols_loop, daemon=True).start()
    threading.Thread(target=poll_telegram_commands, daemon=True).start()
    threading.Thread(target=cache_updater_1m, daemon=True).start()
    threading.Thread(target=cache_updater_60m, daemon=True).start()
    threading.Thread(target=cascade_watcher, daemon=True).start()

    while True:
        try:
            time.sleep(300)
            cleanup_alerted_keys()
            with ohlcv_cache_lock:
                cache_size = len(ohlcv_cache)
            with trades_lock:
                signals_count = len(trades_history)
            log.info(
                "💓 البوت يعمل | كاش: %s مفتاح | إشارات: %s | سريع: %s | كامل: %s",
                cache_size,
                signals_count,
                "✅" if fast_prefetch_done.is_set() else "⏳",
                "✅" if prefetch_done.is_set() else "⏳",
            )
        except Exception as exc:
            log.error("❌ خطأ في main loop: %s\n%s", exc, traceback.format_exc())
            time.sleep(10)


if __name__ == "__main__":
    main()