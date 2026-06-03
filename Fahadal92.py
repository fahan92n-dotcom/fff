import os
import requests
import pandas as pd
import numpy as np
import time
import logging
import threading
from collections import deque
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(name)

#------------------------------------------
#Main Settings
#------------------------------------------

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN",   "8988740597:AAE_I7M7zB5VM2NykUwreQQMws0vk7qlU78")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "7801703329")

BINANCE_BASE      = "https://data-api.binance.vision"
TOP_SYMBOLS_LIMIT = 200
PORT              = int(os.environ.get("PORT", "8080"))
ALERT_EXPIRY_HOURS = 4
NEAR6_EXPIRY_HOURS = 2

TF_MAP = {"1m": "1m", "60m": "1h"}

TRIPLING_PAIRS = [
(9,   27,  3,  "1m",  "1m"),
(12,  36,  4,  "1m",  "1m"),
(15,  45,  5,  "1m",  "1m"),
(18,  54,  6,  "1m",  "1m"),
(21,  63,  7,  "1m",  "1m"),
(24,  72,  8,  "1m",  "1m"),
(27,  81,  9,  "1m",  "1m"),
(30,  90,  10, "1m",  "1m"),
(45,  135, 15, "1m",  "1m"),
(60,  180, 20, "60m", "1m"),
(90,  270, 30, "60m", "1m"),
(120, 360, 40, "60m", "1m"),
(180, 540, 60, "60m", "60m"),
]

TIMEFRAME_CHAIN = [9, 12, 15, 18, 21, 24, 27, 30, 45, 60, 90, 120, 180]
NEXT_TF = {TIMEFRAME_CHAIN[i]: TIMEFRAME_CHAIN[i+1] for i in range(len(TIMEFRAME_CHAIN)-1)}

FAST_FETCH_CANDLES = {"1m": 3500,  "60m": 250}
API_FETCH_CANDLES  = {"1m": 15_000, "60m": 2_000}
CACHE_MAX_CANDLES  = {"1m": 16_000, "60m": 2_500}
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

WARMUP_EMA   = 200
WARMUP_MACD  = 200
WARMUP_SMI   = 100
WARMUP_RSI   = 200
WARMUP_STOCH = 100
WARMUP_DON   = 50
MIN_CANDLES  = 250

#------------------------------------------
#Shared State
#------------------------------------------

alerted_keys        = {}
alerted_keys_lock   = threading.Lock()
trades_history      = deque(maxlen=2000)
trades_lock         = threading.Lock()
near_signals        = {}
near_signals_lock   = threading.Lock()
near_signals_6      = {}
near_signals_6_lock = threading.Lock()
symbols_cache       = []
symbols_cache_lock  = threading.Lock()
ohlcv_cache         = {}
ohlcv_cache_lock    = threading.Lock()

fast_prefetch_done  = threading.Event()
prefetch_done       = threading.Event()

diag_counts = {
"total": 0, "no_data": 0, "smi_oversold": 0, "active_skip": 0,
"macd_red": 0, "donchian_entry": 0, "donchian_confirm": 0,
"macd_confirm": 0, "ema50": 0, "rsi_stoch": 0, "passed": 0,
}
diag_lock         = threading.Lock()
cache_diag_logged = threading.Event()
_local            = threading.local()

#------------------------------------------

#Last Check

#------------------------------------------

last_diag = {"symbol": None, "step": None, "entry_min": None, "time": None}
last_diag_lock = threading.Lock()

#------------------------------------------
#Diagnostics
#------------------------------------------

DIAG_LABELS = {
"no_data"          : "بيانات ناقصة",
"smi_oversold"     : "SMI مش في التشبع البيعي",
"active_skip"      : "SMI الفريم الأكبر لم يتأكد",
"macd_red"         : "MACD الرئيسي مش أحمر",
"donchian_entry"   : "Donchian Ribbon الرئيسي مش أخضر",
"donchian_confirm" : "Donchian Ribbon Confirm مش أخضر",
"macd_confirm"     : "MACD Confirm مش أخضر (×3)",
"ema50"            : "السعر فوق EMA50",
"rsi_stoch"        : "RSI/Stochastic ما اتحقق",
}

STEP_LABELS = {
"no_data"          : "بيانات كافية ✅",
"smi_oversold"     : "① تشبع بيعي SMI ✅",
"active_skip"      : "⭐ الفريم الأكبر ليس في تشبع بيعي ✅",
"macd_red"         : "③ MACD أحمر ✅",
"donchian_entry"   : "④ Donchian Ribbon أخضر ✅",
"donchian_confirm" : "⑤ Donchian Confirm أخضر ✅",
"macd_confirm"     : "⑥ MACD Confirm أخضر (×3) ✅",
"ema50"            : "⑦ السعر تحت EMA50 ✅",
"rsi_stoch"        : "⑧ RSI تقاطع + Stochastic ✅",
}

def build_diag_msg(reset=False):
    with diag_lock:
        t = diag_counts["total"] or 1
        non_total = {k: v for k, v in diag_counts.items() if k not in ["total", "passed"]}
        worst_k   = max(non_total, key=lambda k: non_total[k])
        worst_v   = non_total[worst_k]
        lines = [
            "🔍 <b>تقرير التشخيص</b>", "━━━━━━━━━━━━━━━",
            f"📊 إجمالي الفحوصات: <b>{t}</b>", "",
        ]
        remaining = t
        for k, pass_label in STEP_LABELS.items():
            failed   = diag_counts[k]
            passed   = remaining - failed
            pass_pct = int(passed / t * 100)
            fail_pct = int(failed / t * 100)
            bar      = "█" * (pass_pct // 10) + "░" * (10 - pass_pct // 10)
            lines.append(
                f"{pass_label}\n"
                f"  {bar} نجح: {passed} ({pass_pct}%) | فشل: {failed} ({fail_pct}%)"
            )
            remaining = passed
        lines += [
            "", f"🏆 اجتازت الكل: <b>{diag_counts['passed']}</b>",
            "━━━━━━━━━━━━━━━",
            f"⚠️ أكثر سبب فشل: <b>{DIAG_LABELS.get(worst_k, worst_k)}</b> ({worst_v})",
        ]
        if reset:
            for k in diag_counts:
                diag_counts[k] = 0
        return "\n".join(lines)

def send_diag_report():
    while True:
        time.sleep(3600)
        send_telegram(build_diag_msg(reset=True))

#------------------------------------------
#Helpers
#------------------------------------------

def get_session():
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.headers.update({"Accept-Encoding": "gzip", "User-Agent": "Mozilla/5.0"})
        _local.s = s
    return _local.s

def delete_webhook():
    try:
        r = get_session().post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook",
            json={"drop_pending_updates": True}, timeout=10,
        ).json()
        if r.get("ok"):
            log.info("✅ تم حذف الـ Webhook")
    except Exception as e:
        log.error(f"deleteWebhook error: {e}")

def cleanup_alerted_keys():
    now = datetime.now(timezone.utc)
    with alerted_keys_lock:
        expired = [k for k, t in list(alerted_keys.items()) if now - t > timedelta(hours=ALERT_EXPIRY_HOURS)]
        for k in expired:
            del alerted_keys[k]

def cleanup_near6():
    now = datetime.now(timezone.utc)
    with near_signals_6_lock:
        expired = [k for k, v in list(near_signals_6.items()) if now - v["time"] > timedelta(hours=NEAR6_EXPIRY_HOURS)]
        for k in expired:
            del near_signals_6[k]

def save_signal(symbol, price, entry_min, confirm_min, third_min):
    with trades_lock:
        trades_history.append({
            "time"     : datetime.now(timezone.utc),
            "symbol"   : symbol,
            "price"    : price,
            "timeframe": f"{entry_min}m/{confirm_min}m/{third_min}m",
        })

def send_telegram(msg, chat_id=None):
    target = chat_id or TELEGRAM_CHAT_ID
    try:
        r = get_session().post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": target, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        ).json()
        return r.get("ok", False)
    except Exception as e:
        log.error(f"Telegram send error: {e}")
        return False

def get_report(period="today"):
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

#------------------------------------------
#Binance OHLCV
#------------------------------------------

def _parse_binance_klines(resp):
    df = pd.DataFrame(resp, columns=[
        "ts", "open", "high", "low", "close", "vol",
        "close_time", "quote_vol", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)[["ts", "open", "high", "low", "close", "vol"]]

def get_ohlcv(symbol, tf, limit=500):
    binance_tf = TF_MAP.get(tf, "1m")
    try:
        resp = get_session().get(
            f"{BINANCE_BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": binance_tf, "limit": min(limit, 1000)},
            timeout=10,
        ).json()
        if isinstance(resp, list) and resp:
            return _parse_binance_klines(resp)
    except Exception as e:
        log.error(f"get_ohlcv {symbol} {tf}: {e}")
    return pd.DataFrame()

def get_ohlcv_full(symbol, tf, target):
    binance_tf = TF_MAP.get(tf, "1m")
    tf_ms      = 60_000 if tf == "1m" else 3_600_000
    BIN_MAX    = 1000
    all_dfs, end_ms, fetched, retries = [], int(time.time() * 1000), 0, 0
    
    while fetched < target:
        batch    = min(BIN_MAX, target - fetched)
        start_ms = end_ms - batch * tf_ms
        
        try:
            resp = get_session().get(
                f"{BINANCE_BASE}/api/v3/klines",
                params={"symbol": symbol, "interval": binance_tf,
                        "startTime": start_ms, "endTime": end_ms, "limit": batch},
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
            retries  = 0
            end_ms   = start_ms - 1
            
            if len(df) < batch:
                break
                
        except Exception as e:
            retries += 1
            if retries >= 3:
                break
            time.sleep(2)
            
    return (pd.concat(all_dfs).drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
            if all_dfs else pd.DataFrame())

def cache_merge(symbol, tf, new_df):
    if new_df.empty:
        return
    key  = (symbol, tf)
    maxc = CACHE_MAX_CANDLES.get(tf, 5000)
    with ohlcv_cache_lock:
        old = ohlcv_cache.get(key)
        if old is not None and not old.empty:
            merged = pd.concat([old, new_df]).drop_duplicates(subset="ts").sort_values("ts")
            ohlcv_cache[key] = merged.tail(maxc).reset_index(drop=True)
        else:
            ohlcv_cache[key] = new_df.tail(maxc).reset_index(drop=True)


def get_cached(symbol, tf):
    with ohlcv_cache_lock:
        df = ohlcv_cache.get((symbol, tf))
        return df.copy() if df is not None else pd.DataFrame()


def prefetch_all(symbols):
    def fetch_sym_fast(sym):
        for tf, n in FAST_FETCH_CANDLES.items():
            df = get_ohlcv_full(sym, tf, target=n)
            cache_merge(sym, tf, df)

    def fetch_sym_full(sym):
        for tf, n in API_FETCH_CANDLES.items():
            df = get_ohlcv_full(sym, tf, target=n)
            cache_merge(sym, tf, df)

    log.info("🚀 بدء التحميل السريع بالـ threads...")
    with ThreadPoolExecutor(max_workers=15) as ex:
        ex.map(fetch_sym_fast, symbols)
    fast_prefetch_done.set()
    send_telegram("⚡ <b>التحميل السريع اكتمل — البوت يعمل الآن!</b>")

    log.info("📦 بدء التحميل الكامل...")
    with ThreadPoolExecutor(max_workers=10) as ex:
        ex.map(fetch_sym_full, symbols)
    prefetch_done.set()
    send_telegram("✅ <b>التحميل الكامل اكتمل وجاهز للعمل!</b>")


def _update_batch(symbols, tf, limit):
    def fetch_one(sym):
        df = get_ohlcv(sym, tf, limit=limit)
        if not df.empty:
            cache_merge(sym, tf, df)
    with ThreadPoolExecutor(max_workers=30) as ex:
        ex.map(fetch_one, symbols)


def cache_updater_1m():
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
    while True:
        time.sleep(3600)
        if fast_prefetch_done.is_set():
            with symbols_cache_lock:
                syms = list(symbols_cache)
            if syms:
                _update_batch(syms, "60m", limit=5)
                
#------------------------------------------

#Technical Indicators

#------------------------------------------

def resample_ohlcv(df, minutes):
    if df.empty:
        return pd.DataFrame()
    return (df.copy().set_index("ts")
            .resample(f"{minutes}min", closed="left", label="left", origin=EPOCH)
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "vol": "sum"})
            .dropna().iloc[:-1].reset_index())


def resample_ohlcv_closed(df, minutes):
    if df.empty:
        return pd.DataFrame()
    return (df.copy().set_index("ts")
            .resample(f"{minutes}min", closed="left", label="left", origin=EPOCH)
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "vol": "sum"})
            .dropna().reset_index())


def wilder_rma(series, period):
    return series.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()


def _calc_macd_hist(close):
    ml  = (close.ewm(span=12, min_periods=12, adjust=False).mean()
         - close.ewm(span=26, min_periods=26, adjust=False).mean())
    sig = ml.ewm(span=9, min_periods=9, adjust=False).mean()
    return ml - sig


def _calc_macd_full(close):
    macd_line   = (close.ewm(span=12, min_periods=12, adjust=False).mean()
                 - close.ewm(span=26, min_periods=26, adjust=False).mean())
    signal_line = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def check_macd_red(df):
    if len(df) < WARMUP_MACD:
        return False
    return bool(_calc_macd_hist(df["close"]).iloc[-1] < 0)


def check_macd_green(df):
    if len(df) < WARMUP_MACD:
        return False
    return bool(_calc_macd_hist(df["close"]).iloc[-1] > 0)

#------------------------------------------

#Donchian Ribbon

#------------------------------------------

def calc_donchian_trend(df, length=20):
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


def check_donchian_ribbon(df, direction="green"):
    if len(df) < 22:
        return False
    trend = calc_donchian_trend(df)
    if not trend:
        return False
    last_trend = trend[-1]
    if direction == "green":
        return last_trend == 1
    return last_trend == -1


def check_ema50_below(df):
    ema = df["close"].ewm(span=50, adjust=False).mean()
    return bool(df["close"].iloc[-1] < ema.iloc[-1])


def calc_smi(high, low, close, k=10, d=3, ema_len=10, smooth=1):
    hh      = high.rolling(k, min_periods=k).max()
    ll      = low.rolling(k,  min_periods=k).min()
    diff    = hh - ll
    rdiff   = close - (hh + ll) / 2
    avgrel  = rdiff.ewm(span=d, min_periods=d, adjust=False).mean()
    avgdiff = diff.ewm(span=d,  min_periods=d, adjust=False).mean()
    smi_arr = np.where(avgdiff != 0, (avgrel / (avgdiff / 2)) * 100, 0.0)
    smi     = pd.Series(smi_arr, index=close.index)
    if smooth > 1:
        smi = smi.rolling(smooth, min_periods=smooth).mean()
    sig = smi.ewm(span=ema_len, min_periods=ema_len, adjust=False).mean()
    return smi, sig


def check_smi_oversold(df, threshold=-40):
    if len(df) < WARMUP_SMI:
        return False
    smi, _ = calc_smi(df["high"], df["low"], df["close"])
    return bool(smi.iloc[-1] <= threshold)


def get_smi_value(df):
    if len(df) < WARMUP_SMI:
        return None, None
    smi, sig = calc_smi(df["high"], df["low"], df["close"])
    return round(float(smi.iloc[-1]), 2), round(float(sig.iloc[-1]), 2)


def calc_rsi_tv(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta.clip(upper=0))
    up    = wilder_rma(gain, period)
    down  = wilder_rma(loss, period)
    return 100.0 - (100.0 / (1.0 + up / (down + 1e-10)))


def calc_stoch_tv(close, high, low, k_len=15, k_smooth=3, d_smooth=3):
    lo  = low.rolling(k_len,    min_periods=k_len).min()
    hi  = high.rolling(k_len,   min_periods=k_len).max()
    raw = 100.0 * (close - lo) / (hi - lo + 1e-10)
    k   = raw.rolling(k_smooth, min_periods=k_smooth).mean()
    d   = k.rolling(d_smooth,   min_periods=d_smooth).mean()
    return k, d
    
#------------------------------------------
#check_rsi_stoch
#------------------------------------------

def check_rsi_stoch(df, lookback=5):
    if len(df) < WARMUP_RSI + lookback:
        return False
    rsi     = calc_rsi_tv(df["close"], period=14)
    rsi_sig = rsi.rolling(14).mean()
    k, _    = calc_stoch_tv(df["close"], df["high"], df["low"])
    for i in range(-lookback, 0):
        stoch_cross = (float(k.iloc[i - 1]) < 20) and (float(k.iloc[i]) >= 20)
        rsi_cross = (
            float(rsi.iloc[i - 1]) < float(rsi_sig.iloc[i - 1])
            and float(rsi.iloc[i]) >= float(rsi_sig.iloc[i])
        )
        if stoch_cross or rsi_cross:
            return True
    return False
    
#------------------------------------------
#handle_check5
#------------------------------------------

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
        age_minutes     = (now - df5["ts"].iloc[-1]).total_seconds() / 60  

        if age_minutes > 10:
            try:
                log.warning(...)
                send_telegram(...)
                df_fresh = get_ohlcv(symbol, "1m", limit=1000)
                if not df_fresh.empty:
                    cache_merge(symbol, "1m", df_fresh)
                df_raw = get_cached(symbol, "1m")
                df5    = resample_ohlcv_closed(df_raw, 5)
            except Exception as e:
                log.error(f"Error: {e}")

        if now < last_candle_end:
            df5 = df5.iloc[:-1]

        if df5.empty or len(df5) < MIN_CANDLES:
            send_telegram("⚠️ شموع غير كافية بعد الفلترة", chat_id)
            return

        price     = float(df_raw["close"].iloc[-1])
        candle_ts = df5["ts"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")
        fetch_ts  = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        rsi_series  = calc_rsi_tv(df5["close"], period=14)
        rsi_val     = round(float(rsi_series.iloc[-1]), 2)

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

    rsi_zone   = ("🔴 تشبع بيعي"  if rsi_val  < 30  
                  else ("🟠 تشبع شرائي" if rsi_val  > 70 else "🟡 محايد"))  
    stoch_zone = ("🔴 تشبع بيعي"  if stoch_k  < 20  
                  else ("🟠 تشبع شرائي" if stoch_k  > 80 else "🟡 محايد"))  
    smi_zone   = ("🔴 تشبع بيعي"  if smi_val  <= -40  
                  else ("🟠 تشبع شرائي" if smi_val  >= 40 else "🟡 محايد"))  

    send_telegram(
            f"📊 <b>{symbol} — فريم 5 دقايق</b>\n"
            f"🕯 الشمعة المغلقة: <b>{candle_ts}</b>\n"
            f"🕐 وقت الجلب: {fetch_ts}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💰 السعر الحالي: <b>{price:.2f}$</b>\n"
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
        send_telegram(f"خطا في check5: {e}", chat_id)

#------------------------------------------
#check5_watcher
#------------------------------------------

def get_next_close(tf_minutes):
    now   = datetime.now(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    elapsed_min = (now - epoch).total_seconds() / 60
    next_min    = (int(elapsed_min // tf_minutes) + 1) * tf_minutes
    return epoch + timedelta(minutes=next_min)


def check5_watcher():
    while True:
        try:
            nxt  = get_next_close(5)
            now  = datetime.now(timezone.utc)
            wait = (nxt - now).total_seconds()

                       if wait < -60:
               log.warning(f"⚠️ check5_watcher تأخر {abs(wait):.0f}ث — تخطي للشمعة التالية")
               next_nxt  = get_next_close(5)
               next_wait = (next_nxt - datetime.now(timezone.utc)).total_seconds()
               time.sleep(max(next_wait, 0) + 5)
               continue

           time.sleep(max(wait, 0) + 5)

           if not fast_prefetch_done.is_set():
               continue

           threading.Thread(
               target=handle_check5,
               args=(TELEGRAM_CHAT_ID, "BTCUSDT"),
               daemon=True,
           ).start()

    except Exception as e:  
        log.error(f"check5_watcher error: {e}")  
        time.sleep(10)

#------------------------------------------

# Scanning and Monitoring

#------------------------------------------

def scan_symbol(symbol, entry_min, confirm_min, third_min, ec_api, t_api):
    raw_ec = get_cached(symbol, ec_api)
    raw_t  = get_cached(symbol, t_api)

    with diag_lock:
        diag_counts["total"] += 1

    def save_last(step):
        with last_diag_lock:
            last_diag["symbol"]    = symbol
            last_diag["step"]      = step
            last_diag["entry_min"] = entry_min
            last_diag["time"]      = datetime.now(timezone.utc)

    if raw_ec.empty or raw_t.empty:
        with diag_lock:
            diag_counts["no_data"] += 1
        save_last("no_data")
        return

    df_entry   = resample_ohlcv(raw_ec, entry_min)
    df_confirm = resample_ohlcv(raw_ec, confirm_min)
    df_third   = resample_ohlcv(raw_t,  third_min)  

if df_entry.empty or df_confirm.empty or df_third.empty:  
    with diag_lock:  
        diag_counts["no_data"] += 1  
    save_last("no_data")  
    return  

if not check_smi_oversold(df_entry):  
    with diag_lock:  
        diag_counts["smi_oversold"] += 1  
    save_last("smi_oversold")  
    return  

next_tf = NEXT_TF.get(entry_min)  
if next_tf:  
    df_next = resample_ohlcv(raw_ec, next_tf)  
    if not df_next.empty and check_smi_oversold(df_next):  
        with diag_lock:  
            diag_counts["active_skip"] += 1  
        save_last("active_skip")  
        return  

    if not check_macd_red(df_entry):
       with diag_lock:
           diag_counts["macd_red"] += 1
       save_last("macd_red")
       return

   if not check_donchian_ribbon(df_entry, "green"):
       with diag_lock:
           diag_counts["donchian_entry"] += 1
       save_last("donchian_entry")
       return

if not check_donchian_ribbon(df_confirm, "green"):  
    with diag_lock:  
        diag_counts["donchian_confirm"] += 1  
    save_last("donchian_confirm")  
    return  

if not check_macd_green(df_confirm):  
    with diag_lock:  
        diag_counts["macd_confirm"] += 1  
    save_last("macd_confirm")  
    return  

if not check_ema50_below(df_entry):  
    with diag_lock:  
        diag_counts["ema50"] += 1  
    save_last("ema50")  
    return  

if not check_rsi_stoch(df_third):  
    with diag_lock:  
        diag_counts["rsi_stoch"] += 1  
    save_last("rsi_stoch")  
    return  

key = (symbol, entry_min, confirm_min, third_min)  
now = datetime.now(timezone.utc)  

with alerted_keys_lock:  
    last_alert = alerted_keys.get(key)  
    if last_alert and now - last_alert < timedelta(hours=ALERT_EXPIRY_HOURS):  
        return  
    alerted_keys[key] = now  

try:  
    with diag_lock:  
        diag_counts["passed"] += 1  
    price      = df_entry["close"].iloc[-1]  
    entry_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")  
    save_signal(symbol, price, entry_min, confirm_min, third_min)  
    send_telegram(  
        f"🚨 <b>إشارة دخول:</b> {symbol}\n"  
        f"⏱ الفريم: {entry_min}m / {confirm_min}m / {third_min}m\n"  
        f"💰 سعر الدخول: <b>{price:.6g}</b>\n"  
        f"🕐 وقت الدخول: <b>{entry_time}</b>"  
    )  
        except Exception as e:
            log.error(f"❌ خطأ في إرسال الإشارة {symbol}: {e}")


def candle_watcher(entry_min, confirm_min, third_min, ec_api, t_api):
    while True:
        time.sleep(30)
        if not fast_prefetch_done.is_set():
            continue
        with symbols_cache_lock:
            syms = list(symbols_cache)
        fn = partial(scan_symbol,
                     entry_min=entry_min, confirm_min=confirm_min,
                     third_min=third_min, ec_api=ec_api, t_api=t_api)
        with ThreadPoolExecutor(max_workers=20) as ex:
            list(ex.map(fn, syms))

#------------------------------------------

#Telegram Commands

#------------------------------------------

def poll_telegram_commands():
    last_id = 0
    while True:
        try:
            r = get_session().get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
                params={"offset": last_id + 1, "timeout": 30}, timeout=35,
            ).json()
            for upd in r.get("result", []):
                last_id = upd["update_id"]
                txt     = upd.get("message", {}).get("text", "").strip()
                chat_id = str(upd.get("message", {}).get("chat", {}).get("id", ""))
                if not txt or not chat_id:
                    continue

if txt == "/status":  
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
            elif txt in ("1", "/today"):  
                send_telegram(get_report("today"), chat_id)  
            elif txt in ("2", "/yesterday"):  
                send_telegram(get_report("yesterday"), chat_id)  
            elif txt in ("3", "/week"):  
                send_telegram(get_report("week"), chat_id)  
            elif txt in ("/سبب", "/diag"):  
                with last_diag_lock:  
                    ld = dict(last_diag)  
                if not ld["symbol"]:  
                    send_telegram("⚠️ لا توجد بيانات بعد.", chat_id)  
                else:  
                    step_ar = DIAG_LABELS.get(ld["step"], ld["step"])  
                    t_str   = ld["time"].strftime("%H:%M:%S UTC") if ld["time"] else ""  
                    send_telegram(  
                        f"🔍 <b>آخر فحص:</b>\n"  
                        f"━━━━━━━━━━━━━━━\n"  
                        f"🪙 الرمز: <b>{ld['symbol']}</b>\n"  
                        f"⏱ الفريم: <b>{ld['entry_min']}m</b>\n"  
                        f"❌ سبب الفشل: <b>{step_ar}</b>\n"  
                        f"🕐 الوقت: {t_str}",  
                        chat_id,  
                    )  
            elif txt.startswith("/check5"):  
                parts  = txt.split()  
                symbol = parts[1].upper() if len(parts) > 1 else "BTCUSDT"  
                if not symbol.endswith("USDT"):  
                    symbol += "USDT"  
                threading.Thread(  
                    target=handle_check5, args=(chat_id, symbol), daemon=True  
                ).start()  
            elif txt == "/help":  
                send_telegram(  
                    "📋 <b>الأوامر المتاحة:</b>\n"  
                    "📊 <code>/check5</code> — تقرير BTC فريم 5 دقايق\n"  
                    "📊 <code>/check5 ETH</code> — تقرير ETH فريم 5 دقايق\n"  
                    "1️⃣ <code>1</code> — إشارات اليوم\n"  
                    "2️⃣ <code>2</code> — إشارات أمس\n"  
                    "3️⃣ <code>3</code> — آخر 7 أيام\n"  
                    "🔍 <code>/سبب</code> — آخر رمز فُحص وسبب فشله\n"  
                    "📊 <code>/status</code> — حالة البوت\n"  
                    "📋 <code>/help</code> — قائمة الأوامر",  
                    chat_id,  
                )  
    except Exception:  
        time.sleep(10)

#------------------------------------------
#Symbols Loop
#------------------------------------------

def update_symbols_loop():
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
            [t for t in tickers if isinstance(t, dict) and t.get("symbol", "").endswith("USDT")],  
            key=lambda x: float(x.get("quoteVolume", 0)),  
            reverse=True  
        )[:TOP_SYMBOLS_LIMIT]  

        with symbols_cache_lock:  
            symbols_cache[:] = [t["symbol"] for t in top]  
        log.info(f"✅ عملات: {len(symbols_cache)} — أول 5: {symbols_cache[:5]}")  
        if not fast_prefetch_done.is_set():  
            threading.Thread(target=prefetch_all, args=(list(symbols_cache),), daemon=True).start()  
    except Exception as e:  
        log.error(f"update_symbols_loop: {e}")  
    time.sleep(3600)

#------------------------------------------
#Health Server
#------------------------------------------

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *_):
        pass

#------------------------------------------
#Main
#------------------------------------------

def main():
    import sys
    import traceback

    def handle_exception(exc_type, exc_value, exc_tb):
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        log.error(f"💥 خطأ غير متوقع أوقف البوت:\n{msg}")
        try:
            send_telegram(f"💥 <b>البوت توقف بسبب خطأ:</b>\n<code>{exc_value}</code>")
        except Exception:
            pass
    sys.excepthook = handle_exception

    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info(f"✅ Health server شغّال على port {PORT}")

    delete_webhook()

    threading.Thread(target=update_symbols_loop,    daemon=True).start()
    threading.Thread(target=poll_telegram_commands, daemon=True).start()
    threading.Thread(target=cache_updater_1m,        daemon=True).start()
    threading.Thread(target=cache_updater_60m,       daemon=True).start()
    threading.Thread(target=check5_watcher,          daemon=True).start()
    threading.Thread(target=send_diag_report,        daemon=True).start()

    for params in TRIPLING_PAIRS:
        threading.Thread(target=candle_watcher, args=params, daemon=True).start()

    while True:
        try:
            time.sleep(300)
            cleanup_alerted_keys()
            with ohlcv_cache_lock:
                cache_size = len(ohlcv_cache)
            with trades_lock:
                signals_count = len(trades_history)
            log.info(
                f"💓 البوت يعمل | "
                f"كاش: {cache_size} مفتاح | "
                f"إشارات: {signals_count} | "
                f"سريع: {'✅' if fast_prefetch_done.is_set() else '⏳'} | "
                f"كامل: {'✅' if prefetch_done.is_set() else '⏳'}"
            )
        except Exception as e:
            log.error(f"❌ خطأ في main loop: {e}\n{traceback.format_exc()}")
            time.sleep(10)


if __name__ == "__main__":
    main()