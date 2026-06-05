"""بوت مسح العملات من Binance مع تنبيهات Telegram.""" 
"""بوت مسح العملات من Binance مع Donchian Trend Ribbon من Telegram.""" 
import os
import time
import logging
import threading
import sys
import traceback
from collections import deque
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import HTTPServer, BaseHTTPRequestHandler

import requests
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------
# Main Settings
# ------------------------------------------

TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_TOKEN",  "8907286779:AAFTn1sfkpOnUgwlChN3RIV9xLqQ9EqAnzk")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "-1003972769219")

BINANCE_BASE     = "https://data-api.binance.vision"
TOP_SYMBOLS_LIMIT = 200
PORT             = int(os.environ.get("PORT", "8080"))
ALERT_EXPIRY_HOURS = 4

TF_MAP = {"1m": "1m", "5m": "5m", "60m": "1h"}

# ------------------------------------------
# Donchian Ribbon Settings
# ------------------------------------------
DONCHIAN_LENGTH = 20  # طول قناة Donchian الأساسية
RIBBON_LEVELS = 10    # عدد مستويات الـ Ribbon (0 إلى 9)

# ------------------------------------------
# جدول الاستراتيجيات
# ------------------------------------------
ENTRY_TO_STRATEGY = {
    3:   (9,   27),
    4:   (12,  36),
    5:   (15,  45),
    6:   (18,  54),
    7:   (21,  63),
    8:   (24,  72),
    9:   (27,  81),
    10:  (30,  90),
    15:  (45,  135),
    20:  (60,  180),
    30:  (90,  270),
    40:  (120, 360),
    60:  (180, 540),
    80:  (240, 720),
    120: (360, 1080),
    160: (480, 1440),
}

SUPPORTED_CHECK_TFS = sorted(ENTRY_TO_STRATEGY.keys())

TRIPLING_PAIRS = [
    (9,   27,  3,   "1m",  "1m"),
    (12,  36,  4,   "1m",  "1m"),
    (15,  45,  5,   "1m",  "1m"),
    (18,  54,  6,   "1m",  "1m"),
    (21,  63,  7,   "1m",  "1m"),
    (24,  72,  8,   "1m",  "1m"),
    (27,  81,  9,   "1m",  "1m"),
    (30,  90,  10,  "1m",  "1m"),
    (45,  135, 15,  "1m",  "1m"),
    (60,  180, 20,  "1m",  "1m"),
    (90,  270, 30,  "1m",  "1m"),
    (120, 360, 40,  "1m",  "1m"),
    (180, 540, 60,  "1m",  "1m"),
    (240, 720, 80,  "60m", "1m"),
    (360, 1080,120, "60m", "60m"),
    (480, 1440,160, "60m", "1m"),
    (720, 2160,240, "60m", "60m"),
]

TIMEFRAME_CHAIN = [9, 12, 15, 18, 21, 24, 27, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720]
NEXT_TF = {TIMEFRAME_CHAIN[i]: TIMEFRAME_CHAIN[i + 1] for i in range(len(TIMEFRAME_CHAIN) - 1)}

FAST_FETCH_CANDLES = {"1m": 3500,  "60m": 250}
API_FETCH_CANDLES  = {"1m": 15_000, "60m": 2_000}
CACHE_MAX_CANDLES  = {"1m": 16_000, "60m": 2_500}
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")

WARMUP_DON = 50
MIN_CANDLES = 250

# ------------------------------------------
# Shared State
# ------------------------------------------

alerted_keys        = {}
alerted_keys_lock   = threading.Lock()
trades_history      = deque(maxlen=2000)
trades_lock         = threading.Lock()
symbols_cache       = []
symbols_cache_lock  = threading.Lock()
ohlcv_cache         = {}
ohlcv_cache_lock    = threading.Lock()

fast_prefetch_done  = threading.Event()
prefetch_done       = threading.Event()
check_running       = threading.Lock()

diag_counts = {
    "total": 0, "no_data": 0, "ribbon_not_green": 0, "passed": 0,
}
diag_lock         = threading.Lock()
_local            = threading.local()

last_diag = {"symbol": None, "step": None, "entry_min": None, "time": None}
last_diag_lock = threading.Lock()

# ------------------------------------------
# Diagnostics
# ------------------------------------------

DIAG_LABELS = {
    "no_data"          : "بيانات ناقصة",
    "ribbon_not_green" : "Donchian Trend Ribbon ليس أخضر",
}

STEP_LABELS = {
    "no_data"          : "بيانات كافية ✅",
    "ribbon_not_green" : "🟢 Donchian Trend Ribbon أخضر ✅",
}


def build_diag_msg(reset=False):
    with diag_lock:
        total = diag_counts["total"] or 1
        lines = [
            "🔍 <b>تقرير التشخيص - Donchian Trend Ribbon</b>", "━━━━━━━━━━━━━━━",
            f"📊 إجمالي الفحوصات: <b>{total}</b>", "",
        ]
        
        no_data_count = diag_counts["no_data"]
        ribbon_count = diag_counts["ribbon_not_green"]
        passed_count = diag_counts["passed"]
        
        lines.append(f"✅ البيانات متاحة: <b>{total - no_data_count}</b> ({int((total - no_data_count) / total * 100)}%)")
        lines.append(f"🟢 Ribbon أخضر: <b>{passed_count}</b> ({int(passed_count / total * 100)}%)")
        lines.append(f"🔴 Ribbon أحمر/محايد: <b>{ribbon_count}</b> ({int(ribbon_count / total * 100)}%)")
        
        lines += ["", "━━━━━━━━━━━━━━━"]
        
        if reset:
            for k in diag_counts:
                diag_counts[k] = 0
        return "\n".join(lines)


def send_diag_report():
    while True:
        time.sleep(3600)
        send_telegram(build_diag_msg(reset=True))

# ------------------------------------------
# Helpers
# ------------------------------------------


def get_session():
    if not hasattr(_local, "s"):
        session = requests.Session()
        session.headers.update({"Accept-Encoding": "gzip", "User-Agent": "Mozilla/5.0"})
        _local.s = session
    return _local.s


def delete_webhook():
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
    now = datetime.now(timezone.utc)
    with alerted_keys_lock:
        expired = [
            k for k, t in list(alerted_keys.items())
            if now - t > timedelta(hours=ALERT_EXPIRY_HOURS)
        ]
        for k in expired:
            del alerted_keys[k]


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
    except requests.RequestException as exc:
        log.error("Telegram send error: %s", exc)
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

# ------------------------------------------
# Binance OHLCV
# ------------------------------------------


def _parse_binance_klines(resp):
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
    binance_tf = TF_MAP.get(tf, "1m")
    tf_ms = 60_000 if tf == "1m" else 3_600_000
    bin_max = 1000
    all_dfs, end_ms, fetched, retries = [], int(time.time() * 1000), 0, 0

    while fetched < target:
        batch    = min(bin_max, target - fetched)
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
            retries  = 0
            end_ms   = start_ms - 1
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
    def fetch_one(sym):
        df = get_ohlcv(sym, tf, limit=limit)
        if not df.empty:
            cache_merge(sym, tf, df)
    with ThreadPoolExecutor(max_workers=30) as executor:
        executor.map(fetch_one, symbols)


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

# ------------------------------------------
# Donchian Trend Ribbon Calculations
# ------------------------------------------


def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample OHLCV إلى فريم أكبر — يتزامن مع TradingView."""
    if df.empty:
        return pd.DataFrame()

    resampled = (
        df.copy()
        .set_index("ts")
        .resample(f"{minutes}min", closed="left", label="left", origin=EPOCH)
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "vol": "sum"})
        .dropna()
        .reset_index()
    )

    if resampled.empty:
        return resampled

    now_utc       = pd.Timestamp.now(tz="UTC")
    epoch         = pd.Timestamp("1970-01-01", tz="UTC")
    elapsed_min   = (now_utc - epoch).total_seconds() / 60
    current_open  = epoch + pd.Timedelta(minutes=int(elapsed_min // minutes) * minutes)

    return resampled[resampled["ts"] < current_open].reset_index(drop=True)


def calc_donchian_trend(df, length=DONCHIAN_LENGTH):
    """حساب اتجاه Donchian Ribbon — مطابق لكود Pine Script."""
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
            trend[i] = 1      # 🟢 أخضر (صاعد)
        elif df["close"].iloc[i] < ll.iloc[i]:
            trend[i] = -1     # 🔴 أحمر (هابط)
        else:
            trend[i] = trend[i - 1]
    
    return trend


def calc_donchian_ribbon_full(df, dlen=DONCHIAN_LENGTH):
    """
    حساب Donchian Trend Ribbon الكامل مع 10 مستويات.
    يرجع:
      - maintrend: الاتجاه الرئيسي (1 أخضر، -1 أحمر، 0 محايد)
      - ribbon_colors: قائمة بألوان كل مستوى
    """
    if len(df) < dlen + 2:
        return 0, []
    
    main_trend = calc_donchian_trend(df, dlen)
    if not main_trend:
        return 0, []
    
    maintrend = main_trend[-1]
    if maintrend == 0:
        return 0, []
    
    # احسب كل مستوى من الـ Ribbon
    ribbon_colors = []
    for offset in range(RIBBON_LEVELS):
        sub_len = dlen - offset
        if sub_len < 10:
            break
        sub_trend = calc_donchian_trend(df, sub_len)
        if not sub_trend:
            break
        
        current_trend = sub_trend[-1]
        
        # لون المستوى
        if current_trend == maintrend:
            if maintrend == 1:
                color = "🟢"  # أخضر
            else:
                color = "🔴"  # أحمر
        else:
            color = "⚪"      # محايد (غير متطابق)
        
        ribbon_colors.append({
            "level": offset,
            "length": sub_len,
            "trend": current_trend,
            "color": color,
        })
    
    return maintrend, ribbon_colors


def check_donchian_ribbon_green(df, dlen=DONCHIAN_LENGTH, min_agreement=0.7):
    """
    تحقق إذا كان Donchian Ribbon أخضر (صاعد).
    min_agreement: نسبة المستويات التي يجب أن تتفق مع الاتجاه الرئيسي.
    """
    if len(df) < dlen + 2:
        return False
    
    maintrend, ribbon_colors = calc_donchian_ribbon_full(df, dlen)
    
    if maintrend != 1:  # ليس صاعد
        return False
    
    if not ribbon_colors:
        return False
    
    # احسب نسبة الاتفاق
    agreement_count = sum(1 for r in ribbon_colors if r["trend"] == maintrend)
    agreement_ratio = agreement_count / len(ribbon_colors)
    
    return agreement_ratio >= min_agreement


def get_ribbon_display(df, dlen=DONCHIAN_LENGTH):
    """يرجع نص عرض الـ Ribbon للـ Telegram."""
    if df.empty or len(df) < dlen + 2:
        return "⚠️ بيانات غير كافية"
    
    maintrend, ribbon_colors = calc_donchian_ribbon_full(df, dlen)
    
    if maintrend == 0:
        trend_text = "⚪ محايد"
    elif maintrend == 1:
        trend_text = "🟢 صاعد (أخضر)"
    else:
        trend_text = "🔴 هابط (أحمر)"
    
    lines = [f"<b>Donchian Trend Ribbon</b>"]
    lines.append(f"الاتجاه الرئيسي: {trend_text}")
    lines.append(f"السعر الحالي: {df['close'].iloc[-1]:.6g}$")
    lines.append("\n<b>تفاصيل المستويات:</b>")
    
    for r in ribbon_colors:
        agreement = "✅" if r["trend"] == maintrend else "❌"
        lines.append(
            f"  {r['color']} Level {r['level']}: Length={r['length']} {agreement}"
        )
    
    return "\n".join(lines)

# ------------------------------------------
# Scanning and Monitoring
# ------------------------------------------


def _record_diag(step, symbol, entry_min):
    with diag_lock:
        diag_counts[step] += 1
    with last_diag_lock:
        last_diag["symbol"]    = symbol
        last_diag["step"]      = step
        last_diag["entry_min"] = entry_min
        last_diag["time"]      = datetime.now(timezone.utc)


def _build_scan_frames(raw_1m, raw_60m, entry_min, confirm_min, third_min):
    if entry_min >= 240:
        df_entry   = resample_ohlcv(raw_60m, entry_min)
        df_confirm = resample_ohlcv(raw_60m, confirm_min)
    else:
        df_entry   = resample_ohlcv(raw_1m, entry_min)
        df_confirm = resample_ohlcv(raw_1m, confirm_min)

    if third_min >= 60:
        df_third = resample_ohlcv(raw_60m, third_min)
    else:
        df_third = resample_ohlcv(raw_1m, third_min)

    return df_entry, df_confirm, df_third


def _passes_filters(df_entry, df_confirm, df_third):
    """تحقق من شرط Donchian Ribbon الأساسي فقط."""
    # كل الفريمات يجب أن تكون أخضر
    entry_green = check_donchian_ribbon_green(df_entry)
    confirm_green = check_donchian_ribbon_green(df_confirm)
    third_green = check_donchian_ribbon_green(df_third)
    
    if not (entry_green and confirm_green and third_green):
        return "ribbon_not_green"
    
    return None


def _fire_signal(symbol, entry_min, confirm_min, third_min, df_entry):
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
        entry_time = now.strftime("%Y-%m-%d %H:%M UTC")
        save_signal(symbol, price, entry_min, confirm_min, third_min)
        send_telegram(
            f"🚨 <b>إشارة Donchian Trend Ribbon:</b> {symbol}\n"
            f"🟢 الفريم: {entry_min}m / {confirm_min}m / {third_min}m\n"
            f"💰 السعر: <b>{price:.6g}</b>\n"
            f"🕐 الوقت: <b>{entry_time}</b>"
        )
    except Exception as exc:
        log.error("❌ خطأ في إرسال الإشارة %s: %s", symbol, exc)


def scan_symbol(symbol, entry_min, confirm_min, third_min):
    raw_1m  = get_cached(symbol, "1m")
    raw_60m = get_cached(symbol, "60m")

    with diag_lock:
        diag_counts["total"] += 1

    if raw_1m.empty:
        _record_diag("no_data", symbol, entry_min)
        return

    df_entry, df_confirm, df_third = _build_scan_frames(
        raw_1m, raw_60m, entry_min, confirm_min, third_min
    )

    if df_entry.empty or df_confirm.empty or df_third.empty:
        _record_diag("no_data", symbol, entry_min)
        return

    failed_step = _passes_filters(df_entry, df_confirm, df_third)
    if failed_step:
        _record_diag(failed_step, symbol, entry_min)
        return

    _fire_signal(symbol, entry_min, confirm_min, third_min, df_entry)


def candle_watcher(entry_min, confirm_min, third_min, ec_api, t_api):
    while True:
        time.sleep(30)
        if not fast_prefetch_done.is_set():
            continue
        with symbols_cache_lock:
            syms = list(symbols_cache)
        fn = partial(
            scan_symbol,
            entry_min=entry_min, confirm_min=confirm_min,
            third_min=third_min,
        )
        with ThreadPoolExecutor(max_workers=20) as executor:
            list(executor.map(fn, syms))

# ------------------------------------------
# Telegram Commands
# ------------------------------------------


def _cmd_status(chat_id):
    with trades_lock:
        cnt = len(trades_history)
    with alerted_keys_lock:
        active = len(alerted_keys)
    with ohlcv_cache_lock:
        keys = len(ohlcv_cache)
    send_telegram(
        f"🤖 بوت Donchian Trend Ribbon يعمل\n"
        f"🕐 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"📊 إجمالي الإشارات: {cnt}\n"
        f"🔑 تنبيهات نشطة: {active}\n"
        f"💾 الكاش: {keys} مفتاح\n"
        f"⚡ تحميل سريع: {'✅' if fast_prefetch_done.is_set() else '⏳'}\n"
        f"📦 تحميل كامل: {'✅' if prefetch_done.is_set() else '⏳'}",
        chat_id,
    )


def _cmd_diag(chat_id):
    send_telegram("🔄 جاري تقرير الفحص الفوري...", chat_id)
    time.sleep(2)
    send_telegram(build_diag_msg(reset=False), chat_id)


def _cmd_check(chat_id, txt):
    parts = txt.split()
    cmd = parts[0].lower()
    try:
        entry_min = int(cmd.replace("/check", ""))
    except ValueError:
        send_telegram("❌ صيغة غير صحيحة. مثال: /check5 أو /check5 ETHUSDT", chat_id)
        return

    symbol = parts[1].upper() if len(parts) > 1 else "BTCUSDT"
    if not symbol.endswith("USDT"):
        symbol += "USDT"

    if entry_min not in ENTRY_TO_STRATEGY:
        supported = " | ".join(f"/check{n}" for n in SUPPORTED_CHECK_TFS)
        send_telegram(
            f"❌ فريم <b>{entry_min}m</b> غير مدعوم.\n"
            f"الفريمات المدعومة:\n{supported}",
            chat_id,
        )
        return

    try:
        raw_1m = get_cached(symbol, "1m")
        raw_60m = get_cached(symbol, "60m")

        if raw_1m.empty:
            send_telegram(f"❌ لا يوجد بيانات لـ {symbol}", chat_id)
            return

        main_min, confirm_min = ENTRY_TO_STRATEGY[entry_min]
        
        def _build(minutes):
            if minutes >= 240:
                return resample_ohlcv(raw_60m, minutes) if not raw_60m.empty else pd.DataFrame()
            return resample_ohlcv(raw_1m, minutes)

        df_entry = _build(entry_min)
        df_main = _build(main_min)
        df_confirm = _build(confirm_min)

        msg = f"📊 <b>{symbol} — Donchian Trend Ribbon</b>\n"
        msg += f"🕐 فريم الدخول: {entry_min}m | رئيسي: {main_min}m | تأكيد: {confirm_min}m\n\n"

        for label, df, minutes in [
            ("فريم الدخول", df_entry, entry_min),
            ("فريم الرئيسي", df_main, main_min),
            ("فريم التأكيد", df_confirm, confirm_min),
        ]:
            msg += f"━━━━━━━━━━━━━━━━\n"
            msg += f"📌 <b>{label} — {minutes}m</b>\n"
            if df.empty or len(df) < WARMUP_DON:
                msg += f"⚠️ بيانات غير كافية\n\n"
            else:
                msg += get_ribbon_display(df, DONCHIAN_LENGTH) + "\n\n"

        send_telegram(msg, chat_id)

    except Exception as exc:
        log.error("handle_check error: %s", exc)
        send_telegram(f"❌ خطأ: {exc}", chat_id)


def _cmd_alerts(chat_id):
    now = datetime.now(timezone.utc)
    with alerted_keys_lock:
        active = [
            (k, t) for k, t in alerted_keys.items()
            if now - t < timedelta(hours=ALERT_EXPIRY_HOURS)
        ]
    if not active:
        send_telegram("✅ لا توجد تنبيهات نشطة حالياً.", chat_id)
        return
    active.sort(key=lambda x: x[1], reverse=True)
    lines = [f"🔔 <b>التنبيهات النشطة ({len(active)})</b>", "━━━━━━━━━━━━━━━"]
    for (symbol, entry_min, confirm_min, third_min), t in active[:50]:
        remaining = ALERT_EXPIRY_HOURS * 60 - int((now - t).total_seconds() / 60)
        lines.append(
            f"• {symbol} | {entry_min}m/{confirm_min}m/{third_min}m"
            f" | ⏳ {remaining} دقيقة"
        )
    send_telegram("\n".join(lines), chat_id)


def _dispatch_command(txt, chat_id):
    cmd = txt.split()[0].lower()

    if cmd == "/status":
        _cmd_status(chat_id)
    elif txt in ("1", "/today"):
        send_telegram(get_report("today"), chat_id)
    elif txt in ("2", "/yesterday"):
        send_telegram(get_report("yesterday"), chat_id)
    elif txt in ("3", "/week"):
        send_telegram(get_report("week"), chat_id)
    elif cmd in ("/سبب", "/diag"):
        threading.Thread(target=_cmd_diag, args=(chat_id,), daemon=True).start()
    elif cmd == "/alerts":
        _cmd_alerts(chat_id)
    elif cmd.startswith("/check") and len(cmd) > 6:
        _cmd_check(chat_id, txt)
    elif cmd == "/help":
        check_cmds = "\n".join(
            f"  <code>/check{n}</code> — {n}m/{ENTRY_TO_STRATEGY[n][0]}m/{ENTRY_TO_STRATEGY[n][1]}m"
            for n in SUPPORTED_CHECK_TFS[:8]  # عرض أول 8 فقط
        )
        send_telegram(
            "📋 <b>أوامر بوت Donchian Trend Ribbon</b>\n\n"
            "📊 <b>فحص المؤشرات:</b>\n"
            f"{check_cmds}\n"
            "... و أيضاً /check20 و /check30 و /check60 وغيره\n\n"
            "💡 مثال: <code>/check5 ETH</code> (بدون USDT)\n\n"
            "1️⃣ <code>1</code> — إشارات اليوم\n"
            "2️⃣ <code>2</code> — إشارات أمس\n"
            "3️⃣ <code>3</code> — آخر 7 أيام\n"
            "🔍 <code>/سبب</code> — تقرير الشروط\n"
            "🔔 <code>/alerts</code> — التنبيهات النشطة\n"
            "📊 <code>/status</code> — حالة البوت\n"
            "📋 <code>/help</code> — قائمة الأوامر",
            chat_id,
        )


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
                if txt and chat_id:
                    _dispatch_command(txt, chat_id)
        except Exception:
            time.sleep(10)

# ------------------------------------------
# Symbols Loop
# ------------------------------------------


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
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *_):
        pass

# ------------------------------------------
# Main
# ------------------------------------------


def main():
    def handle_exception(exc_type, exc_value, exc_tb):
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        log.error("💥 خطأ غير متوقع:\n%s", msg)
        try:
            send_telegram(f"💥 <b>البوت توقف بسبب خطأ:</b>\n<code>{exc_value}</code>")
        except Exception:
            pass

    sys.excepthook = handle_exception

    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("✅ Health server على port %s", PORT)

    delete_webhook()

    threading.Thread(target=update_symbols_loop,    daemon=True).start()
    threading.Thread(target=poll_telegram_commands, daemon=True).start()
    threading.Thread(target=cache_updater_1m,       daemon=True).start()
    threading.Thread(target=cache_updater_60m,      daemon=True).start()
    threading.Thread(target=send_diag_report,       daemon=True).start()

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
                "💓 البوت يعمل | كاش: %s | إشارات: %s | سريع: %s | كامل: %s",
                cache_size,
                signals_count,
                "✅" if fast_prefetch_done.is_set() else "⏳",
                "✅" if prefetch_done.is_set() else "⏳",
            )
        except Exception as exc:
            log.error("❌ خطأ في main loop: %s", exc)
            time.sleep(10)


if __name__ == "__main__":
    main()