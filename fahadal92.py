"""بوت مسح العملات من Binance مع تنبيهات Telegram - نسخة Cascade Pipeline مع استراتيجية مزدوجة (شراء/بيع)."""
import os
import time
import logging
import threading
import sys
import traceback
import json
import concurrent.futures
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
    (9, 27, 3, "1m", "1m"), (12, 36, 4, "1m", "1m"), (15, 45, 5, "1m", "1m"),
    (18, 54, 6, "1m", "1m"), (21, 63, 7, "1m", "1m"), (24, 72, 8, "1m", "1m"),
    (27, 81, 9, "1m", "1m"), (30, 90, 10, "1m", "1m"), (45, 135, 15, "1m", "1m"),
    (60, 180, 20, "60m", "1m"), (90, 270, 30, "60m", "1m"), (120, 360, 40, "60m", "1m"),
    (150, 450, 50, "60m", "1m"), (180, 540, 60, "60m", "60m"), (210, 630, 70, "60m", "1m"),
    (240, 720, 80, "60m", "1m"),
]

TIMEFRAME_CHAIN = [9, 12, 15, 18, 21, 24, 27, 30, 45, 60, 90, 120, 150, 180, 210, 240]
NEXT_TF = {TIMEFRAME_CHAIN[i]: TIMEFRAME_CHAIN[i + 1] for i in range(len(TIMEFRAME_CHAIN) - 1)}

FAST_FETCH_CANDLES = {"1m": 100_000, "60m": 15_000}
API_FETCH_CANDLES = {"1m": 100_000, "60m": 15_000}
CACHE_MAX_CANDLES = {"1m": 120_000, "60m": 20_000}

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

cascade_results = defaultdict(dict)
cascade_results_lock = threading.Lock()
cascade_stats = {i: {"total": 0, "passed": 0} for i in range(1, 9)}
cascade_stats_lock = threading.Lock()

last_complete_stats = {i: {"total": 0, "passed": 0} for i in range(1, 9)}
last_complete_results = defaultdict(dict)
last_complete_survivors = {}
last_complete_lock = threading.Lock()
last_complete_scan_time = {"buy": None, "sell": None}
last_complete_scan_time_lock = threading.Lock()

short_cascade_results = defaultdict(dict)
short_cascade_results_lock = threading.Lock()
short_cascade_stats = {i: {"total": 0, "passed": 0} for i in range(1, 9)}
short_cascade_stats_lock = threading.Lock()

last_complete_short_stats = {i: {"total": 0, "passed": 0} for i in range(1, 9)}
last_complete_short_results = defaultdict(dict)
last_complete_short_survivors = {}
last_complete_short_lock = threading.Lock()

fast_prefetch_done = threading.Event()
prefetch_done = threading.Event()

_local = threading.local()

first_scan_notified = False
first_scan_lock = threading.Lock()

# ------------------------------------------
# Labels
# ------------------------------------------

STEP_NAMES = ["smi_oversold", "macd_red", "donchian_base", "donchian_confirm",
              "macd_confirm", "ema50", "donchian_triple", "rsi_stoch"]

STEP_LABELS = {
    "smi_oversold": "① تشبع بيعي SMI",
    "macd_red": "② MACD أحمر",
    "donchian_base": "③ Donchian Ribbon (الفريم الأساسي) أخضر",
    "donchian_confirm": "④ Donchian Ribbon (فريم التأكيد) أخضر",
    "macd_confirm": "⑤ MACD Confirm أخضر",
    "ema50": "⑥ السعر تحت EMA50",
    "donchian_triple": "⑦ Donchian Ribbon (فريم التثليث) أحمر",
    "rsi_stoch": "⑧ RSI/Stochastic تقاطع",
}

SHORT_STEP_NAMES = ["smi_overbought", "macd_green", "donchian_base_red", "donchian_confirm_red",
                    "macd_confirm_red", "ema50_above", "donchian_triple_green", "rsi_stoch_short"]

SHORT_STEP_LABELS = {
    "smi_overbought": "① تشبع شرائي SMI ≥ +40",
    "macd_green": "② MACD أخضر",
    "donchian_base_red": "③ Donchian Ribbon (الفريم الأساسي) أحمر",
    "donchian_confirm_red": "④ Donchian Ribbon (فريم التأكيد) أحمر",
    "macd_confirm_red": "⑤ MACD Confirm أحمر",
    "ema50_above": "⑥ السعر فوق EMA50",
    "donchian_triple_green": "⑦ Donchian Ribbon (فريم التثليث) أخضر",
    "rsi_stoch_short": "⑧ RSI≥65 / Stochastic≤20",
}

# ------------------------------------------
# Helper Functions
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
        expired = [k for k, t in list(alerted_keys.items()) if now - t > timedelta(hours=ALERT_EXPIRY_HOURS)]
        for k in expired:
            del alerted_keys[k]

def save_signal(symbol, price, base_frame, confirm_frame, triple_frame, signal_type="buy"):
    with trades_lock:
        trades_history.append({
            "time": datetime.now(timezone.utc),
            "symbol": symbol,
            "price": price,
            "timeframe": f"{base_frame}m/{confirm_frame}m/{triple_frame}m",
            "type": signal_type,
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

def get_report(period="today", signal_type=None):
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
        if signal_type:
            rows = [r for r in rows if r.get("type") == signal_type]

    if not rows:
        return f"<b>{title}:</b>\nلا توجد إشارات."

    lines = [f"<b>{title} ({len(rows)})</b>\n" + "━" * 15]
    for t in rows:
        icon = "🟢" if t.get("type") == "buy" else "🔴"
        lines.append(f"{icon} {t['symbol']} | {t['timeframe']} | {t['price']:.4g} | {t['time'].strftime('%H:%M UTC')}")
    return "\n".join(lines)

def diagnose_signal_failures():
    """
    تشخيص أهم 3 أسباب لعدم مجيء إشارات
    ترتيب من الأقوى فشل إلى الأضعف
    """
    
    if not fast_prefetch_done.is_set():
        return [
            {
                "rank": 1,
                "reason": "❌ البيانات لم تحمل بعد",
                "severity": "CRITICAL",
                "description": "البوت ما زال يحمل البيانات الأولية",
                "solution": "انتظر 5-30 دقيقة للتحميل الكامل"
            }
        ]
    
    with symbols_cache_lock:
        symbols = list(symbols_cache)
    
    if not symbols:
        return [
            {
                "rank": 1,
                "reason": "❌ لا توجد عملات محملة",
                "severity": "CRITICAL",
                "description": "قائمة العملات فارغة",
                "solution": "تحقق من Binance API"
            }
        ]
    
    failures = []
    
    # ─────────────────────────────────────────────
    # السبب #1: فشل MIN_CANDLES (الأهم!)
    # ─────────────────────────────────────────────
    
    min_candles_failures = 0
    total_candidates = 0
    
    for sym in symbols[:10]:  # فحص أول 10 عملات
        raw_1m = get_cached(sym, "1m")
        raw_60m = get_cached(sym, "60m")
        
        if raw_1m.empty or raw_60m.empty:
            continue
        
        for base_frame, confirm_frame, triple_frame, base_api, triple_api in TRIPLING_PAIRS:
            total_candidates += 1
            raw_base = raw_1m if base_api == "1m" else raw_60m
            
            if raw_base.empty:
                continue
            
            df_base = resample_ohlcv(raw_base, base_frame)
            
            if len(df_base) < MIN_CANDLES:
                min_candles_failures += 1
    
    min_candles_percentage = (min_candles_failures / total_candidates * 100) if total_candidates > 0 else 0
    
    # ─────────────────────────────────────────────
    # السبب #2: فشل Step 6 (حماية RSI)
    # ─────────────────────────────────────────────
    
    step6_failures = 0
    step6_total = 0
    
    with last_complete_lock:
        for step_num in [6]:
            stats = last_complete_stats.get(step_num, {})
            total = stats.get("total", 0)
            passed = stats.get("passed", 0)
            
            if total > 0:
                step6_failures = total - passed
                step6_total = total
    
    step6_percentage = (step6_failures / step6_total * 100) if step6_total > 0 else 0
    
    # ─────────────────────────────────────────────
    # السبب #3: فشل Step 1 (SMI Oversold)
    # ─────────────────────────────────────────────
    
    step1_failures = 0
    step1_total = 0
    
    with last_complete_lock:
        for step_num in [1]:
            stats = last_complete_stats.get(step_num, {})
            total = stats.get("total", 0)
            passed = stats.get("passed", 0)
            
            if total > 0:
                step1_failures = total - passed
                step1_total = total
    
    step1_percentage = (step1_failures / step1_total * 100) if step1_total > 0 else 0
    
    # ─────────────────────────────────────────────
    # ترتيب الأسباب من الأقوى فشل
    # ─────────────────────────────────────────────
    
    reasons = [
        {
            "rank": 1,
            "reason": "❌ فشل MIN_CANDLES (الحد الأدنى من الشموات)",
            "severity": "CRITICAL" if min_candles_percentage > 50 else "HIGH",
            "percentage": min_candles_percentage,
            "total_failed": min_candles_failures,
            "total": total_candidates,
            "description": f"{min_candles_failures} مرشح من {total_candidates} فشلوا في اختبار الحد الأدنى (250 شمعة)",
            "solution": "زيادة API_FETCH_CANDLES من 15_000 إلى 100_000",
            "why": "الأطر الكبيرة (180m, 240m) تحتاج بيانات أكثر"
        },
        {
            "rank": 2,
            "reason": "⚠️ فشل Step 6 (حماية RSI القاسية)",
            "severity": "HIGH" if step6_percentage > 50 else "MEDIUM",
            "percentage": step6_percentage,
            "total_failed": step6_failures,
            "total": step6_total,
            "description": f"{step6_failures} مرشح من {step6_total} فشلوا في خطوة RSI",
            "solution": "تقليل متطلبات RSI (تغيير threshold من 35 إلى 40)",
            "why": "شروط RSI معقدة جداً ومتقاطعة"
        },
        {
            "rank": 3,
            "reason": "⚡ فشل Step 1 (SMI Oversold ≤ -40)",
            "severity": "MEDIUM" if step1_percentage > 70 else "LOW",
            "percentage": step1_percentage,
            "total_failed": step1_failures,
            "total": step1_total,
            "description": f"{step1_failures} مرشح من {step1_total} لم يصلوا لتشبع SMI بيعي",
            "solution": "تخفيف عتبة SMI من -40 إلى -30",
            "why": "السوق لا يدخل تشبع بيعي في كل وقت"
        }
    ]
    
    # ترتيب حسب الفشل (من الأكثر للأقل)
    reasons.sort(key=lambda x: x["percentage"], reverse=True)
    
    # إعادة ترقيم
    for i, reason in enumerate(reasons, 1):
        reason["rank"] = i
    
    return reasons


def send_diagnostics_report():
    """إرسال تقرير التشخيص عبر Telegram"""
    reasons = diagnose_signal_failures()
    
    lines = [
        "🔍 <b>تقرير تشخيص فشل الإشارات</b>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ]
    
    for reason in reasons:
        rank = reason["rank"]
        reason_text = reason["reason"]
        severity = reason["severity"]
        percentage = reason["percentage"]
        total_failed = reason["total_failed"]
        total = reason["total"]
        description = reason["description"]
        solution = reason["solution"]
        why = reason["why"]
        
        icon = "🔴" if severity == "CRITICAL" else ("🟠" if severity == "HIGH" else "🟡")
        
        lines.append(f"""
{icon} <b>السبب #{rank}: {reason_text}</b>
├─ الشدة: {severity}
├─ نسبة الفشل: <b>{percentage:.1f}%</b> ({total_failed}/{total})
├─ التفاصيل: {description}
├─ الحل: <code>{solution}</code>
└─ السبب: {why}
""")
    
    msg = "\n".join(lines)
    send_telegram(msg)


def handle_diag_command(chat_id):
    """معالج أمر /diag_failures"""
    reasons = diagnose_signal_failures()
    
    lines = [
        "🔍 <b>تشخيص أسباب فشل الإشارات</b>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ""
    ]
    
    for reason in reasons:
        rank = reason["rank"]
        reason_text = reason["reason"]
        percentage = f"{reason['percentage']:.1f}%"
        description = reason["description"]
        solution = reason["solution"]
        
        lines.append(f"""
<b>#{rank}: {reason_text}</b>
📊 نسبة الفشل: <b>{percentage}</b>
📝 الوصف: {description}
✅ الحل: {solution}
""")
    
    msg = "\n".join(lines)
    
    # تقسيم الرسالة إلى أجزاء إذا كانت طويلة جداً
    for i in range(0, len(msg), 4000):
        send_telegram(msg[i:i + 4000], chat_id)
        
def get_top_hard_filters(signal_type="buy", top_n=3, max_pass_pct=10.0):
    """
    يرجع أكثر N فلاتر قسوة (نسبة نجاح < max_pass_pct%)
    مرتبة من الأصعب للأخف
    """
    if signal_type == "buy":
        lock = last_complete_lock
        stats = last_complete_stats
        step_names = STEP_NAMES
        step_labels = STEP_LABELS
    else:
        lock = last_complete_short_lock
        stats = last_complete_short_stats
        step_names = SHORT_STEP_NAMES
        step_labels = SHORT_STEP_LABELS

    hard_filters = []

    with lock:
        for step_num in range(1, 9):
            stat = stats.get(step_num, {})
            total = stat.get("total", 0)
            passed = stat.get("passed", 0)

            if total == 0:
                continue  # لا بيانات بعد

            pass_pct = (passed / total) * 100

            if pass_pct <= max_pass_pct:
                name = step_names[step_num - 1]
                label = step_labels[name]
                hard_filters.append({
                    "step_num": step_num,
                    "label": label,
                    "total": total,
                    "passed": passed,
                    "failed": total - passed,
                    "pass_pct": pass_pct,
                })

    # ترتيب من الأصعب (أقل نسبة نجاح) للأخف
    hard_filters.sort(key=lambda x: x["pass_pct"])

    return hard_filters[:top_n]


def handle_hard_filters_command(chat_id, signal_type="buy"):
    """معالج أمر /hard_filters أو /hard_filters_sell"""

    if not fast_prefetch_done.is_set():
        send_telegram("⏳ البوت لم يكمل التحميل بعد، انتظر قليلاً.", chat_id)
        return

    icon_type = "🟢 LONG (شراء)" if signal_type == "buy" else "🔴 SHORT (بيع)"
    filters = get_top_hard_filters(signal_type=signal_type, top_n=3, max_pass_pct=10.0)

    if not filters:
        send_telegram(
            f"✅ <b>{icon_type}</b>\n"
            f"لا توجد فلاتر بنسبة نجاح أقل من 10% — الكود يعمل بشكل طبيعي.",
            chat_id
        )
        return

    lines = [
        f"⚠️ <b>أكثر الفلاتر قسوة — {icon_type}</b>",
        f"<i>(نسبة النجاح أقل من 10%)</i>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ""
    ]

    medals = ["🥇", "🥈", "🥉"]

    for i, f in enumerate(filters):
        pass_pct = f["pass_pct"]
        fail_pct = 100 - pass_pct
        bar_pass = "█" * int(pass_pct / 10) + "░" * (10 - int(pass_pct / 10))

        lines.append(
            f"{medals[i]} <b>خطوة #{f['step_num']}: {f['label']}</b>\n"
            f"  {bar_pass}\n"
            f"  ✅ نجح: <b>{f['passed']}</b> ({pass_pct:.1f}%)\n"
            f"  ❌ فشل: <b>{f['failed']}</b> ({fail_pct:.1f}%)\n"
            f"  📥 دخل: <b>{f['total']}</b>\n"
        )

    msg = "\n".join(lines)

    for i in range(0, len(msg), 4000):
        send_telegram(msg[i:i + 4000], chat_id)
        
        
# ------------------------------------------
# Binance OHLCV
# ------------------------------------------

def _parse_binance_klines(resp):
    df = pd.DataFrame(resp, columns=["ts", "open", "high", "low", "close", "vol", "close_time", "quote_vol", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    for col in ["open", "high", "low", "close", "vol"]:
        df[col] = df[col].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)[["ts", "open", "high", "low", "close", "vol"]]

def get_ohlcv(symbol, tf, limit=500):
    binance_tf = TF_MAP.get(tf, "1m")
    try:
        resp = get_session().get(f"{BINANCE_BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": binance_tf, "limit": min(limit, 1000)}, timeout=10).json()
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
        batch = min(bin_max, target - fetched)
        start_ms = end_ms - batch * tf_ms
        try:
            resp = get_session().get(f"{BINANCE_BASE}/api/v3/klines",
                params={"symbol": symbol, "interval": binance_tf, "startTime": start_ms, "endTime": end_ms, "limit": batch}, timeout=15).json()
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

    return (pd.concat(all_dfs).drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
            if all_dfs else pd.DataFrame())

def cache_merge(symbol, tf, new_df):
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
    with ohlcv_cache_lock:
        df = ohlcv_cache.get((symbol, tf))
    return df.copy() if df is not None else pd.DataFrame()

def check_entry_rsi_clean_buy(df, lookback=50, threshold=30):
   if len(df) < WARMUP_RSI + lookback:
       return True
   rsi = calc_rsi_tv(df["close"], period=14)
   return not bool((rsi.iloc[-lookback:] <= threshold).any())

def check_entry_rsi_clean_sell(df, lookback=50, threshold=70):
   if len(df) < WARMUP_RSI + lookback:
       return True
   rsi = calc_rsi_tv(df["close"], period=14)
   return not bool((rsi.iloc[-lookback:] >= threshold).any())

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
# Technical Indicators
# ------------------------------------------

def resample_ohlcv(df, minutes):
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
    if df.empty:
        return pd.DataFrame()
    return (df.copy().set_index("ts").resample(f"{minutes}min", closed="left", label="left", origin=datetime(1970, 1, 1, tzinfo=timezone.utc))
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "vol": "sum"})
            .dropna().reset_index())

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

def calc_donchian_trend(df, length=20):
    if len(df) < length + 2:
        return []
    hh = df["high"].rolling(length).max()
    ll = df["low"].rolling(length).min()
    trend = [0] * len(df)
    for i in range(1, len(df)):
        if pd.isna(hh.iloc[i-1]) or pd.isna(ll.iloc[i-1]):
            trend[i] = trend[i-1]
            continue
        if df["close"].iloc[i] > hh.iloc[i-1]:
            trend[i] = 1
        elif df["close"].iloc[i] < ll.iloc[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
    return trend

def calc_donchian_trend_vectorized(close_arr, high_arr, low_arr, length):
    n = len(close_arr)
    if n < length + 2:
        return 0

    high_s = pd.Series(high_arr)
    low_s = pd.Series(low_arr)

    prev_hh = high_s.rolling(length).max().shift(1).values
    prev_ll = low_s.rolling(length).min().shift(1).values

    close_arr = np.asarray(close_arr)

    breakout_up = close_arr > prev_hh
    breakout_down = close_arr < prev_ll

    up_indices = np.where(breakout_up)[0]
    down_indices = np.where(breakout_down)[0]

    last_up = up_indices[-1] if len(up_indices) > 0 else -1
    last_down = down_indices[-1] if len(down_indices) > 0 else -1

    if last_up == -1 and last_down == -1:
        return 0
    return 1 if last_up > last_down else -1

def calc_donchian_trend_ribbon_correct(df, length=20):
    if len(df) < length + 2:
        return 0, False
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    layers = []
    for offset in range(10):
        layer_len = length - offset
        if layer_len < 2:
            layers.append(0)
            continue
        t = calc_donchian_trend_vectorized(close, high, low, layer_len)
        layers.append(t)
    current_main = layers[0]
    all_consistent = all(t == current_main for t in layers)
    return current_main, all_consistent

def check_donchian_trend_ribbon(df, direction="green"):
    if len(df) < 35:
        return False
    trend, _ = calc_donchian_trend_ribbon_correct(df, length=20)
    if direction == "green":
        return trend == 1
    return trend == -1

def check_ema50_below(df):
    ema = df["close"].ewm(span=50, adjust=False).mean()
    return bool(df["close"].iloc[-1] < ema.iloc[-1])

def check_ema50_above(df):
    ema = df["close"].ewm(span=50, adjust=False).mean()
    return bool(df["close"].iloc[-1] > ema.iloc[-1])

def calc_smi(high, low, close, k=10, d=3, ema_len=10, smooth=1):
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
    if len(df) < WARMUP_SMI:
        return False
    smi, _ = calc_smi(df["high"], df["low"], df["close"])
    return bool(smi.iloc[-1] <= threshold)

def check_smi_overbought(df, threshold=40):
    if len(df) < WARMUP_SMI:
        return False
    smi, _ = calc_smi(df["high"], df["low"], df["close"])
    return bool(smi.iloc[-1] >= threshold)

def check_ema50_above_since_overbought(df, smi_threshold=40):
    if len(df) < WARMUP_SMI:
        return False
    smi, _ = calc_smi(df["high"], df["low"], df["close"])
    ema = df["close"].ewm(span=50, adjust=False).mean()
    overbought_mask = smi >= smi_threshold
    if not overbought_mask.any():
        return False
    last_idx = overbought_mask[::-1].idxmax()
    return bool((df["close"].loc[last_idx:] > ema.loc[last_idx:]).any())

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

def check_rsi_stoch(df, lookback=5, max_gap=5):
    if len(df) < WARMUP_RSI + lookback:
        return False
    rsi = calc_rsi_tv(df["close"], period=14)
    rsi_sig = rsi.rolling(14).mean()
    k, _ = calc_stoch_tv(df["close"], df["high"], df["low"])
    if float(k.iloc[-1]) <= 20:
        return False
    stoch_crosses = []
    rsi_crosses = []
    for i in range(-lookback, 0):
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

def check_rsi_stoch_short(df, lookback=5, max_gap=5):
    if len(df) < WARMUP_RSI + lookback:
        return False
    rsi = calc_rsi_tv(df["close"], period=14)
    rsi_sig = rsi.rolling(14).mean()
    k, _ = calc_stoch_tv(df["close"], df["high"], df["low"])
    if float(k.iloc[-1]) >= 80:
        return False
    stoch_crosses = []
    rsi_crosses = []
    for i in range(-lookback, 0):
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

# ------------------------------------------
# CASCADE PIPELINE - LONG (BUY)
# ------------------------------------------

def step1(c):
    if not check_smi_oversold(c["df_base"]):
        return False, "smi_oversold"
    base_frame = c["base_frame"]
    raw_base = c["raw_base"]
    for tf in TIMEFRAME_CHAIN:
        if tf <= base_frame:
            continue
        df_higher = c["get_resampled"](c["raw_base"], c["sym"], c["base_api"], tf)
        if not df_higher.empty and check_smi_oversold(df_higher):
            return False, "active_skip"
    return True, "passed"

def step2(c):
    if not check_macd_red(c["df_base"]):
        return False, "macd_red"
    return True, "passed"

def step3(c):
    if not check_donchian_trend_ribbon(c["df_base"], "green"):
        return False, "donchian_base"
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
    if not check_ema50_below(c["df_base"]):
        return False, "ema50"
    if not check_rsi_not_oversold_recently(c["df_triple"], lookback=50, threshold=35):
        return False, "ema50"
    if not check_confirm_rsi_not_oversold(c["df_confirm"], lookback=30, threshold=35):
        return False, "ema50"
    if not check_entry_rsi_clean_buy(c["df_triple"], lookback=50, threshold=30):
        return False, "ema50"
    if not check_confirm_rsi_not_oversold(c["df_confirm"], lookback=30, threshold=30):
        return False, "ema50"
    return True, "passed"

def step7(c):
    if not check_donchian_trend_ribbon(c["df_triple"], "red"):
        return False, "donchian_triple"
    return True, "passed"

def step8(c):
    if not check_rsi_touched_oversold(c["df_triple"]):
        return False, "rsi_stoch"
    if not check_rsi_stoch(c["df_triple"]):
        return False, "rsi_stoch"
    return True, "passed"

steps = [step1, step2, step3, step4, step5, step6, step7, step8]

# ------------------------------------------
# CASCADE PIPELINE - SHORT (SELL)
# ------------------------------------------

def short_step1(c):
    if not check_smi_overbought(c["df_base"], threshold=40):
        return False, "smi_overbought"
    base_frame = c["base_frame"]
    for tf in TIMEFRAME_CHAIN:
        if tf <= base_frame:
            continue
        df_higher = c["get_resampled"](c["raw_base"], c["sym"], c["base_api"], tf)
        if not df_higher.empty and check_smi_overbought(df_higher, threshold=40):
            return False, "active_skip"
    return True, "passed"

def short_step2(c):
    if not check_macd_green(c["df_base"]):
        return False, "macd_green"
    return True, "passed"

def short_step3(c):
    if not check_donchian_trend_ribbon(c["df_base"], "red"):
        return False, "donchian_base_red"
    return True, "passed"

def short_step4(c):
    if not check_donchian_trend_ribbon(c["df_confirm"], "red"):
        return False, "donchian_confirm_red"
    return True, "passed"

def short_step5(c):
    if not check_macd_red(c["df_confirm"]):
        return False, "macd_confirm_red"
    return True, "passed"

def short_step6(c):
    if not check_ema50_above(c["df_base"]):
        return False, "ema50_above"
    if not check_rsi_not_overbought_recently(c["df_triple"], lookback=50, threshold=70):
        return False, "ema50_above"
    if not check_confirm_rsi_not_overbought(c["df_confirm"], lookback=30, threshold=70):
        return False, "ema50_above"
    if not check_entry_rsi_clean_sell(c["df_triple"], lookback=50, threshold=70):
        return False, "ema50_above"
    if not check_confirm_rsi_not_overbought(c["df_confirm"], lookback=30, threshold=70):
        return False, "ema50_above"
    return True, "passed"

def short_step7(c):
    if not check_donchian_trend_ribbon(c["df_triple"], "green"):
        return False, "donchian_triple_green"
    return True, "passed"

def short_step8(c):
    if not check_rsi_overbought_short(c["df_triple"]):
        return False, "rsi_stoch_short"
    if not check_rsi_stoch_short(c["df_triple"]):
        return False, "rsi_stoch_short"
    return True, "passed"

short_steps = [short_step1, short_step2, short_step3, short_step4,
               short_step5, short_step6, short_step7, short_step8]

long_steps = steps

def run_cascade_scan():
    with symbols_cache_lock:
        symbols = list(symbols_cache)
    if not symbols:
        log.warning("⚠️ لا توجد symbols في الكاش")
        return

    with ohlcv_cache_lock:
        cache_size = len(ohlcv_cache)
    if cache_size < len(symbols) * 0.8:
        log.info("⏳ الكاش غير كافٍ بعد (%d مفتاح)، تخطي المسح", cache_size)
        return

    log.info("✅ الكاش كافٍ (%d مفتاح)", cache_size)

    with cascade_stats_lock, cascade_results_lock:
        for i in range(1, 9):
            cascade_stats[i]["total"] = 0
            cascade_stats[i]["passed"] = 0
            cascade_results[i].clear()

    resample_cache = {}
    step_survivors = {}

    def get_resampled(raw_df, sym, tf, minutes):
        key = (sym, tf, minutes)
        if key not in resample_cache:
            resample_cache[key] = resample_ohlcv(raw_df, minutes)
        return resample_cache[key]

    candidates = []
    for sym in symbols:
        raw_ec_1m = get_cached(sym, "1m")
        raw_ec_60m = get_cached(sym, "60m")

        for base_frame, confirm_frame, triple_frame, base_api, triple_api in TRIPLING_PAIRS:
            raw_base = raw_ec_1m if base_api == "1m" else raw_ec_60m
            raw_triple = raw_ec_1m if triple_api == "1m" else raw_ec_60m

            if raw_base.empty or raw_triple.empty:
                continue

            df_base = get_resampled(raw_base, sym, base_api, base_frame)
            df_confirm = get_resampled(raw_base, sym, base_api, confirm_frame)
            df_triple = get_resampled(raw_triple, sym, triple_api, triple_frame)

            if df_base.empty or df_confirm.empty or df_triple.empty:
                continue
            if len(df_base) < MIN_CANDLES:
                continue

            candidates.append({
                "sym": sym, "base_api": base_api, "triple_api": triple_api,
                "base_frame": base_frame, "confirm_frame": confirm_frame, "triple_frame": triple_frame,
                "df_base": df_base, "df_confirm": df_confirm, "df_triple": df_triple,
                "raw_base": raw_base,
                "get_resampled": get_resampled,
            })

    log.info("🔄 Cascade Scan (LONG): %d مرشح قبل الخطوات", len(candidates))

    for step_num, step_fn in enumerate(steps, start=1):
        if not candidates:
            log.info("⏸️ انقطعت المعالجة في الخطوة %d (LONG)", step_num)
            break

        def run_one(c, fn=step_fn):
            try:
                return c, *fn(c)
            except Exception as e:
                log.error("❌ خطأ في الخطوة %d (LONG): %s", step_num, e)
                return c, False, str(e)

        try:
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(run_one, candidate) for candidate in candidates]
                results = []
                for future in concurrent.futures.as_completed(futures, timeout=120):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        log.error("❌ خطأ: %s", e)
        except Exception as e:
            log.error("❌ خطأ في الخطوة %d (LONG): %s", step_num, e)
            break

        passed = []
        now = datetime.now(timezone.utc)
        with cascade_results_lock, cascade_stats_lock:
            cascade_stats[step_num]["total"] = len(results)
            for c, ok, reason in results:
                key = (c["sym"], c["base_frame"], c["confirm_frame"], c["triple_frame"])
                cascade_results[step_num][key] = {"passed": ok, "reason": reason, "time": now}
                if ok:
                    cascade_stats[step_num]["passed"] += 1
                    passed.append(c)

        log.info("📍 خطوة %d (LONG): %d/%d نجحوا", step_num, len(passed), len(results))
        step_survivors[step_num] = passed
        candidates = passed

    if cascade_stats.get(1, {}).get("total", 0) > 0:
        with last_complete_lock, cascade_stats_lock, cascade_results_lock:
            for i in range(1, 9):
                last_complete_stats[i] = dict(cascade_stats.get(i, {}))
                last_complete_results[i] = dict(cascade_results.get(i, {}))
            last_complete_survivors.clear()
            last_complete_survivors.update(step_survivors)
        with last_complete_scan_time_lock:
            last_complete_scan_time["buy"] = datetime.now(timezone.utc)

    log.info("🎉 إشارات نهائية (LONG): %d", len(candidates))
    for c in candidates:
        _fire_signal(c["sym"], c["base_frame"], c["confirm_frame"],
                     c["triple_frame"], c["df_base"], signal_type="buy")
                    

def run_short_cascade_scan():
    with symbols_cache_lock:
        symbols = list(symbols_cache)
    if not symbols:
        return

    with short_cascade_stats_lock, short_cascade_results_lock:
        for i in range(1, 9):
            short_cascade_stats[i]["total"] = 0
            short_cascade_stats[i]["passed"] = 0
            short_cascade_results[i].clear()

    resample_cache = {}
    short_step_survivors = {}

    def get_resampled(raw_df, sym, tf, minutes):
        key = (sym, tf, minutes)
        if key not in resample_cache:
            resample_cache[key] = resample_ohlcv(raw_df, minutes)
        return resample_cache[key]

    short_candidates = []
    for sym in symbols:
        raw_base_1m = get_cached(sym, "1m")
        raw_base_60m = get_cached(sym, "60m")

        for base_frame, confirm_frame, triple_frame, base_api, triple_api in TRIPLING_PAIRS:
            raw_base = raw_base_1m if base_api == "1m" else raw_base_60m
            raw_triple = raw_base_1m if triple_api == "1m" else raw_base_60m

            if raw_base.empty or raw_triple.empty:
                continue

            df_base = get_resampled(raw_base, sym, base_api, base_frame)
            df_confirm = get_resampled(raw_base, sym, base_api, confirm_frame)
            df_triple = get_resampled(raw_triple, sym, triple_api, triple_frame)

            if df_base.empty or df_confirm.empty or df_triple.empty:
                continue
            if len(df_base) < MIN_CANDLES:
                continue


            short_candidates.append({
                "sym": sym, "base_api": base_api, "triple_api": triple_api,
                "base_frame": base_frame, "confirm_frame": confirm_frame, "triple_frame": triple_frame,
                "df_base": df_base, "df_confirm": df_confirm, "df_triple": df_triple,
                "raw_base": raw_base,
                "get_resampled": get_resampled,
            })

    log.info("🔄 Cascade Scan (SHORT): %d مرشح", len(short_candidates))

    candidates = short_candidates

    for step_num, step_fn in enumerate(short_steps, start=1):
        if not candidates:
            log.info("⏸️  انقطعت المعالجة في الخطوة %d (SHORT)", step_num)
            break

        def run_one(c, fn=step_fn):
            try:
                return c, *fn(c)
            except Exception as e:
                log.error("❌ خطأ في الخطوة %d (SHORT): %s", step_num, e)
                return c, False, str(e)

        try:
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = [executor.submit(run_one, candidate) for candidate in candidates]
                results = []

                for future in concurrent.futures.as_completed(futures, timeout=120):
                    try:
                        result = future.result()
                        results.append(result)
                    except concurrent.futures.TimeoutError:
                        log.warning("⚠️  timeout في الخطوة %d (SHORT)", step_num)
                    except Exception as e:
                        log.error("❌ خطأ: %s", e)

        except Exception as e:
            log.error("❌ خطأ في الخطوة %d (SHORT): %s", step_num, e)
            break

        passed = []
        now = datetime.now(timezone.utc)
        short_cascade_stats[step_num] = {"total": 0, "passed": 0}
        short_cascade_results[step_num] = {}

        if results:
            with short_cascade_results_lock, short_cascade_stats_lock:
                short_cascade_stats[step_num]["total"] = len(results)
                for c, ok, reason in results:
                    key = (c["sym"], c["base_frame"], c["confirm_frame"], c["triple_frame"])
                    short_cascade_results[step_num][key] = {"passed": ok, "reason": reason, "time": now}
                    if ok:
                        short_cascade_stats[step_num]["passed"] += 1
                        passed.append(c)

            log.info("📍 خطوة %d (SHORT): %d/%d نجحوا", step_num, len(passed), len(results))
        else:
            log.warning("⚠️  لا توجد نتائج في الخطوة %d", step_num)

        short_step_survivors[step_num] = passed
        candidates = passed

    # خارج حلقة for
    if short_cascade_stats.get(1, {}).get("total", 0) > 0:
        with last_complete_short_lock, short_cascade_stats_lock, short_cascade_results_lock:
            for i in range(1, 9):
                last_complete_short_stats[i] = dict(short_cascade_stats.get(i, {}))
                last_complete_short_results[i] = dict(short_cascade_results.get(i, {}))
            last_complete_short_survivors.clear()
            last_complete_short_survivors.update(short_step_survivors)
        with last_complete_scan_time_lock:
            last_complete_scan_time["sell"] = datetime.now(timezone.utc)

    log.info("🎉 إشارات نهائية (SHORT): %d", len(candidates))
    for c in candidates:
        _fire_signal(
            c["sym"],
            c["base_frame"],
            c["confirm_frame"],
            c["triple_frame"],
            c["df_base"],
            signal_type="sell"
        )

# ------------------------------------------
# Telegram Commands
# ------------------------------------------

def _cmd_cascade_diag(chat_id, signal_type="buy"):
    if signal_type == "buy":
        lock = last_complete_lock
        stats = last_complete_stats
        results = last_complete_results
        title = "🔍 <b>تقرير Cascade Pipeline — الشراء LONG</b>"
        scan_key = "buy"
    else:
        lock = last_complete_short_lock
        stats = last_complete_short_stats
        results = last_complete_short_results
        title = "🔍 <b>تقرير Cascade Pipeline — البيع SHORT</b>"
        scan_key = "sell"

    with last_complete_scan_time_lock:
        last_time = last_complete_scan_time.get(scan_key)

    if last_time:
        age_min = int((datetime.now(timezone.utc) - last_time).total_seconds() / 60)
        time_str = f"{last_time.strftime('%H:%M:%S UTC')} (منذ {age_min} دقيقة)"
    else:
        time_str = "⏳ لا توجد بيانات بعد — لم يكتمل أي سكان كامل"

    with lock:
        lines = [title, f"🕐 آخر تحديث: {time_str}", "━━━━━━━━━━━━━━━━━━━━━━"]

        for step_num in range(1, 9):
            step_name = STEP_NAMES[step_num - 1] if signal_type == "buy" else SHORT_STEP_NAMES[step_num - 1]
            step_label = STEP_LABELS[step_name] if signal_type == "buy" else SHORT_STEP_LABELS[step_name]

            stat = stats[step_num]
            total_t = stat["total"]
            total_p = stat["passed"]
            fail_count = total_t - total_p
            pct = int(total_p / total_t * 100) if total_t else 0
            bar = "█" * (pct // 10) + "░" * (10 - pct // 10)

            lines.append(f"\n{step_label}\n{bar}\n✅ نجح: <b>{total_p}</b> | ❌ فشل: <b>{fail_count}</b> | دخل: <b>{total_t}</b> ({pct}%)")

        msg = "\n".join(lines)

    for i in range(0, len(msg), 4000):
        send_telegram(msg[i:i + 4000], chat_id)

def _cmd_show_step_survivors(chat_id, step_num=6, signal_type="buy"):
    if signal_type == "buy":
        lock = last_complete_lock
        survivors_dict = last_complete_survivors
    else:
        lock = last_complete_short_lock
        survivors_dict = last_complete_short_survivors

    with lock:
        survivors = survivors_dict.get(step_num, [])

    if not survivors:
        send_telegram(f"⚠️ لا توجد عملات نجحت حتى الخطوة {step_num}", chat_id)
        return

    icon = "🟢" if signal_type == "buy" else "🔴"
    lines = [f"{icon} <b>الناجحون حتى الخطوة {step_num} ({len(survivors)} عملات)</b>", "━" * 30]

    for c in survivors:
        lines.append(f"• <b>{c['sym']}</b>\n├─ فريم أساسي: {c['base_frame']}m\n├─ فريم تأكيد: {c['confirm_frame']}m\n└─ فريم تثليث: {c['triple_frame']}m")

    msg = "\n".join(lines)

    for i in range(0, len(msg), 4000):
        send_telegram(msg[i:i + 4000], chat_id)

def _cmd_status(chat_id):
    with ohlcv_cache_lock:
        cache_size = len(ohlcv_cache)
    with trades_lock:
        signals_count = len(trades_history)

    msg = (f"<b>📊 حالة البوت</b>\n"
           f"🔄 الكاش: {cache_size} مفتاح\n"
           f"📈 الإشارات: {signals_count}\n"
           f"⚡ التحميل السريع: {'✅' if fast_prefetch_done.is_set() else '⏳'}\n"
           f"📦 التحميل الكامل: {'✅' if prefetch_done.is_set() else '⏳'}")
    send_telegram(msg, chat_id)

def handle_check5(chat_id, symbol="BTCUSDT"):
    send_telegram(f"🔄 جاري جلب بيانات {symbol} — فريم 5 دقايق...", chat_id)
    try:
        fresh = get_ohlcv(symbol, "1m", limit=100)
        if not fresh.empty:
            cache_merge(symbol, "1m", fresh)

        df_raw = get_cached(symbol, "1m")
        if df_raw.empty:
            send_telegram("❌ فشل جلب البيانات من Binance", chat_id)
            return

        now = datetime.now(timezone.utc)
        df5_full = resample_ohlcv_closed(df_raw, 5)

        if df5_full.empty:
            send_telegram("❌ فشل إعادة العينة", chat_id)
            return

        last_candle_end = df5_full["ts"].iloc[-1] + timedelta(minutes=5)
        if now < last_candle_end:
            df5 = df5_full.iloc[:-1]
        else:
            df5 = df5_full

        if len(df5) < MIN_CANDLES:
            send_telegram(f"⚠️ شموع غير كافية: {len(df5)} (المطلوب {MIN_CANDLES})", chat_id)
            return

        price = df5["close"].iloc[-1]
        candle_ts = df5["ts"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")
        fetch_ts = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        rsi_series = calc_rsi_tv(df5["close"], period=14)
        rsi_val = round(float(rsi_series.iloc[-1]), 2)

        k_series, d_series = calc_stoch_tv(df5["close"], df5["high"], df5["low"])
        stoch_k = round(float(k_series.iloc[-1]), 2)
        stoch_d = round(float(d_series.iloc[-1]), 2)

        macd_line, signal_line, histogram = _calc_macd_full(df5["close"])
        macd_hist_val = round(float(histogram.iloc[-1]), 4)
        macd_line_val = round(float(macd_line.iloc[-1]), 4)
        signal_line_val = round(float(signal_line.iloc[-1]), 4)
        macd_color = "🟢" if macd_hist_val > 0 else "🔴"

        smi_series, smi_sig_series = calc_smi(df5["high"], df5["low"], df5["close"])
        smi_val = round(float(smi_series.iloc[-1]), 2)
        smi_sig = round(float(smi_sig_series.iloc[-1]), 2)

        don_trend = calc_donchian_trend(df5)
        if don_trend:
            don_val = don_trend[-1]
            don_color = "🟢 أخضر (صاعد)" if don_val == 1 else ("🔴 أحمر (هابط)" if don_val == -1 else "⚪ محايد")
        else:
            don_color = "⚪ محايد"

        rsi_zone = "🔴 تشبع بيعي" if rsi_val < 30 else ("🟠 تشبع شرائي" if rsi_val > 70 else "🟡 محايد")
        stoch_zone = "🔴 تشبع بيعي" if stoch_k < 20 else ("🟠 تشبع شرائي" if stoch_k > 80 else "🟡 محايد")
        smi_zone = "🔴 تشبع بيعي" if smi_val <= -40 else ("🟠 تشبع شرائي" if smi_val >= 40 else "🟡 محايد")

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
            f"📉 Stoch K(15,3): <b>{stoch_k}</b> {stoch_zone}\nStoch D(3): <b>{stoch_d}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"⚡ MACD Histogram: {macd_color} <b>{macd_hist_val}</b>\nMACD Line: <b>{macd_line_val}</b>\nSignal Line: <b>{signal_line_val}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"🔵 SMI: <b>{smi_val}</b> {smi_zone}\nSignal: <b>{smi_sig}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📦 شموع الـ5m: {len(df5)} | بيانات الـ1m: {len(df_raw)}",
            chat_id,
        )

    except Exception as e:
        log.error(f"check5 error: {e}")
        send_telegram(f"❌ خطأ في /check5: {e}", chat_id)

def _dispatch_command(txt, chat_id):
    if txt == "/status":
        _cmd_status(chat_id)
    elif txt in ("1", "/today"):
        send_telegram(get_report("today"), chat_id)
    elif txt in ("2", "/yesterday"):
        send_telegram(get_report("yesterday"), chat_id)
    elif txt in ("3", "/week"):
        send_telegram(get_report("week"), chat_id)
    elif txt in ("/سبب_شراء", "/diag_buy"):
        _cmd_cascade_diag(chat_id, "buy")
    elif txt in ("/سبب_بيع", "/diag_sell"):
        _cmd_cascade_diag(chat_id, "sell")
    elif txt in ("/diag_failures", "/أسباب_الفشل"):
        handle_diag_command(chat_id)
    elif txt == "/survivors6":
        _cmd_show_step_survivors(chat_id, step_num=6, signal_type="buy")
    elif txt == "/survivors7":
        _cmd_show_step_survivors(chat_id, step_num=7, signal_type="buy")
    elif txt == "/survivors8":
        _cmd_show_step_survivors(chat_id, step_num=8, signal_type="buy")
    elif txt == "/survivors6_sell":
        _cmd_show_step_survivors(chat_id, step_num=6, signal_type="sell")
    elif txt == "/survivors7_sell":
        _cmd_show_step_survivors(chat_id, step_num=7, signal_type="sell")
    elif txt == "/survivors8_sell":
        _cmd_show_step_survivors(chat_id, step_num=8, signal_type="sell")
    elif txt.startswith("/check5"):
        parts = txt.split()
        symbol = parts[1].upper() if len(parts) > 1 else "BTCUSDT"
        if not symbol.endswith("USDT"):
            symbol += "USDT"
        threading.Thread(target=handle_check5, args=(chat_id, symbol), daemon=True).start()
    elif txt in ("/hard_filters", "/فلاتر_صعبة"):
        handle_hard_filters_command(chat_id, signal_type="buy")
    elif txt in ("/hard_filters_sell", "/فلاتر_صعبة_بيع"):
        handle_hard_filters_command(chat_id, signal_type="sell")
    elif txt in ("/hard_filters_sell", "/فلاتر_صعبة_بيع"):
        handle_hard_filters_command(chat_id, signal_type="sell")


    elif txt == "/scan_now":
        if not fast_prefetch_done.is_set():
            send_telegram("⏳ التحميل لم يكتمل بعد، انتظر.", chat_id)
            return
        else:
            send_telegram("🔄 جاري تشغيل المسح الفوري...", chat_id)
        def do_scan():
            run_cascade_scan()
            run_short_cascade_scan()
            send_telegram("✅ المسح الفوري اكتمل — جرّب /سبب_شراء الآن", chat_id)
        threading.Thread(target=do_scan, daemon=True).start()
        
    elif txt == "/help":
        send_telegram(
            "📋 <b>الأوامر المتاحة:</b>\n"
            "1️⃣ <code>1</code> — إشارات اليوم\n"
            "2️⃣ <code>2</code> — إشارات أمس\n"
            "3️⃣ <code>3</code> — آخر 7 أيام\n"
            "📊 <code>/status</code> — حالة البوت\n"
            "📋 <code>/help</code> — قائمة الأوامر",
            chat_id,
        )
    elif txt == "/debug":
        with ohlcv_cache_lock:
            cache_keys = list(ohlcv_cache.keys())
        with symbols_cache_lock:
            syms = list(symbols_cache)
        sample_sym = syms[0] if syms else None
        sample_info = ""
        if sample_sym:
            raw_1m = get_cached(sample_sym, "1m")
            raw_60m = get_cached(sample_sym, "60m")
            df_base = resample_ohlcv(raw_1m, 9)
            sample_info = (
                f"\n📌 عينة: {sample_sym}"
                f"\n1m candles: {len(raw_1m)}"
                f"\n60m candles: {len(raw_60m)}"
                f"\ndf_base (9m): {len(df_base)}"
                f"\nMIN_CANDLES: {MIN_CANDLES}"
                f"\nيمر؟ {'✅' if len(df_base) >= MIN_CANDLES else '❌'}"
            )
        msg = (
            f"🔧 <b>Debug Info</b>\n"
            f"عملات: {len(syms)}\n"
            f"Cache keys: {len(cache_keys)}\n"
            f"fast_prefetch: {'✅' if fast_prefetch_done.is_set() else '⏳'}\n"
            f"prefetch_done: {'✅' if prefetch_done.is_set() else '⏳'}"
            f"{sample_info}"
        )
        send_telegram(msg, chat_id)
        
# ------------------------------------------
# QUICK CHECK - Steps 7-8 only on saved Step6 survivors
# ------------------------------------------

def run_quick_step78(signal_type="buy"):
    if signal_type == "buy":
        surv_lock = last_complete_lock
        survivors_dict = last_complete_survivors
        step7_fn, step8_fn = step7, step8
    else:
        surv_lock = last_complete_short_lock
        survivors_dict = last_complete_short_survivors
        step7_fn, step8_fn = short_step7, short_step8

    with surv_lock:
        candidates = list(survivors_dict.get(6, []))

    if not candidates:
        return

    resample_cache = {}

    def get_resampled(raw_df, sym, tf, minutes):
        key = (sym, tf, minutes)
        if key not in resample_cache:
            resample_cache[key] = resample_ohlcv(raw_df, minutes)
        return resample_cache[key]

    # إعادة بناء df_triple بأحدث بيانات (raw_base قد يكون تحدّث)
    refreshed = []
    for c in candidates:
        sym = c["sym"]
        triple_api = c["triple_api"]
        raw_triple = get_cached(sym, triple_api)
        if raw_triple.empty:
            continue
        df_triple = get_resampled(raw_triple, sym, triple_api, c["triple_frame"])
        if df_triple.empty or len(df_triple) < MIN_CANDLES:
            continue
        c2 = dict(c)
        c2["df_triple"] = df_triple
        c2["get_resampled"] = get_resampled
        refreshed.append(c2)

    if not refreshed:
        return

    def run_one(c):
        try:
            ok7, _ = step7_fn(c)
            if not ok7:
                return c, False
            ok8, _ = step8_fn(c)
            return c, ok8
        except Exception as e:
            log.error("❌ خطأ في quick_step78 (%s): %s", signal_type, e)
            return c, False

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(run_one, refreshed))

    fired = 0
    for c, ok in results:
        if ok:
            fired += 1
            _fire_signal(c["sym"], c["base_frame"], c["confirm_frame"],
                         c["triple_frame"], c["df_base"], signal_type=signal_type)

    if fired:
        log.info("⚡ Quick check (%s): %d إشارة من %d مرشح محفوظ", signal_type, fired, len(refreshed))


def run_quick_step78_short():
    run_quick_step78(signal_type="sell")


def poll_telegram_commands():
    last_id = 0
    while True:
        try:
            r = get_session().get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates",
                params={"offset": last_id + 1, "timeout": 30},
                timeout=35,
            ).json()
            for upd in r.get("result", []):
                last_id = upd["update_id"]
                txt = upd.get("message", {}).get("text", "").strip()
                chat_id = str(upd.get("message", {}).get("chat", {}).get("id", ""))
                if txt and chat_id:
                    threading.Thread(target=_dispatch_command, args=(txt, chat_id), daemon=True).start()
        except Exception:
            time.sleep(10)

def next_candle_close():
    now = datetime.now(timezone.utc)
    total_seconds = now.minute * 60 + now.second
    min_wait = 999999
    for tf in TIMEFRAME_CHAIN:
        tf_seconds = tf * 60
        remaining = tf_seconds - (total_seconds % tf_seconds)
        if remaining < min_wait:
            min_wait = remaining
    return min_wait + 1
_quick_check_counter = {"n": 0}

def cascade_watcher():
    while True:
        try:
            if fast_prefetch_done.is_set():
                with ohlcv_cache_lock:
                    if len(ohlcv_cache) < 300:  # تأكد الكاش فيه بيانات
                        time.sleep(30)
                        continue
                # ✅ fetch مرة واحدة للاثنين
                with symbols_cache_lock:
                    syms = list(symbols_cache)
                def fetch_fresh(sym):
                    for tf in ["1m", "60m"]:
                        df = get_ohlcv(sym, tf, limit=3)
                        if not df.empty:
                            cache_merge(sym, tf, df)
                with ThreadPoolExecutor(max_workers=30) as executor:
                    executor.map(fetch_fresh, syms)
                                # ⚡ فحص سريع (7-8) على المحفوظين من الخطوة 6 — كل دورة
                qt1 = threading.Thread(target=run_quick_step78, args=("buy",), daemon=True)
                qt2 = threading.Thread(target=run_quick_step78, args=("sell",), daemon=True)
                qt1.start()
                qt2.start()
                qt1.join()
                qt2.join()

                # 🔄 سكان كامل (1-8) — كل 3 دورات فقط لتحديث القائمة المحفوظة
                _quick_check_counter["n"] += 1
                if _quick_check_counter["n"] >= 3:
                    _quick_check_counter["n"] = 0
                    t1 = threading.Thread(target=run_cascade_scan, daemon=True)
                    t2 = threading.Thread(target=run_short_cascade_scan, daemon=True)
                    t1.start()
                    t2.start()
                    t1.join()
                    t2.join()
            time.sleep(next_candle_close())
        except Exception as e:
            log.error("❌ خطأ في cascade_watcher: %s", e)
            time.sleep(5)

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

            top = sorted([t for t in tickers if isinstance(t, dict) and t.get("symbol", "").endswith("USDT")],
                        key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)[:TOP_SYMBOLS_LIMIT]

            with symbols_cache_lock:
                symbols_cache[:] = [t["symbol"] for t in top]
            log.info("✅ عملات: %s — أول 5: %s", len(symbols_cache), symbols_cache[:5])
            if not fast_prefetch_done.is_set():
                threading.Thread(target=prefetch_all, args=(list(symbols_cache),), daemon=True).start()
        except requests.RequestException as exc:
            log.error("update_symbols_loop: %s", exc)
        time.sleep(3600)

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *_):
        pass

def main():
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

    send_telegram("🚀 <b>البوت انطلق — استراتيجية مزدوجة (شراء + بيع)</b>")

    while True:
        try:
            time.sleep(300)
            cleanup_alerted_keys()
            with ohlcv_cache_lock:
                cache_size = len(ohlcv_cache)
            with trades_lock:
                signals_count = len(trades_history)
            log.info("💓 البوت يعمل | كاش: %s مفتاح | إشارات: %s | سريع: %s | كامل: %s",
                    cache_size, signals_count, "✅" if fast_prefetch_done.is_set() else "⏳",
                    "✅" if prefetch_done.is_set() else "⏳")
        except Exception as exc:
            log.error("❌ خطأ في main loop: %s\n%s", exc, traceback.format_exc())
            time.sleep(10)

if __name__ == "__main__":
    main()

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

            top = sorted([t for t in tickers if isinstance(t, dict) and t.get("symbol", "").endswith("USDT")],
                        key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)[:TOP_SYMBOLS_LIMIT]

            with symbols_cache_lock:
                symbols_cache[:] = [t["symbol"] for t in top]
            log.info("✅ عملات: %s — أول 5: %s", len(symbols_cache), symbols_cache[:5])
            if not fast_prefetch_done.is_set():
                threading.Thread(target=prefetch_all, args=(list(symbols_cache),), daemon=True).start()
        except requests.RequestException as exc:
            log.error("update_symbols_loop: %s", exc)
        time.sleep(3600)

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *_):
        pass

def main():
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

    send_telegram("🚀 <b>البوت انطلق — استراتيجية مزدوجة (شراء + بيع)</b>")

    while True:
        try:
            time.sleep(300)
            cleanup_alerted_keys()
            with ohlcv_cache_lock:
                cache_size = len(ohlcv_cache)
            with trades_lock:
                signals_count = len(trades_history)
            log.info("💓 البوت يعمل | كاش: %s مفتاح | إشارات: %s | سريع: %s | كامل: %s",
                    cache_size, signals_count, "✅" if fast_prefetch_done.is_set() else "⏳",
                    "✅" if prefetch_done.is_set() else "⏳")
        except Exception as exc:
            log.error("❌ خطأ في main loop: %s\n%s", exc, traceback.format_exc())
            time.sleep(10)

if __name__ == "__main__":
    main()