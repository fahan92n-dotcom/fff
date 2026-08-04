"""بوت مسح العملات من Binance مع تنبيهات Telegram - نسخة Cascade Pipeline مع استراتيجية مزدوجة (شراء/بيع)."""
import os
import time
import logging
import threading
import sys
import traceback
import json
import gc
import ctypes
import resource
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

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8696456847:AAG06_sYJVIZNjCRwO29OynYFh9GsWYOwXo")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "-1003968771145")

PORT = int(os.environ.get("PORT", "8080"))
ALERT_EXPIRY_HOURS = 4

from binance_data import (
    BINANCE_FUTURES_BASE, BINANCE_SPOT_BASE, BINANCE_BASE, MARKET_MODE,
    TOP_SYMBOLS_LIMIT, CUSTOM_SYMBOLS, TF_MAP,
    CACHE_MAX_CANDLES, API_FETCH_CANDLES, FAST_FETCH_CANDLES,
    UPDATE_BUFFER_SECONDS, UPDATER_30M_INTERVAL_SECONDS,
    symbols_cache, symbols_cache_lock,
    invalid_symbols_cache, invalid_symbols_lock,
    invalid_symbols_reason_cache, invalid_symbols_reason_lock,
    ohlcv_cache, ohlcv_cache_lock,
    fast_prefetch_done, prefetch_done, cache_updated_event,
    get_session, validate_symbols_with_reasons,
    get_ohlcv, get_ohlcv_futures, get_ohlcv_full, get_ohlcv_full_futures,
    cache_merge, get_cached, cleanup_old_symbols_cache,
    prefetch_all, prefetch_all_futures,
    cache_updater_1m, cache_updater_30m, cache_updater_60m,
    cache_updater_1m_futures, cache_updater_30m_futures, cache_updater_60m_futures,
    update_symbols_loop, update_symbols_loop_futures,
    get_last_closed_candle, set_telegram_sender,
)

TRIPLING_PAIRS = [
    (9, 27, 3, "1m", "1m"), (12, 36, 4, "1m", "1m"), (15, 45, 5, "1m", "1m"),
    (18, 54, 6, "1m", "1m"), (21, 63, 7, "1m", "1m"), (24, 72, 8, "1m", "1m"),
    (27, 81, 9, "1m", "1m"), (30, 90, 10, "1m", "1m"), (45, 135, 15, "1m", "1m"),
    (60, 180, 20, "60m", "1m"), (90, 270, 30, "30m", "30m"), (120, 360, 40, "30m", "30m"),
    (150, 450, 50, "30m", "30m"), (180, 540, 60, "60m", "60m"), (210, 630, 70, "60m", "30m"),
    (240, 720, 80, "60m", "30m"),
]
# Source routing is mixed by pair to balance history depth and granularity:
# low triple frames use 1m, medium ranges use 30m, and long confirmation ranges keep 60m.

TIMEFRAME_CHAIN = [9, 12, 15, 18, 21, 24, 27, 30, 45, 60, 90, 120, 150, 180, 210, 240]
NEXT_TF = {TIMEFRAME_CHAIN[i]: TIMEFRAME_CHAIN[i + 1] for i in range(len(TIMEFRAME_CHAIN) - 1)}

# خريطة: base_frame → base_api الصحيح المحدد في TRIPLING_PAIRS لكل فريم.
# تُستخدم في _has_higher_tf_saturation لضمان جلب بيانات الفريم الأعلى
# من مصدره الحقيقي (30m أو 60m) لا من مصدر المرشح نفسه.
# مثال: فريم 90/120/150 → 30m ، فريم 180/210/240 → 60m
TF_TO_API = {p[0]: p[3] for p in TRIPLING_PAIRS}

QUICK_CHECK_INTERVAL_SECONDS = 3

from indicators import (
    WARMUP_EMA, WARMUP_MACD, WARMUP_SMI, WARMUP_RSI, WARMUP_STOCH, WARMUP_DON, MIN_CANDLES,
    DONCHIAN_DLEN,
    _ribbon_cache, _ribbon_cache_lock,
    resample_ohlcv, resample_ohlcv_closed,
    wilder_rma, _calc_macd_hist, _calc_macd_full, _get_macd_window_hours,
    check_macd_line_long, check_macd_line_short, check_macd_red, check_macd_green,
    calc_donchian_trend_pine, _calc_donchian_ribbon_result, check_donchian_trend_ribbon,
    check_ema50_below, check_ema50_above,
    calc_smi, check_smi_oversold, check_smi_overbought, check_ema50_above_since_overbought,
    calc_rsi_tv, calc_stoch_tv,
    check_rsi_touched_oversold, check_rsi_overbought_short,
    check_rsi_not_oversold_recently, check_rsi_not_overbought_recently,
    check_confirm_rsi_not_oversold, check_confirm_rsi_not_overbought,
    check_rsi_closed_oversold, check_rsi_closed_overbought,
    check_rsi_touched_since, check_smi_touched_since,
    check_rsi_stoch, check_rsi_stoch_short,
)

# ------------------------------------------
# Shared State
# ------------------------------------------

alerted_keys = {}
alerted_keys_lock = threading.Lock()
trades_history = deque(maxlen=2000)
trades_lock = threading.Lock()

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

first_scan_notified = False
first_scan_lock = threading.Lock()

step6_ready_since = {}
step6_ready_since_lock = threading.Lock()
step1_ready_since = {}
step1_ready_since_lock = threading.Lock()
step7_ready_since = {}
step7_ready_since_lock = threading.Lock()

step5_entry_time = {}  # (signal_type, sym, base_frame) → timestamp
step5_entry_time_lock = threading.Lock()
STEP5_MAX_WAIT_SECONDS = None  # بدون انتهاء زمني

# الحد الأدنى لنسبة الفريمين حتى يُعدّا "مستقلَّين" ويتعايشا معاً في المرحلة 5.
# مثال: 12m و240m → 240/12 = 20x ≥ 4 → يتعايشان.
#         60m و90m   →  90/60 = 1.5x < 4  → يُحتفظ بالأكبر فقط.
STAGE5_COEXIST_MIN_RATIO = 4.0

def _frames_far_apart(frame1: int, frame2: int) -> bool:
    """هل الفريمان بعيدان بما يكفي للتعايش معاً في المرحلة 5؟"""
    larger, smaller = max(frame1, frame2), min(frame1, frame2)
    return larger / smaller >= STAGE5_COEXIST_MIN_RATIO




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

def get_candidate_key(candidate):
    return (
        candidate["sym"],
        candidate["base_frame"],
        candidate["confirm_frame"],
        candidate["triple_frame"],
    )

def get_signal_key(symbol, base_frame, confirm_frame, triple_frame, signal_type="buy"):
    return (symbol, base_frame, confirm_frame, triple_frame, signal_type)

def _set_ready_since(store, lock, key, ready_ts=None):
    ts = ready_ts or datetime.now(timezone.utc)
    with lock:
        if key not in store:
            store[key] = ts
    return ts

def _get_stage_maps(signal_type):
    if signal_type == "buy":
        return last_complete_survivors, last_complete_lock
    return last_complete_short_survivors, last_complete_short_lock

def _upsert_stage_candidate(survivors_dict, stage_num, candidate):
    key = get_candidate_key(candidate)
    items = list(survivors_dict.get(stage_num, []))
    for idx, existing in enumerate(items):
        if get_candidate_key(existing) == key:
            items[idx] = candidate
            survivors_dict[stage_num] = items
            return
    items.append(candidate)
    survivors_dict[stage_num] = items

def _remove_stage_candidate(survivors_dict, stage_num, candidate_key):
    items = [c for c in survivors_dict.get(stage_num, []) if get_candidate_key(c) != candidate_key]
    survivors_dict[stage_num] = items

def _candidate_keys_in_stages(survivors_dict, stages):
    keys = set()
    for stage_num in stages:
        for candidate in survivors_dict.get(stage_num, []):
            keys.add(get_candidate_key(candidate))
    return keys

def _store_step5_waiters(signal_type, candidates):
    survivors_dict, surv_lock = _get_stage_maps(signal_type)
    now = datetime.now(timezone.utc)
    
    with surv_lock:
        blocked = _candidate_keys_in_stages(survivors_dict, (6, 7))
        
        # استراتيجية جديدة: لكل عملة، يُحتفظ بعدة فريمات إذا كانت بعيدة بما يكفي
        # (نسبة الأكبر / الأصغر >= STAGE5_COEXIST_MIN_RATIO).
        # مثال: 12m و240m يتعايشان. أما 60m و90m فيُحتفظ بالأكبر فقط.
        # stage5_by_symbol: sym → {base_frame: candidate}
        stage5_by_symbol: dict = {}
        
        for candidate in candidates:
            key = get_candidate_key(candidate)
            if key in blocked:
                continue
            
            sym = candidate["sym"]
            base_frame = candidate["base_frame"]
            
            if sym not in stage5_by_symbol:
                stage5_by_symbol[sym] = {base_frame: candidate}
                with step5_entry_time_lock:
                    if (signal_type, sym, base_frame) not in step5_entry_time:
                        step5_entry_time[(signal_type, sym, base_frame)] = now
            else:
                # ابحث عن فريم "قريب" موجود مسبقاً لنفس العملة
                close_frame = None
                for existing_frame in list(stage5_by_symbol[sym].keys()):
                    if not _frames_far_apart(base_frame, existing_frame):
                        close_frame = existing_frame
                        break
                
                if close_frame is not None:
                    # الفريمان قريبان → احتفظ بالأكبر فقط
                    if base_frame > close_frame:
                        with step5_entry_time_lock:
                            step5_entry_time.pop((signal_type, sym, close_frame), None)
                        del stage5_by_symbol[sym][close_frame]
                        stage5_by_symbol[sym][base_frame] = candidate
                        with step5_entry_time_lock:
                            step5_entry_time[(signal_type, sym, base_frame)] = now
                    # else: الموجود أكبر بالفعل، لا تغيير
                else:
                    # الفريمان بعيدان → يتعايشان
                    stage5_by_symbol[sym][base_frame] = candidate
                    with step5_entry_time_lock:
                        if (signal_type, sym, base_frame) not in step5_entry_time:
                            step5_entry_time[(signal_type, sym, base_frame)] = now
        
        # ✅ احذف المرشحات اللي تجاوزت الحد الزمني (إن كان مضبوطاً)
        if STEP5_MAX_WAIT_SECONDS is not None:
            to_remove = []
            with step5_entry_time_lock:
                for (sig_type, sym, base_frame), entry_time in list(step5_entry_time.items()):
                    if sig_type == signal_type:
                        elapsed = (now - entry_time).total_seconds()
                        if elapsed > STEP5_MAX_WAIT_SECONDS:
                            to_remove.append((sym, base_frame))
            for sym, base_frame in to_remove:
                if sym in stage5_by_symbol:
                    stage5_by_symbol[sym].pop(base_frame, None)
                    if not stage5_by_symbol[sym]:
                        stage5_by_symbol.pop(sym, None)
                with step5_entry_time_lock:
                    step5_entry_time.pop((signal_type, sym, base_frame), None)
        
        survivors_dict[5] = [c for frames in stage5_by_symbol.values() for c in frames.values()]

def _promote_candidates(signal_type, from_stage, to_stage, candidates):
    if not candidates:
        return
    survivors_dict, surv_lock = _get_stage_maps(signal_type)
    with surv_lock:
        for candidate in candidates:
            key = get_candidate_key(candidate)
            _remove_stage_candidate(survivors_dict, from_stage, key)
            _upsert_stage_candidate(survivors_dict, to_stage, candidate)

def _set_step8_survivors(signal_type, candidates):
    survivors_dict, surv_lock = _get_stage_maps(signal_type)
    with surv_lock:
        stage8 = {
            get_candidate_key(candidate): candidate
            for candidate in survivors_dict.get(8, [])
        }
        for candidate in candidates:
            stage8[get_candidate_key(candidate)] = candidate
        survivors_dict[8] = list(stage8.values())

def _clear_waiting_candidate(symbol, base_frame, confirm_frame, triple_frame, signal_type="buy"):
    candidate_key = (symbol, base_frame, confirm_frame, triple_frame)
    ready_key = get_signal_key(symbol, base_frame, confirm_frame, triple_frame, signal_type)
    survivors_dict, surv_lock = _get_stage_maps(signal_type)
    with surv_lock:
        for stage_num in (5, 6, 7):
            _remove_stage_candidate(survivors_dict, stage_num, candidate_key)
    with step6_ready_since_lock:
        step6_ready_since.pop(ready_key, None)
    with step1_ready_since_lock:
        step1_ready_since.pop(ready_key, None)
    with step7_ready_since_lock:
        step7_ready_since.pop(ready_key, None)
    with step5_entry_time_lock:
        step5_entry_time.pop((signal_type, symbol, base_frame), None)

def _update_last_complete_step(signal_type, step_num, evaluations):
    if signal_type == "buy":
        stats = last_complete_stats
        results = last_complete_results
        lock = last_complete_lock
    else:
        stats = last_complete_short_stats
        results = last_complete_short_results
        lock = last_complete_short_lock

    now = datetime.now(timezone.utc)
    passed_count = sum(1 for _, ok, _ in evaluations if ok)
    step_results = {}
    for candidate, ok, reason in evaluations:
        step_results[get_candidate_key(candidate)] = {"passed": ok, "reason": reason, "time": now}

    with lock:
        stats[step_num] = {"total": len(evaluations), "passed": passed_count}
        results[step_num] = step_results

def _refresh_waiting_candidate(candidate, get_resampled, need_base=False, need_triple=False):
    refreshed = dict(candidate)
    sym = candidate["sym"]

    if need_base:
        raw_base = get_cached(sym, candidate["base_api"])
        if raw_base.empty:
            return None
        df_base = get_resampled(raw_base, sym, candidate["base_api"], candidate["base_frame"])
        df_confirm = get_resampled(raw_base, sym, candidate["base_api"], candidate["confirm_frame"])
        if df_base.empty or df_confirm.empty or len(df_base) < MIN_CANDLES:
            return None
        refreshed["raw_base"] = raw_base
        refreshed["df_base"] = df_base
        refreshed["df_confirm"] = df_confirm

    if need_triple:
        raw_triple = get_cached(sym, candidate["triple_api"])
        if raw_triple.empty:
            return None
        df_triple = get_resampled(raw_triple, sym, candidate["triple_api"], candidate["triple_frame"])
        if df_triple.empty or len(df_triple) < MIN_CANDLES:
            return None
        refreshed["df_triple"] = df_triple

    refreshed["get_resampled"] = get_resampled
    return refreshed

def _run_step_batch(candidates, step_fn, step_num, signal_label):
    if not candidates:
        return []

    def run_one(candidate, fn=step_fn):
        try:
            return candidate, *fn(candidate)
        except Exception as exc:
            log.error("❌ خطأ في الخطوة %d (%s): %s", step_num, signal_label, exc)
            return candidate, False, str(exc)

    executor = ThreadPoolExecutor(max_workers=20)
    try:
        futures = [executor.submit(run_one, candidate) for candidate in candidates]
        results = []
        try:
            for future in concurrent.futures.as_completed(futures, timeout=120):
                results.append(future.result())
        except concurrent.futures.TimeoutError:
            log.warning("⚠️ بعض المهام لم تكتمل خلال المهلة المحددة في الخطوة %d (%s)", step_num, signal_label)
        return results
    except Exception as exc:
        log.error("❌ خطأ في الخطوة %d (%s): %s", step_num, signal_label, exc)
        return []
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

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

# ربط مُرسل Telegram بطبقة البيانات (prefetch / update_symbols_loop)
set_telegram_sender(send_telegram)

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
        raw_by_tf = {
            "1m": get_cached(sym, "1m"),
            "30m": get_cached(sym, "30m"),
            "60m": get_cached(sym, "60m"),
        }
        
        for base_frame, confirm_frame, triple_frame, base_api, triple_api in TRIPLING_PAIRS:
            total_candidates += 1
            raw_base = raw_by_tf.get(base_api, pd.DataFrame())
            
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
        
        
def _fire_signal(symbol, base_frame, confirm_frame, triple_frame, df, signal_type="buy"):
    key = (symbol, base_frame, confirm_frame, triple_frame, signal_type)
    now = datetime.now(timezone.utc)
    with alerted_keys_lock:
        last = alerted_keys.get(key)
        if last and now - last < timedelta(hours=ALERT_EXPIRY_HOURS):
            return
        alerted_keys[key] = now

    price = float(df["close"].iloc[-1])
    save_signal(symbol, price, base_frame, confirm_frame, triple_frame, signal_type=signal_type)
    _clear_waiting_candidate(symbol, base_frame, confirm_frame, triple_frame, signal_type=signal_type)

    icon = "🟢 شراء (LONG)" if signal_type == "buy" else "🔴 بيع (SHORT)"
    send_telegram(
        f"{icon}\n"
        f"💱 العملة: <b>{symbol}</b>\n"
        f"💰 السعر: <b>{price:.6g}</b>\n"
        f"⏱️ الفريمات: {base_frame}m / {confirm_frame}m / {triple_frame}m\n"
        f"🕐 الوقت: {now.strftime('%H:%M:%S UTC')}"
    )


# ------------------------------------------
# CASCADE PIPELINE - LONG (BUY)
# ------------------------------------------

def step1(c):
    if not check_smi_oversold(c["df_base"]):
        return False, "smi_oversold"
    # ✅ فحص الأولوية: لو فريم أكبر دخل تشبع بيعي → ألغِ هذا المرشح
    # يستخدم TF_TO_API داخلياً لضمان المصدر الصحيح لكل فريم
    if _has_higher_tf_saturation(c, "buy", c["get_resampled"]):
        return False, "active_skip"
    return True, "passed"

def step2(c):
    """✅ MACD الفريم الأساسي - كل الشروط مرتبطة (LONG)"""
    if len(c["df_base"]) < WARMUP_MACD:
        return False, "warmup"
    macd_line, signal_line, histogram = _calc_macd_full(c["df_base"]["close"])

    current_hist = float(histogram.iloc[-1])
    current_macd = float(macd_line.iloc[-1])

    # ✅ الشرط 1: Histogram أحمر (< 0)
    if current_hist >= 0:
        return False, "macd_histogram_not_red"

    # ✅ الشرط 2: الخط الأزرق فوق الهوستقرام
    if current_macd < current_hist:
        return False, "macd_line_not_above_histogram"

    # ✅ الشرط 3: الخط الأزرق ≤ 40% من أقصى ارتفاع (نافذة ديناميكية: 24 ساعة لفريم ≤60 د، 72 ساعة لفريم >60 د)
    window_hours = _get_macd_window_hours(c["base_frame"])
    last_ts = c["df_base"]["ts"].iloc[-1]
    cutoff_ts = last_ts - timedelta(hours=window_hours)
    macd_today = macd_line[c["df_base"]["ts"].values >= np.datetime64(cutoff_ts)]
    if macd_today.empty:
        macd_today = macd_line
    max_window = float(macd_today.max())
    threshold = max_window * 0.40

    if current_macd > threshold:
        return False, "macd_line_exceeds_40_percent"

    return True, "passed"

def step3(c):
    key = (c["sym"], c["base_api"], c["base_frame"])
    if not check_donchian_trend_ribbon(c["df_base"], "green", cache_key=key):
        return False, "donchian_base"
    return True, "passed"

def step4(c):
    key = (c["sym"], c["base_api"], c["confirm_frame"])
    if not check_donchian_trend_ribbon(c["df_confirm"], "green", cache_key=key):
        return False, "donchian_confirm"
    return True, "passed"

def step5(c):
    if not check_macd_green(c["df_confirm"]):
        return False, "macd_confirm"
    return True, "passed"

def step6(c):
    if not check_ema50_below(c["df_base"]):
        return False, "ema50"
    if not check_confirm_rsi_not_oversold(c["df_confirm"], lookback=30, threshold=30):
        return False, "rsi_confirm_recent"
    return True, "passed"

def step7(c):
    key = (c["sym"], c["triple_api"], c["triple_frame"])
    if not check_donchian_trend_ribbon(c["df_triple"], "red", cache_key=key):
        return False, "donchian_triple"
    return True, "passed"

def step8(c):
    since_ts = get_step1_ready_since(c["sym"], c["base_frame"], c["confirm_frame"], c["triple_frame"], "buy")
    if not check_smi_touched_since(c["df_triple"], since_ts, threshold=-40, direction="long"):
        return False, "smi_touch_since_ready"
    if not check_rsi_touched_since(c["df_triple"], since_ts, threshold=35, direction="long"):
        return False, "rsi_touch_since_ready"
    if not check_rsi_stoch(c["df_triple"], since_ts, max_gap=3):
        return False, "rsi_stoch"
    return True, "passed"

steps = [step1, step2, step3, step4, step5, step6, step7, step8]

# ------------------------------------------
# CASCADE PIPELINE - SHORT (SELL)
# ------------------------------------------

def short_step1(c):
    if not check_smi_overbought(c["df_base"], threshold=40):
        return False, "smi_overbought"
    # ✅ فحص الأولوية: لو فريم أكبر دخل تشبع شرائي → ألغِ هذا المرشح
    # يستخدم TF_TO_API داخلياً لضمان المصدر الصحيح لكل فريم
    if _has_higher_tf_saturation(c, "sell", c["get_resampled"]):
        return False, "active_skip"
    return True, "passed"


def short_step2(c):
    """✅ MACD الفريم الأساسي - كل الشروط مرتبطة (SHORT)"""
    if len(c["df_base"]) < WARMUP_MACD:
        return False, "warmup"
    macd_line, signal_line, histogram = _calc_macd_full(c["df_base"]["close"])

    current_hist = float(histogram.iloc[-1])
    current_macd = float(macd_line.iloc[-1])

    # ✅ الشرط 1: Histogram أخضر (> 0)
    if current_hist <= 0:
        return False, "macd_histogram_not_green"

    # ✅ الشرط 2: الخط الأزرق تحت الهوستقرام
    if current_macd > current_hist:
        return False, "macd_line_not_below_histogram"

    # ✅ الشرط 3: الخط الأزرق ≥ 40% من أدنى مستوى (نافذة ديناميكية: 24 ساعة لفريم ≤60 د، 72 ساعة لفريم >60 د)
    window_hours = _get_macd_window_hours(c["base_frame"])
    last_ts = c["df_base"]["ts"].iloc[-1]
    cutoff_ts = last_ts - timedelta(hours=window_hours)
    macd_today = macd_line[c["df_base"]["ts"].values >= np.datetime64(cutoff_ts)]
    if macd_today.empty:
        macd_today = macd_line
    min_window = float(macd_today.min())
    threshold = min_window * 0.40

    if current_macd < threshold:
        return False, "macd_line_below_40_percent"

    return True, "passed"

def short_step3(c):
    key = (c["sym"], c["base_api"], c["base_frame"])
    if not check_donchian_trend_ribbon(c["df_base"], "red", cache_key=key):
        return False, "donchian_base_red"
    return True, "passed"

def short_step4(c):
    key = (c["sym"], c["base_api"], c["confirm_frame"])
    if not check_donchian_trend_ribbon(c["df_confirm"], "red", cache_key=key):
        return False, "donchian_confirm_red"
    return True, "passed"

def short_step5(c):
    if not check_macd_red(c["df_confirm"]):
        return False, "macd_confirm_red"
    return True, "passed"

def short_step6(c):
    if not check_ema50_above(c["df_base"]):
        return False, "ema50_above"
    if not check_confirm_rsi_not_overbought(c["df_confirm"], lookback=30, threshold=70):
        return False, "rsi_confirm_recent_over"
    return True, "passed"
    
def short_step7(c):
    key = (c["sym"], c["triple_api"], c["triple_frame"])
    if not check_donchian_trend_ribbon(c["df_triple"], "green", cache_key=key):
        return False, "donchian_triple_green"
    return True, "passed"

def short_step8(c):
    since_ts = get_step1_ready_since(c["sym"], c["base_frame"], c["confirm_frame"], c["triple_frame"], "sell")
    if not check_smi_touched_since(c["df_triple"], since_ts, threshold=40, direction="short"):
        return False, "smi_touch_since_ready_short"
    if not check_rsi_touched_since(c["df_triple"], since_ts, threshold=65, direction="short"):
        return False, "rsi_touch_since_ready_short"
    if not check_rsi_stoch_short(c["df_triple"], since_ts, max_gap=3):
        return False, "rsi_stoch_short"
    return True, "passed"

short_steps = [short_step1, short_step2, short_step3, short_step4,
               short_step5, short_step6, short_step7, short_step8]

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
        for i in range(1, 6):
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
        raw_by_tf = {
            "1m": get_cached(sym, "1m"),
            "30m": get_cached(sym, "30m"),
            "60m": get_cached(sym, "60m"),
        }

        for base_frame, confirm_frame, triple_frame, base_api, triple_api in TRIPLING_PAIRS:
            raw_base = raw_by_tf.get(base_api, pd.DataFrame())
            raw_triple = raw_by_tf.get(triple_api, pd.DataFrame())

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

    for step_num, step_fn in enumerate(steps[:5], start=1):
        if not candidates:
            log.info("⏸️ انقطعت المعالجة في الخطوة %d (LONG)", step_num)
            break

        def run_one(c, fn=step_fn):
            try:
                return c, *fn(c)
            except Exception as e:
                log.error("❌ خطأ في الخطوة %d (LONG): %s", step_num, e)
                return c, False, str(e)

        results = []
        step_error = False
        executor = ThreadPoolExecutor(max_workers=15)
        try:
            futures = [executor.submit(run_one, candidate) for candidate in candidates]
            try:
                for future in concurrent.futures.as_completed(futures, timeout=120):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        log.error("❌ خطأ: %s", e)
            except concurrent.futures.TimeoutError:
                log.warning("⚠️ بعض المهام لم تكتمل خلال المهلة المحددة في الخطوة %d (LONG)", step_num)
        except Exception as e:
            log.error("❌ خطأ في الخطوة %d (LONG): %s", step_num, e)
            step_error = True
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if step_error:
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
                    if step_num == 1:
                        ready_key = get_signal_key(
                            c["sym"],
                            c["base_frame"],
                            c["confirm_frame"],
                            c["triple_frame"],
                            "buy",
                        )
                        _set_ready_since(step1_ready_since, step1_ready_since_lock, ready_key)
                    passed.append(c)
    
                log.info("📍 خطوة %d (LONG): %d/%d نجحوا", step_num, len(passed), len(results))
        step_survivors[step_num] = passed

        candidates = passed
        
    if cascade_stats.get(1, {}).get("total", 0) > 0:
        _store_step5_waiters("buy", step_survivors.get(5, []))
        with last_complete_lock, cascade_stats_lock, cascade_results_lock:
            for i in range(1, 5):
                last_complete_stats[i] = dict(cascade_stats.get(i, {}))
                last_complete_results[i] = dict(cascade_results.get(i, {}))
                last_complete_survivors[i] = list(step_survivors.get(i, []))
            last_complete_stats[5] = dict(cascade_stats.get(5, {}))
            last_complete_results[5] = dict(cascade_results.get(5, {}))
        with last_complete_scan_time_lock:
            last_complete_scan_time["buy"] = datetime.now(timezone.utc)

    log.info("🪜 مرشحو LONG المحفوظون بعد الخطوة 5: %d", len(step_survivors.get(5, [])))

    resample_cache.clear()
    with _ribbon_cache_lock:
        _ribbon_cache.clear()
                    

def run_short_cascade_scan():
    with symbols_cache_lock:
        symbols = list(symbols_cache)
    if not symbols:
        return

    with short_cascade_stats_lock, short_cascade_results_lock:
        for i in range(1, 6):
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
        raw_by_tf = {
            "1m": get_cached(sym, "1m"),
            "30m": get_cached(sym, "30m"),
            "60m": get_cached(sym, "60m"),
        }

        for base_frame, confirm_frame, triple_frame, base_api, triple_api in TRIPLING_PAIRS:
            raw_base = raw_by_tf.get(base_api, pd.DataFrame())
            raw_triple = raw_by_tf.get(triple_api, pd.DataFrame())

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

    for step_num, step_fn in enumerate(short_steps[:5], start=1):
        if not candidates:
            log.info("⏸️  انقطعت المعالجة في الخطوة %d (SHORT)", step_num)
            break

        def run_one(c, fn=step_fn):
            try:
                return c, *fn(c)
            except Exception as e:
                log.error("❌ خطأ في الخطوة %d (SHORT): %s", step_num, e)
                return c, False, str(e)

        results = []
        step_error = False
        executor = ThreadPoolExecutor(max_workers=15)
        try:
            futures = [executor.submit(run_one, candidate) for candidate in candidates]
            try:
                for future in concurrent.futures.as_completed(futures, timeout=120):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        log.error("❌ خطأ: %s", e)
            except concurrent.futures.TimeoutError:
                log.warning("⚠️ بعض المهام لم تكتمل خلال المهلة المحددة في الخطوة %d (SHORT)", step_num)
        except Exception as e:
            log.error("❌ خطأ في الخطوة %d (SHORT): %s", step_num, e)
            step_error = True
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if step_error:
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
                        if step_num == 1:
                            ready_key = get_signal_key(
                                c["sym"],
                                c["base_frame"],
                                c["confirm_frame"],
                                c["triple_frame"],
                                "sell",
                            )
                            _set_ready_since(step1_ready_since, step1_ready_since_lock, ready_key)
                        passed.append(c)

            log.info("📍 خطوة %d (SHORT): %d/%d نجحوا", step_num, len(passed), len(results))
        else:
            log.warning("⚠️  لا توجد نتائج في الخطوة %d", step_num)

                
        short_step_survivors[step_num] = passed

        candidates = passed

    # خارج حلقة for
    if short_cascade_stats.get(1, {}).get("total", 0) > 0:
        _store_step5_waiters("sell", short_step_survivors.get(5, []))
        with last_complete_short_lock, short_cascade_stats_lock, short_cascade_results_lock:
            for i in range(1, 5):
                last_complete_short_stats[i] = dict(short_cascade_stats.get(i, {}))
                last_complete_short_results[i] = dict(short_cascade_results.get(i, {}))
                last_complete_short_survivors[i] = list(short_step_survivors.get(i, []))
            last_complete_short_stats[5] = dict(short_cascade_stats.get(5, {}))
            last_complete_short_results[5] = dict(short_cascade_results.get(5, {}))
        with last_complete_scan_time_lock:
            last_complete_scan_time["sell"] = datetime.now(timezone.utc)

    log.info("🪜 مرشحو SHORT المحفوظون بعد الخطوة 5: %d", len(short_step_survivors.get(5, [])))

    resample_cache.clear()
    with _ribbon_cache_lock:
        _ribbon_cache.clear()

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
    """عرض العملات الناجحة حتى خطوة معينة"""
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
    lines = [
        f"{icon} <b>الناجحون حتى الخطوة {step_num} ({len(survivors)} عملات)</b>",
        "━" * 30
    ]

    for c in survivors:
        lines.append(
            f"• <b>{c['sym']}</b>\n"
            f"├─ فريم أساسي: {c['base_frame']}m\n"
            f"├─ فريم تأكيد: {c['confirm_frame']}m\n"
            f"└─ فريم تثليث: {c['triple_frame']}m"
        )

    msg = "\n".join(lines)

    for i in range(0, len(msg), 4000):
        send_telegram(msg[i:i + 4000], chat_id)


def _cmd_status(chat_id):
    """عرض حالة البوت الحالية"""
    with ohlcv_cache_lock:
        cache_size = len(ohlcv_cache)
    with trades_lock:
        signals_count = len(trades_history)

    msg = (
        f"<b>📊 حالة البوت</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🔄 الكاش: <b>{cache_size}</b> مفتاح\n"
        f"📈 الإشارات: <b>{signals_count}</b>\n"
        f"⚡ التحميل السريع: {'✅ اكتمل' if fast_prefetch_done.is_set() else '⏳ قيد التحميل'}\n"
        f"📦 التحميل الكامل: {'✅ اكتمل' if prefetch_done.is_set() else '⏳ قيد التحميل'}"
    )
    send_telegram(msg, chat_id)


def handle_check5(chat_id, symbol="BTCUSDT"):
    send_telegram(f"🔄 جاري جلب آخر إغلاق لـ {symbol} — فريم 1 دقيقة (الأمر /check5 يستخدم 1m الآن)...", chat_id)
    try:
        candle = get_last_closed_candle(symbol, "1m")
        
        if not candle:
            send_telegram("❌ فشل جلب البيانات من Binance", chat_id)
            return
        
        price = candle["close"]
        ts = candle["timestamp"]
        
        # جلب البيانات الكاملة للمؤشرات
        fresh = get_ohlcv_full(symbol, "1m", target=3000)
        if not fresh.empty:
            cache_merge(symbol, "1m", fresh)
        
        df_raw = get_cached(symbol, "1m")
        if df_raw.empty:
            send_telegram("❌ فشل جلب البيانات من الكاش", chat_id)
            return
        
        # ✅ احذف الشمعة الأخيرة لو لسه ما اتقفلتش (عشان المؤشرات تتحسب وقت الإغلاق تماماً)
        now_utc = datetime.now(timezone.utc)
        last_candle_ts = df_raw["ts"].iloc[-1]
        candle_close_time = last_candle_ts + pd.Timedelta(minutes=1)
        if now_utc < candle_close_time:
            df_raw = df_raw.iloc[:-1].reset_index(drop=True)

        if df_raw.empty:
            send_telegram("❌ لا توجد شمعة مغلقة كافية بعد", chat_id)
            return

        ts = df_raw["ts"].iloc[-1]
        price = float(df_raw["close"].iloc[-1])

        # حساب المؤشرات من البيانات الخام
        if len(df_raw) >= 50:
            rsi_series = calc_rsi_tv(df_raw["close"], period=14)
            rsi_val = round(float(rsi_series.iloc[-1]), 2)
            
            k_series, d_series = calc_stoch_tv(df_raw["close"], df_raw["high"], df_raw["low"])
            stoch_k = round(float(k_series.iloc[-1]), 2)
            stoch_d = round(float(d_series.iloc[-1]), 2)
            
            macd_line, signal_line, histogram = _calc_macd_full(df_raw["close"])
            macd_hist_val = round(float(histogram.iloc[-1]), 4)
            macd_line_val = round(float(macd_line.iloc[-1]), 4)
            signal_line_val = round(float(signal_line.iloc[-1]), 4)
            macd_color = "🟢" if macd_hist_val > 0 else "🔴"
            smi_series, ema_signal, smi_signal = calc_smi(df_raw["high"], df_raw["low"], df_raw["close"])
            smi_val = round(float(smi_series.iloc[-1]), 2)
            smi_sig = round(float(ema_signal.iloc[-1]), 2)
            
            # ✅ أضف هنا: حساب الاتجاهات (سطرين جدد)
            rsi_trend = "📈 صاعد" if rsi_val > 50 else "📉 هابط"
            macd_trend = "🟢 أخضر" if macd_hist_val > 0 else "🔴 أحمر"
            
            rsi_zone = "🔴 تشبع بيعي" if rsi_val < 30 else ("🟠 تشبع شرائي" if rsi_val > 70 else "🟡 محايد")
            stoch_zone = "🔴 تشبع بيعي" if stoch_k < 20 else ("🟠 تشبع شرائي" if stoch_k > 80 else "🟡 محايد")
            smi_zone = "🔴 تشبع بيعي" if smi_val <= -40 else ("🟠 تشبع شرائي" if smi_val >= 40 else "🟡 محايد")
            
            send_telegram(
                f"📊 <b>{symbol} — فريم 1 دقيقة</b>\n"
                f"🕯️ الشمعة المغلقة: {ts.strftime('%H:%M UTC (%Y-%m-%d)')}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"💰 السعر: <b>${price:.2f}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📈 RSI (14): <b>{rsi_val}</b> {rsi_trend} {rsi_zone}\n"
                f"📉 Stoch K(15,3): <b>{stoch_k}</b> {stoch_zone}\nStoch D(3): <b>{stoch_d}</b>\n"
                f"⚡ MACD: {macd_trend} | Histogram: {macd_color} <b>{macd_hist_val}</b>\nMACD Line: <b>{macd_line_val}</b>\nSignal: <b>{signal_line_val}</b>\n"
                f"🔵 SMI: <b>{smi_val}</b> {smi_zone}\nSignal: <b>{smi_sig}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📦 شموع الـ1m: {len(df_raw)}",
                chat_id,
            )

    except Exception as e:
        log.error(f"check5 error: {e}")
        send_telegram(f"❌ خطأ: {e}", chat_id)

# ------------------------------------------
# تحديث الدوال - التحقق من الشروط 2-5
# ------------------------------------------

def _has_higher_tf_saturation(candidate, signal_type, get_resampled):
    """
    يتحقق من أن الفريم الذي يليه مباشرة في TIMEFRAME_CHAIN فقط (وليس كل الفريمات الأعلى)
    دخل تشبعًا. يستخدم API المصدر الصحيح لهذا الفريم (من TF_TO_API) بدلاً من مصدر المرشح دائمًا.
    مثال: 24 تُفحص فقط ضد 27 (وليس 30، 45، ... إلخ).
    """
    sym = candidate["sym"]
    base_frame = candidate["base_frame"]

    higher_tf = NEXT_TF.get(base_frame)
    if higher_tf is None:
        return False  # لا يوجد فريم أعلى تالٍ في السلسلة

    native_api = TF_TO_API.get(higher_tf, candidate["base_api"])
    raw_native = get_cached(sym, native_api)
    if raw_native.empty:
        return False

    df_higher = get_resampled(raw_native, sym, native_api, higher_tf)
    if df_higher.empty:
        return False

    if signal_type == "buy":
        return check_smi_oversold(df_higher)
    else:
        return check_smi_overbought(df_higher, threshold=40)


def _refresh_and_validate_step5(candidate, get_resampled):
    """
    تحديث البيانات وإعادة فحص الشروط 2-5 قبل step6 (LONG)
    ✅ إذا خرج من التشبع (-40) ولم يجي دخول → احذف المرشح فوراً
    """
    sym = candidate["sym"]
    
    # تحديث البيانات
    raw_base = get_cached(sym, candidate["base_api"])
    if raw_base.empty:
        return None
    
    df_base = get_resampled(raw_base, sym, candidate["base_api"], candidate["base_frame"])
    df_confirm = get_resampled(raw_base, sym, candidate["base_api"], candidate["confirm_frame"])
    
    if df_base.empty or df_confirm.empty or len(df_base) < MIN_CANDLES:
        return None
    
    # ✅ فحص جديد: هل خرج من التشبع SMI؟
    smi, _, _ = calc_smi(df_base["high"], df_base["low"], df_base["close"])
    current_smi = float(smi.iloc[-1])
    
    # لو الـ SMI الحالي > -40 (خرج من التشبع) → احذف فوراً
    # لأن الفرصة انتهت بدون دخول
    if current_smi > -40:
        return None  # ❌ خرج من التشبع بدون دخول = انتهت الفرصة

    # ✅ فحص الأولوية: هل يوجد فريم أكبر دخل تشبعًا الآن؟ إن نعم، ألغِ الفريم الأصغر فورًا
    candidate["raw_base"] = raw_base
    if _has_higher_tf_saturation(candidate, "buy", get_resampled):
        return None  # ❌ فريم أكبر دخل تشبع بيعي = الفريم الأصغر ملغى

    # ✅ الشرط 2: MACD أحمر + MACD Line منخفض
    if not check_macd_red(df_base) or not check_macd_line_long(df_base, base_frame=candidate["base_frame"]):
        return None
    
    # ✅ الشرط 3: Donchian Base أخضر
    key3 = (sym, candidate["base_api"], candidate["base_frame"])
    if not check_donchian_trend_ribbon(df_base, "green", cache_key=key3):
        return None
    
    # ✅ الشرط 4: Donchian Confirm أخضر
    key4 = (sym, candidate["base_api"], candidate["confirm_frame"])
    if not check_donchian_trend_ribbon(df_confirm, "green", cache_key=key4):
        return None
    
    # ✅ الشرط 5: MACD Confirm أخضر
    if not check_macd_green(df_confirm):
        return None
    
    # ✅ كل شيء تمام، حدّث البيانات
    candidate["df_base"] = df_base
    candidate["df_confirm"] = df_confirm
    candidate["get_resampled"] = get_resampled
    
    return candidate


def _refresh_and_validate_step5_short(candidate, get_resampled):
    """
    تحديث البيانات وإعادة فحص الشروط 2-5 قبل step6 (SHORT)
    ✅ إذا خرج من التشبع (+40) ولم يجي دخول → احذف المرشح فوراً
    """
    sym = candidate["sym"]
    
    # تحديث البيانات
    raw_base = get_cached(sym, candidate["base_api"])
    if raw_base.empty:
        return None
    
    df_base = get_resampled(raw_base, sym, candidate["base_api"], candidate["base_frame"])
    df_confirm = get_resampled(raw_base, sym, candidate["base_api"], candidate["confirm_frame"])
    
    if df_base.empty or df_confirm.empty or len(df_base) < MIN_CANDLES:
        return None
    
    # ✅ فحص جديد: هل خرج من التشبع SMI؟
    smi, _, _ = calc_smi(df_base["high"], df_base["low"], df_base["close"])
    current_smi = float(smi.iloc[-1])
    
    # لو الـ SMI الحالي < 40 (خرج من التشبع) → احذف فوراً
    # لأن الفرصة انتهت بدون دخول
    if current_smi < 40:
        return None  # ❌ خرج من التشبع بدون دخول = انتهت الفرصة

    # ✅ فحص الأولوية: هل يوجد فريم أكبر دخل تشبعًا شرائيًا الآن؟ إن نعم، ألغِ الفريم الأصغر فورًا
    candidate["raw_base"] = raw_base
    if _has_higher_tf_saturation(candidate, "sell", get_resampled):
        return None  # ❌ فريم أكبر دخل تشبع شرائي = الفريم الأصغر ملغى

    # ✅ الشرط 2: MACD أخضر + MACD Line مرتفع
    if not check_macd_green(df_base) or not check_macd_line_short(df_base, base_frame=candidate["base_frame"]):
        return None
    
    # ✅ الشرط 3: Donchian Base أحمر
    key3 = (sym, candidate["base_api"], candidate["base_frame"])
    if not check_donchian_trend_ribbon(df_base, "red", cache_key=key3):
        return None
    
    # ✅ الشرط 4: Donchian Confirm أحمر
    key4 = (sym, candidate["base_api"], candidate["confirm_frame"])
    if not check_donchian_trend_ribbon(df_confirm, "red", cache_key=key4):
        return None
    
    # ✅ الشرط 5: MACD Confirm أحمر
    if not check_macd_red(df_confirm):
        return None
    
    # ✅ كل شيء تمام، حدّث البيانات
    candidate["df_base"] = df_base
    candidate["df_confirm"] = df_confirm
    candidate["get_resampled"] = get_resampled
    
    return candidate
        
# ------------------------------------------
# QUICK CHECK - Steps 6-8 on saved Step5/6/7 survivors
# ------------------------------------------

def quick_check_watcher():
    """يفحص الخطوات 6 و7 و8 كل 3 ثوانٍ على الناجحين المحفوظين تراتبياً"""
    while True:
        time.sleep(QUICK_CHECK_INTERVAL_SECONDS)
        # مسح كاش Donchian Ribbon لإعادة حساب القيم من جديد في كل دورة
        # (القيم القديمة قد تعكس شمعة سابقة وتُفسد نتائج step3/step4/step7)
        with _ribbon_cache_lock:
            _ribbon_cache.clear()
        try:
            if fast_prefetch_done.is_set():
                with last_complete_lock:
                    buy_stage5 = list(last_complete_survivors.get(5, []))
                    buy_stage6 = list(last_complete_survivors.get(6, []))
                    buy_stage7 = list(last_complete_survivors.get(7, []))
                with last_complete_short_lock:
                    sell_stage5 = list(last_complete_short_survivors.get(5, []))
                    sell_stage6 = list(last_complete_short_survivors.get(6, []))
                    sell_stage7 = list(last_complete_short_survivors.get(7, []))

                # جلب البيانات الطازة
                refresh_items = {
                    (candidate["sym"], candidate["base_api"])
                    for candidate in (
                        buy_stage5 + buy_stage6 + buy_stage7 +
                        sell_stage5 + sell_stage6 + sell_stage7
                    )
                }

                # جلب بيانات الفريم التالي مباشرة فقط (NEXT_TF)
                # _has_higher_tf_saturation فعليًا بنسختها الحالية
                for candidate in (
                    buy_stage5 + buy_stage6 + buy_stage7 +
                    sell_stage5 + sell_stage6 + sell_stage7
                ):
                    sym = candidate["sym"]
                    base_frame = candidate["base_frame"]
                    higher_tf = NEXT_TF.get(base_frame)
                    if higher_tf is None:
                        continue
                    refresh_items.add((sym, TF_TO_API.get(higher_tf, candidate["base_api"])))

                def fetch_tf(item):
                    sym, tf = item
                    # إصلاح: استخدام الدالة الصحيحة حسب MARKET_MODE لتجنب خلط بيانات Spot/Futures
                    fetch_fn = get_ohlcv_futures if MARKET_MODE == "futures" else get_ohlcv
                    df = fetch_fn(sym, tf, limit=3)
                    if not df.empty:
                        cache_merge(sym, tf, df)

                if refresh_items:
                    with ThreadPoolExecutor(max_workers=20) as executor:
                        executor.map(fetch_tf, refresh_items)

                # ✅ إعادة تحقق من الشروط 2-5 قبل الذهاب للخطوة 6
                resample_cache = {}
                def get_resampled(raw_df, sym, tf, minutes):
                    key = (sym, tf, minutes)
                    if key not in resample_cache:
                        resample_cache[key] = resample_ohlcv(raw_df, minutes)
                    return resample_cache[key]

                # ============ LONG ============
                # التحقق من stage 5
                validated_buy_stage5 = []
                for candidate in buy_stage5:
                    refreshed = _refresh_and_validate_step5(candidate, get_resampled)
                    if refreshed:
                        validated_buy_stage5.append(refreshed)
                    else:
                        candidate_key = get_candidate_key(candidate)
                        with last_complete_lock:
                            _remove_stage_candidate(last_complete_survivors, 5, candidate_key)

                # stage 5 -> 6
                if validated_buy_stage5:
                    step6_results_batch = _run_step_batch(validated_buy_stage5, step6, 6, "LONG")
                    _update_last_complete_step("buy", 6, step6_results_batch)
                    step6_passed = [candidate for candidate, ok, _ in step6_results_batch if ok]
                    if step6_passed:
                        now_ts = datetime.now(timezone.utc)
                        for candidate in step6_passed:
                            ready_key = get_signal_key(
                                candidate["sym"],
                                candidate["base_frame"],
                                candidate["confirm_frame"],
                                candidate["triple_frame"],
                                "buy",
                            )
                            _set_ready_since(step6_ready_since, step6_ready_since_lock, ready_key, now_ts)
                        _promote_candidates("buy", 5, 6, step6_passed)

                # stage 6 -> 7
                with last_complete_lock:
                    step6_queue = list(last_complete_survivors.get(6, []))

                if step6_queue:
                    refreshed_step6 = []
                    for candidate in step6_queue:
                        candidate2 = _refresh_waiting_candidate(candidate, get_resampled, need_triple=True)
                        if candidate2 is not None:
                            refreshed_step6.append(candidate2)
                        else:
                            candidate_key = get_candidate_key(candidate)
                            with last_complete_lock:
                                _remove_stage_candidate(last_complete_survivors, 6, candidate_key)

                    if refreshed_step6:
                        # ✅ فحص الأولوية: أزل أي مرشح يوجد فريم أكبر منه دخل تشبع الآن (Stage 6→7)
                        filtered_step6 = []
                        for c in refreshed_step6:
                            if _has_higher_tf_saturation(c, "buy", get_resampled):
                                candidate_key = get_candidate_key(c)
                                with last_complete_lock:
                                    _remove_stage_candidate(last_complete_survivors, 6, candidate_key)
                            else:
                                filtered_step6.append(c)
                        refreshed_step6 = filtered_step6

                        step7_results_batch = _run_step_batch(refreshed_step6, step7, 7, "LONG")
                        _update_last_complete_step("buy", 7, step7_results_batch)
                        step7_passed = [candidate for candidate, ok, _ in step7_results_batch if ok]
                        if step7_passed:
                            now_ts = datetime.now(timezone.utc)
                            for candidate in step7_passed:
                                ready_key = get_signal_key(
                                    candidate["sym"],
                                    candidate["base_frame"],
                                    candidate["confirm_frame"],
                                    candidate["triple_frame"],
                                    "buy",
                                )
                                _set_ready_since(step7_ready_since, step7_ready_since_lock, ready_key, now_ts)
                            _promote_candidates("buy", 6, 7, step7_passed)

                # stage 7 -> 8 (gated)
                with last_complete_lock:
                    step7_queue = list(last_complete_survivors.get(7, []))

                refreshed_step7 = []
                if step7_queue:
                    for candidate in step7_queue:
                        candidate2 = _refresh_waiting_candidate(candidate, get_resampled, need_triple=True)
                        if candidate2 is not None:
                            refreshed_step7.append(candidate2)
                        else:
                            candidate_key = get_candidate_key(candidate)
                            with last_complete_lock:
                                _remove_stage_candidate(last_complete_survivors, 7, candidate_key)

                if refreshed_step7:
                    # ✅ حارس أخير: أزل أي مرشح يوجد فريم أكبر منه دخل تشبع (Stage 7→8، قبل الإطلاق)
                    final_step7 = []
                    for c in refreshed_step7:
                        if _has_higher_tf_saturation(c, "buy", get_resampled):
                            candidate_key = get_candidate_key(c)
                            with last_complete_lock:
                                _remove_stage_candidate(last_complete_survivors, 7, candidate_key)
                        else:
                            final_step7.append(c)
                    refreshed_step7 = final_step7

                    step8_results_batch = _run_step_batch(refreshed_step7, step8, 8, "LONG")
                    _update_last_complete_step("buy", 8, step8_results_batch)
                    step8_passed = [candidate for candidate, ok, _ in step8_results_batch if ok]

                    if step8_passed:
                        _set_step8_survivors("buy", step8_passed)
                        for candidate in step8_passed:
                            _fire_signal(
                                candidate["sym"],
                                candidate["base_frame"],
                                candidate["confirm_frame"],
                                candidate["triple_frame"],
                                candidate["df_base"],
                                signal_type="buy",
                            )
                        log.info(
                            "⚡ Quick check (LONG): %d إشارة من %d مرشح محفوظ",
                            len(step8_passed),
                            len(refreshed_step7),
                        )

                # ============ SHORT ============
                # التحقق من stage 5
                validated_sell_stage5 = []
                for candidate in sell_stage5:
                    refreshed = _refresh_and_validate_step5_short(candidate, get_resampled)
                    if refreshed:
                        validated_sell_stage5.append(refreshed)
                    else:
                        candidate_key = get_candidate_key(candidate)
                        with last_complete_short_lock:
                            _remove_stage_candidate(last_complete_short_survivors, 5, candidate_key)

                # stage 5 -> 6
                if validated_sell_stage5:
                    step6_results_batch = _run_step_batch(validated_sell_stage5, short_step6, 6, "SHORT")
                    _update_last_complete_step("sell", 6, step6_results_batch)
                    step6_passed = [candidate for candidate, ok, _ in step6_results_batch if ok]
                    if step6_passed:
                        now_ts = datetime.now(timezone.utc)
                        for candidate in step6_passed:
                            ready_key = get_signal_key(
                                candidate["sym"],
                                candidate["base_frame"],
                                candidate["confirm_frame"],
                                candidate["triple_frame"],
                                "sell",
                            )
                            _set_ready_since(step6_ready_since, step6_ready_since_lock, ready_key, now_ts)
                        _promote_candidates("sell", 5, 6, step6_passed)

                # stage 6 -> 7
                with last_complete_short_lock:
                    step6_queue = list(last_complete_short_survivors.get(6, []))

                if step6_queue:
                    refreshed_step6 = []
                    for candidate in step6_queue:
                        candidate2 = _refresh_waiting_candidate(candidate, get_resampled, need_triple=True)
                        if candidate2 is not None:
                            refreshed_step6.append(candidate2)
                        else:
                            candidate_key = get_candidate_key(candidate)
                            with last_complete_short_lock:
                                _remove_stage_candidate(last_complete_short_survivors, 6, candidate_key)

                    if refreshed_step6:
                        # ✅ فحص الأولوية: أزل أي مرشح يوجد فريم أكبر منه دخل تشبع الآن (Stage 6→7)
                        filtered_step6 = []
                        for c in refreshed_step6:
                            if _has_higher_tf_saturation(c, "sell", get_resampled):
                                candidate_key = get_candidate_key(c)
                                with last_complete_short_lock:
                                    _remove_stage_candidate(last_complete_short_survivors, 6, candidate_key)
                            else:
                                filtered_step6.append(c)
                        refreshed_step6 = filtered_step6

                        step7_results_batch = _run_step_batch(refreshed_step6, short_step7, 7, "SHORT")
                        _update_last_complete_step("sell", 7, step7_results_batch)
                        step7_passed = [candidate for candidate, ok, _ in step7_results_batch if ok]
                        if step7_passed:
                            now_ts = datetime.now(timezone.utc)
                            for candidate in step7_passed:
                                ready_key = get_signal_key(
                                    candidate["sym"],
                                    candidate["base_frame"],
                                    candidate["confirm_frame"],
                                    candidate["triple_frame"],
                                    "sell",
                                )
                                _set_ready_since(step7_ready_since, step7_ready_since_lock, ready_key, now_ts)
                            _promote_candidates("sell", 6, 7, step7_passed)

                # stage 7 -> 8 (gated)
                with last_complete_short_lock:
                    step7_queue = list(last_complete_short_survivors.get(7, []))

                refreshed_step7 = []
                if step7_queue:
                    for candidate in step7_queue:
                        candidate2 = _refresh_waiting_candidate(candidate, get_resampled, need_triple=True)
                        if candidate2 is not None:
                            refreshed_step7.append(candidate2)
                        else:
                            candidate_key = get_candidate_key(candidate)
                            with last_complete_short_lock:
                                _remove_stage_candidate(last_complete_short_survivors, 7, candidate_key)

                if refreshed_step7:
                    # ✅ حارس أخير: أزل أي مرشح يوجد فريم أكبر منه دخل تشبع (Stage 7→8، قبل الإطلاق)
                    final_step7 = []
                    for c in refreshed_step7:
                        if _has_higher_tf_saturation(c, "sell", get_resampled):
                            candidate_key = get_candidate_key(c)
                            with last_complete_short_lock:
                                _remove_stage_candidate(last_complete_short_survivors, 7, candidate_key)
                        else:
                            final_step7.append(c)
                    refreshed_step7 = final_step7

                    step8_results_batch = _run_step_batch(refreshed_step7, short_step8, 8, "SHORT")
                    _update_last_complete_step("sell", 8, step8_results_batch)
                    step8_passed = [candidate for candidate, ok, _ in step8_results_batch if ok]

                    if step8_passed:
                        _set_step8_survivors("sell", step8_passed)
                        for candidate in step8_passed:
                            _fire_signal(
                                candidate["sym"],
                                candidate["base_frame"],
                                candidate["confirm_frame"],
                                candidate["triple_frame"],
                                candidate["df_base"],
                                signal_type="sell",
                            )
                        log.info(
                            "⚡ Quick check (SHORT): %d إشارة من %d مرشح محفوظ",
                            len(step8_passed),
                            len(refreshed_step7),
                        )

                # تحديث آخر وقت scan
                with last_complete_scan_time_lock:
                    last_complete_scan_time["buy"] = datetime.now(timezone.utc)
                    last_complete_scan_time["sell"] = datetime.now(timezone.utc)

        except Exception as e:
            log.error("❌ خطأ في quick_check_watcher: %s", e)


def _dispatch_command(txt, chat_id):
    """معالج أوامر Telegram"""
    # تقارير الإشارات
    if txt in ("1", "/today"):
        send_telegram(get_report("today"), chat_id)
    elif txt in ("2", "/yesterday"):
        send_telegram(get_report("yesterday"), chat_id)
    elif txt in ("3", "/week"):
        send_telegram(get_report("week"), chat_id)

    # فحص العملة
    elif txt.startswith("/check5"):
        parts = txt.split()
        symbol = parts[1] if len(parts) > 1 else "BTCUSDT"
        handle_check5(chat_id, symbol)

    # تقارير Cascade
    elif txt in ("/cascade_diag", "/سبب_شراء", "/diag_buy"):
        _cmd_cascade_diag(chat_id, "buy")
    elif txt in ("/cascade_diag_sell", "/سبب_بيع", "/diag_sell"):
        _cmd_cascade_diag(chat_id, "sell")

    # الناجحون من كل خطوة (شراء)
    elif txt == "/survivors6":
        _cmd_show_step_survivors(chat_id, step_num=6, signal_type="buy")
    elif txt == "/survivors7":
        _cmd_show_step_survivors(chat_id, step_num=7, signal_type="buy")
    elif txt == "/survivors8":
        _cmd_show_step_survivors(chat_id, step_num=8, signal_type="buy")

    # الناجحون من كل خطوة (بيع)
    elif txt == "/survivors6_sell":
        _cmd_show_step_survivors(chat_id, step_num=6, signal_type="sell")
    elif txt == "/survivors7_sell":
        _cmd_show_step_survivors(chat_id, step_num=7, signal_type="sell")
    elif txt == "/survivors8_sell":
        _cmd_show_step_survivors(chat_id, step_num=8, signal_type="sell")

    # دعم /survivors برقم (مثل /survivors 6 أو /survivors 6_sell)
    elif txt.startswith("/survivors"):
        parts = txt.split()
        if "_sell" in txt:
            num_part = parts[0].replace("/survivors", "").replace("_sell", "")
            step_num = int(num_part) if num_part.isdigit() else 6
            _cmd_show_step_survivors(chat_id, step_num=step_num, signal_type="sell")
        else:
            step_num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 6
            if 1 <= step_num <= 8:
                _cmd_show_step_survivors(chat_id, step_num=step_num, signal_type="buy")
            else:
                send_telegram("⚠️ رقم الخطوة يجب أن يكون من 1 إلى 8", chat_id)

    # الحالة والفلاتر
    elif txt == "/invalid_symbols":
        with invalid_symbols_lock:
            bad = list(invalid_symbols_cache)

        if bad:
            with invalid_symbols_reason_lock:
                reasons = dict(invalid_symbols_reason_cache)

            market_label = "Futures" if MARKET_MODE == "futures" else "Spot"
            lines = [f"❌ <b>عملات غير متاحة حالياً على Binance {market_label}:</b>"]
            for s in bad:
                lines.append(f"• <code>{s}</code> — {reasons.get(s, 'UNKNOWN')}")
            send_telegram("\n".join(lines), chat_id)
        else:
            send_telegram("✅ كل العملات في القائمة متاحة وتعمل بشكل صحيح.", chat_id)

    elif txt == "/status":
        _cmd_status(chat_id)

    elif txt == "/hard_filters":
        handle_hard_filters_command(chat_id, "buy")

    elif txt == "/hard_filters_sell":
        handle_hard_filters_command(chat_id, "sell")

    # المساعدة
    elif txt == "/help":
        send_telegram(
            "📋 <b>الأوامر المتاحة:</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>📊 التقارير:</b>\n"
            "1️⃣ <code>1</code> أو <code>/today</code> — إشارات اليوم\n"
            "2️⃣ <code>2</code> أو <code>/yesterday</code> — إشارات أمس\n"
            "3️⃣ <code>3</code> أو <code>/week</code> — آخر 7 أيام\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>🔍 التحليل:</b>\n"
            "🟢 <code>/cascade_diag</code> أو <code>/سبب_شراء</code> — تقرير Cascade الشراء\n"
            "🔴 <code>/cascade_diag_sell</code> أو <code>/سبب_بيع</code> — تقرير Cascade البيع\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>🎯 الناجحون (شراء):</b>\n"
            "🟢 <code>/survivors6</code> — الناجحون حتى الخطوة 6\n"
            "🟢 <code>/survivors7</code> — الناجحون حتى الخطوة 7\n"
            "🟢 <code>/survivors8</code> — الناجحون حتى الخطوة 8\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>🎯 الناجحون (بيع):</b>\n"
            "🔴 <code>/survivors6_sell</code> — الناجحون حتى الخطوة 6\n"
            "🔴 <code>/survivors7_sell</code> — الناجحون حتى الخطوة 7\n"
            "🔴 <code>/survivors8_sell</code> — الناجحون حتى الخطوة 8\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>⚠️ الفلاتر القاسية:</b>\n"
            "⚠️ <code>/hard_filters</code> — أقسى الفلاتر (شراء)\n"
            "⚠️ <code>/hard_filters_sell</code> — أقسى الفلاتر (بيع)\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "<b>📈 أخرى:</b>\n"
            "📊 <code>/status</code> — حالة البوت\n"
            "📛 <code>/invalid_symbols</code> — عرض العملات غير المتاحة حالياً\n"
            "🔎 <code>/check5 [symbol]</code> — فحص 1m بدلاً من 5m (الاسم محفوظ للتوافق)\n"
            "📋 <code>/help</code> — هذه القائمة",
            chat_id,
        )
        
def poll_telegram_commands():
    last_id = 0
    allowed_ids = {
        cid.strip()
        for cid in os.environ.get("ALLOWED_CHAT_IDS", TELEGRAM_CHAT_ID).split(",")
        if cid.strip()
    }
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
                    if chat_id in allowed_ids:
                        threading.Thread(target=_dispatch_command, args=(txt, chat_id), daemon=True).start()
                    else:
                        log.warning("⚠️ رسالة من chat_id غير مصرح به: %s — تم تجاهلها", chat_id)
        except requests.RequestException as e:
            log.error("poll_telegram_commands network error: %s", e)
            time.sleep(10)
        except Exception as e:
            log.error("poll_telegram_commands error: %s", e)
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
_scan_lock = threading.Lock()

def cascade_watcher():
    while True:
        try:
            if fast_prefetch_done.is_set():
                with ohlcv_cache_lock:
                    if len(ohlcv_cache) < 200:  # تأكد الكاش فيه بيانات
                        time.sleep(30)
                        continue
                # ✅ fetch مرة واحدة للاثنين — أضفنا 30m لضمان فريش دائمًا
                with symbols_cache_lock:
                    syms = list(symbols_cache)

                def fetch_fresh(sym):
                    # إصلاح: استخدام الدالة الصحيحة حسب MARKET_MODE لتجنب خلط بيانات Spot/Futures
                    fetch_fn = get_ohlcv_futures if MARKET_MODE == "futures" else get_ohlcv
                    for tf in ["1m", "30m", "60m"]:
                        df = fetch_fn(sym, tf, limit=3)
                        if not df.empty:
                            cache_merge(sym, tf, df)

                with ThreadPoolExecutor(max_workers=30) as executor:
                    executor.map(fetch_fresh, syms)

                # 🔄 سكان كامل (1-8) — كل استيقاظة (بدون تخطي دورات)
                # مع قفل يمنع تشغيل سكان جديد قبل انتهاء القديم
                if _scan_lock.acquire(blocking=False):
                    try:
                        t1 = threading.Thread(target=run_cascade_scan, daemon=True)
                        t2 = threading.Thread(target=run_short_cascade_scan, daemon=True)
                        t1.start()
                        t2.start()
                        t1.join()
                        t2.join()
                        with _ribbon_cache_lock:
                            _ribbon_cache.clear()
                        trim_memory()
                    finally:
                        _scan_lock.release()
                else:
                    log.warning("⏭️ تخطي السكان — السكان السابق لسه شغال")

            time.sleep(next_candle_close())
        except Exception as e:
            log.error("❌ خطأ في cascade_watcher: %s", e)
            time.sleep(5)

def trim_memory():
    try:
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        rss_before = None

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as e:
        log.error("malloc_trim error: %s", e)

    if rss_before is not None:
        try:
            rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            log.info("🧹 trim_memory: peak RSS قبل=%s KB، بعد=%s KB (peak قد لا يقل حتى مع نجاح trim)", rss_before, rss_after)
        except Exception:
            pass

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *_):
        pass

def thread_exception_handler(args):
    msg = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    thread_name = args.thread.name if args.thread else "unknown"
    log.error("💥 خطأ في Thread [%s]:\n%s", thread_name, msg)
    try:
        send_telegram(f"⚠️ <b>خطأ في Thread {thread_name}:</b>\n<code>{args.exc_value}</code>")
    except Exception:
        pass


def run_forever(target, name):
    def wrapper():
        while True:
            try:
                target()
            except Exception as e:
                log.error("💥 %s توقف بخطأ، سيُعاد تشغيله خلال 10 ثواني: %s", name, e)
                try:
                    send_telegram(f"🔄 <b>{name}</b> توقف وسيُعاد تشغيله تلقائياً.\n<code>{e}</code>")
                except Exception:
                    pass
                time.sleep(10)

    threading.Thread(target=wrapper, name=name, daemon=True).start()


def main():
    def handle_exception(exc_type, exc_value, exc_tb):
        msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        log.error("💥 خطأ غير متوقع أوقف البوت:\n%s", msg)
        try:
            send_telegram(f"💥 <b>البوت توقف بسبب خطأ:</b>\n<code>{exc_value}</code>")
        except Exception:
            pass

    sys.excepthook = handle_exception
    threading.excepthook = thread_exception_handler

    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("✅ Health server شغّال على port %s", PORT)

    delete_webhook()

    if MARKET_MODE == "futures":
        run_forever(update_symbols_loop_futures, "update_symbols_loop_futures")
        threading.Thread(target=cache_updater_1m_futures, daemon=True).start()
        threading.Thread(target=cache_updater_60m_futures, daemon=True).start()
        threading.Thread(target=cache_updater_30m_futures, daemon=True).start()
    else:
        run_forever(update_symbols_loop, "update_symbols_loop")
        threading.Thread(target=cache_updater_1m, daemon=True).start()
        threading.Thread(target=cache_updater_60m, daemon=True).start()
        threading.Thread(target=cache_updater_30m, daemon=True).start()

    run_forever(poll_telegram_commands, "poll_telegram_commands")
    run_forever(cascade_watcher, "cascade_watcher")
    run_forever(quick_check_watcher, "quick_check_watcher")

    send_telegram("🚀 <b>البوت انطلق — استراتيجية مزدوجة (شراء + بيع)</b>")

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