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
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer, BaseHTTPRequestHandler

import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------
# Main Settings
# ------------------------------------------

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8696456847:AAG06_sYJVIZNjCRwO29OynYFh9GsWYOwXo")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "-1003968771145")

PORT = int(os.environ.get("PORT", "8080"))

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

from indicators import (
    MIN_CANDLES,
    _ribbon_cache, _ribbon_cache_lock,
    resample_ohlcv,
    _calc_macd_full,
    calc_smi,
    calc_rsi_tv, calc_stoch_tv,
)
from state_manager import (
    ALERT_EXPIRY_HOURS, STEP5_MAX_WAIT_SECONDS, STAGE5_COEXIST_MIN_RATIO,
    alerted_keys, alerted_keys_lock, trades_history, trades_lock,
    cascade_results, cascade_results_lock, cascade_stats, cascade_stats_lock,
    short_cascade_results, short_cascade_results_lock,
    short_cascade_stats, short_cascade_stats_lock,
    last_complete_stats, last_complete_results, last_complete_survivors,
    last_complete_lock, last_complete_short_stats, last_complete_short_results,
    last_complete_short_survivors, last_complete_short_lock,
    last_complete_scan_time, last_complete_scan_time_lock,
    step1_ready_since, step1_ready_since_lock,
    step6_ready_since, step6_ready_since_lock,
    step7_ready_since, step7_ready_since_lock,
    step5_entry_time, step5_entry_time_lock,
    cleanup_alerted_keys, claim_signal, save_signal,
    get_candidate_key, get_signal_key, get_ready_since, get_step1_ready_since,
    _set_ready_since, _get_stage_maps, _upsert_stage_candidate,
    _remove_stage_candidate, _candidate_keys_in_stages, _frames_far_apart,
    _store_step5_waiters, _promote_candidates, _set_step8_survivors,
    _clear_waiting_candidate, _update_last_complete_step,
)
from cascade_pipeline import (
    TRIPLING_PAIRS, TIMEFRAME_CHAIN, NEXT_TF, TF_TO_API,
    QUICK_CHECK_INTERVAL_SECONDS,
    STEP_NAMES, STEP_LABELS, SHORT_STEP_NAMES, SHORT_STEP_LABELS,
    step1, step2, step3, step4, step5, step6, step7, step8, steps,
    short_step1, short_step2, short_step3, short_step4,
    short_step5, short_step6, short_step7, short_step8, short_steps,
    _has_higher_tf_saturation, _refresh_waiting_candidate,
    _refresh_and_validate_step5, _refresh_and_validate_step5_short,
    _run_step_batch, run_cascade_scan, run_short_cascade_scan,
    quick_check_watcher, set_signal_handler,
)

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
    now = claim_signal(key)
    if now is None:
        return

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

set_signal_handler(_fire_signal)

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

    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        log.error("check5 error: %s", exc)
        send_telegram(f"❌ خطأ: {exc}", chat_id)


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
        except requests.RequestException as exc:
            log.error("poll_telegram_commands network error: %s", exc)
            time.sleep(10)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            log.error("poll_telegram_commands response error: %s", exc)
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
        except Exception:  # Intentional daemon boundary: retry after unexpected failures.
            log.exception("❌ خطأ في cascade_watcher")
            time.sleep(5)

def trim_memory():
    try:
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, ValueError):
        rss_before = None

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError) as exc:
        log.error("malloc_trim error: %s", exc)

    if rss_before is not None:
        try:
            rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            log.info("🧹 trim_memory: peak RSS قبل=%s KB، بعد=%s KB (peak قد لا يقل حتى مع نجاح trim)", rss_before, rss_after)
        except (OSError, ValueError) as exc:
            log.debug("تعذر قراءة RSS بعد التنظيف: %s", exc)

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