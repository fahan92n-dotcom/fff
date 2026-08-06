"""بوت مسح العملات من Binance مع تنبيهات Telegram - نسخة Cascade Pipeline مع استراتيجية مزدوجة (شراء/بيع)."""
import logging
from datetime import datetime, timezone, timedelta
from html import escape as html_escape

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------
# Main Settings
# ------------------------------------------

from config import (
    ALLOWED_CHAT_IDS,
    PORT,
    TELEGRAM_CHAT_ID,
    TELEGRAM_TOKEN,
)

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
    _clear_waiting_candidate, abandon_waiting_candidate,
    _purge_orphaned_ready_timestamps, _update_last_complete_step,
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
    quick_check_watcher, set_signal_handler, audit_broken_frames,
)
from main import (
    HealthHandler,
    cascade_watcher,
    configure_integrations,
    heartbeat_once,
    main,
    next_candle_close,
    run_forever,
    start_background_services,
    thread_exception_handler,
    trim_memory,
)
from telegram_bot import (
    _fire_signal,
    delete_webhook,
    poll_telegram_commands,
    send_telegram,
    set_command_handler,
)

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

def _diag_reason(
    reason,
    severity,
    description,
    solution,
    why="",
    percentage=0.0,
    total_failed=0,
    total=0,
    rank=1,
):
    """Build one diagnostics reason with a stable schema for Telegram formatting."""
    return {
        "rank": rank,
        "reason": reason,
        "severity": severity,
        "percentage": float(percentage),
        "total_failed": int(total_failed),
        "total": int(total),
        "description": description,
        "solution": solution,
        "why": why,
    }


def diagnose_signal_failures():
    """
    تشخيص أهم 3 أسباب لعدم مجيء إشارات
    ترتيب من الأقوى فشل إلى الأضعف
    """
    if not fast_prefetch_done.is_set():
        return [
            _diag_reason(
                reason="❌ البيانات لم تحمل بعد",
                severity="CRITICAL",
                percentage=100.0,
                description="البوت ما زال يحمل البيانات الأولية",
                solution="انتظر 5-30 دقيقة للتحميل الكامل",
                why="التشخيص يحتاج كاش مكتمل قبل حساب نسب الفشل",
            )
        ]

    with symbols_cache_lock:
        symbols = list(symbols_cache)

    if not symbols:
        return [
            _diag_reason(
                reason="❌ لا توجد عملات محملة",
                severity="CRITICAL",
                percentage=100.0,
                description="قائمة العملات فارغة",
                solution="تحقق من Binance API وإعداد MARKET_MODE",
                why="بدون عملات لا يمكن تشغيل خطوات الـ Cascade",
            )
        ]

    # ─────────────────────────────────────────────
    # السبب #1: فشل MIN_CANDLES
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

    min_candles_percentage = (
        (min_candles_failures / total_candidates * 100) if total_candidates > 0 else 0.0
    )

    # ─────────────────────────────────────────────
    # السبب #2: فشل Step 6 (EMA50)
    # ─────────────────────────────────────────────
    step6_failures = 0
    step6_total = 0

    with last_complete_lock:
        stats = last_complete_stats.get(6, {})
        total = stats.get("total", 0)
        passed = stats.get("passed", 0)
        if total > 0:
            step6_failures = total - passed
            step6_total = total

    step6_percentage = (
        (step6_failures / step6_total * 100) if step6_total > 0 else 0.0
    )

    # ─────────────────────────────────────────────
    # السبب #3: فشل Step 1 (SMI Oversold)
    # ─────────────────────────────────────────────
    step1_failures = 0
    step1_total = 0

    with last_complete_lock:
        stats = last_complete_stats.get(1, {})
        total = stats.get("total", 0)
        passed = stats.get("passed", 0)
        if total > 0:
            step1_failures = total - passed
            step1_total = total

    step1_percentage = (
        (step1_failures / step1_total * 100) if step1_total > 0 else 0.0
    )

    reasons = [
        _diag_reason(
            reason="❌ فشل MIN_CANDLES (الحد الأدنى من الشموع)",
            severity="CRITICAL" if min_candles_percentage > 50 else "HIGH",
            percentage=min_candles_percentage,
            total_failed=min_candles_failures,
            total=total_candidates,
            description=(
                f"{min_candles_failures} مرشح من {total_candidates} "
                f"فشلوا في اختبار الحد الأدنى ({MIN_CANDLES} شمعة)"
            ),
            solution="زيادة API_FETCH_CANDLES أو تقليل عدد الأطر الكبيرة",
            why="الأطر الكبيرة (180m, 240m) تحتاج بيانات أكثر",
        ),
        _diag_reason(
            reason="⚠️ فشل Step 6 (شرط EMA50)",
            severity="HIGH" if step6_percentage > 50 else "MEDIUM",
            percentage=step6_percentage,
            total_failed=step6_failures,
            total=step6_total,
            description=(
                f"{step6_failures} مرشح من {step6_total} "
                "فشلوا في شرط السعر مقابل EMA50 منذ تشبع الفريم الأساسي"
            ),
            solution="راجع اتجاه السعر بالنسبة لـ EMA50 بعد تشبع Step 1",
            why="Step 6 يتحقق من لمس/اختراق EMA50 منذ لحظة التشبع وليس من RSI",
        ),
        _diag_reason(
            reason="⚡ فشل Step 1 (SMI Oversold ≤ -40)",
            severity="MEDIUM" if step1_percentage > 70 else "LOW",
            percentage=step1_percentage,
            total_failed=step1_failures,
            total=step1_total,
            description=(
                f"{step1_failures} مرشح من {step1_total} "
                "لم يصلوا لتشبع SMI بيعي"
            ),
            solution="تخفيف عتبة SMI من -40 إلى -30",
            why="السوق لا يدخل تشبع بيعي في كل وقت",
        ),
    ]

    reasons.sort(key=lambda item: item["percentage"], reverse=True)
    for index, reason in enumerate(reasons, 1):
        reason["rank"] = index
    return reasons


def _format_diagnostics_report(reasons, title):
    """Format diagnostics reasons into one Telegram-ready message."""
    lines = [
        f"🔍 <b>{title}</b>",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]
    for reason in reasons:
        severity = reason["severity"]
        icon = (
            "🔴"
            if severity == "CRITICAL"
            else ("🟠" if severity == "HIGH" else "🟡")
        )
        lines.append(
            f"""
{icon} <b>السبب #{reason['rank']}: {reason['reason']}</b>
├─ الشدة: {severity}
├─ نسبة الفشل: <b>{reason['percentage']:.1f}%</b> ({reason['total_failed']}/{reason['total']})
├─ التفاصيل: {reason['description']}
├─ الحل: <code>{reason['solution']}</code>
└─ السبب: {reason['why']}
"""
        )
    return "\n".join(lines)


def send_diagnostics_report(chat_id=None):
    """إرسال تقرير التشخيص عبر Telegram"""
    msg = _format_diagnostics_report(
        diagnose_signal_failures(),
        "تقرير تشخيص فشل الإشارات",
    )
    send_telegram(msg, chat_id)


def handle_diag_command(chat_id):
    """معالج أمر /diag_failures"""
    msg = _format_diagnostics_report(
        diagnose_signal_failures(),
        "تشخيص أسباب فشل الإشارات",
    )
    for i in range(0, len(msg), 4000):
        send_telegram(msg[i:i + 4000], chat_id)


def _arabic_frame_count(count):
    """Arabic plural phrasing for broken-frame counts."""
    if count == 1:
        return "فريم واحد معطوب"
    if count == 2:
        return "فريمان معطوبان"
    if 3 <= count <= 10:
        return f"{count} فريمات معطوبة"
    return f"{count} فريماً معطوباً"


def format_broken_frames_report(report):
    """Format audit_broken_frames() output into Telegram HTML chunks."""
    if not report.get("ready"):
        return [
            "⏳ البيانات الأولية لم تكتمل بعد.\n"
            "انتظر انتهاء التحميل ثم أعد طلب "
            "<code>/broken_frames</code>."
        ]

    broken_by_symbol = report.get("broken_by_symbol") or {}
    symbols_checked = report.get("symbols_checked", 0)
    total_pairs = report.get("total_pairs", len(TRIPLING_PAIRS))
    broken_frame_count = report.get("broken_frame_count", 0)
    ok_count = len(report.get("ok_symbols") or [])

    if not broken_by_symbol:
        return [
            "✅ <b>كل الفريمات شغّالة</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"تم فحص <b>{symbols_checked}</b> عملة × "
            f"<b>{total_pairs}</b> فريم ثلاثي.\n"
            "ما في أي فريم متخطّى بسبب نقص بيانات أو شموع."
        ]

    header = (
        "🛠️ <b>تقرير الفريمات المعطوبة</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"عملات فيها مشاكل: <b>{len(broken_by_symbol)}</b> / {symbols_checked}\n"
        f"إجمالي الفريمات المعطوبة: <b>{broken_frame_count}</b>\n"
        f"عملات سليمة بالكامل: <b>{ok_count}</b>\n"
    )

    blocks = [header]
    for symbol in sorted(broken_by_symbol):
        issues = broken_by_symbol[symbol]
        lines = [
            f"• <code>{html_escape(symbol)}</code> — "
            f"{_arabic_frame_count(len(issues))}:"
        ]
        for issue in issues:
            frames = (
                f"{issue['base_frame']}m / "
                f"{issue['confirm_frame']}m / "
                f"{issue['triple_frame']}m"
            )
            detail = html_escape(str(issue.get("detail") or issue.get("reason")))
            lines.append(f"  ◦ <code>{frames}</code> — {detail}")
        blocks.append("\n".join(lines))

    # Pack into Telegram-safe chunks (~4000 chars).
    chunks = []
    current = ""
    for block in blocks:
        candidate = block if not current else f"{current}\n\n{block}"
        if len(candidate) > 4000 and current:
            chunks.append(current)
            current = block
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def handle_broken_frames_command(chat_id, symbol=None):
    """معالج أمر /broken_frames — يعرض الفريمات التي يتخطاها الماسح."""
    symbols = None
    if symbol:
        symbols = [symbol.upper().replace("/", "").strip()]
    report = audit_broken_frames(symbols=symbols)
    for chunk in format_broken_frames_report(report):
        send_telegram(chunk, chat_id)

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

            stat = stats.get(step_num) or {}
            total_t = int(stat.get("total", 0) or 0)
            total_p = int(stat.get("passed", 0) or 0)
            fail_count = total_t - total_p
            pct = int(total_p / total_t * 100) if total_t else 0
            bar = "█" * (pct // 10) + "░" * (10 - pct // 10)

            lines.append(f"\n{step_label}\n{bar}\n✅ نجح: <b>{total_p}</b> | ❌ فشل: <b>{fail_count}</b> | دخل: <b>{total_t}</b> ({pct}%)")

        msg = "\n".join(lines)

    for i in range(0, len(msg), 4000):
        if not send_telegram(msg[i:i + 4000], chat_id):
            log.error("cascade_diag send failed for signal_type=%s", signal_type)
            send_telegram(
                "❌ فشل إرسال تقرير Cascade (تحقق من تنسيق HTML في التسميات).",
                chat_id,
            )
            return

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


def _normalize_command_text(txt):
    """Strip BotFather @bot_username suffix from the first command token."""
    text = (txt or "").strip()
    if not text:
        return text
    parts = text.split(maxsplit=1)
    command = parts[0]
    if "@" in command:
        command = command.split("@", 1)[0]
    if len(parts) == 1:
        return command
    return f"{command} {parts[1]}"


def _dispatch_command(txt, chat_id):
    """معالج أوامر Telegram"""
    txt = _normalize_command_text(txt)
    try:
        _dispatch_command_inner(txt, chat_id)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        log.exception("command handler failed for %r", txt)
        send_telegram(
            f"❌ خطأ أثناء تنفيذ الأمر: <code>{html_escape(str(exc))}</code>",
            chat_id,
        )


def _dispatch_command_inner(txt, chat_id):
    """Route one normalized Telegram command."""
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

    elif txt in ("/diag_failures", "/diag"):
        handle_diag_command(chat_id)

    elif txt.startswith("/broken_frames") or txt.startswith("/فريمات"):
        parts = txt.split()
        symbol = parts[1] if len(parts) > 1 else None
        handle_broken_frames_command(chat_id, symbol)

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
            "🧪 <code>/diag_failures</code> أو <code>/diag</code> — تشخيص أسباب ضعف الإشارات\n"
            "🛠️ <code>/broken_frames</code> أو <code>/فريمات</code> — الفريمات المعطوبة لكل عملة\n"
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
set_command_handler(_dispatch_command)
configure_integrations(_dispatch_command)


if __name__ == "__main__":
    main(_dispatch_command)