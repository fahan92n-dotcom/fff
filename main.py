"""Process entry point and background-service supervision."""

import ctypes
import gc
import logging
import resource
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

from binance_data import (
    MARKET_MODE,
    cache_merge,
    cache_updater_1m,
    cache_updater_1m_futures,
    cache_updater_30m,
    cache_updater_30m_futures,
    cache_updater_60m,
    cache_updater_60m_futures,
    fast_prefetch_done,
    get_ohlcv,
    get_ohlcv_futures,
    ohlcv_cache,
    ohlcv_cache_lock,
    prefetch_done,
    set_telegram_sender,
    symbols_cache,
    symbols_cache_lock,
    update_symbols_loop,
    update_symbols_loop_futures,
)
from cascade_pipeline import (
    TIMEFRAME_CHAIN,
    audit_broken_frames,
    quick_check_watcher,
    run_cascade_scan,
    run_short_cascade_scan,
    set_signal_handler,
)
from config import PORT
from indicators import clear_ribbon_cache
from state_manager import (
    BROKEN_FRAMES_SNAPSHOT_INTERVAL,
    cleanup_alerted_keys,
    last_broken_frames_snapshot_at,
    broken_frames_history_lock,
    record_broken_frames_snapshot,
    trades_history,
    trades_lock,
)
from telegram_bot import (
    _fire_signal,
    delete_webhook,
    poll_telegram_commands,
    send_telegram,
    set_command_handler,
)


log = logging.getLogger(__name__)
_scan_lock = threading.Lock()


def configure_integrations(command_handler=None):
    """Wire callbacks without creating import cycles between service modules."""
    set_telegram_sender(send_telegram)
    set_signal_handler(_fire_signal)
    if command_handler is not None:
        set_command_handler(command_handler)


def next_candle_close():
    """Return seconds until the next configured base-frame boundary."""
    now_seconds = time.time()
    waits = [
        timeframe * 60 - (int(now_seconds) % (timeframe * 60))
        for timeframe in TIMEFRAME_CHAIN
    ]
    return min(waits) + 1


def _fetch_fresh_symbols():
    with symbols_cache_lock:
        symbols = list(symbols_cache)
    fetch_fn = get_ohlcv_futures if MARKET_MODE == "futures" else get_ohlcv

    def fetch_symbol(symbol):
        for timeframe in ("1m", "30m", "60m"):
            frame = fetch_fn(symbol, timeframe, limit=3)
            if not frame.empty:
                cache_merge(symbol, timeframe, frame)

    with ThreadPoolExecutor(max_workers=30) as executor:
        list(executor.map(fetch_symbol, symbols))


def _run_full_scan_pair():
    if not _scan_lock.acquire(blocking=False):
        log.warning("⏭️ تخطي السكان — السكان السابق لسه شغال")
        return False
    try:
        workers = [
            threading.Thread(target=target, daemon=True)
            for target in (run_cascade_scan, run_short_cascade_scan)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        clear_ribbon_cache()
        trim_memory()
        return True
    finally:
        _scan_lock.release()


def cascade_watcher():
    """Refresh market data and run both cascade sides at candle boundaries."""
    while True:
        try:
            if fast_prefetch_done.is_set():
                with ohlcv_cache_lock:
                    cache_ready = len(ohlcv_cache) >= 200
                if cache_ready:
                    _fetch_fresh_symbols()
                    _run_full_scan_pair()
                else:
                    time.sleep(30)
                    continue
            time.sleep(next_candle_close())
        except Exception:  # pylint: disable=broad-exception-caught
            # Intentional daemon boundary: the next market cycle must still run.
            log.exception("❌ خطأ في cascade_watcher")
            time.sleep(5)


def trim_memory():
    """Collect Python objects and ask glibc to release free arenas."""
    try:
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, ValueError):
        rss_before = None

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (AttributeError, OSError) as exc:
        log.error("malloc_trim error: %s", exc)

    if rss_before is None:
        return
    try:
        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        log.info(
            "🧹 trim_memory: peak RSS قبل=%s KB، بعد=%s KB",
            rss_before,
            rss_after,
        )
    except (OSError, ValueError) as exc:
        log.debug("تعذر قراءة RSS بعد التنظيف: %s", exc)


class HealthHandler(BaseHTTPRequestHandler):
    """Minimal Railway health endpoint."""

    def do_GET(self):  # pylint: disable=invalid-name
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *_):
        return


def thread_exception_handler(args):
    """Log and notify uncaught worker-thread failures."""
    message = "".join(
        traceback.format_exception(
            args.exc_type,
            args.exc_value,
            args.exc_traceback,
        )
    )
    thread_name = args.thread.name if args.thread else "unknown"
    log.error("💥 خطأ في Thread [%s]:\n%s", thread_name, message)
    send_telegram(
        f"⚠️ <b>خطأ في Thread {thread_name}:</b>\n"
        f"<code>{args.exc_value}</code>"
    )


def run_forever(target, name):
    """Run a long-lived target under a restart boundary."""

    def wrapper():
        while True:
            try:
                target()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # Intentional supervisor boundary: restart any failed service.
                log.exception(
                    "💥 %s توقف بخطأ، سيُعاد تشغيله خلال 10 ثواني",
                    name,
                )
                send_telegram(
                    f"🔄 <b>{name}</b> توقف وسيُعاد تشغيله تلقائياً.\n"
                    f"<code>{exc}</code>"
                )
                time.sleep(10)

    threading.Thread(target=wrapper, name=name, daemon=True).start()


def _start_market_threads():
    if MARKET_MODE == "futures":
        run_forever(update_symbols_loop_futures, "update_symbols_loop_futures")
        targets = (
            cache_updater_1m_futures,
            cache_updater_60m_futures,
            cache_updater_30m_futures,
        )
    else:
        run_forever(update_symbols_loop, "update_symbols_loop")
        targets = (
            cache_updater_1m,
            cache_updater_60m,
            cache_updater_30m,
        )
    for target in targets:
        threading.Thread(target=target, daemon=True).start()


def start_background_services():
    """Start health, market, Telegram, and cascade services."""
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    log.info("✅ Health server شغّال على port %s", PORT)

    delete_webhook()
    _start_market_threads()
    for target, name in (
        (poll_telegram_commands, "poll_telegram_commands"),
        (cascade_watcher, "cascade_watcher"),
        (quick_check_watcher, "quick_check_watcher"),
    ):
        run_forever(target, name)
    send_telegram("🚀 <b>البوت انطلق — استراتيجية مزدوجة (شراء + بيع)</b>")
    return server


def _should_snapshot_broken_frames():
    """True when the periodic broken-frames audit is due."""
    if not fast_prefetch_done.is_set():
        return False
    now = datetime.now(timezone.utc)
    with broken_frames_history_lock:
        last_at = last_broken_frames_snapshot_at.get("at")
    if last_at is None:
        return True
    return now - last_at >= BROKEN_FRAMES_SNAPSHOT_INTERVAL


def heartbeat_once():
    """Clean expired state and log one process heartbeat."""
    cleanup_alerted_keys()
    if _should_snapshot_broken_frames():
        try:
            record_broken_frames_snapshot(audit_broken_frames(), force=False)
        except Exception:  # pylint: disable=broad-exception-caught
            log.exception("broken-frames snapshot failed during heartbeat")
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


def main(command_handler):
    """Configure integrations and run the process until interrupted."""
    configure_integrations(command_handler)

    def handle_exception(exc_type, exc_value, exc_tb):
        message = "".join(
            traceback.format_exception(exc_type, exc_value, exc_tb)
        )
        log.error("💥 خطأ غير متوقع أوقف البوت:\n%s", message)
        send_telegram(
            f"💥 <b>البوت توقف بسبب خطأ:</b>\n<code>{exc_value}</code>"
        )

    sys.excepthook = handle_exception
    threading.excepthook = thread_exception_handler
    start_background_services()

    while True:
        time.sleep(300)
        heartbeat_once()
