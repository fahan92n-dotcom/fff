"""Standalone entry point for the Pullback strategy bot."""

from __future__ import annotations

import logging
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Allow `python -m pullback_bot` / `python pullback_bot/main.py` to import
# shared indicator helpers that live at the repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pullback_bot.config import PORT, TELEGRAM_CHAT_ID, TELEGRAM_TOKEN
from pullback_bot.strategy import (
    MONTH_DAYS,
    WEEK_DAYS,
    handle_pullback_week_command,
)
from pullback_bot.telegram_app import (
    delete_webhook,
    poll_telegram_commands,
    send_telegram,
    set_command_handler,
)

log = logging.getLogger(__name__)


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, fmt, *args):
        return


def _dispatch_command(txt, chat_id):
    text = (txt or "").strip()
    lower = text.lower()
    if lower in ("/start", "/help"):
        send_telegram(
            "📋 <b>بوت استراتيجية Pullback (منفصل)</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "القواعد:\n"
            "• إلغاء الأصغر: تشبّع SMI أكبر فقط (≤−40 بيع / ≥+40 شراء)\n"
            "• الدخول على المملوك: التشبّع يُحسب فقط إذا RSI أصلاً صحيح "
            "وقت التكوّن (&gt;50 شراء / &lt;50 بيع)؛ إن تكوّن غلط يبقى الإبيسود "
            "ميت حتى لو RSI انقلب لاحقاً — مع الإغلاق فوق/تحت EMA60\n"
            "• داخله تشبّع عكسي؛ بعد تكوّنه يبدأ حساب الدخول\n"
            "• على فريم الدخول: ننتظر حتى يصير الشرطان غير متحققين (عكس)،\n"
            "  ثم ندخل أول ما يتحققان معاً — لا رفض إذا كانوا متحققين أولاً\n"
            "• شراء: دونشيان أخضر + فوق EMA60\n"
            "• بيع: دونشيان أحمر + تحت EMA60\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "1️⃣ <code>1</code> أو <code>/week</code> — آخر 7 أيام\n"
            "2️⃣ <code>2</code> أو <code>/month</code> — آخر 30 يومًا\n"
            "معيار النجاح: +1% | الخسارة: ارتداد 0.70%",
            chat_id,
        )
        return
    if lower in ("/week", "1"):
        handle_pullback_week_command(chat_id, send_telegram, days=WEEK_DAYS)
        return
    if lower in ("/month", "2"):
        handle_pullback_week_command(chat_id, send_telegram, days=MONTH_DAYS)
        return
    send_telegram(
        "أمر غير معروف. أرسل <code>/help</code> أو <code>/week</code> أو <code>/month</code>.",
        chat_id,
    )


def run():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.error(
            "Set PULLBACK_TELEGRAM_TOKEN / PULLBACK_TELEGRAM_CHAT_ID "
            "(or TELEGRAM_TOKEN / TELEGRAM_CHAT_ID)."
        )
    set_command_handler(_dispatch_command)
    delete_webhook()

    health = HTTPServer(("0.0.0.0", PORT), _HealthHandler)
    threading.Thread(target=health.serve_forever, daemon=True).start()
    log.info("Pullback bot health on :%s", PORT)

    send_telegram(
        "✅ بوت <b>Pullback</b> اشتغل (منفصل عن Cascade).\n"
        "أرسل <code>/week</code> لآخر 7 أيام أو <code>/month</code> لآخر 30 يومًا."
    )
    poll_telegram_commands()


if __name__ == "__main__":
    run()
