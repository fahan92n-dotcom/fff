"""Telegram transport, polling, and signal notification integration."""

import logging
import threading
import time
from collections.abc import Callable

import requests

from binance_data import get_session
from config import ALLOWED_CHAT_IDS, TELEGRAM_CHAT_ID, TELEGRAM_TOKEN
from state_manager import (
    _clear_waiting_candidate,
    claim_signal,
    save_signal,
)


log = logging.getLogger(__name__)
_command_handler: Callable[[str, str], None] | None = None


def set_command_handler(handler):
    """Register the application command dispatcher used by the poller."""
    global _command_handler
    _command_handler = handler


def _api_url(method):
    if not TELEGRAM_TOKEN:
        return None
    return f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"


def delete_webhook():
    """Delete an existing webhook before long polling starts."""
    url = _api_url("deleteWebhook")
    if url is None:
        log.error("TELEGRAM_TOKEN is not configured; webhook was not deleted")
        return False
    try:
        response = get_session().post(
            url,
            json={"drop_pending_updates": True},
            timeout=10,
        ).json()
    except requests.RequestException as exc:
        log.error("deleteWebhook error: %s", exc)
        return False
    if response.get("ok"):
        log.info("✅ تم حذف الـ Webhook")
        return True
    log.error("Telegram rejected deleteWebhook: %s", response)
    return False


def send_telegram(message, chat_id=None):
    """Send one HTML Telegram message and return whether Telegram accepted it."""
    target = chat_id or TELEGRAM_CHAT_ID
    url = _api_url("sendMessage")
    if url is None or not target:
        log.error("Telegram token/chat id is not configured")
        return False
    try:
        response = get_session().post(
            url,
            json={"chat_id": target, "text": message, "parse_mode": "HTML"},
            timeout=10,
        ).json()
    except requests.RequestException as exc:
        log.error("Telegram send error: %s", exc)
        return False
    if not response.get("ok", False):
        log.error("Telegram rejected sendMessage: %s", response)
        return False
    return True


def _fire_signal(
    symbol,
    base_frame,
    confirm_frame,
    triple_frame,
    frame,
    signal_type="buy",
):
    """Persist and send one deduplicated stage-8 signal."""
    key = (symbol, base_frame, confirm_frame, triple_frame, signal_type)
    now = claim_signal(key)
    if now is None:
        return False

    price = float(frame["close"].iloc[-1])
    save_signal(
        symbol,
        price,
        base_frame,
        confirm_frame,
        triple_frame,
        signal_type=signal_type,
    )
    _clear_waiting_candidate(
        symbol,
        base_frame,
        confirm_frame,
        triple_frame,
        signal_type=signal_type,
    )

    icon = "🟢 شراء (LONG)" if signal_type == "buy" else "🔴 بيع (SHORT)"
    send_telegram(
        f"{icon}\n"
        f"💱 العملة: <b>{symbol}</b>\n"
        f"💰 السعر: <b>{price:.6g}</b>\n"
        f"⏱️ الفريمات: {base_frame}m / {confirm_frame}m / {triple_frame}m\n"
        f"🕐 الوقت: {now.strftime('%H:%M:%S UTC')}"
    )
    return True


def _dispatch_update(update):
    message = update.get("message", {})
    text = message.get("text", "").strip()
    chat_id = str(message.get("chat", {}).get("id", ""))
    if not text or not chat_id:
        return
    if chat_id not in ALLOWED_CHAT_IDS:
        log.warning(
            "⚠️ رسالة من chat_id غير مصرح به: %s — تم تجاهلها",
            chat_id,
        )
        return
    if _command_handler is None:
        log.error("Telegram command handler is not configured")
        return
    threading.Thread(
        target=_command_handler,
        args=(text, chat_id),
        daemon=True,
    ).start()


def poll_telegram_commands():
    """Long-poll Telegram and dispatch authorized commands."""
    url = _api_url("getUpdates")
    if url is None:
        log.error("TELEGRAM_TOKEN is not configured; polling is disabled")
        time.sleep(60)
        return

    last_id = 0
    while True:
        try:
            response = get_session().get(
                url,
                params={"offset": last_id + 1, "timeout": 30},
                timeout=35,
            ).json()
            for update in response.get("result", []):
                last_id = update["update_id"]
                _dispatch_update(update)
        except requests.RequestException as exc:
            log.error("poll_telegram_commands network error: %s", exc)
            time.sleep(10)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            log.error("poll_telegram_commands response error: %s", exc)
            time.sleep(10)
