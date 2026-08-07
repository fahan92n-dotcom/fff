"""Minimal Telegram transport for the Pullback bot only."""

import logging
import threading
import time
from collections.abc import Callable

import requests

from pullback_bot.config import ALLOWED_CHAT_IDS, TELEGRAM_CHAT_ID, TELEGRAM_TOKEN

log = logging.getLogger(__name__)
_command_handler: Callable[[str, str], None] | None = None
_SESSION = requests.Session()


def set_command_handler(handler):
    global _command_handler
    _command_handler = handler


def _api_url(method):
    if not TELEGRAM_TOKEN:
        return None
    return f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{method}"


def delete_webhook():
    url = _api_url("deleteWebhook")
    if url is None:
        log.error("PULLBACK_TELEGRAM_TOKEN is not configured")
        return False
    try:
        response = _SESSION.post(
            url, json={"drop_pending_updates": True}, timeout=10
        ).json()
    except requests.RequestException as exc:
        log.error("deleteWebhook error: %s", exc)
        return False
    return bool(response.get("ok"))


def send_telegram(message, chat_id=None):
    target = chat_id or TELEGRAM_CHAT_ID
    url = _api_url("sendMessage")
    if url is None or not target:
        log.error("Pullback Telegram token/chat id is not configured")
        return False
    try:
        response = _SESSION.post(
            url,
            json={"chat_id": target, "text": message, "parse_mode": "HTML"},
            timeout=10,
        ).json()
    except requests.RequestException as exc:
        log.error("Telegram send error: %s", exc)
        return False
    return bool(response.get("ok", False))


def _dispatch_update(update):
    message = update.get("message", {})
    text = message.get("text", "").strip()
    chat_id = str(message.get("chat", {}).get("id", ""))
    if not text or not chat_id:
        return
    if chat_id not in ALLOWED_CHAT_IDS:
        log.warning("ignored unauthorized chat_id=%s", chat_id)
        return
    if _command_handler is None:
        log.error("Pullback command handler is not configured")
        return
    threading.Thread(
        target=_command_handler, args=(text, chat_id), daemon=True
    ).start()


def poll_telegram_commands():
    url = _api_url("getUpdates")
    if url is None:
        log.error("PULLBACK_TELEGRAM_TOKEN missing; polling disabled")
        time.sleep(60)
        return
    last_id = 0
    while True:
        try:
            response = _SESSION.get(
                url,
                params={"offset": last_id + 1, "timeout": 30},
                timeout=35,
            ).json()
            for update in response.get("result", []):
                last_id = update["update_id"]
                _dispatch_update(update)
        except requests.RequestException as exc:
            log.error("poll network error: %s", exc)
            time.sleep(10)
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            log.error("poll response error: %s", exc)
            time.sleep(10)
