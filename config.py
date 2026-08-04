"""Environment-backed application configuration."""

import os


def _read_port(value):
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")
    return port


TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
PORT = _read_port(os.environ.get("PORT", "8080"))
ALLOWED_CHAT_IDS = frozenset(
    chat_id.strip()
    for chat_id in os.environ.get(
        "ALLOWED_CHAT_IDS",
        TELEGRAM_CHAT_ID,
    ).split(",")
    if chat_id.strip()
)
