"""Config for the standalone Pullback bot (separate Telegram credentials)."""

import os


def _read_port(value):
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")
    return port


# Prefer Pullback-specific secrets; fall back to TELEGRAM_* so an existing
# single-bot deploy can switch from Cascade to Pullback without renaming envs.
TELEGRAM_TOKEN = (
    os.environ.get("PULLBACK_TELEGRAM_TOKEN")
    or os.environ.get("TELEGRAM_TOKEN")
    or ""
)
TELEGRAM_CHAT_ID = (
    os.environ.get("PULLBACK_TELEGRAM_CHAT_ID")
    or os.environ.get("TELEGRAM_CHAT_ID")
    or ""
)
PORT = _read_port(os.environ.get("PULLBACK_PORT", os.environ.get("PORT", "8081")))
ALLOWED_CHAT_IDS = frozenset(
    chat_id.strip()
    for chat_id in os.environ.get(
        "PULLBACK_ALLOWED_CHAT_IDS",
        os.environ.get("ALLOWED_CHAT_IDS", TELEGRAM_CHAT_ID),
    ).split(",")
    if chat_id.strip()
)
