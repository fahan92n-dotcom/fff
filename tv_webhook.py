"""TradingView alert webhook → Telegram results (no step readings)."""

from __future__ import annotations

import hmac
import json
import os
from html import escape as html_escape
from urllib.parse import parse_qs, urlparse

MAX_BODY_BYTES = 8192
ALLOWED_EVENTS = frozenset({"entry", "result"})
ALLOWED_SIDES = frozenset({"buy", "sell"})


def webhook_secret():
    """Shared token expected on /tv?token=..."""
    return os.environ.get("TV_WEBHOOK_SECRET", "")


def _query_token(query):
    values = query.get("token") or query.get("key") or []
    if not values:
        return ""
    return str(values[0])


def _authorized(query, payload_token):
    secret = webhook_secret()
    if not secret:
        return False
    offered = _query_token(query) or (payload_token or "")
    if not offered:
        return False
    return hmac.compare_digest(offered, secret)


def parse_payload(raw):
    """Decode one TradingView alert body into a dict."""
    if raw is None:
        raise ValueError("empty body")
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    text = text.strip()
    if not text:
        raise ValueError("empty body")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("payload must be an object")
    return data


def format_result_message(data):
    """Build a short Arabic Telegram message for one TV result."""
    event = data.get("event")
    side = data.get("side")
    if event not in ALLOWED_EVENTS or side not in ALLOWED_SIDES:
        raise ValueError("invalid event or side")
    tag = html_escape(str(data.get("tag") or "—"))
    symbol = html_escape(str(data.get("symbol") or "—"))
    price = data.get("price")
    try:
        price_txt = f"{float(price):.6g}"
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid price") from exc
    side_ar = "شراء" if side == "buy" else "بيع"
    if event == "entry":
        icon = "🟢" if side == "buy" else "🔴"
        return (
            f"{icon} <b>صفقة {side_ar}</b>\n"
            f"💱 <b>{symbol}</b>\n"
            f"📐 {tag}\n"
            f"💰 {price_txt}"
        )
    win = data.get("win")
    if win is True or win == 1 or win == "true":
        return (
            f"✅ <b>ناجحة</b> — {side_ar} {tag}\n"
            f"💱 <b>{symbol}</b>\n"
            f"💰 {price_txt}\n"
            f"📈 الهدف 0.67%"
        )
    if win is False or win == 0 or win == "false":
        return (
            f"❌ <b>فاشلة</b> — {side_ar} {tag}\n"
            f"💱 <b>{symbol}</b>\n"
            f"💰 {price_txt}\n"
            f"📉 ارتداد 0.53%"
        )
    raise ValueError("invalid win flag")


def ingest(raw, query, send_telegram):
    """Validate one POST and forward a Telegram result.

    Returns ``(status_code, body_bytes)``.
    """
    try:
        data = parse_payload(raw)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return 400, b"bad payload"
    if not _authorized(query, data.get("token")):
        return 401, b"unauthorized"
    try:
        message = format_result_message(data)
    except ValueError:
        return 400, b"bad fields"
    if not send_telegram(message):
        return 502, b"telegram failed"
    return 200, b"ok"


def handle_http(handler, send_telegram):
    """Serve POST /tv on an existing BaseHTTPRequestHandler."""
    parsed = urlparse(handler.path)
    if parsed.path.rstrip("/") != "/tv":
        handler.send_response(404)
        handler.end_headers()
        handler.wfile.write(b"not found")
        return
    raw_len = handler.headers.get("Content-Length", "0")
    try:
        length = int(raw_len)
    except (TypeError, ValueError):
        handler.send_response(400)
        handler.end_headers()
        handler.wfile.write(b"bad length")
        return
    if length < 0 or length > MAX_BODY_BYTES:
        handler.send_response(413)
        handler.end_headers()
        handler.wfile.write(b"too large")
        return
    raw = handler.rfile.read(length) if length else b""
    status, reply = ingest(raw, parse_qs(parsed.query), send_telegram)
    handler.send_response(status)
    handler.end_headers()
    handler.wfile.write(reply)
