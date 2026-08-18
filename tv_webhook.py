"""TradingView alert webhook: store Cascade results; Telegram asks on demand."""

from __future__ import annotations

import hmac
import json
import os
import threading
from datetime import datetime, timezone
from html import escape as html_escape
from pathlib import Path
from urllib.parse import parse_qs, urlparse

MAX_BODY_BYTES = 8192
ALLOWED_EVENTS = frozenset({"entry", "result"})
ALLOWED_SIDES = frozenset({"buy", "sell"})
HIST_WIN_PCT = 0.67
HIST_LOSS_PCT = 0.53
SCORE_COMMANDS = frozenset(
    {
        "/نتائج",
        "/نتيجة",
        "/score",
        "/tvscore",
        "/tv_score",
        "نتائج",
        "نتيجة",
        "score",
    }
)

_LOCK = threading.Lock()
_STATE = {
    "wins": 0,
    "losses": 0,
    "open": 0,
    "symbols": {},
    "updated_at": None,
}


def webhook_secret():
    """Shared token expected on /tv?token=..."""
    return os.environ.get("TV_WEBHOOK_SECRET", "")


def score_path():
    """JSON file that survives process restarts on the same disk."""
    return Path(os.environ.get("TV_SCORE_PATH", "tv_score.json"))


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


def _as_count(value):
    """Parse a non-negative int count, or None if the field is absent/invalid."""
    if value is None or value == "":
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    if count < 0:
        return None
    return count


def _empty_state():
    return {
        "wins": 0,
        "losses": 0,
        "open": 0,
        "symbols": {},
        "updated_at": None,
    }


def reset_score():
    """Clear in-memory totals and the score file (tests)."""
    with _LOCK:
        _STATE.clear()
        _STATE.update(_empty_state())
        try:
            score_path().unlink()
        except FileNotFoundError:
            pass


def _load_unlocked():
    path = score_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        _STATE.clear()
        _STATE.update(_empty_state())
        return
    except OSError:
        return
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return
    if not isinstance(data, dict):
        return
    wins = _as_count(data.get("wins")) or 0
    losses = _as_count(data.get("losses")) or 0
    open_n = _as_count(data.get("open")) or 0
    symbols = data.get("symbols")
    if not isinstance(symbols, dict):
        symbols = {}
    _STATE.clear()
    _STATE.update(
        {
            "wins": wins,
            "losses": losses,
            "open": open_n,
            "symbols": symbols,
            "updated_at": data.get("updated_at"),
        }
    )


def _save_unlocked():
    path = score_path()
    payload = json.dumps(_STATE, ensure_ascii=False, indent=2)
    path.write_text(payload, encoding="utf-8")


def snapshot():
    """Copy of the running win/loss totals."""
    with _LOCK:
        if _STATE.get("updated_at") is None and not _STATE.get("symbols"):
            _load_unlocked()
        return {
            "wins": int(_STATE.get("wins") or 0),
            "losses": int(_STATE.get("losses") or 0),
            "open": int(_STATE.get("open") or 0),
            "symbols": dict(_STATE.get("symbols") or {}),
            "updated_at": _STATE.get("updated_at"),
        }


def _validate_event(data):
    event = data.get("event")
    side = data.get("side")
    if event not in ALLOWED_EVENTS or side not in ALLOWED_SIDES:
        raise ValueError("invalid event or side")
    price = data.get("price")
    try:
        float(price)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid price") from exc
    return event, side


def apply_event(data):
    """Merge one TradingView entry/result into the stored score. No Telegram."""
    event, _side = _validate_event(data)
    symbol = str(data.get("symbol") or "—").strip() or "—"
    wins = _as_count(data.get("wins"))
    losses = _as_count(data.get("losses"))
    open_n = _as_count(data.get("open"))
    win_flag = data.get("win")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with _LOCK:
        _load_unlocked()
        symbols = _STATE.setdefault("symbols", {})
        row = symbols.setdefault(symbol, {"wins": 0, "losses": 0, "open": 0})
        if wins is not None:
            row["wins"] = wins
        elif event == "result":
            if win_flag is True or win_flag == 1 or win_flag == "true":
                row["wins"] = int(row.get("wins") or 0) + 1
            elif win_flag is False or win_flag == 0 or win_flag == "false":
                row["losses"] = int(row.get("losses") or 0) + 1
            else:
                raise ValueError("invalid win flag")
        if losses is not None:
            row["losses"] = losses
        if open_n is not None:
            row["open"] = open_n
        elif event == "entry":
            row["open"] = int(row.get("open") or 0) + 1
        elif event == "result":
            row["open"] = max(0, int(row.get("open") or 0) - 1)
        _STATE["wins"] = sum(int(item.get("wins") or 0) for item in symbols.values())
        _STATE["losses"] = sum(int(item.get("losses") or 0) for item in symbols.values())
        _STATE["open"] = sum(int(item.get("open") or 0) for item in symbols.values())
        _STATE["updated_at"] = now
        _save_unlocked()
        return {
            "wins": int(_STATE.get("wins") or 0),
            "losses": int(_STATE.get("losses") or 0),
            "open": int(_STATE.get("open") or 0),
            "symbols": dict(_STATE.get("symbols") or {}),
            "updated_at": _STATE.get("updated_at"),
        }


def is_score_command(text):
    """True when the user is asking for TradingView win/loss totals."""
    raw = (text or "").strip()
    if not raw:
        return False
    first = raw.split(maxsplit=1)[0]
    if "@" in first:
        first = first.split("@", 1)[0]
    return first in SCORE_COMMANDS or first.lower() in SCORE_COMMANDS


def format_score_message(data=None):
    """Arabic summary of stored TradingView win/loss counts."""
    state = data if data is not None else snapshot()
    wins = int(state.get("wins") or 0)
    losses = int(state.get("losses") or 0)
    open_n = int(state.get("open") or 0)
    updated = html_escape(str(state.get("updated_at") or "—"))
    symbols = state.get("symbols") or {}
    if wins == 0 and losses == 0 and open_n == 0 and not symbols:
        return (
            "📊 <b>نتائج Cascade من TradingView</b>\n"
            "ما وصله نتائج بعد. فعّل التنبيه على الشارت:\n"
            "Alert → Any alert() function call → "
            "Webhook <code>/tv?token=SECRET</code>\n"
            "بعدها أرسل <code>/نتائج</code> متى ما تبي الرقم."
        )
    lines = [
        "📊 <b>نتائج Cascade من TradingView</b>",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"✅ ناجحة: <b>{wins}</b>",
        f"❌ فاشلة: <b>{losses}</b>",
        f"⏳ مفتوحة: <b>{open_n}</b>",
        f"📈 ربح +{HIST_WIN_PCT:g}%  |  📉 خسارة {HIST_LOSS_PCT:g}%",
        f"🕐 آخر تحديث: {updated}",
    ]
    if len(symbols) > 1:
        lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        for name in sorted(symbols):
            row = symbols[name]
            safe = html_escape(str(name))
            lines.append(
                f"💱 <b>{safe}</b> — ناجحة {int(row.get('wins') or 0)} | "
                f"فاشلة {int(row.get('losses') or 0)} | "
                f"مفتوحة {int(row.get('open') or 0)}"
            )
    lines.append("أرسل <code>/نتائج</code> متى ما تبي العدد.")
    return "\n".join(lines)


def handle_score_command(chat_id, send_telegram):
    """Reply to one on-demand Telegram score request."""
    send_telegram(format_score_message(), chat_id)


def ingest(raw, query, send_telegram=None):
    """Validate one POST and store the result without sending Telegram.

    ``send_telegram`` is unused: totals are sent only when the user asks.
    Returns ``(status_code, body_bytes)``.
    """
    del send_telegram
    try:
        data = parse_payload(raw)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return 400, b"bad payload"
    if not _authorized(query, data.get("token")):
        return 401, b"unauthorized"
    try:
        apply_event(data)
    except (OSError, ValueError):
        return 400, b"bad fields"
    return 200, b"ok"


def handle_http(handler, send_telegram=None):
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
