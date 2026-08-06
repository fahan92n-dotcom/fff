"""Strategy experiment variants — compare filters on last-week market data."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from html import escape as html_escape

from binance_data import CUSTOM_SYMBOLS, fast_prefetch_done, symbols_cache, symbols_cache_lock
from week_scan import (
    LOSS_PCT,
    WIN_PCT,
    _ensure_symbol_raw,
    format_week_trades_report,
    scan_week_trades,
)

log = logging.getLogger(__name__)

_experiments_lock = threading.Lock()
_experiments_running = False

# Default BTC correlation threshold for experiment #5 (tunable later).
DEFAULT_BTC_CORR_MIN = 0.50
DEFAULT_BTC_CORR_LOOKBACK = 50


@dataclass(frozen=True)
class StrategyVariant:
    """One experiment configuration. Empty overrides == live baseline."""

    id: str
    title_ar: str
    skip_donchian_confirm: bool = False
    ema_on_confirm: bool = False
    # None disables RSI-confirm; omit key in to_dict baseline via sentinel below.
    confirm_rsi_lookback: int | None = 30
    btc_corr_min: float | None = None
    btc_corr_lookback: int = DEFAULT_BTC_CORR_LOOKBACK
    # When False, confirm_rsi_lookback is left unset so step6 uses live default 30.
    override_rsi_lookback: bool = True

    def to_dict(self):
        payload = {
            "skip_donchian_confirm": self.skip_donchian_confirm,
            "ema_on_confirm": self.ema_on_confirm,
            "btc_corr_min": self.btc_corr_min,
            "btc_corr_lookback": self.btc_corr_lookback,
        }
        if self.override_rsi_lookback:
            payload["confirm_rsi_lookback"] = self.confirm_rsi_lookback
        return payload


EXPERIMENT_VARIANTS = (
    StrategyVariant(
        id="baseline",
        title_ar="1) النسخة الحالية بدون تعديلات",
        override_rsi_lookback=False,
    ),
    StrategyVariant(
        id="ema_confirm",
        title_ar=(
            "2) EMA50 فريم التأكيد وقت الدخول: "
            "شراء فوق EMA50 / بيع تحت EMA50"
        ),
        ema_on_confirm=True,
        override_rsi_lookback=False,
    ),
    StrategyVariant(
        id="no_donchian_confirm",
        title_ar="3) إلغاء Donchian فريم التأكيد",
        skip_donchian_confirm=True,
        override_rsi_lookback=False,
    ),
    StrategyVariant(
        id="rsi_off",
        title_ar="4أ) إلغاء RSI التأكيد",
        confirm_rsi_lookback=None,
        override_rsi_lookback=True,
    ),
    StrategyVariant(
        id="rsi_50",
        title_ar="4ب) RSI التأكيد على 50 شمعة",
        confirm_rsi_lookback=50,
        override_rsi_lookback=True,
    ),
    StrategyVariant(
        id="btc_corr",
        title_ar=(
            f"5) ارتباط CC مع BTC ≥ {DEFAULT_BTC_CORR_MIN:g} "
            f"على الفريم الأساسي ({DEFAULT_BTC_CORR_LOOKBACK} شمعة)"
        ),
        btc_corr_min=DEFAULT_BTC_CORR_MIN,
        btc_corr_lookback=DEFAULT_BTC_CORR_LOOKBACK,
        override_rsi_lookback=False,
    ),
)


def score_scan_result(result):
    """
    Rank helper for a week-scan result.

    Primary: expectancy in %-points = wins*WIN_PCT - losses*LOSS_PCT
    Secondary: win rate on closed trades
    Tertiary: closed trade count
    """
    wins = len(result.get("wins") or [])
    losses = len(result.get("losses") or [])
    opens = len(result.get("opens") or [])
    closed = wins + losses
    expectancy = wins * WIN_PCT - losses * LOSS_PCT
    win_rate = (wins / closed * 100.0) if closed else 0.0
    return {
        "wins": wins,
        "losses": losses,
        "opens": opens,
        "closed": closed,
        "total": int(result.get("total") or 0),
        "expectancy": expectancy,
        "win_rate": win_rate,
        "avg_expectancy": (expectancy / closed) if closed else 0.0,
    }


def rank_key(summary):
    return (
        summary["expectancy"],
        summary["win_rate"],
        summary["closed"],
    )


def run_strategy_experiments(
    symbols=None,
    *,
    days=7,
    variants=None,
    progress_callback=None,
):
    """Run each variant on the same preloaded OHLCV and return ranked rows."""
    variants = tuple(variants or EXPERIMENT_VARIANTS)
    if symbols is None:
        with symbols_cache_lock:
            symbols = list(symbols_cache) or list(CUSTOM_SYMBOLS)
    else:
        symbols = list(symbols)

    preloaded = {}
    total_symbols = len(symbols)
    for index, symbol in enumerate(symbols, start=1):
        if progress_callback is not None and (
            index == 1 or index == total_symbols or index % 20 == 0
        ):
            progress_callback("preload", index, total_symbols, symbol, None)
        preloaded[symbol] = _ensure_symbol_raw(symbol)

    btc_raw = preloaded.get("BTCUSDT") or _ensure_symbol_raw("BTCUSDT")
    rows = []
    for variant_index, variant in enumerate(variants, start=1):
        if progress_callback is not None:
            progress_callback(
                "variant",
                variant_index,
                len(variants),
                None,
                variant,
            )

        def _scan_progress(index, total, symbol, _variant=variant):
            if progress_callback is not None:
                progress_callback(
                    "scan",
                    index,
                    total,
                    symbol,
                    _variant,
                )

        result = scan_week_trades(
            symbols,
            days=days,
            variant=variant.to_dict(),
            preloaded_raw=preloaded,
            btc_raw_by_tf=btc_raw,
            progress_callback=_scan_progress,
        )
        summary = score_scan_result(result)
        rows.append(
            {
                "variant": variant,
                "result": result,
                "summary": summary,
            }
        )

    rows.sort(key=lambda row: rank_key(row["summary"]), reverse=True)
    return {
        "ready": True,
        "symbols_scanned": total_symbols,
        "rows": rows,
        "winner": rows[0] if rows else None,
    }


def format_experiments_report(bundle):
    """Telegram HTML chunks comparing all variants and naming the winner."""
    if not bundle.get("ready"):
        return ["⚠️ تعذر تشغيل تجارب الاستراتيجية."]

    rows = bundle.get("rows") or []
    if not rows:
        return ["⚠️ ما فيه نتائج تجارب."]

    winner = bundle.get("winner") or rows[0]
    win_v = winner["variant"]
    win_s = winner["summary"]

    header = (
        "🧪 <b>تجارب الاستراتيجية — آخر 7 أيام</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        f"عملات: <b>{bundle.get('symbols_scanned', 0)}</b>\n"
        f"معيار النجاح: <b>+{WIN_PCT:g}%</b> | "
        f"الخسارة: <b>{LOSS_PCT:g}%</b> ضد الصفقة\n"
        f"الترتيب حسب التوقع: "
        f"(نجاح×{WIN_PCT:g}) − (خسارة×{LOSS_PCT:g})\n"
        "\n"
        f"🏆 <b>الأفضل الآن:</b> {html_escape(win_v.title_ar)}\n"
        f"صفقات مغلقة: <b>{win_s['closed']}</b> | "
        f"✅ {win_s['wins']} | ❌ {win_s['losses']} | "
        f"⏳ {win_s['opens']}\n"
        f"نسبة النجاح: <b>{win_s['win_rate']:.1f}%</b>\n"
        f"التوقع الإجمالي: <b>{win_s['expectancy']:+.2f}</b> نقطة٪\n"
    )

    lines = ["📊 <b>مقارنة النسخ:</b>"]
    for rank, row in enumerate(rows, start=1):
        variant = row["variant"]
        summary = row["summary"]
        medal = "🏆" if rank == 1 else f"{rank})"
        lines.append(
            f"{medal} {html_escape(variant.title_ar)}\n"
            f"   ✅ {summary['wins']} | ❌ {summary['losses']} | "
            f"⏳ {summary['opens']} | "
            f"نجاح {summary['win_rate']:.1f}% | "
            f"توقع {summary['expectancy']:+.2f}"
        )

    note = (
        "\n📌 الاستراتيجية الحية ما تتغير تلقائيًا — "
        "بعد ما تعتمد الأفضل نثبّتها في الكود."
    )
    return [header, "\n".join(lines) + note]


def handle_experiments_command(chat_id, send_telegram):
    """Telegram entry: compare strategy variants on last-week market data."""
    global _experiments_running

    if not _experiments_lock.acquire(blocking=False):
        send_telegram("⏳ تجارب الاستراتيجية شغّالة الآن — انتظر.", chat_id)
        return
    if _experiments_running:
        _experiments_lock.release()
        send_telegram("⏳ تجارب الاستراتيجية شغّالة الآن — انتظر.", chat_id)
        return

    _experiments_running = True
    try:
        if not fast_prefetch_done.is_set():
            send_telegram(
                "📡 جاري تحميل البيانات وتشغيل التجارب "
                "(قد يطول أول مرة)...",
                chat_id,
            )
        else:
            send_telegram(
                "🧪 جاري تشغيل تجارب الاستراتيجية على آخر 7 أيام...\n"
                "1) Baseline\n"
                "2) تأكيد EMA50 وقت الدخول (شراء فوق / بيع تحت)\n"
                "3) بدون Donchian تأكيد\n"
                "4أ) بدون RSI تأكيد  4ب) RSI=50\n"
                "5) CC مع BTC ≥ 0.50 على الفريم الأساسي\n"
                "ثم نرتّب ونطلع الأفضل.",
                chat_id,
            )

        last_msg_key = {"k": None}

        def on_progress(phase, index, total, symbol, variant):
            if phase == "preload" and (
                index in (1, total) or index % 25 == 0
            ):
                key = ("preload", index)
                if last_msg_key["k"] != key:
                    last_msg_key["k"] = key
                    send_telegram(
                        f"📦 تجهيز البيانات: {index}/{total} "
                        f"(<code>{html_escape(symbol or '')}</code>)",
                        chat_id,
                    )
            elif phase == "variant":
                title = variant.title_ar if variant is not None else ""
                send_telegram(
                    f"🔬 تجربة {index}/{total}: {html_escape(title)}",
                    chat_id,
                )

        bundle = run_strategy_experiments(progress_callback=on_progress)
        for chunk in format_experiments_report(bundle):
            send_telegram(chunk, chat_id)

        # Also send the winner's detailed week report for inspection.
        winner = bundle.get("winner")
        if winner is not None:
            send_telegram(
                f"📋 تفاصيل صفقة النسخة الأفضل "
                f"({html_escape(winner['variant'].title_ar)}):",
                chat_id,
            )
            for chunk in format_week_trades_report(winner["result"]):
                send_telegram(chunk, chat_id)
    except Exception as exc:
        log.exception("experiments command failed")
        send_telegram(
            f"❌ فشلت تجارب الاستراتيجية: "
            f"<code>{html_escape(str(exc))}</code>",
            chat_id,
        )
    finally:
        _experiments_running = False
        _experiments_lock.release()
