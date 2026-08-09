"""
binance_leverage_limits.py — أقصى قيمة مركز مسموحة لكل عقد آجل على Binance عند رافعة معيّنة.

تعتمد Binance على شرائح قيمة اسمية (notional brackets): كلما كبرت قيمة
المركز انخفضت الرافعة القصوى المسموحة. تقرأ الأداة هذه الشرائح وتستخرج لكل
رمز أكبر قيمة مركز يمكن فتحها عند الرافعة المطلوبة، مرتّبة تنازلياً.

مصدر البيانات المفضّل هو مسار عام لا يحتاج مفتاحاً. إن تعذّر وكان
BINANCE_API_KEY و BINANCE_API_SECRET موجودين في البيئة، تستخدم الأداة المسار
الموقّع بدلاً منه. كلا المسارين للقراءة فقط.

⚠️ يحجب Binance الطلبات القادمة من بعض المواقع الجغرافية (HTTP 451)، فشغّل
الأداة من نفس البيئة التي يعمل منها البوت.
"""
import os
import csv
import time
import hmac
import hashlib
import argparse
import logging
import urllib.parse

import requests

log = logging.getLogger(__name__)

BINANCE_FUTURES_BASE = "https://fapi.binance.com"
PUBLIC_BRACKETS_URL = (
    "https://www.binance.com/bapi/futures/v1/public/future/common/brackets"
)
REQUEST_TIMEOUT = 30


class BracketsUnavailable(RuntimeError):
    """تعذّر جلب شرائح الرافعة من أي مصدر متاح."""


def _normalise_public(payload):
    """يحوّل رد المسار العام إلى {symbol: [(max_leverage, notional_cap), ...]}."""
    brackets = {}
    for entry in payload.get("data") or []:
        tiers = [
            (tier.get("maxOpenPosLeverage"), tier.get("bracketNotionalCap"))
            for tier in entry.get("riskBrackets") or []
        ]
        if tiers:
            brackets[entry["symbol"]] = tiers
    return brackets


def _normalise_signed(payload):
    """يحوّل رد المسار الموقّع إلى نفس الشكل."""
    brackets = {}
    for entry in payload:
        tiers = [
            (tier.get("initialLeverage"), tier.get("notionalCap"))
            for tier in entry.get("brackets") or []
        ]
        if tiers:
            brackets[entry["symbol"]] = tiers
    return brackets


def fetch_public_brackets():
    """يجلب الشرائح من المسار العام بلا مفاتيح."""
    response = requests.get(
        PUBLIC_BRACKETS_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return _normalise_public(response.json())


def fetch_signed_brackets(api_key, api_secret):
    """يجلب الشرائح من fapi عبر طلب موقّع."""
    query = urllib.parse.urlencode({"timestamp": int(time.time() * 1000)})
    signature = hmac.new(
        api_secret.encode(), query.encode(), hashlib.sha256
    ).hexdigest()
    response = requests.get(
        f"{BINANCE_FUTURES_BASE}/fapi/v1/leverageBracket?{query}&signature={signature}",
        headers={"X-MBX-APIKEY": api_key},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return _normalise_signed(response.json())


def load_brackets():
    """يحاول المسار العام أولاً ثم الموقّع، ويشرح سبب الفشل إن تعذّر الاثنان."""
    try:
        return fetch_public_brackets()
    except requests.RequestException as exc:
        log.warning("تعذّر المسار العام: %s", exc)
        public_error = exc

    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if not (api_key and api_secret):
        raise BracketsUnavailable(
            "المسار العام غير متاح من هذا الموقع "
            f"({public_error}). اضبط BINANCE_API_KEY و BINANCE_API_SECRET "
            "أو شغّل الأداة من بيئة يصلها Binance."
        )
    return fetch_signed_brackets(api_key, api_secret)


def max_notional_at_leverage(tiers, leverage):
    """أكبر قيمة مركز تسمح بها الرافعة المطلوبة، أو None إن لم تسمح بها شريحة."""
    caps = [cap for max_leverage, cap in tiers
            if max_leverage and cap and max_leverage >= leverage]
    return max(caps) if caps else None


def build_rows(leverage, brackets=None, quote="USDT"):
    """يبني قائمة مرتّبة تنازلياً بأقصى قيمة مركز لكل رمز عند الرافعة المطلوبة."""
    brackets = brackets if brackets is not None else load_brackets()

    rows = []
    for symbol, tiers in brackets.items():
        if quote and not symbol.endswith(quote):
            continue
        notional = max_notional_at_leverage(tiers, leverage)
        if not notional:
            continue
        rows.append({
            "symbol": symbol,
            "max_leverage": max(lev for lev, _ in tiers if lev),
            "max_amount_usdt": notional,
            "margin_needed_usdt": notional / leverage,
        })

    rows.sort(key=lambda row: row["max_amount_usdt"], reverse=True)
    return rows


def print_table(rows, leverage, total=None):
    """يطبع الجدول بصيغة مقروءة في الطرفية."""
    print(f"{'#':>4}  {'SYMBOL':<20} {'MAXLEV':>7} {'MAX AMOUNT':>18} {'YOUR MARGIN':>14}")
    print("-" * 68)
    for index, row in enumerate(rows, 1):
        print(f"{index:>4}  {row['symbol']:<20} {row['max_leverage']:>7.0f} "
              f"{row['max_amount_usdt']:>18,.0f} {row['margin_needed_usdt']:>14,.0f}")
    print(f"\n{total if total is not None else len(rows)} symbols "
          f"allow {leverage:g}x leverage.")


def write_csv(rows, path):
    """يحفظ النتائج في ملف CSV."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """نقطة الدخول لسطر الأوامر."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leverage", type=float, default=100)
    parser.add_argument("--top", type=int, default=0, help="0 = كل النتائج")
    parser.add_argument("--quote", default="USDT", help="عملة التسعير، فارغة = الكل")
    parser.add_argument("--csv", metavar="PATH")
    args = parser.parse_args()

    try:
        rows = build_rows(args.leverage, quote=args.quote)
    except BracketsUnavailable as exc:
        raise SystemExit(str(exc)) from exc

    if not rows:
        print(f"No symbol allows {args.leverage:g}x leverage.")
        return

    print_table(rows[:args.top] if args.top else rows, args.leverage, total=len(rows))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
