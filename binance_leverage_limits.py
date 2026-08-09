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
BAPI_BASE = "https://www.binance.com/bapi/futures/v1"
BAPI_V2_BASE = "https://www.binance.com/bapi/futures/v2"
REQUEST_TIMEOUT = 30

# لا توثّق Binance مساراً عاماً للشرائح، وقد تغيّر الصيغة دون إشعار، لذا تُجرّب
# المرشّحات بالترتيب ويُبلَّغ عن سبب فشل كل واحد.
PUBLIC_SOURCES = (
    ("GET", f"{BAPI_BASE}/friendly/future/common/brackets", None),
    ("GET", f"{BAPI_BASE}/public/future/common/brackets", None),
    ("GET", f"{BAPI_BASE}/friendly/future/common/brackets?quoteAsset=USDT", None),
    ("GET", f"{BAPI_BASE}/public/future/common/brackets?quoteAsset=USDT", None),
    ("GET", f"{BAPI_V2_BASE}/friendly/future/common/brackets", None),
    ("GET", f"{BAPI_V2_BASE}/public/future/common/brackets", None),
    ("POST", f"{BAPI_BASE}/friendly/future/common/brackets", {}),
    ("POST", f"{BAPI_BASE}/public/future/common/brackets", {}),
    ("POST", f"{BAPI_BASE}/friendly/future/common/brackets", {"quoteAsset": "USDT"}),
)


class BracketsUnavailable(RuntimeError):
    """تعذّر جلب شرائح الرافعة من أي مصدر متاح."""


def _normalise_public(payload):
    """يحوّل رد المسار العام إلى {symbol: [(max_leverage, notional_cap), ...]}.

    تختلف أسماء الحقول بين إصدارات المسار العام، فتُقبل التسميات المعروفة كلها.
    """
    entries = payload.get("data") if isinstance(payload, dict) else payload
    brackets = {}
    for entry in entries or []:
        symbol = entry.get("symbol")
        raw_tiers = entry.get("riskBrackets") or entry.get("brackets") or []
        tiers = []
        for tier in raw_tiers:
            leverage = (tier.get("maxOpenPosLeverage")
                        or tier.get("initialLeverage")
                        or tier.get("maxLeverage"))
            cap = (tier.get("bracketNotionalCap")
                   or tier.get("notionalCap")
                   or tier.get("maxNotionalValue"))
            if leverage and cap:
                tiers.append((leverage, cap))
        if symbol and tiers:
            brackets[symbol] = tiers
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
    """يجرّب المرشّحات العامة بالترتيب، ويعيد أول رد يحمل شرائح فعلية.

    يعيد (brackets, attempts) حيث attempts سجل نصّي لما جرى مع كل مرشّح.
    """
    attempts = []
    for method, url, body in PUBLIC_SOURCES:
        label = f"{method} {url}"
        try:
            response = requests.request(
                method, url, json=body,
                headers={"User-Agent": "Mozilla/5.0", "clienttype": "web"},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            attempts.append(f"{label} -> {type(exc).__name__}")
            continue

        if response.status_code != 200:
            snippet = " ".join(response.text[:200].split())
            attempts.append(f"{label} -> HTTP {response.status_code} {snippet}")
            continue

        try:
            brackets = _normalise_public(response.json())
        except ValueError:
            attempts.append(f"{label} -> رد غير JSON")
            continue

        if brackets:
            attempts.append(f"{label} -> OK ({len(brackets)} رمز)")
            return brackets, attempts
        attempts.append(f"{label} -> HTTP 200 بلا شرائح")

    return {}, attempts


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
    """يحاول المسارات العامة أولاً ثم الموقّع، ويشرح سبب الفشل إن تعذّر الجميع."""
    brackets, attempts = fetch_public_brackets()
    if brackets:
        return brackets

    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if api_key and api_secret:
        return fetch_signed_brackets(api_key, api_secret)

    report = "\n  ".join(attempts)
    raise BracketsUnavailable(
        "تعذّر جلب شرائح الرافعة من أي مسار عام:\n  " + report
        + "\n\nلا توثّق Binance مساراً عاماً للشرائح، فالحل الموثوق هو مفتاح "
          "API للقراءة فقط: اضبط BINANCE_API_KEY و BINANCE_API_SECRET ثم أعد "
          "التشغيل."
    )


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
