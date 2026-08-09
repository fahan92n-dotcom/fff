"""
bybit_leverage_limits.py — أقصى قيمة مركز مسموحة لكل عقد خطّي على Bybit عند رافعة معيّنة.

يقرأ risk-limit العامة (قيمة اسمية بالـ USDT مباشرة). قد يحجب Bybit بعض
المواقع الجغرافياً (HTTP 403)؛ شغّل الأداة من سيرفر يصل Bybit مثل سنغافورة.
"""
import csv
import argparse
import logging

import requests

log = logging.getLogger(__name__)

TOOL_VERSION = "2026-08-09e"
BYBIT_BASE = "https://api.bybit.com"
REQUEST_TIMEOUT = 30


class BracketsUnavailable(RuntimeError):
    """تعذّر جلب شرائح المخاطر من Bybit."""


def _get(path, params=None):
    response = requests.get(
        f"{BYBIT_BASE}{path}",
        params=params,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == 403:
        raise BracketsUnavailable(
            "Bybit يحجب هذا الموقع (HTTP 403). شغّل الأداة من سيرفر يصل Bybit "
            "(مثل Droplet سنغافورة)."
        )
    response.raise_for_status()
    payload = response.json()
    if payload.get("retCode") != 0:
        raise BracketsUnavailable(f"Bybit {path}: {payload}")
    return payload["result"]


def fetch_risk_limits(symbol=None):
    """كل شرائح المخاطر مع تقليب الصفحات (15 رمزاً في الصفحة للخطّي)."""
    rows = []
    cursor = None
    while True:
        params = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        if cursor:
            params["cursor"] = cursor
        result = _get("/v5/market/risk-limit", params)
        rows.extend(result.get("list") or [])
        cursor = result.get("nextPageCursor") or None
        if not cursor or symbol:
            break
    return rows


def max_notional_at_leverage(tiers, leverage):
    """أكبر riskLimitValue تسمح به الرافعة، مع رافعة الشريحة."""
    eligible = []
    for tier in tiers:
        max_lev = float(tier.get("maxLeverage") or 0)
        cap = float(tier.get("riskLimitValue") or 0)
        if max_lev >= leverage and cap > 0:
            eligible.append((cap, max_lev))
    if not eligible:
        return None, None
    return max(eligible, key=lambda item: item[0])


def build_rows(leverage):
    """يبني الترتيب التنازلي عند الرافعة المطلوبة."""
    try:
        all_tiers = fetch_risk_limits()
    except BracketsUnavailable:
        raise
    except requests.RequestException as exc:
        raise BracketsUnavailable(str(exc)) from exc

    by_symbol = {}
    for tier in all_tiers:
        by_symbol.setdefault(tier["symbol"], []).append(tier)

    rows = []
    for symbol, tiers in by_symbol.items():
        if not symbol.endswith("USDT"):
            continue
        notional, tier_leverage = max_notional_at_leverage(tiers, leverage)
        if not notional:
            continue
        rows.append({
            "symbol": symbol,
            "at_leverage": leverage,
            "tier_leverage": tier_leverage,
            "max_amount_usdt": notional,
            "margin_needed_usdt": notional / leverage,
        })
    rows.sort(key=lambda row: row["max_amount_usdt"], reverse=True)
    return rows


def print_table(rows, leverage, total=None):
    print(f"أقصى مبلغ صفقة على Bybit عند رافعة {leverage:g}x")
    print(f"{'#':>4}  {'SYMBOL':<20} {'AT':>6} {'TIER':>6} "
          f"{'MAX AMOUNT':>18} {'YOUR MARGIN':>14}")
    print("-" * 74)
    for index, row in enumerate(rows, 1):
        print(f"{index:>4}  {row['symbol']:<20} {row['at_leverage']:>5.0f}x "
              f"{row['tier_leverage']:>5.2f}x {row['max_amount_usdt']:>18,.0f} "
              f"{row['margin_needed_usdt']:>14,.0f}")
    print(f"\n{total if total is not None else len(rows)} symbols "
          f"allow {leverage:g}x leverage.")


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leverage", type=float, default=100)
    parser.add_argument("--top", type=int, default=0)
    parser.add_argument("--csv", metavar="PATH")
    args = parser.parse_args()
    print(f"bybit_leverage_limits {TOOL_VERSION}")

    try:
        rows = build_rows(args.leverage)
    except BracketsUnavailable as exc:
        raise SystemExit(str(exc)) from exc

    if not rows:
        print(f"No Bybit linear symbol allows {args.leverage:g}x leverage.")
        return
    print_table(rows[:args.top] if args.top else rows, args.leverage, total=len(rows))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
