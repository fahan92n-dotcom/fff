"""
mexc_leverage_limits.py — أقصى حجم صفقة مسموح لكل عقد آجل على MEXC عند رافعة معيّنة.

يقرأ جدول العقود العام من MEXC ويحسب لكل رمز أعلى قيمة صفقة (notional)
يمكن فتحها عند الرافعة المطلوبة، مرتّبة تنازلياً. الأداة للقراءة فقط
ولا تتصل بأي مفتاح API ولا تتفاعل مع مسار البوت.
"""
import csv
import argparse
import logging

import requests

log = logging.getLogger(__name__)

MEXC_CONTRACT_BASE = "https://contract.mexc.com/api/v1/contract"
REQUEST_TIMEOUT = 30

# MEXC تدرج عقود أسهم وسلع إلى جانب العملات الرقمية على نفس المنصة.
NON_CRYPTO_SUFFIXES = ("STOCK_USDT",)
NON_CRYPTO_SYMBOLS = frozenset({
    "XAU_USDT", "XAUT_USDT", "XAG_USDT", "XPT_USDT", "XPD_USDT",
    "SILVER_USDT", "COPPER_USDT", "UKOIL_USDT", "USOIL_USDT", "NGAS_USDT",
    "TESLA_USDT", "NVIDIA_USDT", "SOXL_USDT", "EWY_USDT",
    "SPX500_USDT", "NAS100_USDT", "COINBASE_USDT", "ROBINHOOD_USDT",
})


def _fetch(path):
    """يطلب مساراً عاماً من MEXC ويعيد حقل data."""
    response = requests.get(
        f"{MEXC_CONTRACT_BASE}/{path}",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["data"]


def max_volume_at_leverage(contract, leverage):
    """أكبر حجم مركز (بعدد العقود) تسمح به الرافعة المطلوبة.

    لكل عقد شرائح مخاطر: كلما كبر المركز انخفضت الرافعة المتاحة. تعيد الدالة
    سقف أوسع شريحة ما زالت تسمح بالرافعة المطلوبة، أو None إذا لم تسمح بها
    أي شريحة.

    ترد MEXC الشرائح بأحد نمطين: جدول صريح في riskLimitCustom، أو نمط زيادة
    تدريجية تُشتق فيه كل شريحة من riskBaseVol و riskIncrVol و riskIncrImr.
    """
    custom_tiers = contract.get("riskLimitCustom")
    if contract.get("riskLimitMode") == "CUSTOM" and custom_tiers:
        allowed = [tier["maxVol"] for tier in custom_tiers
                   if (tier.get("maxLeverage") or 0) >= leverage]
        return max(allowed) if allowed else None

    base_imr = contract["initialMarginRate"]
    incr_imr = contract.get("riskIncrImr") or 0
    tier_count = max(1, contract.get("riskLevelLimit") or 1)

    allowed_volume = None
    for tier in range(1, tier_count + 1):
        imr = base_imr + (tier - 1) * incr_imr
        if imr <= 0 or 1 / imr < leverage:
            break
        allowed_volume = (contract["riskBaseVol"]
                          + (tier - 1) * (contract.get("riskIncrVol") or 0))
    return allowed_volume


def _is_crypto(symbol):
    return (symbol not in NON_CRYPTO_SYMBOLS
            and not symbol.endswith(NON_CRYPTO_SUFFIXES))


def build_rows(leverage, crypto_only=False):
    """يبني قائمة مرتّبة تنازلياً بأقصى قيمة صفقة لكل عقد عند الرافعة المطلوبة."""
    contracts = _fetch("detail")
    prices = {ticker["symbol"]: ticker["lastPrice"] for ticker in _fetch("ticker")}

    rows = []
    for contract in contracts:
        symbol = contract["symbol"]
        if contract.get("settleCoin") != "USDT" or contract.get("state") != 0:
            continue
        if (contract.get("maxLeverage") or 0) < leverage:
            continue
        if crypto_only and not _is_crypto(symbol):
            continue

        price = prices.get(symbol)
        volume = max_volume_at_leverage(contract, leverage)
        if not price or not volume:
            continue

        notional = volume * contract["contractSize"] * price
        per_order = contract.get("limitMaxVol") or contract["maxVol"]
        rows.append({
            "symbol": symbol,
            "price": price,
            "max_leverage": contract["maxLeverage"],
            "max_position_usdt": notional,
            "margin_needed_usdt": notional / leverage,
            "max_order_usdt": per_order * contract["contractSize"] * price,
        })

    rows.sort(key=lambda row: row["max_position_usdt"], reverse=True)
    return rows


def print_table(rows, leverage, total=None):
    """يطبع الجدول بصيغة مقروءة في الطرفية."""
    print(f"{'#':>4}  {'SYMBOL':<20} {'PRICE':>13} {'MAXLEV':>7} "
          f"{'MAX AMOUNT':>18} {'YOUR MARGIN':>14} {'MAX ORDER':>16}")
    print("-" * 98)
    for index, row in enumerate(rows, 1):
        print(f"{index:>4}  {row['symbol']:<20} {row['price']:>13,.6g} "
              f"{row['max_leverage']:>7.0f} {row['max_position_usdt']:>18,.0f} "
              f"{row['margin_needed_usdt']:>14,.0f} {row['max_order_usdt']:>16,.0f}")
    print(f"\n{total if total is not None else len(rows)} contracts "
          f"allow {leverage:g}x leverage.")


def write_csv(rows, path):
    """يحفظ النتائج في ملف CSV."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    """نقطة الدخول لسطر الأوامر."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leverage", type=float, default=100)
    parser.add_argument("--top", type=int, default=0, help="0 = كل النتائج")
    parser.add_argument("--crypto-only", action="store_true",
                        help="استبعاد عقود الأسهم والسلع والمؤشرات")
    parser.add_argument("--csv", metavar="PATH")
    args = parser.parse_args()

    rows = build_rows(args.leverage, crypto_only=args.crypto_only)
    if not rows:
        print(f"No contract allows {args.leverage:g}x leverage.")
        return

    print_table(rows[:args.top] if args.top else rows, args.leverage, total=len(rows))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
