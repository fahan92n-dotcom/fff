"""
kucoin_leverage_limits.py — أقصى قيمة مركز مسموحة لكل عقد آجل على KuCoin عند رافعة معيّنة.

يقرأ العقود النشطة وشرائح risk-limit العامة. لعقود USDT يُعامل maxRiskLimit
كقيمة اسمية بالـ USDT مباشرة.
"""
import csv
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

log = logging.getLogger(__name__)

KUCOIN_FUTURES = "https://api-futures.kucoin.com"
REQUEST_TIMEOUT = 30
MAX_WORKERS = 8
TOOL_VERSION = "2026-08-09f"


def _get(path):
    response = requests.get(
        f"{KUCOIN_FUTURES}{path}",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if str(payload.get("code")) != "200000":
        raise RuntimeError(f"KuCoin {path}: {payload}")
    return payload["data"]


def fetch_contracts():
    """عقود USDT الدائمة المفتوحة."""
    return [
        item for item in _get("/api/v1/contracts/active")
        if item.get("quoteCurrency") == "USDT"
        and item.get("status") == "Open"
        and not item.get("isInverse")
    ]


def fetch_risk_limits(symbol):
    return _get(f"/api/v1/contracts/risk-limit/{symbol}")


def max_notional_at_leverage(tiers, leverage):
    """أكبر maxRiskLimit تسمح به الرافعة، مع رافعة الشريحة."""
    eligible = []
    for tier in tiers:
        max_lev = float(tier.get("maxLeverage") or 0)
        cap = float(tier.get("maxRiskLimit") or 0)
        if max_lev >= leverage and cap > 0:
            eligible.append((cap, max_lev))
    if not eligible:
        return None, None
    return max(eligible, key=lambda item: item[0])


def build_rows(leverage):
    contracts = fetch_contracts()
    candidates = [
        item for item in contracts
        if float(item.get("maxLeverage") or 0) >= leverage
    ]

    rows = []

    def one(contract):
        try:
            tiers = fetch_risk_limits(contract["symbol"])
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            log.warning("تخطي %s: %s", contract["symbol"], exc)
            return None
        notional, tier_leverage = max_notional_at_leverage(tiers, leverage)
        if not notional:
            return None
        return {
            "symbol": contract["symbol"],
            "at_leverage": leverage,
            "tier_leverage": tier_leverage,
            "max_amount_usdt": notional,
            "margin_needed_usdt": notional / leverage,
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(one, contract) for contract in candidates]
        for future in as_completed(futures):
            row = future.result()
            if row:
                rows.append(row)

    rows.sort(key=lambda row: row["max_amount_usdt"], reverse=True)
    return rows


def print_table(rows, leverage, total=None):
    print(f"kucoin_leverage_limits {TOOL_VERSION}")
    print(f"أقصى مبلغ صفقة على KuCoin عند رافعة {leverage:g}x")
    print(f"{'#':>4}  {'SYMBOL':<18} {'AT':>6} {'TIER':>6} "
          f"{'MAX AMOUNT':>18} {'YOUR MARGIN':>14}")
    print("-" * 72)
    for index, row in enumerate(rows, 1):
        print(f"{index:>4}  {row['symbol']:<18} {row['at_leverage']:>5.0f}x "
              f"{row['tier_leverage']:>5.0f}x {row['max_amount_usdt']:>18,.0f} "
              f"{row['margin_needed_usdt']:>14,.0f}")
    print(f"\n{total if total is not None else len(rows)} symbols "
          f"allow {leverage:g}x leverage.")
    print("ملاحظة: XBTUSDTM = BTC على KuCoin")


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

    rows = build_rows(args.leverage)
    if not rows:
        print(f"kucoin_leverage_limits {TOOL_VERSION}")
        print(f"No KuCoin USDT contract allows {args.leverage:g}x leverage.")
        return
    print_table(rows[:args.top] if args.top else rows, args.leverage, total=len(rows))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
