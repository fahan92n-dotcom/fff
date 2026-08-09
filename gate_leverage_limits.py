"""
gate_leverage_limits.py — أقصى قيمة مركز على Gate.io USDT futures عند رافعة معيّنة.

شرائح risk_limit_tiers: risk_limit سقف القيمة الاسمية بالـ USDT، و leverage_max
أقصى رافعة للشريحة.
"""
import csv
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

log = logging.getLogger(__name__)
GATE = "https://api.gateio.ws/api/v4"
TIMEOUT = 30
WORKERS = 8
TOOL_VERSION = "2026-08-09g"


def _get(path, params=None):
    response = requests.get(
        f"{GATE}{path}", params=params,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def fetch_contracts():
    return [
        item for item in _get("/futures/usdt/contracts")
        if not item.get("in_delisting") and item.get("type") == "direct"
    ]


def fetch_tiers(contract):
    return _get("/futures/usdt/risk_limit_tiers", {"contract": contract})


def max_notional_at_leverage(tiers, leverage):
    eligible = []
    for tier in tiers:
        max_lev = float(tier.get("leverage_max") or 0)
        cap = float(tier.get("risk_limit") or 0)
        if max_lev >= leverage and cap > 0:
            eligible.append((cap, max_lev))
    if not eligible:
        return None, None
    return max(eligible, key=lambda item: item[0])


def build_rows(leverage):
    contracts = [
        item for item in fetch_contracts()
        if float(item.get("leverage_max") or 0) >= leverage
    ]
    rows = []

    def one(contract):
        name = contract["name"]
        try:
            tiers = fetch_tiers(name)
        except (requests.RequestException, ValueError, TypeError) as exc:
            log.warning("تخطي %s: %s", name, exc)
            return None
        notional, tier_leverage = max_notional_at_leverage(tiers, leverage)
        if not notional:
            return None
        return {
            "symbol": name,
            "at_leverage": leverage,
            "tier_leverage": tier_leverage,
            "max_amount_usdt": notional,
            "margin_needed_usdt": notional / leverage,
        }

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(one, contract) for contract in contracts]
        for future in as_completed(futures):
            row = future.result()
            if row:
                rows.append(row)
    rows.sort(key=lambda row: row["max_amount_usdt"], reverse=True)
    return rows


def print_table(rows, leverage, total=None):
    print(f"gate_leverage_limits {TOOL_VERSION}")
    print(f"أقصى مبلغ صفقة على Gate.io عند رافعة {leverage:g}x")
    print(f"{'#':>4}  {'SYMBOL':<18} {'AT':>6} {'TIER':>8} "
          f"{'MAX AMOUNT':>18} {'YOUR MARGIN':>14}")
    print("-" * 74)
    for index, row in enumerate(rows, 1):
        print(f"{index:>4}  {row['symbol']:<18} {row['at_leverage']:>5.0f}x "
              f"{row['tier_leverage']:>7.2f}x {row['max_amount_usdt']:>18,.0f} "
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
    rows = build_rows(args.leverage)
    if not rows:
        print(f"gate_leverage_limits {TOOL_VERSION}")
        print(f"No Gate USDT contract allows {args.leverage:g}x leverage.")
        return
    print_table(rows[:args.top] if args.top else rows, args.leverage, total=len(rows))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
