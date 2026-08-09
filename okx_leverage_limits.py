"""
okx_leverage_limits.py — أقصى قيمة مركز مسموحة لكل عقد SWAP على OKX عند رافعة معيّنة.

يقرأ instruments + position-tiers العامة، ويحسب لكل رمز أكبر قيمة اسمية
(maxSz × ctVal × السعر) ما زالت شريحتها تسمح بالرافعة المطلوبة.
"""
import csv
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

log = logging.getLogger(__name__)

OKX_BASE = "https://www.okx.com"
REQUEST_TIMEOUT = 30
MAX_WORKERS = 8


def _get(path, params=None):
    response = requests.get(
        f"{OKX_BASE}{path}",
        params=params,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if str(payload.get("code")) != "0":
        raise RuntimeError(f"OKX {path}: {payload}")
    return payload["data"]


def fetch_instruments():
    """عقود SWAP الحيّة المسوّاة بـ USDT."""
    return [
        item for item in _get("/api/v5/public/instruments", {"instType": "SWAP"})
        if item.get("settleCcy") == "USDT" and item.get("state") == "live"
    ]


def fetch_tickers():
    """آخر سعر لكل عقد."""
    return {item["instId"]: float(item["last"])
            for item in _get("/api/v5/market/tickers", {"instType": "SWAP"})
            if item.get("last")}


def fetch_tiers(uly):
    """شرائح المركز لعملة أساس واحدة (cross)."""
    return _get("/api/v5/public/position-tiers", {
        "instType": "SWAP",
        "tdMode": "cross",
        "uly": uly,
    })


def max_contracts_at_leverage(tiers, leverage):
    """أكبر maxSz (بعدد العقود) تسمح به الرافعة، مع رافعة الشريحة."""
    eligible = []
    for tier in tiers:
        max_lever = float(tier.get("maxLever") or 0)
        max_sz = float(tier.get("maxSz") or 0)
        if max_lever >= leverage and max_sz > 0:
            eligible.append((max_sz, max_lever))
    if not eligible:
        return None, None
    return max(eligible, key=lambda item: item[0])


def build_rows(leverage):
    """يبني الترتيب التنازلي لكل العقود التي تسمح بالرافعة."""
    instruments = fetch_instruments()
    tickers = fetch_tickers()
    candidates = [
        item for item in instruments
        if float(item.get("lever") or 0) >= leverage
    ]
    ulys = sorted({item["uly"] for item in candidates})
    tiers_by_uly = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_tiers, uly): uly for uly in ulys}
        for future in as_completed(futures):
            uly = futures[future]
            try:
                tiers_by_uly[uly] = future.result()
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                log.warning("تخطي %s: %s", uly, exc)

    rows = []
    for item in candidates:
        price = tickers.get(item["instId"])
        tiers = tiers_by_uly.get(item["uly"]) or []
        contracts, tier_leverage = max_contracts_at_leverage(tiers, leverage)
        if not price or not contracts:
            continue
        ct_val = float(item["ctVal"])
        notional = contracts * ct_val * price
        rows.append({
            "symbol": item["instId"],
            "at_leverage": leverage,
            "tier_leverage": tier_leverage,
            "max_amount_usdt": notional,
            "margin_needed_usdt": notional / leverage,
        })
    rows.sort(key=lambda row: row["max_amount_usdt"], reverse=True)
    return rows


def print_table(rows, leverage, total=None):
    print(f"أقصى مبلغ صفقة على OKX عند رافعة {leverage:g}x")
    print(f"{'#':>4}  {'SYMBOL':<22} {'AT':>6} {'TIER':>6} "
          f"{'MAX AMOUNT':>18} {'YOUR MARGIN':>14}")
    print("-" * 76)
    for index, row in enumerate(rows, 1):
        print(f"{index:>4}  {row['symbol']:<22} {row['at_leverage']:>5.0f}x "
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

    rows = build_rows(args.leverage)
    if not rows:
        print(f"No OKX SWAP allows {args.leverage:g}x leverage.")
        return
    print_table(rows[:args.top] if args.top else rows, args.leverage, total=len(rows))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
