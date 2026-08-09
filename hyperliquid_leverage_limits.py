"""
hyperliquid_leverage_limits.py — أقصى قيمة مركز على Hyperliquid عند رافعة معيّنة.

Hyperliquid تنشر maxLeverage لكل أصل مع جدول هوامش متدرج (marginTables).
الحد عند رافعة L هو lowerBound للشريحة التالية التي تهبط دون L، أو بلا سقف
إن لم توجد. ملاحظة: أعلى رافعة على المنصة حالياً 40x (BTC)، فلا يوجد عقد
يسمح بـ 100x.
"""
import csv
import argparse
import logging
import math

import requests

log = logging.getLogger(__name__)

HL_INFO = "https://api.hyperliquid.xyz/info"
REQUEST_TIMEOUT = 30


def fetch_meta_and_ctxs():
    response = requests.post(
        HL_INFO,
        json={"type": "metaAndAssetCtxs"},
        headers={"Content-Type": "application/json"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def max_notional_at_leverage(margin_tiers, leverage):
    """أعلى قيمة اسمية تسمح بالرافعة، مع رافعة الشريحة التي تغطيها.

    الشرائح مرتبة بـ lowerBound تصاعدياً. الشريحة التالية التي maxLeverage فيها
    أقل من المطلوب تضع السقف. إن لم توجد، السقف غير محدود.
    """
    tiers = sorted(
        ((float(t["lowerBound"]), float(t["maxLeverage"])) for t in margin_tiers),
        key=lambda item: item[0],
    )
    if not tiers or tiers[0][1] < leverage:
        return None, None

    # الرافعة المتاحة عند notional=0 هي رافعة الشريحة الأولى.
    current_lev = tiers[0][1]
    for index, (lower, max_lev) in enumerate(tiers):
        if max_lev < leverage:
            # السقف هو بداية هذي الشريحة؛ الرافعة المستخدمة هي السابقة.
            prev_lev = tiers[index - 1][1] if index else current_lev
            return lower, prev_lev
        current_lev = max_lev
    return math.inf, current_lev


def build_rows(leverage):
    meta, ctxs = fetch_meta_and_ctxs()
    tables = {table_id: spec["marginTiers"] for table_id, spec in meta["marginTables"]}
    rows = []
    for asset, ctx in zip(meta["universe"], ctxs):
        if asset.get("isDelisted"):
            continue
        if float(asset.get("maxLeverage") or 0) < leverage:
            continue
        tiers = tables.get(asset["marginTableId"]) or [
            {"lowerBound": "0.0", "maxLeverage": asset["maxLeverage"]}
        ]
        notional, tier_leverage = max_notional_at_leverage(tiers, leverage)
        if notional is None:
            continue
        rows.append({
            "symbol": asset["name"],
            "at_leverage": leverage,
            "tier_leverage": tier_leverage,
            "max_amount_usdt": notional,
            "margin_needed_usdt": (notional / leverage) if math.isfinite(notional) else math.inf,
            "mark_px": float(ctx.get("markPx") or 0),
        })
    rows.sort(key=lambda row: (
        math.isfinite(row["max_amount_usdt"]),
        row["max_amount_usdt"],
    ), reverse=True)
    return rows


def _fmt(value):
    if not math.isfinite(value):
        return f"{'unlimited':>18}"
    return f"{value:>18,.0f}"


def print_table(rows, leverage, total=None):
    print(f"أقصى مبلغ صفقة على Hyperliquid عند رافعة {leverage:g}x")
    print(f"{'#':>4}  {'SYMBOL':<14} {'AT':>6} {'TIER':>6} "
          f"{'MAX AMOUNT':>18} {'YOUR MARGIN':>14}")
    print("-" * 68)
    for index, row in enumerate(rows, 1):
        print(f"{index:>4}  {row['symbol']:<14} {row['at_leverage']:>5.0f}x "
              f"{row['tier_leverage']:>5.0f}x {_fmt(row['max_amount_usdt'])} "
              f"{_fmt(row['margin_needed_usdt'])}")
    print(f"\n{total if total is not None else len(rows)} assets "
          f"allow {leverage:g}x leverage.")


def write_csv(rows, path):
    serializable = []
    for row in rows:
        item = dict(row)
        if not math.isfinite(item["max_amount_usdt"]):
            item["max_amount_usdt"] = "unlimited"
            item["margin_needed_usdt"] = "unlimited"
        serializable.append(item)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable[0]))
        writer.writeheader()
        writer.writerows(serializable)


def main():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leverage", type=float, default=100)
    parser.add_argument("--top", type=int, default=0)
    parser.add_argument("--csv", metavar="PATH")
    args = parser.parse_args()

    rows = build_rows(args.leverage)
    if not rows:
        print(f"No Hyperliquid asset allows {args.leverage:g}x leverage "
              f"(platform max is currently 40x on BTC).")
        return
    print_table(rows[:args.top] if args.top else rows, args.leverage, total=len(rows))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()
