#!/usr/bin/env python3
"""تشخيص سريع لشرائح رافعة Bybit — مناسب لشاشة الجوال."""
import json
from collections import Counter
import requests

BASE = "https://api.bybit.com"
H = {"User-Agent": "Mozilla/5.0"}


def get(path, params=None):
    r = requests.get(f"{BASE}{path}", params=params, headers=H, timeout=30)
    print("=" * 56)
    print(path, params or "")
    print("HTTP", r.status_code)
    if r.status_code != 200:
        print(r.text[:200])
        return None
    data = r.json()
    print("retCode", data.get("retCode"), "retMsg", data.get("retMsg"))
    return data.get("result")


def main():
    # 1) risk-limit بدون رمز
    result = get("/v5/market/risk-limit", {"category": "linear"})
    if result is not None:
        lst = result.get("list") or []
        print("list_len", len(lst))
        if lst:
            print("keys", sorted(lst[0].keys()))
            print("sample0", {k: lst[0].get(k) for k in (
                "symbol", "maxLeverage", "riskLimitValue", "maintenanceMargin",
                "initialMargin", "isLowestRisk", "id")})
            levers = []
            for item in lst:
                try:
                    levers.append(float(item.get("maxLeverage") or 0))
                except (TypeError, ValueError):
                    pass
            print("max_leverage_seen", max(levers) if levers else None)
            print("leverage_counts", Counter(int(x) for x in levers if x).most_common(10))
            # أعلى 5 رموز من حيث الرافعة
            best = sorted(lst, key=lambda x: float(x.get("maxLeverage") or 0), reverse=True)[:8]
            for item in best:
                print("top", item.get("symbol"), "lev", item.get("maxLeverage"),
                      "cap", item.get("riskLimitValue"))

    # 2) BTC تحديداً
    result = get("/v5/market/risk-limit", {"category": "linear", "symbol": "BTCUSDT"})
    if result is not None:
        lst = result.get("list") or []
        print("BTC tiers", len(lst))
        for item in lst[:10]:
            print({k: item.get(k) for k in (
                "symbol", "maxLeverage", "riskLimitValue", "id", "isLowestRisk")})

    # 3) instruments: ما أعلى رافعة معلنة؟
    result = get("/v5/market/instruments-info", {"category": "linear", "limit": 1000})
    if result is not None:
        lst = result.get("list") or []
        perps = [x for x in lst if x.get("contractType") == "LinearPerpetual"
                 and x.get("status") == "Trading" and x.get("quoteCoin") == "USDT"]
        print("usdt_perps", len(perps))
        if perps:
            print("inst_keys_sample", sorted(perps[0].keys())[:20])
            # leverageFilter إن وجد
            if "leverageFilter" in perps[0]:
                best = sorted(
                    perps,
                    key=lambda x: float((x.get("leverageFilter") or {}).get("maxLeverage") or 0),
                    reverse=True,
                )[:10]
                for item in best:
                    lf = item.get("leverageFilter") or {}
                    print("inst", item["symbol"], "maxLev", lf.get("maxLeverage"))


if __name__ == "__main__":
    main()
