#!/usr/bin/env python3
"""تشخيص قصير لشكل رد شرائح الرافعة على Binance — مناسب لشاشة الجوال."""
import json
import requests

HEADERS = {"User-Agent": "Mozilla/5.0", "clienttype": "web", "lang": "en"}
URLS = [
    "https://www.binance.com/bapi/futures/v1/friendly/future/common/brackets",
    "https://www.binance.com/bapi/futures/v1/friendly/future/common/brackets?quoteAsset=USDT",
    "https://www.binance.com/bapi/futures/v1/friendly/future/common/brackets?symbol=BTCUSDT",
    "https://fapi.binance.com/fapi/v1/ping",
    "https://fapi.binance.com/fapi/v1/exchangeInfo",
]


def summarize(url, response):
    print("=" * 56)
    print(url)
    print("HTTP", response.status_code, "bytes", len(response.content))
    if response.status_code != 200:
        print(response.text[:220].replace("\n", " "))
        return
    if "exchangeInfo" in url:
        payload = response.json()
        symbols = [s["symbol"] for s in payload.get("symbols", [])
                   if s.get("contractType") == "PERPETUAL"
                   and s.get("quoteAsset") == "USDT"
                   and s.get("status") == "TRADING"]
        print("perpetual_usdt", len(symbols), "sample", symbols[:5])
        return
    if url.endswith("/ping"):
        print("ping_ok", response.text)
        return
    try:
        payload = response.json()
    except ValueError:
        print("not_json", response.text[:160])
        return
    print("top", sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__)
    data = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(data, str):
        print("data=STRING len", len(data))
        try:
            data = json.loads(data)
            print("decoded", type(data).__name__)
        except ValueError:
            print("prefix", data[:100])
            return
    print("data_type", type(data).__name__)
    if isinstance(data, dict):
        keys = sorted(data.keys())
        print("data_keys", keys[:15], "count", len(keys))
        first_key = keys[0] if keys else None
        first = data.get(first_key) if first_key else None
        print("first_key", first_key, "type", type(first).__name__)
        if isinstance(first, dict):
            print("first_keys", sorted(first.keys())[:15])
            for key, value in first.items():
                if isinstance(value, list) and value:
                    item = value[0]
                    print("list", key, "len", len(value),
                          "item0", sorted(item.keys())[:12] if isinstance(item, dict) else type(item).__name__)
                    break
        elif isinstance(first, list) and first:
            item = first[0]
            print("first_list_len", len(first),
                  "item0", sorted(item.keys())[:12] if isinstance(item, dict) else type(item).__name__)
    elif isinstance(data, list):
        print("data_len", len(data))
        if data:
            item = data[0]
            print("item0_type", type(item).__name__)
            if isinstance(item, dict):
                print("item0_keys", sorted(item.keys())[:15])
                for key, value in item.items():
                    if isinstance(value, list) and value:
                        nested = value[0]
                        print("list", key, "len", len(value),
                              "item0", sorted(nested.keys())[:12] if isinstance(nested, dict) else type(nested).__name__)
                        break
            else:
                print("item0", repr(item)[:120])
    else:
        print("data", repr(data)[:120])


def main():
    for url in URLS:
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException as exc:
            print("=" * 56)
            print(url)
            print("ERR", type(exc).__name__, exc)
            continue
        summarize(url, response)


if __name__ == "__main__":
    main()
