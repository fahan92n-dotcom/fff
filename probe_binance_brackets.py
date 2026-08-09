#!/usr/bin/env python3
"""Print a short structure sample of Binance public brackets."""
import json
import requests

URLS = [
    "https://www.binance.com/bapi/futures/v1/friendly/future/common/brackets",
    "https://www.binance.com/bapi/futures/v1/friendly/future/common/brackets?quoteAsset=USDT",
]
headers = {"User-Agent": "Mozilla/5.0", "clienttype": "web"}

for url in URLS:
    print("=" * 60)
    print(url)
    try:
        r = requests.get(url, headers=headers, timeout=30)
    except Exception as exc:
        print("ERR", type(exc).__name__, exc)
        continue
    print("HTTP", r.status_code, "bytes", len(r.content))
    if r.status_code != 200:
        print(r.text[:300])
        continue
    try:
        payload = r.json()
    except ValueError:
        print("not json", r.text[:200])
        continue

    print("top_keys", sorted(payload.keys()) if isinstance(payload, dict) else type(payload).__name__)
    data = payload.get("data") if isinstance(payload, dict) else payload
    if isinstance(data, str):
        print("data is STRING len", len(data), "prefix", data[:120])
        try:
            data = json.loads(data)
            print("decoded data type", type(data).__name__)
        except ValueError:
            continue
    print("data_type", type(data).__name__)
    if isinstance(data, dict):
        print("data_keys", sorted(data.keys())[:20])
        first_key = next(iter(data), None)
        first = data.get(first_key) if first_key else None
        print("first_key", first_key, "first_type", type(first).__name__)
        if isinstance(first, dict):
            print("first_keys", sorted(first.keys())[:20])
            for k, v in first.items():
                if isinstance(v, list) and v:
                    print("list_field", k, "len", len(v), "item0_keys",
                          sorted(v[0].keys())[:15] if isinstance(v[0], dict) else type(v[0]).__name__)
                    break
        elif isinstance(first, list) and first:
            print("first_list_len", len(first), "item0", type(first[0]).__name__,
                  sorted(first[0].keys())[:15] if isinstance(first[0], dict) else first[0])
    elif isinstance(data, list):
        print("data_len", len(data))
        if data:
            item = data[0]
            print("item0_type", type(item).__name__)
            if isinstance(item, dict):
                print("item0_keys", sorted(item.keys())[:20])
                for k, v in item.items():
                    if isinstance(v, list) and v:
                        print("list_field", k, "len", len(v), "item0_keys",
                              sorted(v[0].keys())[:15] if isinstance(v[0], dict) else type(v[0]).__name__)
                        break
            else:
                print("item0_sample", repr(item)[:200])
    else:
        print("data_sample", repr(data)[:200])
