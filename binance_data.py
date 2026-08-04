"""
binance_data.py — جلب OHLCV من Binance، التحقق من الرموز، والكاش المشترك.

⚠️ أي تعديل على هذا الملف يؤثر مباشرة على مسار البيانات في fahadal92.py.
نقل صافٍ للكود — بدون تغيير سلوكي مقصود.
"""
import os
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests
import pandas as pd

log = logging.getLogger(__name__)

# ------------------------------------------
# Binance / Market Settings
# ------------------------------------------

BINANCE_FUTURES_BASE = "https://fapi.binance.com"
BINANCE_SPOT_BASE = "https://api.binance.com"
BINANCE_BASE = BINANCE_SPOT_BASE
MARKET_MODE = os.environ.get("MARKET_MODE", "futures").lower()  # futures | spot
TOP_SYMBOLS_LIMIT = 100

# ------------------------------------------
# Custom Fixed Symbol List (يحل محل أعلى 100 عملة بالحجم)
# ------------------------------------------
CUSTOM_SYMBOLS = [
    "BTCUSDT", "XRPUSDT", "ETHUSDT", "SOLUSDT", "TRXUSDT",
    "HYPEUSDT", "DOGEUSDT", "ZECUSDT", "XLMUSDT", "XMRUSDT",
    "LINKUSDT", "ADAUSDT", "BCHUSDT", "LTCUSDT", "SUIUSDT",
    "HBARUSDT", "AVAXUSDT", "ZROUSDT", "NEARUSDT", "TAOUSDT",
    "UNIUSDT", "ONDOUSDT", "SKYUSDT", "DOTUSDT", "PENDLEUSDT",
    "AAVEUSDT", "WLDUSDT", "MORPHOUSDT", "BSVUSDT", "QNTUSDT",
    "KASUSDT", "ENAUSDT", "ALGOUSDT", "PUMPUSDT", "XTZUSDT",
    "ARBUSDT", "LITUSDT", "INJUSDT", "APTUSDT", "CAKEUSDT",
    "ETHFIUSDT", "DASHUSDT", "VIRTUALUSDT", "AEROUSDT", "VETUSDT",
    "PENGUUSDT", "PYTHUSDT", "SUNUSDT", "FETUSDT", "SEIUSDT",
    "CRVUSDT", "TIAUSDT", "LDOUSDT", "IMXUSDT",
    "CFXUSDT", "JASMYUSDT", "SYRUPUSDT", "XPLUSDT", "OPUSDT",
    "KAIAUSDT", "1000FLOKIUSDT", "EIGENUSDT", "RAYSOLUSDT", "CHZUSDT",
    "RUNEUSDT", "TWTUSDT", "SANDUSDT", "MANAUSDT", "FARTCOINUSDT",
    "THETAUSDT", "ARUSDT", "BATUSDT", "SFPUSDT", "1INCHUSDT",
    "DYDXUSDT", "DEEPUSDT", "GALAUSDT", "EGLDUSDT", "GRASSUSDT",
    "SNXUSDT", "ORDIUSDT", "ZENUSDT", "BANKUSDT", "HOODUSDT",
    "MRVLUSDT", "CRCLUSDT", "SNDKUSDT", "MUUSDT", "SKHYNIXUSDT", "KORUUSDT",
    "RENDERUSDT", "1000SHIBUSDT", "POLUSDT", "ATOMUSDT", "NOTUSDT", "POPCATUSDT",
    "ALCHUSDT", "ZILUSDT", "TOWNSUSDT", "BLESSUSDT",

]

TF_MAP = {"1m": "1m", "30m": "30m", "60m": "1h"}

CACHE_MAX_CANDLES = {"1m": 45_000, "30m": 5_500, "60m": 4_500}
API_FETCH_CANDLES = {"1m": 45_000, "30m": 5_500, "60m": 4_500}
FAST_FETCH_CANDLES = {"1m": 45_000, "30m": 5_500, "60m": 4_500}
UPDATE_BUFFER_SECONDS = 20  # small safety buffer before next 30m boundary (API/network jitter)
UPDATER_30M_INTERVAL_SECONDS = 30 * 60 - UPDATE_BUFFER_SECONDS

# ------------------------------------------
# Shared State (OHLCV cache / symbols)
# ------------------------------------------

symbols_cache = []
symbols_cache_lock = threading.Lock()
invalid_symbols_cache = []
invalid_symbols_lock = threading.Lock()
invalid_symbols_reason_cache = {}
invalid_symbols_reason_lock = threading.Lock()
ohlcv_cache = {}
ohlcv_cache_lock = threading.Lock()

fast_prefetch_done = threading.Event()
prefetch_done = threading.Event()
cache_updated_event = threading.Event()

_local = threading.local()

# Optional Telegram notifier — يُربَط من fahadal92 عبر set_telegram_sender
# لتجنّب اعتماد دائري بين الموديولين.
_telegram_sender = None


def set_telegram_sender(fn):
    """يربط دالة إرسال Telegram (مثل send_telegram) لاستخدامها في prefetch/update loops."""
    global _telegram_sender
    _telegram_sender = fn


def _notify_telegram(msg):
    sender = _telegram_sender
    if sender is not None:
        sender(msg)


# ------------------------------------------
# HTTP Session
# ------------------------------------------

def get_session():
    if not hasattr(_local, "s"):
        session = requests.Session()
        session.headers.update({"Accept-Encoding": "gzip", "User-Agent": "Mozilla/5.0"})
        _local.s = session
    return _local.s


# ------------------------------------------
# Symbol validation
# ------------------------------------------

def validate_symbols_with_reasons(symbols, market="futures"):
    valid, invalid = [], []
    reasons = {}

    try:
        if market == "futures":
            url = f"{BINANCE_FUTURES_BASE}/fapi/v1/exchangeInfo"
            market_tag = "FUTURES"
        else:
            url = f"{BINANCE_SPOT_BASE}/api/v3/exchangeInfo"
            market_tag = "SPOT"

        resp = get_session().get(url, timeout=20).json()
        raw_symbols = resp.get("symbols", [])
        by_symbol = {s.get("symbol"): s for s in raw_symbols if s.get("symbol")}

        for sym in symbols:
            meta = by_symbol.get(sym)
            if not meta:
                invalid.append(sym)
                reasons[sym] = f"NOT_FOUND_IN_{market_tag}_EXCHANGE_INFO"
                continue

            status = str(meta.get("status", "")).upper()
            if status != "TRADING":
                invalid.append(sym)
                reasons[sym] = f"NOT_TRADING({status})"
                continue

            valid.append(sym)

    except requests.RequestException as exc:
        log.error("❌ validate_symbols_with_reasons network error: %s", exc)
        return list(symbols), [], {}
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        log.error("❌ validate_symbols_with_reasons response error: %s", exc)
        return list(symbols), [], {}

    return valid, invalid, reasons


# ------------------------------------------
# Binance OHLCV
# ------------------------------------------

def _parse_binance_klines(resp):
    df = pd.DataFrame(resp, columns=["ts", "open", "high", "low", "close", "vol", "close_time", "quote_vol", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    for col in ["open", "high", "low", "close", "vol"]:
        df[col] = df[col].astype(float)
    df["ts"] = pd.to_datetime(df["ts"].astype(int), unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)[["ts", "open", "high", "low", "close", "vol"]]


def _get_ohlcv_impl(symbol, tf, limit, klines_url):
    binance_tf = TF_MAP.get(tf, "1m")
    try:
        resp = get_session().get(
            klines_url,
            params={"symbol": symbol, "interval": binance_tf, "limit": min(limit, 1000)},
            timeout=10,
        ).json()
        if isinstance(resp, list) and resp:
            return _parse_binance_klines(resp)
    except requests.RequestException as exc:
        log.error("get_ohlcv %s %s: %s", symbol, tf, exc)
    return pd.DataFrame()


def get_ohlcv(symbol, tf, limit=500):
    return _get_ohlcv_impl(symbol, tf, limit, f"{BINANCE_SPOT_BASE}/api/v3/klines")


def get_ohlcv_futures(symbol, tf, limit=500):
    return _get_ohlcv_impl(symbol, tf, limit, f"{BINANCE_FUTURES_BASE}/fapi/v1/klines")


def _get_ohlcv_full_impl(symbol, tf, target, klines_url, market_label):
    binance_tf = TF_MAP.get(tf, "1m")
    tf_ms_map = {"1m": 60_000, "30m": 1_800_000, "60m": 3_600_000}
    tf_ms = tf_ms_map.get(tf, 60_000)
    bin_max = 1000
    all_dfs, end_ms, fetched, retries = [], int(time.time() * 1000), 0, 0

    while fetched < target:
        batch = min(bin_max, target - fetched)
        start_ms = end_ms - batch * tf_ms
        try:
            r = get_session().get(
                klines_url,
                params={
                    "symbol": symbol,
                    "interval": binance_tf,
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": batch,
                },
                timeout=15,
            )

            if r.status_code in (429, 418):
                retry_after = int(r.headers.get("Retry-After", 30))
                log.warning("⏳ %s rate-limit %s على %s، انتظار %s ثانية", market_label, r.status_code, symbol, retry_after)
                time.sleep(retry_after)
                continue

            resp = r.json()
            if not isinstance(resp, list) or not resp:
                retries += 1
                if retries >= 3:
                    break
                time.sleep(2 ** retries)
                continue

            df = _parse_binance_klines(resp)
            all_dfs.insert(0, df)
            fetched += len(df)
            retries = 0
            first_ts_ms = int(df["ts"].iloc[0].timestamp() * 1000)
            end_ms = first_ts_ms - 1
            if len(df) < batch:
                break

        except requests.RequestException:
            retries += 1
            if retries >= 3:
                break
            time.sleep(2)

    return (
        pd.concat(all_dfs)
        .drop_duplicates(subset="ts")
        .sort_values("ts")
        .reset_index(drop=True)
        if all_dfs
        else pd.DataFrame()
    )


def get_ohlcv_full(symbol, tf, target):
    return _get_ohlcv_full_impl(symbol, tf, target, f"{BINANCE_SPOT_BASE}/api/v3/klines", "Spot")


def get_ohlcv_full_futures(symbol, tf, target):
    return _get_ohlcv_full_impl(symbol, tf, target, f"{BINANCE_FUTURES_BASE}/fapi/v1/klines", "Futures")


def cache_merge(symbol, tf, new_df):
    if new_df.empty:
        return
    key = (symbol, tf)
    maxc = CACHE_MAX_CANDLES.get(tf, 5000)
    with ohlcv_cache_lock:
        old = ohlcv_cache.get(key)
        if old is not None and not old.empty:
            merged = pd.concat([old, new_df]).drop_duplicates(subset="ts", keep="last").sort_values("ts")
            ohlcv_cache[key] = merged.tail(maxc).reset_index(drop=True)
        else:
            ohlcv_cache[key] = new_df.tail(maxc).reset_index(drop=True)


def get_cached(symbol, tf):
    with ohlcv_cache_lock:
        df = ohlcv_cache.get((symbol, tf))
    return df.copy() if df is not None else pd.DataFrame()


def cleanup_old_symbols_cache():
    with symbols_cache_lock:
        active_symbols = set(symbols_cache)
    with ohlcv_cache_lock:
        stale_keys = [k for k in ohlcv_cache if k[0] not in active_symbols]
        for k in stale_keys:
            del ohlcv_cache[k]
    if stale_keys:
        log.info("🧹 حذف %d مفتاح كاش قديم", len(stale_keys))


def _prefetch_all_impl(symbols, get_full_fn, fast_msg, full_msg):
    def fetch_sym_fast(sym):
        for tf, n in FAST_FETCH_CANDLES.items():
            df = get_full_fn(sym, tf, target=n)
            cache_merge(sym, tf, df)

    def fetch_sym_full(sym):
        for tf, n in API_FETCH_CANDLES.items():
            df = get_full_fn(sym, tf, target=n)
            cache_merge(sym, tf, df)

    log.info("🚀 بدء التحميل السريع %s...", fast_msg)
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(fetch_sym_fast, symbols)
    fast_prefetch_done.set()
    _notify_telegram(f"⚡ <b>التحميل السريع {fast_msg} اكتمل — البوت يعمل الآن!</b>")

    log.info("📦 بدء التحميل الكامل %s...", full_msg)
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(fetch_sym_full, symbols)
    prefetch_done.set()
    _notify_telegram(f"✅ <b>التحميل الكامل {full_msg} اكتمل وجاهز للعمل!</b>")


def prefetch_all(symbols):
    _prefetch_all_impl(symbols, get_ohlcv_full, "Spot", "Spot")


def prefetch_all_futures(symbols):
    _prefetch_all_impl(symbols, get_ohlcv_full_futures, "Futures", "Futures")


def _update_batch_impl(symbols, tf, limit, fetch_fn):
    def fetch_one(sym):
        df = fetch_fn(sym, tf, limit=limit)
        if not df.empty:
            cache_merge(sym, tf, df)
    with ThreadPoolExecutor(max_workers=30) as executor:
        executor.map(fetch_one, symbols)


def _update_batch(symbols, tf, limit):
    _update_batch_impl(symbols, tf, limit, get_ohlcv)


def _update_batch_futures(symbols, tf, limit):
    _update_batch_impl(symbols, tf, limit, get_ohlcv_futures)


def _cache_updater_1m_impl(update_batch_fn):
    while True:
        if not fast_prefetch_done.is_set():
            time.sleep(5)
            continue
        with symbols_cache_lock:
            syms = list(symbols_cache)
        if syms:
            update_batch_fn(syms, "1m", limit=5)
            cache_updated_event.set()
        time.sleep(55)


def cache_updater_1m():
    _cache_updater_1m_impl(_update_batch)


def cache_updater_1m_futures():
    _cache_updater_1m_impl(_update_batch_futures)


def _cache_updater_30m_impl(update_batch_fn):
    while True:
        if not fast_prefetch_done.is_set():
            time.sleep(5)
            continue
        with symbols_cache_lock:
            syms = list(symbols_cache)
        if syms:
            update_batch_fn(syms, "30m", limit=5)
        time.sleep(UPDATER_30M_INTERVAL_SECONDS)


def cache_updater_30m():
    _cache_updater_30m_impl(_update_batch)


def cache_updater_30m_futures():
    _cache_updater_30m_impl(_update_batch_futures)


def _cache_updater_60m_impl(update_batch_fn):
    while True:
        time.sleep(3600)
        if fast_prefetch_done.is_set():
            with symbols_cache_lock:
                syms = list(symbols_cache)
            if syms:
                update_batch_fn(syms, "60m", limit=5)


def cache_updater_60m():
    _cache_updater_60m_impl(_update_batch)


def cache_updater_60m_futures():
    _cache_updater_60m_impl(_update_batch_futures)


def _refresh_symbols_once(market, first_run):
    valid_symbols, invalid_symbols, invalid_reasons = validate_symbols_with_reasons(
        CUSTOM_SYMBOLS,
        market=market,
    )

    with symbols_cache_lock:
        symbols_cache[:] = valid_symbols
    with invalid_symbols_lock:
        invalid_symbols_cache[:] = invalid_symbols
    with invalid_symbols_reason_lock:
        invalid_symbols_reason_cache.clear()
        invalid_symbols_reason_cache.update(invalid_reasons)

    log.info(
        "✅ العملات الصالحة: %s — أول 5: %s",
        len(symbols_cache),
        symbols_cache[:5],
    )
    if invalid_symbols:
        log.warning("❌ العملات غير الصالحة: %s", invalid_symbols)

    cleanup_old_symbols_cache()
    market_label = "Futures" if market == "futures" else "Spot"
    if invalid_symbols:
        msg_lines = [
            f"⚠️ <b>عملات غير متاحة على Binance {market_label} ولن يتم مسحها:</b>",
            "━━━━━━━━━━━━━━━━━━━━",
        ]
        msg_lines += [f"❌ <code>{symbol}</code>" for symbol in invalid_symbols]
        _notify_telegram("\n".join(msg_lines))
    elif first_run:
        _notify_telegram(
            f"✅ جميع العملات ({len(valid_symbols)}) صالحة ومتاحة على "
            f"{market_label}."
        )

    if not fast_prefetch_done.is_set():
        prefetch_target = (
            prefetch_all_futures if market == "futures" else prefetch_all
        )
        threading.Thread(
            target=prefetch_target,
            args=(list(symbols_cache),),
            daemon=True,
        ).start()
    return False


def _update_symbols_loop(market):
    first_run = True
    while True:
        first_run = _refresh_symbols_once(market, first_run)
        time.sleep(3600)


def update_symbols_loop_futures():
    _update_symbols_loop("futures")


def update_symbols_loop():
    _update_symbols_loop("spot")


def get_last_closed_candle(symbol, tf):
    """جلب آخر شمعة مُغلقة 100% من Binance (Spot أو Futures حسب MARKET_MODE)."""
    try:
        fetch = get_ohlcv_futures if MARKET_MODE == "futures" else get_ohlcv
        df = fetch(symbol, tf, limit=2)

        if df.empty or len(df) < 2:
            log.warning("⚠️ بيانات ناقصة لـ %s فريم %s", symbol, tf)
            return None

        now = datetime.now(timezone.utc)
        last_candle = df.iloc[-1]
        last_ts = last_candle["ts"]

        tf_minutes = {"1m": 1, "30m": 30, "60m": 60}.get(tf, 1)
        candle_close_time = last_ts + pd.Timedelta(minutes=tf_minutes)

        # الشمعة الأخيرة مغلقة 100%
        if now >= candle_close_time:
            return {
                "close": float(last_candle["close"]),
                "open": float(last_candle["open"]),
                "high": float(last_candle["high"]),
                "low": float(last_candle["low"]),
                "timestamp": last_ts,
                "closed": True
            }

        # الشمعة السابقة مغلقة تماماً
        if len(df) >= 2:
            prev_candle = df.iloc[-2]
            return {
                "close": float(prev_candle["close"]),
                "open": float(prev_candle["open"]),
                "high": float(prev_candle["high"]),
                "low": float(prev_candle["low"]),
                "timestamp": prev_candle["ts"],
                "closed": True
            }

        return None

    except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
        log.error(
            "❌ خطأ في get_last_closed_candle: %s | symbol: %s | tf: %s",
            exc,
            symbol,
            tf,
        )
        return None
