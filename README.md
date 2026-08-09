# Fahad

Binance cascade scanner with Telegram notifications.

## Structure

- `main.py` — process lifecycle, health server, and worker supervision
- `config.py` — environment-backed runtime configuration
- `telegram_bot.py` — Telegram transport, polling, and signal delivery
- `cascade_pipeline.py` — scan and quick-check orchestration
- `cascade_steps.py` — LONG/SHORT stage predicates
- `state_manager.py` — thread-safe state transitions
- `state_store.py` — owned mutable collections and their locks
- `binance_data.py` — Binance access and OHLCV caches
- `indicators.py` — indicator calculations
- `fahadal92.py` — legacy compatibility exports and Telegram commands
- `mexc_leverage_limits.py` — standalone report of MEXC max position size per leverage
- `binance_leverage_limits.py` — the same report for Binance notional brackets
- `okx_leverage_limits.py` / `bybit_leverage_limits.py` / `hyperliquid_leverage_limits.py` / `kucoin_leverage_limits.py` — OKX, Bybit, Hyperliquid, KuCoin

## Configuration

Set `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, and optionally
`ALLOWED_CHAT_IDS`, `PORT`, and the Binance-related variables before running.
No Telegram credential is stored in the source tree.

## Run

```bash
python fahadal92.py
```

Max position size allowed on each MEXC USDT perpetual at a given leverage,
ranked high to low:

```bash
python mexc_leverage_limits.py --leverage 100 --crypto-only --csv mexc_100x.csv
python binance_leverage_limits.py --leverage 100 --csv binance_100x.csv
python okx_leverage_limits.py --leverage 100 --csv okx_100x.csv
python bybit_leverage_limits.py --leverage 100 --csv bybit_100x.csv
python hyperliquid_leverage_limits.py --leverage 40 --csv hl_40x.csv
python kucoin_leverage_limits.py --leverage 100 --csv kucoin_100x.csv
```

The Binance report needs a location Binance serves; from a restricted one it
exits with HTTP 451. Set `BINANCE_API_KEY` and `BINANCE_API_SECRET` to fall
back to the signed endpoint when the public one is unavailable. Bybit similarly
answers HTTP 403 from some regions — run it from a host Bybit serves
(Singapore works). Hyperliquid's platform max is currently 40x (BTC only).

## Verify

```bash
python -m unittest discover -v
pylint \
  binance_data.py binance_leverage_limits.py bybit_leverage_limits.py \
  cascade_pipeline.py cascade_steps.py config.py fahadal92.py \
  hyperliquid_leverage_limits.py indicators.py main.py \
  mexc_leverage_limits.py kucoin_leverage_limits.py okx_leverage_limits.py state_manager.py \
  state_store.py telegram_bot.py
```

CI runs both commands and separately rejects unexplained broad exception
handlers.
