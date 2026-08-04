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

## Configuration

Set `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, and optionally
`ALLOWED_CHAT_IDS`, `PORT`, and the Binance-related variables before running.
No Telegram credential is stored in the source tree.

## Run

```bash
python fahadal92.py
```

## Verify

```bash
python -m unittest discover -v
pylint \
  binance_data.py cascade_pipeline.py cascade_steps.py config.py \
  fahadal92.py indicators.py main.py state_manager.py state_store.py \
  telegram_bot.py
```

CI runs both commands and separately rejects unexplained broad exception
handlers.
