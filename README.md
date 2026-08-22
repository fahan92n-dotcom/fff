# Fahad

Default deploy runs the **Pullback** bot: **BTCUSDT only**
(SMI / EMA60 / Donchian). Cascade (multi-coin Futures) remains in the
repo for reference but is not the start command.

## Structure

- `pullback_bot/` — BTC Pullback strategy bot (default)
- `main.py` — Cascade process lifecycle (legacy multi-coin scanner)
- `config.py` — environment-backed runtime configuration
- `telegram_bot.py` — Cascade Telegram transport
- `cascade_pipeline.py` — Cascade scan orchestration
- `cascade_steps.py` — LONG/SHORT stage predicates
- `state_manager.py` — thread-safe state transitions
- `state_store.py` — owned mutable collections and their locks
- `binance_data.py` — Binance access and OHLCV caches
- `indicators.py` — indicator calculations
- `fahadal92.py` — Cascade entry / Telegram commands (legacy)
- `pine/cascade_8steps.pine` — TradingView Cascade 8 strategy (5-minute chart)

## Configuration

Set `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` (or `PULLBACK_TELEGRAM_*`), and
optionally `ALLOWED_CHAT_IDS` / `PORT` before running.
Locally you can copy `.env.example` → `.env` (gitignored); the Pullback
bot loads it via python-dotenv. No Telegram credential is stored in git.

## Run

```bash
# Default: BTC Pullback only
python -m pullback_bot

# Legacy Cascade (100 coins Futures) — do not use if you want BTC only
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
