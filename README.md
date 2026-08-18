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
- `tv_webhook.py` — TradingView webhook stores wins/losses; Telegram `/نتائج` asks on demand
- `pine/cascade_8steps.pine` — TradingView Cascade 8-step strategy

## Configuration

Set `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` (or `PULLBACK_TELEGRAM_*`), and
optionally `ALLOWED_CHAT_IDS` / `PORT` before running.

TradingView results: set `TV_WEBHOOK_SECRET`, expose `POST /tv?token=<secret>`,
then in TradingView create an alert on `Cascade 8 — كل الفريمات` with
**Any alert() function call** and that webhook URL. The bot stores wins and
losses silently. In Telegram send `/نتائج` or `/score` whenever you want the
counts — it does not message you after every trade.

For a **complete previous calendar month** of Cascade trades (wins /
losses / still open, including 6m/7m/8m entries), send `/شهر` in Telegram.
The Pullback bot and the Cascade bot both accept it. That replays the
8 steps on Binance 1m candles (not the TradingView chart tally). Pine
on the chart cannot do that month accurately because `request.security`
on a lower timeframe only sees the last lower-TF bar of each chart bar.
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
  telegram_bot.py tv_webhook.py
```

CI runs both commands and separately rejects unexplained broad exception
handlers.
