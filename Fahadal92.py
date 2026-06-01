
# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
import sys
import traceback

def handle_exception(exc_type, exc_value, exc_tb):
msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
log.error(f"💥 خطأ غير متوقع أوقف البوت:\n{msg}")
try:
send_telegram(f"💥 <b>البوت توقف بسبب خطأ:</b>\n<code>{exc_value}</code>")
except Exception:
pass
sys.excepthook = handle_exception

server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
threading.Thread(target=server.serve_forever, daemon=True).start()
log.info(f"✅ Health server شغّال على port {PORT}")

delete_webhook()

threading.Thread(target=update_symbols_loop, daemon=True).start()
threading.Thread(target=poll_telegram_commands, daemon=True).start()
threading.Thread(target=cache_updater_1m, daemon=True).start()
threading.Thread(target=cache_updater_60m, daemon=True).start()
threading.Thread(target=check5_watcher, daemon=True).start()
threading.Thread(target=send_diag_report, daemon=True).start()

for params in TRIPLING_PAIRS:
threading.Thread(target=candle_watcher, args=params, daemon=True).start()

while True:
try:
time.sleep(300)
with ohlcv_cache_lock:
cache_size = len(ohlcv_cache)
with trades_lock:
signals_count = len(trades_history)
log.info(
f"💓 البوت يعمل | "
f"كاش: {cache_size} مفتاح | "
f"إشارات: {signals_count} | "
f"سريع: {'✅' if fast_prefetch_done.is_set() else '⏳'} | "
f"كامل: {'✅' if prefetch_done.is_set() else '⏳'}"
)
except Exception as e:
log.error(f"❌ خطأ في main loop: {e}\n{traceback.format_exc()}")
time.sleep(10)


if __name__ == "__main__":
main()