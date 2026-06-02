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