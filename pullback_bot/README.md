# بوت استراتيجية Pullback (منفصل)

تطبيق مستقل عن بوت Cascade القديم. له توكن تيليجرام خاص وأوامر خاصة.

## التشغيل

من جذر المستودع:

```bash
export PULLBACK_TELEGRAM_TOKEN="..."
export PULLBACK_TELEGRAM_CHAT_ID="..."
python -m pullback_bot
```

فحص الأسبوع بدون بوت:

```bash
python -m pullback_bot.strategy
```

## المتغيرات

| المتغير | الوصف |
|--|--|
| `PULLBACK_TELEGRAM_TOKEN` أو `TELEGRAM_TOKEN` | توكن بوت تيليجرام |
| `PULLBACK_TELEGRAM_CHAT_ID` أو `TELEGRAM_CHAT_ID` | معرّف المحادثة |
| `PULLBACK_ALLOWED_CHAT_IDS` / `ALLOWED_CHAT_IDS` | اختياري، قائمة مفصولة بفواصل |
| `PULLBACK_PORT` / `PORT` | منفذ الصحة (افتراضي 8081) |

للتشغيل المحلي: انسخ `.env.example` إلى `.env` واملأ القيم.
البوت يقرأ `.env` تلقائياً. على Railway ضع نفس المتغيرات كأسرار — لا ترفع `.env`.

## الأوامر (تيليجرام)

- `/week` أو `1` — صفقات BTCUSDT Pullback آخر 7 أيام
- `/month` أو `2` — صفقات BTCUSDT Pullback آخر 30 يومًا
- `/شهر` أو `4` — صفقات Cascade الشهر الماضي كاملة (ناجحة/فاشلة، كل الأزواج)
- `/نتائج` أو `3` — عدّاد TradingView الحي
- `/help` — المساعدة

## النشر المنفصل (Railway)

أنشئ خدمة Railway جديدة تشير لنفس المستودع مع:

- Start command: `python -m pullback_bot`
- أضف أسرار `PULLBACK_TELEGRAM_*` فقط (بدون مشاركة توكن Cascade)

بوت Cascade يبقى على `python fahadal92.py` كما هو.
