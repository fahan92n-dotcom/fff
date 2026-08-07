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
| `PULLBACK_TELEGRAM_TOKEN` | توكن بوت تيليجرام **الجديد** (لا تستخدم توكن Cascade) |
| `PULLBACK_TELEGRAM_CHAT_ID` | معرّف المحادثة |
| `PULLBACK_ALLOWED_CHAT_IDS` | اختياري، قائمة مفصولة بفواصل |
| `PULLBACK_PORT` | منفذ الصحة (افتراضي 8081) |

## الأوامر (تيليجرام)

- `/week` أو `1` — صفقات BTCUSDT آخر 7 أيام
- `/help` — المساعدة

## النشر المنفصل (Railway)

أنشئ خدمة Railway جديدة تشير لنفس المستودع مع:

- Start command: `python -m pullback_bot`
- أضف أسرار `PULLBACK_TELEGRAM_*` فقط (بدون مشاركة توكن Cascade)

بوت Cascade يبقى على `python fahadal92.py` كما هو.
