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

- `/week` أو `1` — صفقات BTCUSDT آخر 7 أيام
- `/help` — المساعدة

### كيف تشغّل `/week`

الأوامر **ما تُنفَّذ من القناة**. لازم تراسل البوت في الخاص:

1. افتح البوت (مثال: `@Hadar5_bot`) واضغط Start / أرسل `/start`
2. أرسل أي رسالة؛ إذا طلع "غير مصرح" انسخ الـ Chat ID اللي يرسله
3. ضعه في `PULLBACK_ALLOWED_CHAT_IDS` مع معرّف القناة، مثال:
   `PULLBACK_ALLOWED_CHAT_IDS=-1003912070746,7801703329`
4. أعد تشغيل البوت، ثم أرسل `/week` أو `1` من نفس الشات الخاص

القناة (`PULLBACK_TELEGRAM_CHAT_ID`) لاستقبال رسائل التشغيل/التنبيهات فقط.

## النشر المنفصل (Railway)

أنشئ خدمة Railway جديدة تشير لنفس المستودع مع:

- Start command: `python -m pullback_bot`
- أضف أسرار `PULLBACK_TELEGRAM_*` فقط (بدون مشاركة توكن Cascade)

بوت Cascade يبقى على `python fahadal92.py` كما هو.
