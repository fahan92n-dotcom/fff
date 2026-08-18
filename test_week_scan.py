"""Tests for historical /week strategy scan and win/loss classification."""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

import fahadal92 as bot
import week_scan


def _bars(start, count, minutes=1, price=100.0, drift=0.0, high_off=0.5, low_off=0.5):
    rows = []
    px = price
    for index in range(count):
        ts = start + timedelta(minutes=minutes * index)
        open_px = px
        close_px = px + drift
        high_px = max(open_px, close_px) + high_off
        low_px = min(open_px, close_px) - low_off
        rows.append(
            {
                "ts": ts,
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "vol": 1.0,
            }
        )
        px = close_px
    return pd.DataFrame(rows)


class TestOutcomeLevels(unittest.TestCase):
    def test_short_frames_use_tight_levels(self):
        for frame in (15, 18, 27, 30):
            self.assertEqual(
                week_scan.outcome_levels(frame),
                (week_scan.SHORT_WIN_PCT, week_scan.SHORT_LOSS_PCT),
            )

    def test_long_frames_use_wide_levels(self):
        for frame in (45, 60, 150, 210, 240):
            self.assertEqual(
                week_scan.outcome_levels(frame),
                (week_scan.LONG_WIN_PCT, week_scan.LONG_LOSS_PCT),
            )


class TestPeriodBounds(unittest.TestCase):
    def test_today_starts_at_utc_midnight(self):
        now = datetime(2026, 8, 14, 15, 30, tzinfo=timezone.utc)
        start, end = week_scan.period_bounds("today", now=now)
        self.assertEqual(start, datetime(2026, 8, 14, tzinfo=timezone.utc))
        self.assertEqual(end, now)

    def test_week_covers_seven_days(self):
        now = datetime(2026, 8, 14, tzinfo=timezone.utc)
        start, end = week_scan.period_bounds("week", now=now)
        self.assertEqual(end, now)
        self.assertEqual(start, now - timedelta(days=7))

    def test_month_is_previous_utc_calendar_month(self):
        now = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
        start, end = week_scan.period_bounds("month", now=now)
        self.assertEqual(start, datetime(2026, 7, 1, tzinfo=timezone.utc))
        self.assertEqual(end, datetime(2026, 8, 1, tzinfo=timezone.utc))

    def test_month_from_january_uses_previous_year(self):
        now = datetime(2026, 1, 5, tzinfo=timezone.utc)
        start, end = week_scan.period_bounds("month", now=now)
        self.assertEqual(start, datetime(2025, 12, 1, tzinfo=timezone.utc))
        self.assertEqual(end, datetime(2026, 1, 1, tzinfo=timezone.utc))

    def test_month_1m_target_covers_month_plus_warmup(self):
        now = datetime(2026, 8, 18, tzinfo=timezone.utc)
        target = week_scan.month_1m_target(now=now)
        self.assertGreaterEqual(target, week_scan.MIN_1M_BARS)
        self.assertLessEqual(target, 120_000)
        start, _end = week_scan.period_bounds("month", now=now)
        span = (now - start).total_seconds() / 60.0
        self.assertGreater(target, span)


class TestEvaluateOutcome(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2026, 8, 1, tzinfo=timezone.utc)

    def test_buy_win_at_one_percent(self):
        # Entry 100 → TP 101. First bar high reaches 101.2 before any SL.
        future = pd.DataFrame(
            [
                {
                    "ts": self.start + timedelta(minutes=1),
                    "open": 100.0,
                    "high": 101.2,
                    "low": 99.8,
                    "close": 101.0,
                    "vol": 1,
                }
            ]
        )
        outcome, exit_price, _ = week_scan.evaluate_outcome("buy", 100.0, future)
        self.assertEqual(outcome, "win")
        self.assertAlmostEqual(exit_price, 101.0)

    def test_buy_loss_at_zero_point_seven_five(self):
        future = pd.DataFrame(
            [
                {
                    "ts": self.start + timedelta(minutes=1),
                    "open": 100.0,
                    "high": 100.2,
                    "low": 99.1,
                    "close": 99.2,
                    "vol": 1,
                }
            ]
        )
        outcome, exit_price, _ = week_scan.evaluate_outcome("buy", 100.0, future)
        self.assertEqual(outcome, "loss")
        self.assertAlmostEqual(exit_price, 99.25)

    def test_buy_win_short_frame_levels(self):
        # Entry 100 → TP 100.67 with short-frame 0.67%.
        future = pd.DataFrame(
            [
                {
                    "ts": self.start + timedelta(minutes=1),
                    "open": 100.0,
                    "high": 100.8,
                    "low": 99.9,
                    "close": 100.7,
                    "vol": 1,
                }
            ]
        )
        win_pct, loss_pct = week_scan.outcome_levels(15)
        outcome, exit_price, _ = week_scan.evaluate_outcome(
            "buy", 100.0, future, win_pct=win_pct, loss_pct=loss_pct
        )
        self.assertEqual(outcome, "win")
        self.assertAlmostEqual(exit_price, 100.67)

    def test_buy_loss_short_frame_levels(self):
        # Entry 100 → SL 99.48 with short-frame 0.52%.
        future = pd.DataFrame(
            [
                {
                    "ts": self.start + timedelta(minutes=1),
                    "open": 100.0,
                    "high": 100.1,
                    "low": 99.4,
                    "close": 99.45,
                    "vol": 1,
                }
            ]
        )
        win_pct, loss_pct = week_scan.outcome_levels(9)
        outcome, exit_price, _ = week_scan.evaluate_outcome(
            "buy", 100.0, future, win_pct=win_pct, loss_pct=loss_pct
        )
        self.assertEqual(outcome, "loss")
        self.assertAlmostEqual(exit_price, 99.48)

    def test_sell_win_at_one_percent(self):
        future = pd.DataFrame(
            [
                {
                    "ts": self.start + timedelta(minutes=1),
                    "open": 100.0,
                    "high": 100.1,
                    "low": 98.8,
                    "close": 99.0,
                    "vol": 1,
                }
            ]
        )
        outcome, exit_price, _ = week_scan.evaluate_outcome("sell", 100.0, future)
        self.assertEqual(outcome, "win")
        self.assertAlmostEqual(exit_price, 99.0)

    def test_sell_loss_at_zero_point_seven_five(self):
        future = pd.DataFrame(
            [
                {
                    "ts": self.start + timedelta(minutes=1),
                    "open": 100.0,
                    "high": 100.9,
                    "low": 99.9,
                    "close": 100.5,
                    "vol": 1,
                }
            ]
        )
        outcome, exit_price, _ = week_scan.evaluate_outcome("sell", 100.0, future)
        self.assertEqual(outcome, "loss")
        self.assertAlmostEqual(exit_price, 100.75)

    def test_same_bar_both_levels_counts_loss(self):
        future = pd.DataFrame(
            [
                {
                    "ts": self.start + timedelta(minutes=1),
                    "open": 100.0,
                    "high": 101.5,
                    "low": 99.0,
                    "close": 100.5,
                    "vol": 1,
                }
            ]
        )
        outcome, _, _ = week_scan.evaluate_outcome("buy", 100.0, future)
        self.assertEqual(outcome, "loss")

    def test_open_when_neither_hit(self):
        future = pd.DataFrame(
            [
                {
                    "ts": self.start + timedelta(minutes=1),
                    "open": 100.0,
                    "high": 100.4,
                    "low": 99.6,
                    "close": 100.1,
                    "vol": 1,
                }
            ]
        )
        outcome, exit_price, exit_ts = week_scan.evaluate_outcome(
            "buy", 100.0, future
        )
        self.assertEqual(outcome, "open")
        self.assertIsNone(exit_price)
        self.assertIsNone(exit_ts)


class TestFormatWeekReport(unittest.TestCase):
    def test_splits_wins_and_losses(self):
        now = datetime(2026, 8, 6, tzinfo=timezone.utc)
        result = {
            "ready": True,
            "start": now - timedelta(days=7),
            "end": now,
            "symbols_scanned": 2,
            "total": 2,
            "wins": [
                {
                    "symbol": "BTCUSDT",
                    "type": "buy",
                    "base_frame": 60,
                    "confirm_frame": 180,
                    "triple_frame": 20,
                    "time": now - timedelta(days=1),
                    "price": 100.0,
                    "outcome": "win",
                }
            ],
            "losses": [
                {
                    "symbol": "ETHUSDT",
                    "type": "sell",
                    "base_frame": 30,
                    "confirm_frame": 90,
                    "triple_frame": 10,
                    "time": now - timedelta(days=2),
                    "price": 50.0,
                    "outcome": "loss",
                }
            ],
            "opens": [],
        }
        chunks = week_scan.format_week_trades_report(result)
        text = "\n".join(chunks)
        self.assertIn("الناجحة", text)
        self.assertIn("الفاشلة", text)
        self.assertIn("مستمرة", text)
        self.assertIn("BTCUSDT", text)
        self.assertIn("ETHUSDT", text)
        self.assertIn("0.67%", text)
        self.assertIn("0.52%", text)
        self.assertIn("1%", text)
        self.assertIn("0.75%", text)
        self.assertIn("حسب فريم الدخول", text)
        self.assertIn("20د:", text)
        self.assertIn("10د:", text)

    def test_entry_tf_summary_counts_6_7_8(self):
        now = datetime(2026, 8, 6, tzinfo=timezone.utc)
        result = {
            "ready": True,
            "start": now - timedelta(days=31),
            "end": now,
            "symbols_scanned": 3,
            "total": 3,
            "wins": [
                {
                    "symbol": "BTCUSDT",
                    "type": "buy",
                    "base_frame": 18,
                    "confirm_frame": 54,
                    "triple_frame": 6,
                    "time": now,
                    "price": 100.0,
                    "outcome": "win",
                }
            ],
            "losses": [
                {
                    "symbol": "ETHUSDT",
                    "type": "sell",
                    "base_frame": 21,
                    "confirm_frame": 63,
                    "triple_frame": 7,
                    "time": now,
                    "price": 50.0,
                    "outcome": "loss",
                }
            ],
            "opens": [
                {
                    "symbol": "SOLUSDT",
                    "type": "buy",
                    "base_frame": 24,
                    "confirm_frame": 72,
                    "triple_frame": 8,
                    "time": now,
                    "price": 20.0,
                    "outcome": "open",
                }
            ],
        }
        text = "\n".join(
            week_scan.format_week_trades_report(result, period="month")
        )
        self.assertIn("6د:", text)
        self.assertIn("7د:", text)
        self.assertIn("8د:", text)
        self.assertIn("الناجحة", text)
        self.assertIn("الفاشلة", text)

    def test_today_report_labels_open_trades_as_ongoing(self):
        now = datetime(2026, 8, 14, tzinfo=timezone.utc)
        result = {
            "ready": True,
            "start": now.replace(hour=0),
            "end": now,
            "symbols_scanned": 1,
            "total": 1,
            "wins": [],
            "losses": [],
            "opens": [
                {
                    "symbol": "BTCUSDT",
                    "type": "buy",
                    "base_frame": 30,
                    "confirm_frame": 90,
                    "triple_frame": 10,
                    "time": now,
                    "price": 100.0,
                    "outcome": "open",
                }
            ],
        }
        text = "\n".join(
            week_scan.format_week_trades_report(result, period="today")
        )
        self.assertIn("صفقات اليوم", text)
        self.assertIn("الناجحة", text)
        self.assertIn("الفاشلة", text)
        self.assertIn("مستمرة", text)
        self.assertIn("BTCUSDT", text)
        self.assertNotIn("مفتوحة", text)


class TestDedupeAndCommand(unittest.TestCase):
    def test_dedupe_keeps_earliest_within_window(self):
        t0 = datetime(2026, 8, 1, tzinfo=timezone.utc)
        signals = [
            {
                "symbol": "BTCUSDT",
                "type": "buy",
                "base_frame": 60,
                "confirm_frame": 180,
                "triple_frame": 20,
                "time": t0,
                "price": 1,
                "outcome": "win",
            },
            {
                "symbol": "BTCUSDT",
                "type": "buy",
                "base_frame": 60,
                "confirm_frame": 180,
                "triple_frame": 20,
                "time": t0 + timedelta(hours=1),
                "price": 2,
                "outcome": "loss",
            },
            {
                "symbol": "BTCUSDT",
                "type": "buy",
                "base_frame": 60,
                "confirm_frame": 180,
                "triple_frame": 20,
                "time": t0 + timedelta(hours=5),
                "price": 3,
                "outcome": "win",
            },
        ]
        kept = week_scan._dedupe_signals(signals, hours=4)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0]["price"], 1)
        self.assertEqual(kept[1]["price"], 3)

    def test_week_command_routes_to_market_scan(self):
        called = {}

        def fake_handle(chat_id, send_fn):
            called["chat_id"] = chat_id
            called["send"] = send_fn

        with patch.object(bot, "handle_week_command", side_effect=fake_handle):
            bot._dispatch_command("/week", "42")
            bot._dispatch_command("3", "42")

        self.assertEqual(called["chat_id"], "42")
        self.assertIs(called["send"], bot.send_telegram)

    def test_month_command_routes_to_market_scan(self):
        called = {}

        def fake_handle(chat_id, send_fn):
            called["chat_id"] = chat_id
            called["send"] = send_fn

        with patch.object(bot, "handle_month_command", side_effect=fake_handle):
            bot._dispatch_command("/شهر", "42")
            bot._dispatch_command("4", "42")
            bot._dispatch_command("/month", "42")

        self.assertEqual(called["chat_id"], "42")
        self.assertIs(called["send"], bot.send_telegram)

    def test_today_command_routes_to_market_scan(self):
        called = {}

        def fake_handle(chat_id, send_fn):
            called["chat_id"] = chat_id
            called["send"] = send_fn

        with patch.object(bot, "handle_today_command", side_effect=fake_handle):
            bot._dispatch_command("/today", "42")
            bot._dispatch_command("1", "42")

        self.assertEqual(called["chat_id"], "42")
        self.assertIs(called["send"], bot.send_telegram)

    def test_handle_week_sends_formatted_report(self):
        now = datetime(2026, 8, 6, tzinfo=timezone.utc)
        fake_result = {
            "ready": True,
            "start": now - timedelta(days=7),
            "end": now,
            "symbols_scanned": 1,
            "total": 1,
            "wins": [
                {
                    "symbol": "BTCUSDT",
                    "type": "buy",
                    "base_frame": 60,
                    "confirm_frame": 180,
                    "triple_frame": 20,
                    "time": now - timedelta(hours=3),
                    "price": 100.0,
                    "outcome": "win",
                }
            ],
            "losses": [],
            "opens": [],
        }
        sent = []
        with (
            patch.object(week_scan, "fast_prefetch_done") as done,
            patch.object(
                week_scan,
                "scan_week_trades",
                return_value=fake_result,
            ),
        ):
            done.is_set.return_value = True
            week_scan.handle_week_command(
                "9",
                lambda message, chat_id=None: sent.append(message),
            )

        self.assertTrue(sent)
        joined = "\n".join(sent)
        self.assertIn("صفقات الاستراتيجية", joined)
        self.assertIn("الناجحة", joined)
        self.assertIn("الفاشلة", joined)
        self.assertIn("مستمرة", joined)
        self.assertIn("BTCUSDT", joined)

    def test_handle_today_sends_formatted_report(self):
        now = datetime(2026, 8, 14, 15, tzinfo=timezone.utc)
        fake_result = {
            "ready": True,
            "start": now.replace(hour=0, minute=0, second=0, microsecond=0),
            "end": now,
            "symbols_scanned": 1,
            "total": 2,
            "wins": [
                {
                    "symbol": "BTCUSDT",
                    "type": "buy",
                    "base_frame": 15,
                    "confirm_frame": 45,
                    "triple_frame": 5,
                    "time": now - timedelta(hours=2),
                    "price": 100.0,
                    "outcome": "win",
                }
            ],
            "losses": [
                {
                    "symbol": "ETHUSDT",
                    "type": "sell",
                    "base_frame": 60,
                    "confirm_frame": 180,
                    "triple_frame": 20,
                    "time": now - timedelta(hours=1),
                    "price": 50.0,
                    "outcome": "loss",
                }
            ],
            "opens": [
                {
                    "symbol": "SOLUSDT",
                    "type": "buy",
                    "base_frame": 30,
                    "confirm_frame": 90,
                    "triple_frame": 10,
                    "time": now - timedelta(minutes=20),
                    "price": 80.0,
                    "outcome": "open",
                }
            ],
        }
        sent = []
        with (
            patch.object(week_scan, "fast_prefetch_done") as done,
            patch.object(
                week_scan,
                "scan_week_trades",
                return_value=fake_result,
            ) as scan,
        ):
            done.is_set.return_value = True
            week_scan.handle_today_command(
                "9",
                lambda message, chat_id=None: sent.append(message),
            )

        scan.assert_called_once()
        kwargs = scan.call_args.kwargs
        self.assertEqual(kwargs["start"].hour, 0)
        self.assertEqual(kwargs["start"].minute, 0)
        self.assertEqual(kwargs["start"].second, 0)
        self.assertEqual((kwargs["end"] - kwargs["start"]).days, 0)
        joined = "\n".join(sent)
        self.assertIn("صفقات اليوم", joined)
        self.assertIn("الناجحة", joined)
        self.assertIn("الفاشلة", joined)
        self.assertIn("مستمرة", joined)
        self.assertIn("SOLUSDT", joined)
        self.assertNotIn("الناجحون", joined)
        self.assertNotIn("مفتوحة", joined)

    def test_handle_month_sends_formatted_report(self):
        now = datetime(2026, 8, 18, tzinfo=timezone.utc)
        start = datetime(2026, 7, 1, tzinfo=timezone.utc)
        end = datetime(2026, 8, 1, tzinfo=timezone.utc)
        fake_result = {
            "ready": True,
            "start": start,
            "end": end,
            "symbols_scanned": 1,
            "total": 1,
            "wins": [
                {
                    "symbol": "BTCUSDT",
                    "type": "buy",
                    "base_frame": 60,
                    "confirm_frame": 180,
                    "triple_frame": 20,
                    "time": start + timedelta(days=2),
                    "price": 100.0,
                    "outcome": "win",
                }
            ],
            "losses": [],
            "opens": [],
        }
        sent = []
        with (
            patch.object(week_scan, "fast_prefetch_done") as done,
            patch.object(
                week_scan,
                "scan_week_trades",
                return_value=fake_result,
            ) as scan,
            patch.object(
                week_scan,
                "period_bounds",
                return_value=(start, end),
            ),
        ):
            done.is_set.return_value = True
            week_scan.handle_month_command(
                "9",
                lambda message, chat_id=None: sent.append(message),
            )

        scan.assert_called_once()
        kwargs = scan.call_args.kwargs
        self.assertEqual(kwargs["start"], start)
        self.assertEqual(kwargs["end"], end)
        self.assertIn("min_1m", kwargs)
        self.assertGreaterEqual(kwargs["min_1m"], week_scan.MIN_1M_BARS)
        joined = "\n".join(sent)
        self.assertIn("الشهر الماضي", joined)
        self.assertIn("الناجحة", joined)
        self.assertIn("BTCUSDT", joined)


class TestWaitingBlindWindow(unittest.TestCase):
    """
    المنتظر يبقى حيًا خلال نافذة شمعة الأساس الجارية حتى لو كانت ستُغلق
    بلا تشبع: دخول يكتمل على فريم الدخول داخل تلك النافذة يجب أن يُسجَّل
    (مطابقة للبوت الحي الذي لا يرى الشمعة غير المتشبعة قبل إغلاقها).
    """

    def test_entry_inside_final_base_candle_window_is_recorded(self):
        t0 = datetime(2026, 8, 3, tzinfo=timezone.utc)
        raw = _bars(t0, 5000, minutes=1, price=100.0)
        sat_index = 300  # بعد warmup؛ شمعة 15m تفتح عند t0+4500 دقيقة
        sat_ts = t0 + timedelta(minutes=sat_index * 15)
        entry_candle_ts = sat_ts + timedelta(minutes=20)  # شمعة 5m داخل نافذة الشمعة التالية

        def fake_calc_smi(high, low, close):
            smi = pd.Series(0.0, index=close.index)
            if len(smi) > sat_index:
                smi.iloc[sat_index] = 50.0
            return smi, smi, smi

        def fake_still_valid(candidate, signal_type):
            return _utc_safe(
                candidate["df_base"]["ts"].iloc[-1].to_pydatetime()
            ) == sat_ts

        def fake_entry(candidate, signal_type):
            last_ts = _utc_safe(
                candidate["df_triple"]["ts"].iloc[-1].to_pydatetime()
            )
            if last_ts != entry_candle_ts:
                return None
            return {
                "entry_ts": last_ts,
                "price": float(candidate["df_triple"]["close"].iloc[-1]),
                "triple_frame": 5,
            }

        with (
            patch.object(week_scan, "calc_smi", side_effect=fake_calc_smi),
            patch.object(
                week_scan, "_passes_steps_1_5", return_value=True
            ),
            patch.object(
                week_scan, "_stage5_still_valid", side_effect=fake_still_valid
            ),
            patch.object(week_scan, "_try_step8_entry", side_effect=fake_entry),
        ):
            signals = week_scan._scan_pair_side(
                "BTCUSDT",
                (15, 45, 5, "1m", "1m"),
                "sell",
                {"1m": raw},
                t0,
                t0 + timedelta(minutes=5000),
                raw,
            )

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["time"], entry_candle_ts)


class TestSliceClosed(unittest.TestCase):
    def test_keeps_only_fully_closed_candles(self):
        start = datetime(2026, 8, 1, tzinfo=timezone.utc)
        df = _bars(start, 5, minutes=10, price=10.0)
        # asof exactly at end of 3rd candle (index 2): ts=start+20m, end=start+30m
        asof = start + timedelta(minutes=30)
        sliced = week_scan._slice_closed(df, 10, asof)
        self.assertEqual(len(sliced), 3)
        self.assertEqual(_utc_safe(sliced["ts"].iloc[-1]), start + timedelta(minutes=20))

    def test_remainder_27m_is_closed_at_utc_midnight(self):
        stub = pd.DataFrame(
            [
                {
                    "ts": pd.Timestamp("2026-08-13 23:51:00+00:00"),
                    "open": 1.0,
                    "high": 1.0,
                    "low": 1.0,
                    "close": 1.0,
                    "vol": 1.0,
                }
            ]
        )
        at_midnight = week_scan._slice_closed(
            stub, 27, datetime(2026, 8, 14, 0, 0, tzinfo=timezone.utc)
        )
        before_old_end = week_scan._slice_closed(
            stub, 27, datetime(2026, 8, 14, 0, 10, tzinfo=timezone.utc)
        )
        self.assertEqual(len(at_midnight), 1)
        self.assertEqual(len(before_old_end), 1)
        still_open = week_scan._slice_closed(
            stub, 27, datetime(2026, 8, 13, 23, 59, tzinfo=timezone.utc)
        )
        self.assertEqual(len(still_open), 0)


def _utc_safe(ts):
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


if __name__ == "__main__":
    unittest.main()
