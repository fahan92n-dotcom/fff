"""Directional predicates used by the cascade orchestration layer.

This module owns the eight LONG and SHORT strategy checks.  It deliberately
contains no scan loops, executors, or mutable survivor collections.
"""

from dataclasses import dataclass
from typing import Callable

from binance_data import get_cached
from indicators import (
    WARMUP_MACD,
    check_confirm_rsi_not_overbought,
    check_confirm_rsi_not_oversold,
    check_donchian_trend_ribbon,
    check_ema50_closed_above_since,
    check_ema50_closed_below_since,
    check_macd_green,
    check_macd_line_long,
    check_macd_line_short,
    check_macd_red,
    check_rsi_stoch,
    check_rsi_stoch_short,
    check_rsi_touched_since,
    check_smi_overbought,
    check_smi_oversold,
    find_smi_touch_index,
)
from state_manager import get_step1_ready_since


TRIPLING_PAIRS = [
    (9, 27, 3, "1m", "1m"),
    (12, 36, 4, "1m", "1m"),
    (15, 45, 5, "1m", "1m"),
    (18, 54, 6, "1m", "1m"),
    (21, 63, 7, "1m", "1m"),
    (24, 72, 8, "1m", "1m"),
    (27, 81, 9, "1m", "1m"),
    (30, 90, 10, "1m", "1m"),
    (45, 135, 15, "1m", "1m"),
    (60, 180, 20, "60m", "1m"),
    (90, 270, 30, "30m", "30m"),
    (120, 360, 40, "30m", "30m"),
    (150, 450, 50, "30m", "30m"),
    (180, 540, 60, "60m", "60m"),
    (210, 630, 70, "60m", "30m"),
    (240, 720, 80, "60m", "30m"),
]

TIMEFRAME_CHAIN = [pair[0] for pair in TRIPLING_PAIRS]
NEXT_TF = {
    TIMEFRAME_CHAIN[index]: TIMEFRAME_CHAIN[index + 1]
    for index in range(len(TIMEFRAME_CHAIN) - 1)
}
TF_TO_API = {pair[0]: pair[3] for pair in TRIPLING_PAIRS}

STEP_NAMES = [
    "smi_oversold",
    "macd_red",
    "donchian_base",
    "donchian_confirm",
    "macd_confirm",
    "ema50",
    "donchian_triple",
    "rsi_stoch",
]
STEP_LABELS = {
    "smi_oversold": "① تشبع بيعي SMI",
    "macd_red": "② MACD أحمر",
    "donchian_base": "③ Donchian Ribbon (الفريم الأساسي) أخضر",
    "donchian_confirm": "④ Donchian Ribbon (فريم التأكيد) أخضر",
    "macd_confirm": "⑤ MACD Confirm أخضر",
    "ema50": "⑥ السعر تحت EMA50",
    "donchian_triple": "⑦ Donchian Ribbon (فريم التثليث) أحمر",
    "rsi_stoch": "⑧ SMI ثم RSI≤35 وStoch>20",
}
SHORT_STEP_NAMES = [
    "smi_overbought",
    "macd_green",
    "donchian_base_red",
    "donchian_confirm_red",
    "macd_confirm_red",
    "ema50_above",
    "donchian_triple_green",
    "rsi_stoch_short",
]
SHORT_STEP_LABELS = {
    "smi_overbought": "① تشبع شرائي SMI ≥ +40",
    "macd_green": "② MACD أخضر",
    "donchian_base_red": "③ Donchian Ribbon (الفريم الأساسي) أحمر",
    "donchian_confirm_red": "④ Donchian Ribbon (فريم التأكيد) أحمر",
    "macd_confirm_red": "⑤ MACD Confirm أحمر",
    "ema50_above": "⑥ السعر فوق EMA50",
    "donchian_triple_green": "⑦ Donchian Ribbon (فريم التثليث) أخضر",
    "rsi_stoch_short": "⑧ SMI ثم RSI≥65 وStoch<80",
}

StepResult = tuple[bool, str]
StepCheck = Callable[[dict], StepResult]


@dataclass(frozen=True)
class SideRules:
    """Checks and reason strings that differ between LONG and SHORT."""

    signal_type: str
    saturation_check: Callable
    base_histogram_check: Callable
    macd_line_check: Callable
    base_ribbon: str
    base_ribbon_reason: str
    confirm_histogram_check: Callable
    confirm_histogram_reason: str
    ema_check: Callable
    ema_reason: str
    confirm_rsi_check: Callable
    confirm_rsi_threshold: int
    confirm_rsi_reason: str
    triple_ribbon: str
    triple_ribbon_reason: str
    smi_threshold: int
    touch_direction: str
    smi_touch_reason: str
    rsi_threshold: int
    rsi_touch_reason: str
    final_check: Callable
    final_reason: str


LONG_RULES = SideRules(
    signal_type="buy",
    saturation_check=check_smi_oversold,
    base_histogram_check=check_macd_red,
    macd_line_check=check_macd_line_long,
    base_ribbon="green",
    base_ribbon_reason="donchian_base",
    confirm_histogram_check=check_macd_green,
    confirm_histogram_reason="macd_confirm",
    ema_check=check_ema50_closed_below_since,
    ema_reason="ema50",
    confirm_rsi_check=check_confirm_rsi_not_oversold,
    confirm_rsi_threshold=30,
    confirm_rsi_reason="rsi_confirm_recent",
    triple_ribbon="red",
    triple_ribbon_reason="donchian_triple",
    smi_threshold=-40,
    touch_direction="long",
    smi_touch_reason="smi_touch_since_ready",
    rsi_threshold=35,
    rsi_touch_reason="rsi_touch_since_ready",
    final_check=check_rsi_stoch,
    final_reason="rsi_stoch",
)

SHORT_RULES = SideRules(
    signal_type="sell",
    saturation_check=check_smi_overbought,
    base_histogram_check=check_macd_green,
    macd_line_check=check_macd_line_short,
    base_ribbon="red",
    base_ribbon_reason="donchian_base_red",
    confirm_histogram_check=check_macd_red,
    confirm_histogram_reason="macd_confirm_red",
    ema_check=check_ema50_closed_above_since,
    ema_reason="ema50_above",
    confirm_rsi_check=check_confirm_rsi_not_overbought,
    confirm_rsi_threshold=70,
    confirm_rsi_reason="rsi_confirm_recent_over",
    triple_ribbon="green",
    triple_ribbon_reason="donchian_triple_green",
    smi_threshold=40,
    touch_direction="short",
    smi_touch_reason="smi_touch_since_ready_short",
    rsi_threshold=65,
    rsi_touch_reason="rsi_touch_since_ready_short",
    final_check=check_rsi_stoch_short,
    final_reason="rsi_stoch_short",
)


def _has_higher_tf_saturation(candidate, signal_type, get_resampled):
    """Check saturation on the candidate's immediate next timeframe only."""
    higher_tf = NEXT_TF.get(candidate["base_frame"])
    if higher_tf is None:
        return False

    native_api = TF_TO_API.get(higher_tf, candidate["base_api"])
    raw_native = get_cached(candidate["sym"], native_api)
    if raw_native.empty:
        return False

    higher_frame = get_resampled(
        raw_native,
        candidate["sym"],
        native_api,
        higher_tf,
    )
    if higher_frame.empty:
        return False
    if signal_type == "buy":
        return check_smi_oversold(higher_frame)
    if signal_type == "sell":
        return check_smi_overbought(higher_frame, threshold=40)
    raise ValueError(f"Unsupported signal type: {signal_type}")


def _ready_since(candidate, rules):
    return get_step1_ready_since(
        candidate["sym"],
        candidate["base_frame"],
        candidate["confirm_frame"],
        candidate["triple_frame"],
        rules.signal_type,
    )


def _step1(candidate, rules):
    if rules.signal_type == "buy":
        saturated = rules.saturation_check(candidate["df_base"])
        reason = "smi_oversold"
    else:
        saturated = rules.saturation_check(candidate["df_base"], threshold=40)
        reason = "smi_overbought"
    if not saturated:
        return False, reason
    if _has_higher_tf_saturation(
        candidate,
        rules.signal_type,
        candidate["get_resampled"],
    ):
        return False, "active_skip"
    return True, "passed"


def _step2(candidate, rules):
    if len(candidate["df_base"]) < WARMUP_MACD:
        return False, "warmup"
    if not rules.base_histogram_check(candidate["df_base"]):
        suffix = "red" if rules.signal_type == "buy" else "green"
        return False, f"macd_histogram_not_{suffix}"
    if not rules.macd_line_check(
        candidate["df_base"],
        pct=0.40,
        base_frame=candidate["base_frame"],
    ):
        return False, "macd_line_band"
    return True, "passed"


def _ribbon_step(candidate, rules, frame_key, api_key, direction, reason):
    key = (
        candidate["sym"],
        candidate[api_key],
        candidate[f"{frame_key}_frame"],
    )
    if not check_donchian_trend_ribbon(
        candidate[f"df_{frame_key}"],
        direction,
        cache_key=key,
    ):
        return False, reason
    return True, "passed"


def _step3(candidate, rules):
    return _ribbon_step(
        candidate,
        rules,
        "base",
        "base_api",
        rules.base_ribbon,
        rules.base_ribbon_reason,
    )


def _step4(candidate, rules):
    return _ribbon_step(
        candidate,
        rules,
        "confirm",
        "base_api",
        rules.base_ribbon,
        "donchian_confirm"
        if rules.signal_type == "buy"
        else "donchian_confirm_red",
    )


def _step5(candidate, rules):
    if not rules.confirm_histogram_check(candidate["df_confirm"]):
        return False, rules.confirm_histogram_reason
    return True, "passed"


def _step6(candidate, rules):
    since_ts = _ready_since(candidate, rules)
    if not rules.ema_check(candidate["df_base"], since_ts):
        return False, rules.ema_reason
    if not rules.confirm_rsi_check(
        candidate["df_confirm"],
        lookback=30,
        threshold=rules.confirm_rsi_threshold,
    ):
        return False, rules.confirm_rsi_reason
    return True, "passed"


def _step7(candidate, rules):
    return _ribbon_step(
        candidate,
        rules,
        "triple",
        "triple_api",
        rules.triple_ribbon,
        rules.triple_ribbon_reason,
    )


def _step8(candidate, rules):
    """
    Step 8 على شموع مغلقة بالترتيب:
    1) إغلاق كامل لتشبع SMI
    2) بعدها RSI وصل مستواه (بدون تقاطع المتوسط) و Stochastic بالاتجاه
       خلال ±3 شموع — لا يهم أيهما أسبق بين RSI و Stoch
    """
    since_ts = _ready_since(candidate, rules)
    frame = candidate["df_triple"]
    smi_index = find_smi_touch_index(
        frame,
        since_ts,
        threshold=rules.smi_threshold,
        direction=rules.touch_direction,
    )
    if smi_index is None:
        return False, rules.smi_touch_reason

    # RSI + Stoch فقط بعد إغلاق شمعة تشبع SMI.
    after_smi_ts = frame["ts"].iloc[smi_index]
    if not check_rsi_touched_since(
        frame,
        after_smi_ts,
        threshold=rules.rsi_threshold,
        direction=rules.touch_direction,
    ):
        return False, rules.rsi_touch_reason
    if not rules.final_check(frame, after_smi_ts, max_gap=3):
        return False, rules.final_reason
    return True, "passed"


def step1(candidate):
    return _step1(candidate, LONG_RULES)


def step2(candidate):
    return _step2(candidate, LONG_RULES)


def step3(candidate):
    return _step3(candidate, LONG_RULES)


def step4(candidate):
    return _step4(candidate, LONG_RULES)


def step5(candidate):
    return _step5(candidate, LONG_RULES)


def step6(candidate):
    return _step6(candidate, LONG_RULES)


def step7(candidate):
    return _step7(candidate, LONG_RULES)


def step8(candidate):
    return _step8(candidate, LONG_RULES)


def short_step1(candidate):
    return _step1(candidate, SHORT_RULES)


def short_step2(candidate):
    return _step2(candidate, SHORT_RULES)


def short_step3(candidate):
    return _step3(candidate, SHORT_RULES)


def short_step4(candidate):
    return _step4(candidate, SHORT_RULES)


def short_step5(candidate):
    return _step5(candidate, SHORT_RULES)


def short_step6(candidate):
    return _step6(candidate, SHORT_RULES)


def short_step7(candidate):
    return _step7(candidate, SHORT_RULES)


def short_step8(candidate):
    return _step8(candidate, SHORT_RULES)


steps: list[StepCheck] = [
    step1,
    step2,
    step3,
    step4,
    step5,
    step6,
    step7,
    step8,
]
short_steps: list[StepCheck] = [
    short_step1,
    short_step2,
    short_step3,
    short_step4,
    short_step5,
    short_step6,
    short_step7,
    short_step8,
]
