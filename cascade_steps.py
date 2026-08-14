"""Directional predicates used by the cascade orchestration layer.

This module owns the eight LONG and SHORT strategy checks.  It deliberately
contains no scan loops, executors, or mutable survivor collections.
"""

from dataclasses import dataclass
from typing import Callable

from binance_data import get_cached
from indicators import (
    WARMUP_MACD,
    WARMUP_SMI,
    calc_smi,
    check_btc_correlation,
    check_confirm_rsi_not_overbought,
    check_confirm_rsi_not_oversold,
    check_donchian_trend_ribbon,
    check_ema50_above,
    check_ema50_below,
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
    find_saturation_start_index,
    find_smi_touch_index,
)
from state_manager import get_step1_ready_since


# (base, confirm, triple, base_api, triple_api)
# كل هدف يجب أن ينقسم على مصدره حتى تطابق الشموع Binance/TradingView.
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
    (120, 360, 40, "30m", "1m"),
    (150, 450, 50, "30m", "1m"),
    (180, 540, 60, "60m", "60m"),
    (210, 630, 70, "30m", "1m"),
    (240, 720, 80, "60m", "1m"),
]

TIMEFRAME_CHAIN = [pair[0] for pair in TRIPLING_PAIRS]
NEXT_TF = {
    TIMEFRAME_CHAIN[index]: TIMEFRAME_CHAIN[index + 1]
    for index in range(len(TIMEFRAME_CHAIN) - 1)
}
TF_TO_API = {pair[0]: pair[3] for pair in TRIPLING_PAIRS}


def iter_cascade_frames():
    """Every closed candle the cascade builds: base, confirm, and entry.

    Alignment is per minute-count in ``resample_ohlcv``, not a 27m/150m special
    case. Adding a pair here automatically opts that TF into the UTC session grid.
    """
    for base, confirm, triple, base_api, triple_api in TRIPLING_PAIRS:
        yield base, base_api, "base"
        yield confirm, base_api, "confirm"
        yield triple, triple_api, "triple"

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
    "macd_red": "② MACD أحمر (عند أول إغلاق تشبع)",
    "donchian_base": "③ Donchian Ribbon (الفريم الأساسي) أخضر",
    "donchian_confirm": "④ Donchian Ribbon (فريم التأكيد) أخضر",
    "macd_confirm": "⑤ MACD Confirm أخضر",
    "ema50": "⑥ تحت EMA50 منذ التشبع + RSI تأكيد",
    "donchian_triple": "⑦ Donchian Ribbon (فريم التثليث) أحمر",
    "rsi_stoch": "⑧ SMI → لمس RSI≤35 → تقاطع RSI → Stoch>20 خلال 3",
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
    "macd_green": "② MACD أخضر (عند أول إغلاق تشبع)",
    "donchian_base_red": "③ Donchian Ribbon (الفريم الأساسي) أحمر",
    "donchian_confirm_red": "④ Donchian Ribbon (فريم التأكيد) أحمر",
    "macd_confirm_red": "⑤ MACD Confirm أحمر",
    "ema50_above": "⑥ فوق EMA50 منذ التشبع + RSI تأكيد",
    "donchian_triple_green": "⑦ Donchian Ribbon (فريم التثليث) أخضر",
    # Use &lt; (not raw <) — Telegram parse_mode=HTML rejects Stoch<80 as a bad tag.
    "rsi_stoch_short": "⑧ SMI → لمس RSI≥65 → تقاطع RSI → Stoch&lt;80 خلال 3",
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


def _higher_tf_is_saturated(smi_value, signal_type):
    if signal_type == "buy":
        return smi_value <= -40
    if signal_type == "sell":
        return smi_value >= 40
    raise ValueError(f"Unsupported signal type: {signal_type}")


def _has_higher_tf_saturation(candidate, signal_type, get_resampled):
    """
    نلغي الفريم الأصغر (شراء وبيع) إذا كان الفريم الأكبر التالي:
    1) متشبع الآن، أو
    2) ما زلنا ضمن أول أو ثاني شمعة مغلقة بعد انتهاء التشبع.

    مثال: تشبع 30m انتهى → أثناء إغلاق الشمعة 1 و 2 بعده → تشبع 27m يُرفض.
    من الشمعة الثالثة بدون تشبع على الأكبر، الأصغر يُسمح له.
    """
    higher_tf = NEXT_TF.get(candidate["base_frame"])
    if higher_tf is None:
        return False

    native_api = TF_TO_API.get(higher_tf, candidate["base_api"])
    get_raw = candidate.get("get_raw")
    raw_native = (
        get_raw(candidate["sym"], native_api)
        if callable(get_raw)
        else get_cached(candidate["sym"], native_api)
    )
    if raw_native is None or getattr(raw_native, "empty", True):
        return False

    higher_frame = get_resampled(
        raw_native,
        candidate["sym"],
        native_api,
        higher_tf,
    )
    if higher_frame.empty or len(higher_frame) < WARMUP_SMI:
        return False

    smi, _, _ = calc_smi(
        higher_frame["high"],
        higher_frame["low"],
        higher_frame["close"],
    )
    saturated = [
        _higher_tf_is_saturated(float(value), signal_type) for value in smi
    ]
    if saturated[-1]:
        return True

    # ابحث آخر شمعة متشبعة؛ إن كنا على أول أو ثاني شمعة بعدها → إلغاء الأصغر
    last_sat_offset = None
    for offset in range(len(saturated) - 1, -1, -1):
        if saturated[offset]:
            last_sat_offset = offset
            break
    if last_sat_offset is None:
        return False
    candles_after_exit = (len(saturated) - 1) - last_sat_offset
    return 1 <= candles_after_exit <= 2


def _ready_since(candidate, rules):
    # Historical replay injects ready_since on the candidate to avoid live STATE.
    if candidate.get("ready_since") is not None:
        return candidate["ready_since"]
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
    """
    فحص MACD مرة واحدة فقط: يُقيَّم على أول شمعة إغلاق متشبعة في نوبة
    التشبع الحالية (لا على آخر شمعة في كل دورة)، فلا يتغيّر قراره
    مع الشموع اللاحقة ما دامت النوبة مستمرة.
    """
    df_base = candidate["df_base"]
    if len(df_base) < WARMUP_MACD:
        return False, "warmup"
    start_index = find_saturation_start_index(
        df_base,
        threshold=-40 if rules.signal_type == "buy" else 40,
        direction="long" if rules.signal_type == "buy" else "short",
    )
    if start_index is None:
        return False, "smi_not_saturated"
    df_eval = df_base.iloc[: start_index + 1]
    if len(df_eval) < WARMUP_MACD:
        return False, "warmup"
    if not rules.base_histogram_check(df_eval):
        suffix = "red" if rules.signal_type == "buy" else "green"
        return False, f"macd_histogram_not_{suffix}"
    if not rules.macd_line_check(
        df_eval,
        pct=0.40,
        base_frame=candidate["base_frame"],
    ):
        return False, "macd_line_band"
    return True, "passed"


def _ribbon_step(candidate, rules, frame_key, api_key, direction, reason):
    if candidate.get("disable_ribbon_cache"):
        key = None
    else:
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


def _variant(candidate):
    """Optional experiment overrides attached on the candidate (live path: {})."""
    value = candidate.get("variant")
    return value if isinstance(value, dict) else {}


def _step4(candidate, rules):
    if _variant(candidate).get("skip_donchian_confirm"):
        return True, "passed"
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

    variant = _variant(candidate)

    # Experiment: at entry, confirm tip vs EMA50 —
    # buy requires confirm ABOVE EMA50; sell requires confirm BELOW EMA50.
    # Example: base 60m / confirm 180m / entry 20m → check 180m vs EMA50.
    if variant.get("ema_on_confirm"):
        if rules.signal_type == "buy":
            if not check_ema50_above(candidate["df_confirm"]):
                return False, "ema50_confirm"
        elif not check_ema50_below(candidate["df_confirm"]):
            return False, "ema50_confirm"

    # Experiment: RSI confirm lookback (None disables the filter).
    if "confirm_rsi_lookback" in variant:
        rsi_lookback = variant.get("confirm_rsi_lookback")
    else:
        rsi_lookback = 30
    if rsi_lookback is not None:
        if not rules.confirm_rsi_check(
            candidate["df_confirm"],
            lookback=int(rsi_lookback),
            threshold=rules.confirm_rsi_threshold,
        ):
            return False, rules.confirm_rsi_reason

    # Experiment: correlation with BTC on the BASE frame (alts only).
    btc_corr_min = variant.get("btc_corr_min")
    if btc_corr_min is not None and candidate.get("sym") != "BTCUSDT":
        df_btc = candidate.get("df_btc_base")
        lookback = int(variant.get("btc_corr_lookback") or 50)
        if not check_btc_correlation(
            candidate["df_base"],
            df_btc,
            lookback=lookback,
            min_corr=float(btc_corr_min),
        ):
            return False, "btc_corr"

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
    Step 8 على شموع مغلقة:
    1) إغلاق كامل لتشبع SMI
    2) لمس RSI (35 تشبع بيعي / 65 تشبع شرائي)
    3) تقاطع RSI مع متوسطه
    4) خلال 3 شموع بعده: Stoch فوق 20 / تحت 80
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
