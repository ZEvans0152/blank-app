"""Signal engine: turns indicator confluence into scored BUY/SELL signals.

Philosophy: no single indicator is a signal. Each rule that fires contributes
a weight; an alert is only produced when one side accumulates at least
``min_score`` AND beats the opposite side by 2+ points. The daily timeframe
adds a trend-alignment bonus so counter-trend setups need more confirmation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .indicators import add_indicators

# Minimum completed candles required on the trigger timeframe. Below this the
# slower indicators (BB-20, stoch smoothing, ATR warm-up) are unreliable.
MIN_TRIGGER_BARS = 60

VOLUME_SPIKE_RATIO = 1.5


@dataclass
class Reason:
    text: str
    weight: int


@dataclass
class Signal:
    direction: str  # "BUY" or "SELL"
    strength: str  # "STRONG" or "MODERATE"
    score: int
    opposite_score: int
    entry: float
    stop: Optional[float]
    target: Optional[float]


@dataclass
class Analysis:
    symbol: str
    timeframe: str
    bar_time: str
    price: float
    bull_score: int
    bear_score: int
    bull_reasons: list[Reason] = field(default_factory=list)
    bear_reasons: list[Reason] = field(default_factory=list)
    context_notes: list[str] = field(default_factory=list)
    snapshot: dict = field(default_factory=dict)
    signal: Optional[Signal] = None


def _ok(*values) -> bool:
    """True when every value is a real number (indicator warm-up guard)."""
    for v in values:
        if v is None:
            return False
        try:
            if math.isnan(float(v)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _trigger_rules(df: pd.DataFrame) -> tuple[list[Reason], list[Reason]]:
    """Evaluate entry rules on the last two completed candles."""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    bull: list[Reason] = []
    bear: list[Reason] = []

    # --- RSI ---------------------------------------------------------------
    if _ok(last["rsi"], prev["rsi"]):
        if prev["rsi"] <= 30.0 < last["rsi"]:
            bull.append(
                Reason(f"RSI(14) crossed up out of oversold (now {last['rsi']:.1f})", 2)
            )
        elif last["rsi"] < 35.0 and last["rsi"] > prev["rsi"]:
            bull.append(
                Reason(f"RSI(14) turning up in oversold territory ({last['rsi']:.1f})", 1)
            )
        if prev["rsi"] >= 70.0 > last["rsi"]:
            bear.append(
                Reason(f"RSI(14) crossed down out of overbought (now {last['rsi']:.1f})", 2)
            )
        elif last["rsi"] > 65.0 and last["rsi"] < prev["rsi"]:
            bear.append(
                Reason(f"RSI(14) rolling over in overbought territory ({last['rsi']:.1f})", 1)
            )

    # --- MACD crossover ------------------------------------------------------
    if _ok(last["macd"], last["macd_signal"], prev["macd"], prev["macd_signal"]):
        if prev["macd"] <= prev["macd_signal"] and last["macd"] > last["macd_signal"]:
            if last["macd"] < 0:
                bull.append(Reason("MACD bullish crossover below the zero line (early reversal)", 3))
            else:
                bull.append(Reason("MACD bullish crossover", 2))
        if prev["macd"] >= prev["macd_signal"] and last["macd"] < last["macd_signal"]:
            if last["macd"] > 0:
                bear.append(Reason("MACD bearish crossover above the zero line (early reversal)", 3))
            else:
                bear.append(Reason("MACD bearish crossover", 2))

    # --- Bollinger Band mean-reversion --------------------------------------
    if _ok(last["bb_lower"], prev["bb_lower"], last["bb_upper"], prev["bb_upper"]):
        if prev["close"] < prev["bb_lower"] and last["close"] > last["bb_lower"]:
            bull.append(Reason("Close rebounded back inside the lower Bollinger Band", 2))
        if prev["close"] > prev["bb_upper"] and last["close"] < last["bb_upper"]:
            bear.append(Reason("Close fell back inside the upper Bollinger Band", 2))

    # --- Stochastic crossover in extreme zones ------------------------------
    if _ok(last["stoch_k"], last["stoch_d"], prev["stoch_k"], prev["stoch_d"]):
        if (
            prev["stoch_k"] <= prev["stoch_d"]
            and last["stoch_k"] > last["stoch_d"]
            and prev["stoch_k"] < 25.0
        ):
            bull.append(Reason("Stochastic %K crossed above %D in the oversold zone", 1))
        if (
            prev["stoch_k"] >= prev["stoch_d"]
            and last["stoch_k"] < last["stoch_d"]
            and prev["stoch_k"] > 75.0
        ):
            bear.append(Reason("Stochastic %K crossed below %D in the overbought zone", 1))

    # --- 20-bar channel break on volume --------------------------------------
    prior_hh = df["hh20"].iloc[-2]
    prior_ll = df["ll20"].iloc[-2]
    vol_ratio = last["vol_ratio"]
    if _ok(prior_hh, vol_ratio) and last["close"] > prior_hh and vol_ratio >= VOLUME_SPIKE_RATIO:
        bull.append(
            Reason(f"Breakout above the 20-bar high on {vol_ratio:.1f}× average volume", 2)
        )
    if _ok(prior_ll, vol_ratio) and last["close"] < prior_ll and vol_ratio >= VOLUME_SPIKE_RATIO:
        bear.append(
            Reason(f"Breakdown below the 20-bar low on {vol_ratio:.1f}× average volume", 2)
        )

    # --- Volume confirmation (only meaningful when something else fired) ----
    if _ok(vol_ratio) and vol_ratio >= VOLUME_SPIKE_RATIO:
        confirm = Reason(f"Volume {vol_ratio:.1f}× its 20-bar average confirms the move", 1)
        if bull and not bear:
            bull.append(confirm)
        elif bear and not bull:
            bear.append(confirm)

    return bull, bear


def _context_rules(
    daily: Optional[pd.DataFrame],
) -> tuple[list[Reason], list[Reason], list[str]]:
    """Daily-timeframe rules: trend alignment bonus and MA crosses."""
    bull: list[Reason] = []
    bear: list[Reason] = []
    notes: list[str] = []
    if daily is None or len(daily) < 2:
        notes.append("Daily context unavailable")
        return bull, bear, notes

    last = daily.iloc[-1]
    prev = daily.iloc[-2]

    trending = _ok(last["adx"]) and last["adx"] >= 20.0
    if _ok(last["adx"]):
        notes.append(
            f"ADX {last['adx']:.0f} — {'trending' if trending else 'range-bound'} market"
        )

    if _ok(last["sma200"]):
        if last["close"] > last["sma200"]:
            notes.append("Price above the 200-day SMA (long-term uptrend)")
            bull.append(Reason("Aligned with the daily uptrend (price > SMA200)", 1))
        else:
            notes.append("Price below the 200-day SMA (long-term downtrend)")
            bear.append(Reason("Aligned with the daily downtrend (price < SMA200)", 1))

    if _ok(last["sma50"], last["sma200"], prev["sma50"], prev["sma200"]):
        if prev["sma50"] <= prev["sma200"] and last["sma50"] > last["sma200"]:
            bull.append(Reason("Golden cross: 50-day SMA crossed above 200-day SMA", 3))
        if prev["sma50"] >= prev["sma200"] and last["sma50"] < last["sma200"]:
            bear.append(Reason("Death cross: 50-day SMA crossed below 200-day SMA", 3))

    return bull, bear, notes


def _levels(
    direction: str, entry: float, daily: Optional[pd.DataFrame], trigger: pd.DataFrame
) -> tuple[Optional[float], Optional[float]]:
    """ATR-based stop/target. Prefers daily ATR; falls back to the trigger frame."""
    atr_value = None
    for frame in (daily, trigger):
        if frame is not None and len(frame) and _ok(frame["atr"].iloc[-1]):
            atr_value = float(frame["atr"].iloc[-1])
            break
    if not atr_value or atr_value <= 0:
        return None, None
    if direction == "BUY":
        return entry - 2.0 * atr_value, entry + 3.0 * atr_value
    return entry + 2.0 * atr_value, entry - 3.0 * atr_value


def analyze(
    symbol: str,
    trigger_df: pd.DataFrame,
    context_df: Optional[pd.DataFrame],
    min_score: int = 4,
    strong_score: int = 7,
    timeframe: str = "15m",
) -> Optional[Analysis]:
    """Run the full analysis. Returns None when there's too little data."""
    if trigger_df is None or len(trigger_df) < MIN_TRIGGER_BARS:
        return None

    trig = add_indicators(trigger_df)
    daily = add_indicators(context_df) if context_df is not None and len(context_df) >= 20 else None

    bull, bear = _trigger_rules(trig)
    ctx_bull, ctx_bear, notes = _context_rules(daily)

    # Trend context only ever amplifies an existing setup — the daily trend by
    # itself is not an entry signal.
    if bull:
        bull.extend(ctx_bull)
    if bear:
        bear.extend(ctx_bear)

    bull_score = sum(r.weight for r in bull)
    bear_score = sum(r.weight for r in bear)

    last = trig.iloc[-1]
    price = float(last["close"])
    bar_time = str(trig.index[-1])

    snapshot = {
        "price": price,
        "rsi": float(last["rsi"]) if _ok(last["rsi"]) else None,
        "macd_hist": float(last["macd_hist"]) if _ok(last["macd_hist"]) else None,
        "stoch_k": float(last["stoch_k"]) if _ok(last["stoch_k"]) else None,
        "stoch_d": float(last["stoch_d"]) if _ok(last["stoch_d"]) else None,
        "atr": float(last["atr"]) if _ok(last["atr"]) else None,
        "vol_ratio": float(last["vol_ratio"]) if _ok(last["vol_ratio"]) else None,
    }
    if _ok(last["bb_upper"], last["bb_lower"]) and last["bb_upper"] != last["bb_lower"]:
        snapshot["percent_b"] = float(
            (last["close"] - last["bb_lower"]) / (last["bb_upper"] - last["bb_lower"])
        )

    analysis = Analysis(
        symbol=symbol,
        timeframe=timeframe,
        bar_time=bar_time,
        price=price,
        bull_score=bull_score,
        bear_score=bear_score,
        bull_reasons=bull,
        bear_reasons=bear,
        context_notes=notes,
        snapshot=snapshot,
    )

    direction = None
    if bull_score >= min_score and bull_score >= bear_score + 2:
        direction = "BUY"
        score, opposite = bull_score, bear_score
    elif bear_score >= min_score and bear_score >= bull_score + 2:
        direction = "SELL"
        score, opposite = bear_score, bull_score

    if direction:
        stop, target = _levels(direction, price, daily, trig)
        analysis.signal = Signal(
            direction=direction,
            strength="STRONG" if score >= strong_score else "MODERATE",
            score=score,
            opposite_score=opposite,
            entry=price,
            stop=stop,
            target=target,
        )
    return analysis
