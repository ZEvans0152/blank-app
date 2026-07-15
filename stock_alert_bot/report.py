"""Deep multi-timeframe technical analysis report (the /full command).

Unlike alerts (which only fire on confluence), this renders the complete
picture for one symbol: trend posture, momentum, volatility, volume, key
levels, and the intraday signal-engine verdict.
"""

from __future__ import annotations

import html
from typing import Optional

import pandas as pd

from .config import Config
from .formatting import DISCLAIMER, _fmt_price
from .indicators import add_indicators
from .market_data import fetch_ohlcv
from .signals import _ok, analyze


def _esc(value) -> str:
    return html.escape(str(value))


def _rsi_zone(value: float) -> str:
    if value >= 70:
        return "overbought"
    if value >= 55:
        return "bullish"
    if value > 45:
        return "neutral"
    if value > 30:
        return "bearish"
    return "oversold"


def _stoch_zone(value: float) -> str:
    if value >= 80:
        return "overbought"
    if value <= 20:
        return "oversold"
    return "mid-range"


def _trend_posture(last: pd.Series) -> Optional[str]:
    if not _ok(last["close"], last["sma20"], last["sma50"], last["sma200"]):
        return None
    c, s20, s50, s200 = last["close"], last["sma20"], last["sma50"], last["sma200"]
    if c > s20 > s50 > s200:
        return "strong uptrend — price above stacked 20/50/200-day SMAs"
    if c < s20 < s50 < s200:
        return "strong downtrend — price below stacked 20/50/200-day SMAs"
    above = [name for name, v in (("20d", s20), ("50d", s50), ("200d", s200)) if c > v]
    below = [name for name, v in (("20d", s20), ("50d", s50), ("200d", s200)) if c <= v]
    if len(above) == 3:
        return "uptrend — price above the 20/50/200-day SMAs (not yet stacked)"
    if len(below) == 3:
        return "downtrend — price below the 20/50/200-day SMAs (not yet stacked)"
    parts = []
    if above:
        parts.append("above " + "/".join(above))
    if below:
        parts.append("below " + "/".join(below))
    return "mixed — price " + ", ".join(parts) + " SMA"


def _daily_bias(last: pd.Series) -> tuple[int, int]:
    """Count bullish factors on the daily frame out of those measurable."""
    checks = [
        (("close", "sma200"), lambda r: r["close"] > r["sma200"]),
        (("close", "sma50"), lambda r: r["close"] > r["sma50"]),
        (("sma50", "sma200"), lambda r: r["sma50"] > r["sma200"]),
        (("macd", "macd_signal"), lambda r: r["macd"] > r["macd_signal"]),
        (("rsi",), lambda r: r["rsi"] > 50),
        (("plus_di", "minus_di"), lambda r: r["plus_di"] > r["minus_di"]),
    ]
    bullish = total = 0
    for cols, test in checks:
        if _ok(*(last[c] for c in cols)):
            total += 1
            if test(last):
                bullish += 1
    return bullish, total


def _bias_label(bullish: int, total: int) -> str:
    if total == 0:
        return "unknown"
    ratio = bullish / total
    if ratio >= 0.85:
        return "strongly bullish"
    if ratio >= 0.6:
        return "bullish"
    if ratio > 0.4:
        return "mixed"
    if ratio > 0.15:
        return "bearish"
    return "strongly bearish"


def build_full_report(symbol: str, cfg: Config) -> Optional[str]:
    """Fetch data and render the full report as Telegram HTML.

    Returns None when there isn't enough data to say anything useful.
    """
    daily_raw = fetch_ohlcv(symbol, cfg.context_interval, cfg.context_period)
    if daily_raw is None or len(daily_raw) < 30:
        return None
    intraday_raw = fetch_ohlcv(symbol, cfg.trigger_interval, cfg.trigger_period)

    daily = add_indicators(daily_raw)
    last = daily.iloc[-1]
    prev = daily.iloc[-2] if len(daily) >= 2 else last
    price = float(last["close"])

    lines = [f"\U0001f4ca <b>Full analysis — {_esc(symbol)}</b>"]

    # --- Header: price, change, 52-week position -----------------------------
    change_txt = ""
    if _ok(prev["close"]) and prev["close"]:
        change = 100.0 * (price - float(prev["close"])) / float(prev["close"])
        change_txt = f" ({change:+.2f}% vs prior session)"
    lines.append(f"Price: <b>{_fmt_price(price)}</b>{change_txt}")

    year = daily.tail(252)
    hi52, lo52 = float(year["high"].max()), float(year["low"].min())
    if hi52 > lo52:
        pos = 100.0 * (price - lo52) / (hi52 - lo52)
        lines.append(
            f"52-week range: {_fmt_price(lo52)} – {_fmt_price(hi52)}"
            f" (trading at {pos:.0f}% of range)"
        )

    # --- Trend (daily) --------------------------------------------------------
    lines += ["", "<b>Trend (daily)</b>"]
    posture = _trend_posture(last)
    if posture:
        lines.append(f"• {_esc(posture)}")
    if _ok(last["adx"], last["plus_di"], last["minus_di"]):
        direction = "buyers" if last["plus_di"] > last["minus_di"] else "sellers"
        strength = (
            "strong trend" if last["adx"] >= 40
            else "trending" if last["adx"] >= 20
            else "weak/range-bound"
        )
        lines.append(
            f"• ADX {last['adx']:.0f} ({strength}), DI shows {direction} in control"
        )

    # --- Momentum ---------------------------------------------------------------
    lines += ["", "<b>Momentum</b>"]
    if _ok(last["rsi"]):
        lines.append(f"• RSI(14) daily: {last['rsi']:.1f} — {_rsi_zone(float(last['rsi']))}")
    if _ok(last["macd"], last["macd_signal"], last["macd_hist"]):
        state = "bullish" if last["macd"] > last["macd_signal"] else "bearish"
        side = "above" if last["macd"] > 0 else "below"
        lines.append(
            f"• MACD daily: {state} (line {side} zero, histogram {last['macd_hist']:+.3f})"
        )
    if _ok(last["stoch_k"], last["stoch_d"]):
        lines.append(
            f"• Stochastic daily: %K {last['stoch_k']:.0f} / %D {last['stoch_d']:.0f}"
            f" — {_stoch_zone(float(last['stoch_k']))}"
        )

    # --- Volatility ----------------------------------------------------------------
    lines += ["", "<b>Volatility</b>"]
    if _ok(last["atr"]) and price:
        lines.append(
            f"• ATR(14) daily: {_fmt_price(float(last['atr']))}"
            f" ({100.0 * float(last['atr']) / price:.1f}% of price)"
        )
    if _ok(last["bb_upper"], last["bb_lower"], last["bb_mid"]) and last["bb_mid"]:
        width = (last["bb_upper"] - last["bb_lower"]) / last["bb_mid"]
        widths = ((daily["bb_upper"] - daily["bb_lower"]) / daily["bb_mid"]).dropna().tail(120)
        squeeze = ""
        if len(widths) >= 60 and width <= widths.quantile(0.2):
            squeeze = " — squeeze: bands unusually tight, a breakout often follows"
        if last["bb_upper"] != last["bb_lower"]:
            pct_b = (price - last["bb_lower"]) / (last["bb_upper"] - last["bb_lower"])
            lines.append(f"• Bollinger %B: {pct_b:.2f}, bandwidth {100 * width:.1f}%{squeeze}")

    # --- Volume ------------------------------------------------------------------
    if _ok(last["vol_ratio"]):
        lines += ["", "<b>Volume</b>"]
        lines.append(f"• Last session: {last['vol_ratio']:.1f}× its 20-day average")

    # --- Key levels ------------------------------------------------------------
    lines += ["", "<b>Key levels</b>"]
    if _ok(last["high"], last["low"]):
        p = (float(last["high"]) + float(last["low"]) + price) / 3.0
        r1 = 2 * p - float(last["low"])
        s1 = 2 * p - float(last["high"])
        r2 = p + (float(last["high"]) - float(last["low"]))
        s2 = p - (float(last["high"]) - float(last["low"]))
        lines.append(
            f"• Pivots: S2 {_fmt_price(s2)} · S1 {_fmt_price(s1)}"
            f" · P {_fmt_price(p)} · R1 {_fmt_price(r1)} · R2 {_fmt_price(r2)}"
        )
    if _ok(last["hh20"], last["ll20"]):
        lines.append(
            f"• 20-day range: {_fmt_price(float(last['ll20']))}"
            f" – {_fmt_price(float(last['hh20']))}"
        )

    # --- Intraday signal engine ----------------------------------------------------
    lines += ["", f"<b>Intraday ({cfg.trigger_interval}) signal engine</b>"]
    intraday_analysis = None
    if intraday_raw is not None:
        intraday_analysis = analyze(
            symbol,
            intraday_raw,
            daily_raw,
            min_score=cfg.min_score,
            strong_score=cfg.strong_score,
            timeframe=cfg.trigger_interval,
        )
    if intraday_analysis is None:
        lines.append("• Not enough intraday data")
    elif intraday_analysis.signal:
        sig = intraday_analysis.signal
        emoji = "\U0001f7e2" if sig.direction == "BUY" else "\U0001f534"
        lines.append(
            f"• {emoji} <b>{sig.direction}</b> ({sig.strength}, score {sig.score}"
            f" vs {sig.opposite_score})"
        )
        reasons = (
            intraday_analysis.bull_reasons
            if sig.direction == "BUY"
            else intraday_analysis.bear_reasons
        )
        lines += [f"   – {_esc(r.text)}" for r in reasons]
        if sig.stop is not None and sig.target is not None:
            lines.append(
                f"   – Levels: entry ≈ {_fmt_price(sig.entry)}, stop ≈ "
                f"{_fmt_price(sig.stop)}, target ≈ {_fmt_price(sig.target)}"
            )
    else:
        lines.append(
            f"• No entry signal right now (bull {intraday_analysis.bull_score}"
            f" / bear {intraday_analysis.bear_score})"
        )

    # --- Verdict -----------------------------------------------------------------
    bullish, total = _daily_bias(last)
    lines += [
        "",
        f"<b>Bias</b>: {_bias_label(bullish, total)}"
        f" ({bullish}/{total} daily factors bullish)",
        "",
        f"<i>{_esc(DISCLAIMER)}</i>",
    ]
    return "\n".join(lines)
