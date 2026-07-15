"""Render analyses/signals as Telegram HTML messages (and plain text for CLI)."""

from __future__ import annotations

import html
import re

from .signals import Analysis

DISCLAIMER = "Automated technical analysis — informational only, not financial advice."


def _esc(value) -> str:
    return html.escape(str(value))


def _fmt_price(value: float) -> str:
    if value >= 1000:
        return f"{value:,.2f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def format_alert(analysis: Analysis) -> str:
    """Alert message for a confirmed signal."""
    sig = analysis.signal
    assert sig is not None
    emoji = "\U0001f7e2" if sig.direction == "BUY" else "\U0001f534"  # green/red circle
    reasons = analysis.bull_reasons if sig.direction == "BUY" else analysis.bear_reasons

    lines = [
        f"{emoji} <b>{sig.direction} signal — {_esc(analysis.symbol)}</b>"
        f" ({sig.strength}, score {sig.score} vs {sig.opposite_score})",
        f"<i>{_esc(analysis.timeframe)} candles · {_esc(analysis.bar_time)}"
        f" · price {_fmt_price(analysis.price)}</i>",
        "",
        "<b>Why</b>",
    ]
    lines += [f"• {_esc(r.text)}" for r in reasons]

    if analysis.context_notes:
        lines += ["", "<b>Daily context</b>"]
        lines += [f"• {_esc(n)}" for n in analysis.context_notes]

    if sig.stop is not None and sig.target is not None:
        lines += [
            "",
            "<b>ATR-based levels</b>",
            f"Entry ≈ {_fmt_price(sig.entry)} · Stop ≈ {_fmt_price(sig.stop)}"
            f" · Target ≈ {_fmt_price(sig.target)}",
        ]

    lines += ["", f"<i>{_esc(DISCLAIMER)}</i>"]
    return "\n".join(lines)


def format_analysis(analysis: Analysis) -> str:
    """Full on-demand report for /check — shown even when no signal fired."""
    snap = analysis.snapshot
    lines = [
        f"<b>{_esc(analysis.symbol)}</b> — {_fmt_price(analysis.price)}",
        f"<i>{_esc(analysis.timeframe)} candles · {_esc(analysis.bar_time)}</i>",
        "",
    ]

    if analysis.signal:
        sig = analysis.signal
        emoji = "\U0001f7e2" if sig.direction == "BUY" else "\U0001f534"
        lines.append(
            f"{emoji} <b>{sig.direction}</b> ({sig.strength}, score {sig.score}"
            f" vs {sig.opposite_score})"
        )
    else:
        lines.append(
            f"⚖️ No actionable signal (bull {analysis.bull_score}"
            f" / bear {analysis.bear_score})"
        )

    indicator_bits = []
    if snap.get("rsi") is not None:
        indicator_bits.append(f"RSI(14): {snap['rsi']:.1f}")
    if snap.get("macd_hist") is not None:
        indicator_bits.append(f"MACD hist: {snap['macd_hist']:+.3f}")
    if snap.get("stoch_k") is not None and snap.get("stoch_d") is not None:
        indicator_bits.append(f"Stoch %K/%D: {snap['stoch_k']:.0f}/{snap['stoch_d']:.0f}")
    if snap.get("percent_b") is not None:
        indicator_bits.append(f"Bollinger %B: {snap['percent_b']:.2f}")
    if snap.get("vol_ratio") is not None:
        indicator_bits.append(f"Volume: {snap['vol_ratio']:.1f}× 20-bar avg")
    if snap.get("atr") is not None:
        indicator_bits.append(f"ATR(14): {_fmt_price(snap['atr'])}")
    if indicator_bits:
        lines += ["", "<b>Indicators</b>"] + [f"• {_esc(b)}" for b in indicator_bits]

    for label, reasons in (
        ("Bullish factors", analysis.bull_reasons),
        ("Bearish factors", analysis.bear_reasons),
    ):
        if reasons:
            lines += ["", f"<b>{label}</b>"]
            lines += [f"• {_esc(r.text)} (+{r.weight})" for r in reasons]

    if analysis.context_notes:
        lines += ["", "<b>Daily context</b>"]
        lines += [f"• {_esc(n)}" for n in analysis.context_notes]

    lines += ["", f"<i>{_esc(DISCLAIMER)}</i>"]
    return "\n".join(lines)


_TAG_RE = re.compile(r"</?(?:b|i|u|s|code|pre)>")


def html_to_text(message: str) -> str:
    """Strip Telegram HTML for console output."""
    return html.unescape(_TAG_RE.sub("", message))
