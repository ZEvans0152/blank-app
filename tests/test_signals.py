import numpy as np
import pandas as pd

from stock_alert_bot.formatting import format_alert, format_analysis, html_to_text
from stock_alert_bot.signals import MIN_TRIGGER_BARS, analyze


def _ohlcv(close, volume=None, spread=0.4):
    close = np.asarray(close, dtype=float)
    n = len(close)
    volume = (
        np.asarray(volume, dtype=float)
        if volume is not None
        else np.full(n, 10_000.0)
    )
    return pd.DataFrame(
        {
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": volume,
        }
    )


def _oversold_bounce_frame():
    """Steady decline that overshoots, then a high-volume reversal candle.

    Engineered to fire several bullish rules at once: RSI recovering from
    oversold, a close back inside the lower Bollinger Band, a stochastic
    cross, and volume confirmation.
    """
    steady = np.linspace(100, 88, 62)          # slow controlled decline
    plunge = np.array([86.5, 84.5, 82.0])      # acceleration below the band
    bounce = np.array([85.5])                  # sharp reversal candle
    close = np.concatenate([steady, plunge, bounce])
    volume = np.full(len(close), 10_000.0)
    volume[-1] = 40_000.0                      # 4x spike on the reversal
    return _ohlcv(close, volume)


def test_not_enough_bars_returns_none():
    df = _ohlcv(np.linspace(100, 101, MIN_TRIGGER_BARS - 1))
    assert analyze("TEST", df, None) is None


def test_flat_market_produces_no_signal():
    rng = np.random.default_rng(11)
    close = 100 + rng.normal(0, 0.05, 120)  # noise, no structure
    analysis = analyze("FLAT", _ohlcv(close), None)
    assert analysis is not None
    assert analysis.signal is None


def test_oversold_bounce_fires_buy_signal():
    analysis = analyze("BOUNCE", _oversold_bounce_frame(), None, min_score=4)
    assert analysis is not None
    assert analysis.bull_score > analysis.bear_score
    assert analysis.signal is not None
    assert analysis.signal.direction == "BUY"
    assert analysis.signal.entry == analysis.price
    # ATR levels: stop below entry, target above for a BUY.
    assert analysis.signal.stop < analysis.signal.entry < analysis.signal.target


def test_overbought_rejection_fires_sell_signal():
    # Mirror image of the bounce: melt-up that overshoots, then reversal down.
    steady = np.linspace(100, 112, 62)
    spike = np.array([113.5, 115.5, 118.0])
    reject = np.array([114.5])
    close = np.concatenate([steady, spike, reject])
    volume = np.full(len(close), 10_000.0)
    volume[-1] = 40_000.0
    analysis = analyze("TOP", _ohlcv(close, volume), None, min_score=4)
    assert analysis is not None
    assert analysis.signal is not None
    assert analysis.signal.direction == "SELL"
    assert analysis.signal.stop > analysis.signal.entry > analysis.signal.target


def test_daily_context_adds_trend_alignment():
    trigger = _oversold_bounce_frame()
    rng = np.random.default_rng(2)
    up_daily = _ohlcv(50 + np.arange(260) * 0.3 + rng.normal(0, 0.3, 260))
    with_ctx = analyze("CTX", trigger, up_daily, min_score=4)
    without_ctx = analyze("CTX", trigger, None, min_score=4)
    assert with_ctx.bull_score > without_ctx.bull_score
    assert any("uptrend" in note.lower() for note in with_ctx.context_notes)


def test_formatting_roundtrip():
    analysis = analyze("BOUNCE", _oversold_bounce_frame(), None, min_score=4)
    alert = format_alert(analysis)
    assert "BUY signal — BOUNCE" in alert
    assert "not financial advice" in alert
    report = html_to_text(format_analysis(analysis))
    assert "<b>" not in report and "BOUNCE" in report
