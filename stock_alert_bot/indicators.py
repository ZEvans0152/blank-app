"""Technical indicators implemented with pandas/numpy — no TA-Lib required.

Every function returns Series aligned to the input index and follows the
standard textbook definition (Wilder smoothing where that is the convention),
so values match what trading platforms display. Warm-up bars are NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (RMA): an EMA with alpha = 1/period."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder(gain, period)
    avg_loss = _wilder(loss, period)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # Conventions when there are no losses in the window:
    out = out.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    out = out.mask((avg_loss == 0.0) & (avg_gain == 0.0), 50.0)
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return mid + num_std * std, mid, mid - num_std * std


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    k_smooth: int = 3,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()
    rng = (highest - lowest).replace(0.0, np.nan)
    raw_k = 100.0 * (close - lowest) / rng
    k = raw_k.rolling(k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(d_period, min_periods=d_period).mean()
    return k, d


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return _wilder(true_range(high, low, close), period)


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index. Returns (adx, +DI, -DI)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0.0), up, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0.0), down, 0.0), index=high.index
    )
    atr_s = _wilder(true_range(high, low, close), period).replace(0.0, np.nan)
    plus_di = 100.0 * _wilder(plus_dm, period) / atr_s
    minus_di = 100.0 * _wilder(minus_dm, period) / atr_s
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    return _wilder(dx, period), plus_di, minus_di


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an OHLCV frame with the full indicator set appended.

    Expects lowercase columns: open, high, low, close, volume.
    """
    out = df.copy()
    c, h, l = out["close"], out["high"], out["low"]
    v = out["volume"].astype(float)

    out["sma20"] = sma(c, 20)
    out["sma50"] = sma(c, 50)
    out["sma200"] = sma(c, 200)
    out["rsi"] = rsi(c)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(c)
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = bollinger(c)
    out["stoch_k"], out["stoch_d"] = stochastic(h, l, c)
    out["atr"] = atr(h, l, c)
    out["adx"], out["plus_di"], out["minus_di"] = adx(h, l, c)
    out["vol_sma20"] = sma(v, 20)
    out["vol_ratio"] = v / out["vol_sma20"].replace(0.0, np.nan)
    # Rolling 20-bar extremes; shift(1) in the rules gives the prior-bar range
    # so a breakout compares today's close against yesterday's channel.
    out["hh20"] = h.rolling(20, min_periods=20).max()
    out["ll20"] = l.rolling(20, min_periods=20).min()
    return out
