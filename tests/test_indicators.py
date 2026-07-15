import numpy as np
import pandas as pd
import pytest

from stock_alert_bot.indicators import (
    add_indicators,
    atr,
    bollinger,
    ema,
    macd,
    rsi,
    sma,
    stochastic,
)


def _series(values):
    return pd.Series(values, dtype=float)


def test_sma_basic():
    s = _series([1, 2, 3, 4, 5])
    out = sma(s, 3)
    assert np.isnan(out.iloc[0]) and np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_ema_matches_hand_computation():
    s = _series([10, 11, 12, 13, 14])
    out = ema(s, 3)  # alpha = 0.5
    # Pandas seeds the EMA with the first value.
    expected = 10.0
    for value in [11, 12, 13, 14]:
        expected = 0.5 * value + 0.5 * expected
    assert out.iloc[-1] == pytest.approx(expected)


def test_rsi_extremes_and_bounds():
    up = _series(np.arange(1, 40, dtype=float))
    down = _series(np.arange(40, 1, -1, dtype=float))
    flat = _series(np.full(40, 50.0))
    assert rsi(up).iloc[-1] == pytest.approx(100.0)
    assert rsi(down).iloc[-1] == pytest.approx(0.0)
    assert rsi(flat).iloc[-1] == pytest.approx(50.0)

    rng = np.random.default_rng(7)
    noisy = _series(100 + np.cumsum(rng.normal(0, 1, 300)))
    values = rsi(noisy).dropna()
    assert ((values >= 0) & (values <= 100)).all()


def test_macd_is_ema_difference():
    rng = np.random.default_rng(1)
    close = _series(100 + np.cumsum(rng.normal(0, 1, 120)))
    line, sig, hist = macd(close)
    expected = ema(close, 12) - ema(close, 26)
    pd.testing.assert_series_equal(line, expected)
    assert (hist.dropna() == (line - sig).dropna()).all()


def test_bollinger_collapses_on_constant_series():
    close = _series(np.full(30, 42.0))
    upper, mid, lower = bollinger(close)
    assert upper.iloc[-1] == pytest.approx(42.0)
    assert mid.iloc[-1] == pytest.approx(42.0)
    assert lower.iloc[-1] == pytest.approx(42.0)


def test_stochastic_bounds_and_extremes():
    n = 40
    close = _series(np.linspace(10, 30, n))
    high = close + 0.01
    low = close - 1.0
    k, d = stochastic(high, low, close)
    # In a steady uptrend closing near the highs, %K should be near 100.
    assert k.iloc[-1] > 90
    assert ((k.dropna() >= 0) & (k.dropna() <= 100)).all()
    assert ((d.dropna() >= 0) & (d.dropna() <= 100)).all()


def test_atr_zero_on_flat_market_positive_otherwise():
    n = 30
    flat_close = _series(np.full(n, 10.0))
    assert atr(flat_close, flat_close, flat_close).iloc[-1] == pytest.approx(0.0)

    rng = np.random.default_rng(3)
    close = _series(100 + np.cumsum(rng.normal(0, 1, n)))
    high = close + 1
    low = close - 1
    assert atr(high, low, close).iloc[-1] > 0


def test_add_indicators_produces_expected_columns():
    rng = np.random.default_rng(5)
    n = 250
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.2, n),
            "high": close + np.abs(rng.normal(0, 0.8, n)),
            "low": close - np.abs(rng.normal(0, 0.8, n)),
            "close": close,
            "volume": rng.integers(1_000, 50_000, n).astype(float),
        }
    )
    out = add_indicators(df)
    for col in (
        "sma20", "sma50", "sma200", "rsi", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_mid", "bb_lower", "stoch_k", "stoch_d", "atr", "adx",
        "vol_ratio", "hh20", "ll20",
    ):
        assert col in out.columns, col
        assert not np.isnan(out[col].iloc[-1]), col
    # Band ordering must hold everywhere the bands exist.
    bands = out.dropna(subset=["bb_upper", "bb_lower"])
    assert (bands["bb_upper"] >= bands["bb_mid"]).all()
    assert (bands["bb_mid"] >= bands["bb_lower"]).all()
