"""Market data via Yahoo Finance.

Primary path is yfinance; if that fails (some proxies reset its curl_cffi TLS
handshake), we fall back to Yahoo's public v8 chart API over plain requests
and keep using it for the rest of the process.

Yahoo quotes are near-real-time but can be delayed up to ~15 minutes depending
on the exchange; good enough for swing/momentum alerts on 15m+ candles.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from datetime import timedelta
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

log = logging.getLogger(__name__)

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
_chart_session: Optional[requests.Session] = None
_prefer_chart_api = False

SYMBOL_RE = re.compile(r"^[A-Z0-9.\-=^]{1,15}$")

_INTERVAL_TO_TIMEDELTA = {
    "1m": timedelta(minutes=1),
    "2m": timedelta(minutes=2),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "60m": timedelta(hours=1),
    "1h": timedelta(hours=1),
    "90m": timedelta(minutes=90),
}


def is_valid_symbol(symbol: str) -> bool:
    return bool(SYMBOL_RE.match(symbol))


def is_always_open(symbol: str) -> bool:
    """Crypto (BTC-USD) and forex (EURUSD=X) trade outside NYSE hours."""
    s = symbol.upper()
    return s.endswith(("-USD", "-USDT", "-USDC", "=X"))


def drop_unclosed_bar(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Drop the last row if that candle is still forming.

    Signals must only be computed on completed candles, otherwise an intrabar
    wiggle can fire an alert that disappears by the close of the bar.
    """
    span = _INTERVAL_TO_TIMEDELTA.get(interval)
    if span is None or df.empty:
        return df
    last_start = df.index[-1]
    if not isinstance(last_start, pd.Timestamp):
        return df
    now = pd.Timestamp.now(tz=last_start.tz) if last_start.tz else pd.Timestamp.now()
    if last_start + span > now:
        return df.iloc[:-1]
    return df


def _fetch_via_yfinance(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df is None or df.empty:
        return None
    # Newer yfinance returns MultiIndex columns even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=lambda name: str(name).lower())
    required = ["open", "high", "low", "close", "volume"]
    if any(col not in df.columns for col in required):
        log.warning("unexpected yfinance columns for %s: %s", symbol, list(df.columns))
        return None
    return df[required]


def _fetch_via_chart_api(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from Yahoo's v8 chart endpoint with a plain requests session."""
    global _chart_session
    if _chart_session is None:
        _chart_session = requests.Session()
        _chart_session.headers["User-Agent"] = _BROWSER_UA

    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        + urllib.parse.quote(symbol)
    )
    resp = _chart_session.get(
        url, params={"range": period, "interval": interval}, timeout=20
    )
    resp.raise_for_status()
    chart = resp.json().get("chart") or {}
    results = chart.get("result")
    if chart.get("error") or not results:
        return None

    result = results[0]
    timestamps = result.get("timestamp")
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    if not timestamps or not quote.get("close"):
        return None

    df = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=pd.to_datetime(timestamps, unit="s", utc=True),
        dtype=float,
    )

    # Back-adjust OHLC for splits/dividends the same way auto_adjust does.
    adjclose = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    if adjclose is not None:
        ratio = pd.Series(adjclose, index=df.index, dtype=float) / df["close"]
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] * ratio

    tz_name = (result.get("meta") or {}).get("exchangeTimezoneName")
    if tz_name:
        try:
            df.index = df.index.tz_convert(tz_name)
        except Exception:
            pass
    return df


def fetch_ohlcv(symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    """Return a normalized OHLCV frame (lowercase columns) or None on failure."""
    global _prefer_chart_api
    df = None

    if not _prefer_chart_api:
        try:
            df = _fetch_via_yfinance(symbol, interval, period)
        except Exception:
            log.warning("yfinance failed for %s (%s/%s)", symbol, interval, period)

    if df is None or df.empty:
        try:
            df = _fetch_via_chart_api(symbol, interval, period)
        except Exception:
            log.exception("chart API failed for %s (%s/%s)", symbol, interval, period)
            return None
        if df is not None and not df.empty and not _prefer_chart_api:
            log.info("yfinance unavailable; using Yahoo chart API from now on")
            _prefer_chart_api = True

    if df is None or df.empty:
        log.warning("no data returned for %s (%s/%s)", symbol, interval, period)
        return None

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = drop_unclosed_bar(df, interval)
    return df if not df.empty else None
