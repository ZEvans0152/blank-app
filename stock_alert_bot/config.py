"""Configuration loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def load_dotenv_file(path: str = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines, '#' comments, no interpolation.

    Values already present in the environment are never overridden.
    """
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _csv_symbols(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


@dataclass
class Config:
    telegram_token: str

    # Scanner cadence and alert throttling
    scan_interval_minutes: int = 15
    alert_cooldown_minutes: int = 240

    # Candle sizes: the trigger frame generates entries, the context frame
    # (daily) establishes the larger trend.
    trigger_interval: str = "15m"
    trigger_period: str = "5d"
    context_interval: str = "1d"
    context_period: str = "1y"

    # Signal thresholds (see signals.py for rule weights)
    min_score: int = 4
    strong_score: int = 7

    # Only scan exchange-traded symbols during US market hours. Crypto/forex
    # symbols are always scanned regardless of this flag.
    market_hours_only: bool = True

    default_watchlist: list[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "NVDA", "SPY", "TSLA"]
    )
    max_watchlist_size: int = 30

    # If set, only these Telegram chat ids may use the bot.
    allowed_chat_ids: Optional[set[int]] = None

    # Chats to auto-subscribe on startup (fresh deployments alert these chats
    # immediately, no /start needed).
    alert_chat_ids: list[int] = field(default_factory=list)

    state_file: str = ".stock_alert_bot_state.json"

    @classmethod
    def from_env(cls, require_token: bool = True) -> "Config":
        load_dotenv_file()
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if require_token and not token:
            raise SystemExit(
                "TELEGRAM_BOT_TOKEN is not set. Create a bot with @BotFather on "
                "Telegram, then export TELEGRAM_BOT_TOKEN or put it in a .env file."
            )

        allowed: Optional[set[int]] = None
        raw_allowed = os.environ.get("ALLOWED_CHAT_IDS", "").strip()
        if raw_allowed:
            allowed = {int(x) for x in raw_allowed.split(",") if x.strip()}

        raw_watchlist = os.environ.get("DEFAULT_WATCHLIST", "").strip()

        raw_alert = os.environ.get("ALERT_CHAT_IDS", "").strip()
        alert_chat_ids = [int(x) for x in raw_alert.split(",") if x.strip()]

        return cls(
            telegram_token=token,
            scan_interval_minutes=_env_int("SCAN_INTERVAL_MINUTES", 15),
            alert_cooldown_minutes=_env_int("ALERT_COOLDOWN_MINUTES", 240),
            trigger_interval=os.environ.get("TRIGGER_INTERVAL", "15m"),
            trigger_period=os.environ.get("TRIGGER_PERIOD", "5d"),
            min_score=_env_int("MIN_SCORE", 4),
            strong_score=_env_int("STRONG_SCORE", 7),
            market_hours_only=_env_bool("MARKET_HOURS_ONLY", True),
            default_watchlist=(
                _csv_symbols(raw_watchlist)
                if raw_watchlist
                else ["AAPL", "MSFT", "NVDA", "SPY", "TSLA"]
            ),
            allowed_chat_ids=allowed,
            alert_chat_ids=alert_chat_ids,
            state_file=os.environ.get("STATE_FILE", ".stock_alert_bot_state.json"),
        )
