"""Entry point: `python -m stock_alert_bot [--check SYMBOL | --scan-once]`."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import Config
from .formatting import format_analysis, html_to_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="stock_alert_bot",
        description="Technical-analysis stock alerts over Telegram.",
    )
    parser.add_argument(
        "--check",
        metavar="SYMBOL",
        help="analyze one symbol, print the report to the console, and exit "
        "(no Telegram token required)",
    )
    parser.add_argument(
        "--scan-once",
        action="store_true",
        help="run a single scan (sending any alerts to subscribed chats) and exit",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.check:
        cfg = Config.from_env(require_token=False)
        # Imported lazily so --help works without pandas/yfinance installed.
        from .market_data import fetch_ohlcv, is_valid_symbol
        from .signals import analyze

        symbol = args.check.strip().upper()
        if not is_valid_symbol(symbol):
            print(f"'{symbol}' does not look like a valid Yahoo Finance symbol.")
            return 2
        trigger = fetch_ohlcv(symbol, cfg.trigger_interval, cfg.trigger_period)
        if trigger is None:
            print(f"Could not fetch data for {symbol}.")
            return 1
        context = fetch_ohlcv(symbol, cfg.context_interval, cfg.context_period)
        analysis = analyze(
            symbol,
            trigger,
            context,
            min_score=cfg.min_score,
            strong_score=cfg.strong_score,
            timeframe=cfg.trigger_interval,
        )
        if analysis is None:
            print(f"Not enough completed candles to analyze {symbol}.")
            return 1
        print(html_to_text(format_analysis(analysis)))
        return 0

    cfg = Config.from_env(require_token=True)
    from .bot import Bot

    bot = Bot(cfg)
    if args.scan_once:
        bot.scan()
        return 0

    try:
        bot.run()
    except KeyboardInterrupt:
        print("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
