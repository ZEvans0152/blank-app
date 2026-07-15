# 📈 Stock Alert Bot — technical-analysis signals over Telegram

A self-contained Python bot that watches your symbols, runs real technical
analysis on every scan, and messages you on Telegram when a high-confluence
buy or sell setup appears.

## How the analysis works

No single indicator ever triggers an alert. Each rule that fires contributes a
weight, and an alert is sent only when one side reaches the score threshold
**and** beats the opposite side by 2+ points.

**Trigger rules** (15-minute candles by default):

| Rule | Weight |
| --- | --- |
| RSI(14) crosses up out of oversold / down out of overbought | 2 |
| RSI turning in the extreme zone (not yet crossed) | 1 |
| MACD(12,26,9) crossover — on the "early reversal" side of the zero line | 2 (3) |
| Close rebounds back inside the lower/upper Bollinger Band (20, 2σ) | 2 |
| Stochastic (14,3,3) %K/%D crossover in the oversold/overbought zone | 1 |
| 20-bar channel breakout/breakdown on ≥1.5× average volume | 2 |
| Volume ≥1.5× its 20-bar average confirming the setup | 1 |

**Daily context** (only amplifies an existing setup, never fires alone):

| Rule | Weight |
| --- | --- |
| Setup aligned with the 200-day SMA trend | 1 |
| Golden cross / death cross (SMA50 × SMA200) | 3 |
| ADX(14) trend-strength note in every alert | — |

Defaults: alert at score ≥ 4, labeled **STRONG** at ≥ 7. Every alert includes
ATR-based suggested levels (stop = 2×ATR, target = 3×ATR) and the reasons that
fired. Signals are computed on **completed candles only**, and a per-symbol,
per-direction cooldown (4h default) prevents alert spam.

Data comes from Yahoo Finance via `yfinance` — free, no API key, but quotes
can be delayed up to ~15 minutes on some exchanges. Crypto (`BTC-USD`) and
forex (`EURUSD=X`) symbols work too and are scanned around the clock, while
stocks are only scanned during US market hours (configurable).

## Setup

1. Install dependencies (from the repo root):

   ```
   pip install -r requirements.txt
   ```

2. Create a Telegram bot: message [@BotFather](https://t.me/BotFather), send
   `/newbot`, and copy the token.

3. Configure:

   ```
   cp .env.example .env
   # edit .env and set TELEGRAM_BOT_TOKEN=...
   ```

   Optionally set `ALLOWED_CHAT_IDS` to lock the bot to your own chat id
   (message [@userinfobot](https://t.me/userinfobot) to find it).

4. Run the bot:

   ```
   python -m stock_alert_bot
   ```

5. Open your bot in Telegram and send `/start`.

## Telegram commands

| Command | What it does |
| --- | --- |
| `/watch AAPL BTC-USD` | add symbols to your watchlist |
| `/unwatch TSLA` | remove symbols |
| `/watchlist` | show your watchlist |
| `/check NVDA` | quick signal check for one symbol, right now |
| `/full NVDA` | deep multi-timeframe report: trend posture, momentum, volatility, volume, pivot levels, 52-week position, and the intraday signal-engine verdict |
| `/scan` | scan your whole watchlist immediately |
| `/mute` / `/unmute` | pause / resume alerts |
| `/settings` | show scanner configuration |
| `/stop` | unsubscribe and delete your data |

## CLI

```
python -m stock_alert_bot --check AAPL   # one-off console analysis, no token needed
python -m stock_alert_bot --scan-once    # single scan + alerts, then exit
python -m stock_alert_bot -v             # run the bot with debug logging
```

## Configuration

All via environment variables (or `.env`) — see `.env.example` for the full
list: scan interval, candle size, score thresholds, cooldown, default
watchlist, market-hours behavior, and state-file location. Chat subscriptions,
watchlists, and cooldowns persist in a local JSON state file across restarts.

## Running 24/7

The bot is a single long-running process — put it on any always-on machine
(home server, Raspberry Pi, a ~$5/mo VPS, or a container platform like
Fly.io/Railway).

**Docker (recommended):**

```
cp .env.example .env        # set TELEGRAM_BOT_TOKEN (and ALERT_CHAT_IDS)
docker compose up -d --build
docker compose logs -f      # watch it run
```

State (subscriptions, watchlists, cooldowns) persists in `./data/` across
restarts and upgrades. `restart: unless-stopped` brings the bot back after
crashes and reboots.

**systemd (bare Linux, no Docker):** see `deploy/stock-alert-bot.service` —
copy it into `/etc/systemd/system/`, adjust the paths, then
`systemctl enable --now stock-alert-bot`.

Tip: set `ALERT_CHAT_IDS` to your chat id so a rebuilt deployment starts
alerting you immediately even with a blank state volume, and set
`ALLOWED_CHAT_IDS` to the same value to keep the bot private.

## Tests

```
pip install pytest
python -m pytest tests/ -v
```

## Disclaimer

This bot performs automated technical analysis for informational purposes
only. It is **not financial advice**; markets can and do invalidate any
technical setup. Never trade money you cannot afford to lose.
