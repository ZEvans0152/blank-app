# Autonomous Money Agent Dashboard

A Streamlit control center for a Phase 3 autonomous opportunity agent across prediction markets and crypto-style signals. The application defaults to simulated execution so strategies, position sizing, and risk limits can be evaluated before any real-money integration.

## Are we ready to start live trading?

No. This repository is not ready for live-money trading yet. The current application is a simulation and dashboard harness, not a production trading system. It does not yet include real Kalshi, Polymarket, or exchange order-book adapters; live order execution; Telegram approval and kill-switch wiring; credential validation; persistent audit logging; or compliance/venue eligibility checks.

## Features

- Autonomous signal generation loop for Kalshi, Polymarket, and crypto-style opportunities.
- Configurable risk policy with max trade size, daily loss limit, exposure cap, spread filter, liquidity filter, and consecutive-loss circuit breaker.
- Simulated order execution with stop-loss, take-profit, and max-holding-time exits.
- Dashboard metrics for total equity, realized P&L, unrealized P&L, win rate, open positions, and launch-readiness checks.
- Tabs for equity curve, open positions, signal history, closed trades, risk policy, and live-trading readiness.
- Live-trading mode placeholder with guarded warnings so API adapters can be added deliberately.

## Required before live trading

- Multi-week paper-trading history with enough trades to evaluate drawdown, slippage assumptions, and strategy performance.
- Real Kalshi, Polymarket, and crypto market data adapters.
- Venue-specific order execution adapters with dry-run and sandbox tests.
- Telegram alerts, approvals, and kill switch connected to the execution layer.
- Trading-only API keys, withdrawal lockouts, and secret validation.
- Compliance and venue eligibility review for every market and jurisdiction involved.
- Persistent database storage for signals, orders, fills, positions, errors, and audit logs.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Safety note

This project is not financial advice and does not guarantee profits. Autonomous trading can lose money. Start in paper mode, use small dedicated balances, disable withdrawals on API keys, and keep manual kill switches enabled before connecting any live venue.
