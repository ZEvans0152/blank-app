from __future__ import annotations

import random
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class RiskConfig:
    starting_bankroll: float
    max_trade_size: float
    max_daily_loss: float
    max_open_positions: int
    max_total_exposure_pct: float
    min_edge_pct: float
    max_spread_pct: float
    min_liquidity: float
    stop_after_losses: int


DB_PATH = "paper_trading.db"

MARKET_TEMPLATES = [
    ("Kalshi", "Fed decision: rates unchanged", "macro"),
    ("Kalshi", "CPI print above consensus", "macro"),
    ("Kalshi", "Rainfall exceeds forecast", "weather"),
    ("Polymarket", "Election polling leader wins", "politics"),
    ("Polymarket", "Crypto ETF approval window", "crypto"),
    ("Polymarket", "Tech product launches before deadline", "tech"),
    ("Crypto", "BTC/USDT momentum", "crypto"),
    ("Crypto", "ETH funding-rate reversal", "crypto"),
    ("Crypto", "SOL volatility breakout", "crypto"),
]

READINESS_REQUIREMENTS = [
    {
        "requirement": "Paper trading has a statistically useful track record",
        "status": "In progress",
        "detail": "Signals and closed paper trades are now persisted to SQLite for review.",
        "status": "Not met",
        "detail": "No persistent backtest or multi-week paper-trading history exists yet.",
    },
    {
        "requirement": "Real Kalshi, Polymarket, and crypto market data adapters exist",
        "status": "Not met",
        "detail": "Current signals are simulated templates, not live order books or venue APIs.",
    },
    {
        "requirement": "Live order execution adapters are implemented and dry-run tested",
        "status": "Not met",
        "detail": "The guarded live mode intentionally does not place real orders.",
    },
    {
        "requirement": "Telegram approval, alerts, and kill switch are wired in",
        "status": "Not met",
        "detail": "Dashboard notes the kill switch requirement, but bot integration is not connected.",
    },
    {
        "requirement": "Secrets, trading-only API keys, and withdrawal lockouts are configured",
        "status": "Not met",
        "detail": "No .env contract, secret validation, or exchange key permission checks exist yet.",
    },
    {
        "requirement": "Compliance and venue eligibility review is complete",
        "status": "Not met",
        "detail": "Prediction-market and crypto access rules need to be verified before live trading.",
    },
]


st.set_page_config(page_title="Autonomous Money Agent", page_icon="🤖", layout="wide")


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_signals (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                platform TEXT NOT NULL,
                market TEXT NOT NULL,
                category TEXT NOT NULL,
                side TEXT NOT NULL,
                market_price REAL NOT NULL,
                model_probability REAL NOT NULL,
                edge REAL NOT NULL,
                spread REAL NOT NULL,
                liquidity REAL NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                suggested_stake REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_trades (
                id TEXT PRIMARY KEY,
                closed_at TEXT NOT NULL,
                platform TEXT NOT NULL,
                market TEXT NOT NULL,
                category TEXT NOT NULL,
                side TEXT NOT NULL,
                stake REAL NOT NULL,
                pnl REAL NOT NULL,
                roi REAL NOT NULL,
                reason TEXT NOT NULL
            )
            """
        )


def save_signal(signal: dict) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO paper_signals VALUES (
                :id, :created_at, :platform, :market, :category, :side, :market_price,
                :model_probability, :edge, :spread, :liquidity, :confidence, :status,
                :reason, :suggested_stake
            )
            """,
            {
                **signal,
                "created_at": signal["time"].isoformat(),
                "suggested_stake": signal.get("suggested_stake", 0.0),
            },
        )


def save_trade(trade: dict) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO paper_trades VALUES (
                :id, :closed_at, :platform, :market, :category, :side, :stake, :pnl, :roi, :reason
            )
            """,
            {**trade, "closed_at": trade["closed_at"].isoformat()},
        )


def load_trade_history() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM paper_trades ORDER BY closed_at DESC", conn
        )


def load_signal_history() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM paper_signals ORDER BY created_at DESC LIMIT 500", conn
        )


def learning_biases() -> dict[tuple[str, str], float]:
    history = load_trade_history()
    if history.empty:
        return {}
    grouped = history.groupby(["platform", "category"])["roi"].mean().to_dict()
    return {key: float(np.clip(value, -0.08, 0.08)) for key, value in grouped.items()}


init_db()


if "agent_running" not in st.session_state:
    st.session_state.agent_running = False
if "cash" not in st.session_state:
    st.session_state.cash = 10_000.0
if "trades" not in st.session_state:
    st.session_state.trades = []
if "signals" not in st.session_state:
    st.session_state.signals = []
if "positions" not in st.session_state:
    st.session_state.positions = []
if "equity" not in st.session_state:
    st.session_state.equity = [
        {"time": datetime.now(timezone.utc) - timedelta(minutes=30), "equity": 10_000.0}
    ]
if "loss_streak" not in st.session_state:
    st.session_state.loss_streak = 0


def generate_signal() -> dict:
    platform, market, category = random.choice(MARKET_TEMPLATES)
    learned_bias = learning_biases().get((platform, category), 0.0)
    market_price = float(np.clip(np.random.normal(0.5, 0.16), 0.05, 0.95))
    model_probability = float(
        np.clip(
            market_price + learned_bias + np.random.normal(0.045, 0.055),
            0.02,
            0.98,
        )
    market_price = float(np.clip(np.random.normal(0.5, 0.16), 0.05, 0.95))
    model_probability = float(
        np.clip(market_price + np.random.normal(0.045, 0.055), 0.02, 0.98)
    )
    spread = float(np.random.uniform(0.005, 0.095))
    liquidity = float(np.random.lognormal(mean=8.2, sigma=0.75))
    side = "YES" if model_probability > market_price else "NO"
    edge = abs(model_probability - market_price)
    confidence = float(
        np.clip(0.45 + edge * 3.0 + np.random.normal(0, 0.06), 0.1, 0.98)
    )
    return {
        "id": str(uuid.uuid4())[:8],
        "time": datetime.now(timezone.utc),
        "platform": platform,
        "market": market,
        "category": category,
        "side": side,
        "market_price": market_price,
        "model_probability": model_probability,
        "edge": edge,
        "spread": spread,
        "liquidity": liquidity,
        "confidence": confidence,
        "learned_bias": learned_bias,
        "status": "new",
        "reason": "",
    }


def current_unrealized_pnl() -> float:
    return sum(
        position["unrealized_pnl"]
        for position in st.session_state.positions
        if position["status"] == "open"
    )


def readiness_summary() -> tuple[int, int]:
    unmet = sum(1 for item in READINESS_REQUIREMENTS if item["status"] != "Met")
    return len(READINESS_REQUIREMENTS) - unmet, unmet


def total_equity() -> float:
    return st.session_state.cash + sum(
        position["stake"] + position["unrealized_pnl"]
        for position in st.session_state.positions
        if position["status"] == "open"
    )


def daily_realized_pnl() -> float:
    cutoff = datetime.now(timezone.utc) - timedelta(days=1)
    return sum(trade["pnl"] for trade in st.session_state.trades if trade["closed_at"] >= cutoff)


def evaluate_signal(signal: dict, risk: RiskConfig) -> tuple[bool, str, float]:
    open_positions = [p for p in st.session_state.positions if p["status"] == "open"]
    open_exposure = sum(p["stake"] for p in open_positions)
    equity = total_equity()



def total_equity() -> float:
    return st.session_state.cash + sum(
        position["stake"] + position["unrealized_pnl"]
        for position in st.session_state.positions
        if position["status"] == "open"
    )


def daily_realized_pnl() -> float:
    cutoff = datetime.now(timezone.utc) - timedelta(days=1)
    return sum(trade["pnl"] for trade in st.session_state.trades if trade["closed_at"] >= cutoff)


def evaluate_signal(signal: dict, risk: RiskConfig) -> tuple[bool, str, float]:
    open_positions = [p for p in st.session_state.positions if p["status"] == "open"]
    open_exposure = sum(p["stake"] for p in open_positions)
    equity = total_equity()

    if st.session_state.loss_streak >= risk.stop_after_losses:
        return False, "Loss-streak circuit breaker is active", 0.0
    if daily_realized_pnl() <= -risk.max_daily_loss:
        return False, "Daily loss limit reached", 0.0
    if len(open_positions) >= risk.max_open_positions:
        return False, "Maximum open positions reached", 0.0
    if signal["edge"] < risk.min_edge_pct / 100:
        return False, "Estimated edge is below threshold", 0.0
    if signal["spread"] > risk.max_spread_pct / 100:
        return False, "Bid/ask spread is too wide", 0.0
    if signal["liquidity"] < risk.min_liquidity:
        return False, "Liquidity is below threshold", 0.0
    if open_exposure >= equity * risk.max_total_exposure_pct / 100:
        return False, "Portfolio exposure limit reached", 0.0

    edge_weight = min(signal["edge"] / 0.12, 1.0)
    confidence_weight = signal["confidence"]
    stake = risk.max_trade_size * max(0.15, edge_weight * confidence_weight)
    remaining_exposure = equity * risk.max_total_exposure_pct / 100 - open_exposure
    stake = max(0.0, min(stake, risk.max_trade_size, remaining_exposure, st.session_state.cash))
    if stake < 1:
        return False, "Suggested position size is below $1", 0.0
    return True, "Approved by autonomous risk policy", stake


def open_position(signal: dict, stake: float) -> None:
    st.session_state.cash -= stake
    st.session_state.positions.append(
        {
            "id": str(uuid.uuid4())[:8],
            "signal_id": signal["id"],
            "opened_at": datetime.now(timezone.utc),
            "platform": signal["platform"],
            "market": signal["market"],
            "side": signal["side"],
            "entry_price": signal["market_price"],
            "current_price": signal["market_price"],
            "stake": stake,
            "unrealized_pnl": 0.0,
            "status": "open",
            "edge": signal["edge"],
            "confidence": signal["confidence"],
            "category": signal["category"],
            "holding_cycles": 0,
        }
    )


def mark_positions() -> None:
    for position in st.session_state.positions:
        if position["status"] != "open":
            continue
        position["holding_cycles"] = position.get("holding_cycles", 0) + 1
        drift = position["edge"] * 0.12
        shock = float(np.random.normal(drift, 0.035))
        position["current_price"] = float(np.clip(position["current_price"] + shock, 0.01, 0.99))
        price_return = (position["current_price"] - position["entry_price"]) / max(position["entry_price"], 0.01)
        if position["side"] == "NO":
            price_return *= -1
        position["unrealized_pnl"] = position["stake"] * price_return


def close_position(position: dict, reason: str) -> None:
    pnl = position["unrealized_pnl"]
    position["status"] = "closed"
    position["closed_at"] = datetime.now(timezone.utc)
    st.session_state.cash += position["stake"] + pnl
    trade = {
        "id": position["id"],
        "platform": position["platform"],
        "market": position["market"],
        "side": position["side"],
        "category": position["category"],
        "stake": position["stake"],
        "pnl": pnl,
        "roi": pnl / position["stake"],
        "reason": reason,
        "closed_at": datetime.now(timezone.utc),
    }
    st.session_state.trades.append(trade)
    save_trade(trade)
    st.session_state.trades.append(
        {
            "id": position["id"],
            "platform": position["platform"],
            "market": position["market"],
            "side": position["side"],
            "stake": position["stake"],
            "pnl": pnl,
            "roi": pnl / position["stake"],
            "reason": reason,
            "closed_at": datetime.now(timezone.utc),
        }
    )
    st.session_state.loss_streak = st.session_state.loss_streak + 1 if pnl < 0 else 0


def autonomous_cycle(risk: RiskConfig, scans: int) -> None:
    mark_positions()
    for position in list(st.session_state.positions):
        if position["status"] != "open":
            continue
        roi = position["unrealized_pnl"] / position["stake"]
        age_minutes = (datetime.now(timezone.utc) - position["opened_at"]).total_seconds() / 60
        if roi <= -0.18:
            close_position(position, "stop loss")
        elif roi >= 0.28:
            close_position(position, "take profit")
        elif age_minutes >= 45 or position.get("holding_cycles", 0) >= 5:
        elif age_minutes >= 45:
            close_position(position, "max holding time")

    for _ in range(scans):
        signal = generate_signal()
        approved, reason, stake = evaluate_signal(signal, risk)
        signal["status"] = "approved" if approved else "rejected"
        signal["reason"] = reason
        signal["suggested_stake"] = stake
        st.session_state.signals.append(signal)
        save_signal(signal)
        if approved and st.session_state.agent_running:
            open_position(signal, stake)

    st.session_state.equity.append({"time": datetime.now(timezone.utc), "equity": total_equity()})


st.title("🤖 Autonomous Money Agent Dashboard")
st.caption(
    "Phase 3 control center with autonomous scanning, position sizing, risk limits, "
    "paper/live-mode separation, and P&L tracking. This app defaults to simulated execution."
)

ready_count, unmet_count = readiness_summary()
if unmet_count:
    st.error(
        "Not ready for live trading. "
        f"{unmet_count} of {len(READINESS_REQUIREMENTS)} launch requirements are unmet."
    )
else:
    st.success("All launch requirements are marked met. Keep small capital limits enabled.")

with st.sidebar:
    st.header("Autonomous Controls")
    mode = st.selectbox("Execution mode", ["Paper trading", "Live trading - guarded"], index=0)
    if mode.startswith("Live"):
        st.warning(
            "Live mode is blocked until the readiness checklist is complete and "
            "venue-specific order adapters are implemented."
        )
    st.session_state.agent_running = st.toggle("Agent running", value=st.session_state.agent_running)
    scans = st.slider("Signals per cycle", min_value=1, max_value=25, value=8)
    risk = RiskConfig(
        starting_bankroll=10_000,
        max_trade_size=st.number_input("Max trade size ($)", 1.0, 5_000.0, 75.0, 5.0),
        max_daily_loss=st.number_input("Daily loss limit ($)", 1.0, 5_000.0, 250.0, 10.0),
        max_open_positions=st.number_input("Max open positions", 1, 100, 12, 1),
        max_total_exposure_pct=st.slider("Max portfolio exposure (%)", 1, 100, 25),
        min_edge_pct=st.slider("Minimum model edge (%)", 0.1, 25.0, 4.0, 0.1),
        max_spread_pct=st.slider("Maximum spread (%)", 0.1, 25.0, 5.0, 0.1),
        min_liquidity=st.number_input("Minimum liquidity ($)", 0.0, 1_000_000.0, 2_500.0, 250.0),
        stop_after_losses=st.number_input("Stop after consecutive losses", 1, 20, 4, 1),
    )
    if st.button("Run autonomous cycle", type="primary"):
        autonomous_cycle(risk, scans)
    if st.button("Reset session simulation"):
        for key in ["trades", "signals", "positions"]:
            st.session_state[key] = []
        st.session_state.cash = risk.starting_bankroll
        st.session_state.loss_streak = 0
        st.session_state.equity = [{"time": datetime.now(timezone.utc), "equity": risk.starting_bankroll}]
    if st.button("Clear paper-learning database"):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM paper_signals")
            conn.execute("DELETE FROM paper_trades")

open_positions = [p for p in st.session_state.positions if p["status"] == "open"]
realized_pnl = sum(t["pnl"] for t in st.session_state.trades)
unrealized_pnl = current_unrealized_pnl()
win_rate = (
    sum(1 for t in st.session_state.trades if t["pnl"] > 0) / len(st.session_state.trades)
    if st.session_state.trades
    else 0
)

metric_cols = st.columns(6)
metric_cols[0].metric(
    "Total equity",
    f"${total_equity():,.2f}",
    f"${total_equity() - risk.starting_bankroll:,.2f}",
)
)

ready_count, unmet_count = readiness_summary()
if unmet_count:
    st.error(
        "Not ready for live trading. "
        f"{unmet_count} of {len(READINESS_REQUIREMENTS)} launch requirements are unmet."
    )
else:
    st.success("All launch requirements are marked met. Keep small capital limits enabled.")

with st.sidebar:
    st.header("Autonomous Controls")
    mode = st.selectbox("Execution mode", ["Paper trading", "Live trading - guarded"], index=0)
    if mode.startswith("Live"):
        st.warning(
            "Live mode is blocked until the readiness checklist is complete and "
            "venue-specific order adapters are implemented."
        )
    st.session_state.agent_running = st.toggle("Agent running", value=st.session_state.agent_running)
    scans = st.slider("Signals per cycle", min_value=1, max_value=25, value=8)
    risk = RiskConfig(
        starting_bankroll=10_000,
        max_trade_size=st.number_input("Max trade size ($)", 1.0, 5_000.0, 75.0, 5.0),
        max_daily_loss=st.number_input("Daily loss limit ($)", 1.0, 5_000.0, 250.0, 10.0),
        max_open_positions=st.number_input("Max open positions", 1, 100, 12, 1),
        max_total_exposure_pct=st.slider("Max portfolio exposure (%)", 1, 100, 25),
        min_edge_pct=st.slider("Minimum model edge (%)", 0.1, 25.0, 4.0, 0.1),
        max_spread_pct=st.slider("Maximum spread (%)", 0.1, 25.0, 5.0, 0.1),
        min_liquidity=st.number_input("Minimum liquidity ($)", 0.0, 1_000_000.0, 2_500.0, 250.0),
        stop_after_losses=st.number_input("Stop after consecutive losses", 1, 20, 4, 1),
    )
    if st.button("Run autonomous cycle", type="primary"):
        autonomous_cycle(risk, scans)
    if st.button("Reset simulation"):
        for key in ["trades", "signals", "positions"]:
            st.session_state[key] = []
        st.session_state.cash = risk.starting_bankroll
        st.session_state.loss_streak = 0
        st.session_state.equity = [{"time": datetime.now(timezone.utc), "equity": risk.starting_bankroll}]

open_positions = [p for p in st.session_state.positions if p["status"] == "open"]
realized_pnl = sum(t["pnl"] for t in st.session_state.trades)
unrealized_pnl = current_unrealized_pnl()
win_rate = (
    sum(1 for t in st.session_state.trades if t["pnl"] > 0) / len(st.session_state.trades)
    if st.session_state.trades
    else 0
)

metric_cols = st.columns(6)
metric_cols[0].metric(
    "Total equity",
    f"${total_equity():,.2f}",
    f"${total_equity() - risk.starting_bankroll:,.2f}",
)
metric_cols[1].metric("Realized P&L", f"${realized_pnl:,.2f}")
metric_cols[2].metric("Unrealized P&L", f"${unrealized_pnl:,.2f}")
metric_cols[3].metric("Open positions", len(open_positions))
metric_cols[4].metric("Win rate", f"{win_rate:.1%}")
metric_cols[5].metric("Launch checks", f"{ready_count}/{len(READINESS_REQUIREMENTS)}")

if not st.session_state.agent_running:
    st.info("Agent is paused. Toggle 'Agent running' before autonomous cycles can open new positions.")
if mode.startswith("Paper"):
    st.success("Paper mode is active. No real orders are being sent.")

chart_tab, positions_tab, signals_tab, trades_tab, learning_tab, risk_tab, readiness_tab = st.tabs(
    [
        "Equity",
        "Open Positions",
        "Signals",
        "Closed Trades",
        "Learning",
        "Risk Policy",
        "Readiness",
    ]
chart_tab, positions_tab, signals_tab, trades_tab, risk_tab, readiness_tab = st.tabs(
    ["Equity", "Open Positions", "Signals", "Closed Trades", "Risk Policy", "Readiness"]
)

with chart_tab:
    equity_df = pd.DataFrame(st.session_state.equity)
    if len(equity_df) > 1:
        chart = alt.Chart(equity_df).mark_line(point=True).encode(
            x=alt.X("time:T", title="Time"),
            y=alt.Y("equity:Q", title="Equity", scale=alt.Scale(zero=False)),
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.write("Run an autonomous cycle to build the equity curve.")

with positions_tab:
    if open_positions:
        st.dataframe(pd.DataFrame(open_positions), width="stretch", hide_index=True)
    else:
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("Run an autonomous cycle to build the equity curve.")

with positions_tab:
    if open_positions:
        st.dataframe(pd.DataFrame(open_positions), use_container_width=True, hide_index=True)
    else:
        st.write("No open positions.")

with signals_tab:
    if st.session_state.signals:
        signals_df = pd.DataFrame(st.session_state.signals).sort_values("time", ascending=False)
        st.dataframe(signals_df, width="stretch", hide_index=True)
        st.dataframe(signals_df, use_container_width=True, hide_index=True)
    else:
        st.write("No signals yet.")

with trades_tab:
    historical_trades = load_trade_history()
    if not historical_trades.empty:
        st.dataframe(historical_trades, width="stretch", hide_index=True)
    else:
        st.write("No closed paper trades yet.")

with learning_tab:
    historical_trades = load_trade_history()
    historical_signals = load_signal_history()
    st.subheader("Paper Learning Memory")
    st.caption(
        "Closed paper trades are persisted to SQLite and summarized by venue/category "
        "so future simulated signals can apply a small learned probability bias."
    )
    if historical_trades.empty:
        st.info("Run paper-trading cycles until positions close to build learning data.")
    else:
        learning_df = (
            historical_trades.groupby(["platform", "category"])
            .agg(
                trades=("id", "count"),
                win_rate=("pnl", lambda values: (values > 0).mean()),
                avg_roi=("roi", "mean"),
                total_pnl=("pnl", "sum"),
            )
            .reset_index()
            .sort_values("total_pnl", ascending=False)
        )
        st.dataframe(learning_df, width="stretch", hide_index=True)
    st.subheader("Recent Persisted Signals")
    if historical_signals.empty:
        st.write("No persisted signals yet.")
    else:
        st.dataframe(historical_signals, width="stretch", hide_index=True)
    if st.session_state.trades:
        trades_df = pd.DataFrame(st.session_state.trades).sort_values("closed_at", ascending=False)
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
    else:
        st.write("No closed trades yet.")

with risk_tab:
    st.subheader("Autonomous Execution Guardrails")
    st.json(risk.__dict__)
    st.markdown(
        """
        **Live execution checklist before connecting real money:**
        - Use trading-only API keys with withdrawals disabled.
        - Keep a small dedicated trading balance.
        - Add per-platform order adapters and dry-run tests.
        - Verify market resolution rules before enabling prediction-market trades.
        - Keep the Telegram kill switch enabled.
        """
    )

with readiness_tab:
    st.subheader("Live Trading Readiness")
    st.warning("Do not fund or enable autonomous live trading until every item is met.")
    readiness_df = pd.DataFrame(READINESS_REQUIREMENTS)
    st.dataframe(readiness_df, width="stretch", hide_index=True)
    st.dataframe(readiness_df, use_container_width=True, hide_index=True)
    st.markdown(
        """
        **Answer:** we are not ready to start live trading from this repo yet. The current
        application is a dashboard and simulation harness; it has no real data ingestion, no
        exchange order adapters, no Telegram control path, and no credential safety checks.
        """
    )
