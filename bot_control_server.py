from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

DB_PATH = os.getenv("PAPER_TRADING_DB", "paper_trading.db")
TELEGRAM_API_BASE = "https://api.telegram.org"
DASHBOARD_PATH = Path(__file__).with_name("web_dashboard.html")

app = FastAPI(title="Autonomous Money Agent Control API")


class TelegramMessage(BaseModel):
    chat_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1, max_length=4096)


class BotCommand(BaseModel):
    command: str = Field(..., min_length=1, max_length=64)
    chat_id: str | None = None


def telegram_token() -> str:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise HTTPException(
            status_code=503,
            detail="TELEGRAM_BOT_TOKEN is not configured on the server.",
        )
    return token


def telegram_request(method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    token = telegram_token()
    response = requests.post(
        f"{TELEGRAM_API_BASE}/bot{token}/{method}",
        json=payload or {},
        timeout=15,
    )
    data = response.json() if response.content else {}
    if not response.ok or not data.get("ok", False):
        raise HTTPException(
            status_code=response.status_code,
            detail=data.get("description", "Telegram API request failed."),
        )
    return data["result"]


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    result = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return result is not None


def read_table(table_name: str) -> pd.DataFrame:
    if not Path(DB_PATH).exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        if not table_exists(conn, table_name):
            return pd.DataFrame()
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def paper_summary() -> dict[str, Any]:
    trades = read_table("paper_trades")
    signals = read_table("paper_signals")
    if trades.empty:
        return {
            "signals": int(len(signals)),
            "closed_trades": 0,
            "total_pnl": 0.0,
            "avg_roi": 0.0,
            "win_rate": 0.0,
            "by_platform": [],
            "recent_trades": [],
        }

    by_platform = (
        trades.groupby("platform")
        .agg(
            trades=("id", "count"),
            total_pnl=("pnl", "sum"),
            avg_roi=("roi", "mean"),
        )
        .reset_index()
        .sort_values("total_pnl", ascending=False)
        .to_dict(orient="records")
    )
    return {
        "signals": int(len(signals)),
        "closed_trades": int(len(trades)),
        "total_pnl": float(trades["pnl"].sum()),
        "avg_roi": float(trades["roi"].mean()),
        "win_rate": float((trades["pnl"] > 0).mean()),
        "by_platform": by_platform,
        "recent_trades": trades.sort_values("closed_at", ascending=False)
        .head(25)
        .to_dict(orient="records"),
    }


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_PATH.read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "telegram_configured": bool(os.getenv("TELEGRAM_BOT_TOKEN", "").strip()),
        "paper_db_exists": Path(DB_PATH).exists(),
    }


@app.get("/api/paper/summary")
def get_paper_summary() -> dict[str, Any]:
    return paper_summary()


@app.get("/api/telegram/me")
def get_telegram_me() -> dict[str, Any]:
    return telegram_request("getMe")


@app.post("/api/telegram/send")
def send_telegram_message(message: TelegramMessage) -> dict[str, Any]:
    return telegram_request(
        "sendMessage",
        {"chat_id": message.chat_id, "text": message.text},
    )


@app.post("/api/bot/command")
def send_bot_command(command: BotCommand) -> dict[str, Any]:
    text = f"/{command.command.lstrip('/')}"
    if command.chat_id:
        return telegram_request("sendMessage", {"chat_id": command.chat_id, "text": text})
    return {"queued": False, "command": text, "detail": "Provide chat_id to send via Telegram."}
