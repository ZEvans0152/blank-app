"""Persistent bot state: subscribed chats, watchlists, alert cooldowns.

Stored as a single JSON file, written atomically. The bot is single-threaded,
so no locking is required.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class State:
    def __init__(self, path: str):
        self._path = Path(path)
        self._data = {"chats": {}, "cooldowns": {}, "last_update_id": 0}
        if self._path.is_file():
            try:
                loaded = json.loads(self._path.read_text())
                if isinstance(loaded, dict):
                    self._data.update(loaded)
            except (json.JSONDecodeError, OSError):
                log.exception("could not read state file %s; starting fresh", path)

    def save(self) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent or Path(".")), prefix=".state-", suffix=".json"
        )
        try:
            with os.fdopen(tmp_fd, "w") as fh:
                json.dump(self._data, fh, indent=2, sort_keys=True)
            os.replace(tmp_path, self._path)
        except OSError:
            log.exception("could not write state file %s", self._path)
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # --- update offset -----------------------------------------------------
    @property
    def last_update_id(self) -> int:
        return int(self._data.get("last_update_id", 0))

    @last_update_id.setter
    def last_update_id(self, value: int) -> None:
        self._data["last_update_id"] = int(value)

    # --- chats ---------------------------------------------------------------
    def chats(self) -> dict[int, dict]:
        return {int(cid): chat for cid, chat in self._data["chats"].items()}

    def get_chat(self, chat_id: int) -> dict | None:
        return self._data["chats"].get(str(chat_id))

    def ensure_chat(self, chat_id: int, default_watchlist: list[str]) -> dict:
        chat = self._data["chats"].setdefault(
            str(chat_id), {"watchlist": list(default_watchlist), "muted": False}
        )
        return chat

    def remove_chat(self, chat_id: int) -> bool:
        return self._data["chats"].pop(str(chat_id), None) is not None

    def set_muted(self, chat_id: int, muted: bool) -> None:
        chat = self._data["chats"].get(str(chat_id))
        if chat is not None:
            chat["muted"] = muted

    # --- watchlists -----------------------------------------------------------
    def add_symbols(self, chat_id: int, symbols: list[str], cap: int) -> list[str]:
        chat = self._data["chats"].get(str(chat_id))
        if chat is None:
            return []
        added = []
        for sym in symbols:
            if sym not in chat["watchlist"] and len(chat["watchlist"]) < cap:
                chat["watchlist"].append(sym)
                added.append(sym)
        return added

    def remove_symbols(self, chat_id: int, symbols: list[str]) -> list[str]:
        chat = self._data["chats"].get(str(chat_id))
        if chat is None:
            return []
        removed = []
        for sym in symbols:
            if sym in chat["watchlist"]:
                chat["watchlist"].remove(sym)
                removed.append(sym)
        return removed

    # --- alert cooldowns --------------------------------------------------------
    @staticmethod
    def _cooldown_key(chat_id: int, symbol: str, direction: str) -> str:
        return f"{chat_id}:{symbol}:{direction}"

    def cooldown_active(
        self, chat_id: int, symbol: str, direction: str, minutes: int
    ) -> bool:
        raw = self._data["cooldowns"].get(self._cooldown_key(chat_id, symbol, direction))
        if not raw:
            return False
        try:
            sent_at = datetime.fromisoformat(raw)
        except ValueError:
            return False
        return _utcnow() - sent_at < timedelta(minutes=minutes)

    def set_cooldown(self, chat_id: int, symbol: str, direction: str) -> None:
        self._data["cooldowns"][
            self._cooldown_key(chat_id, symbol, direction)
        ] = _utcnow().isoformat()

    def prune_cooldowns(self, minutes: int) -> None:
        horizon = _utcnow() - timedelta(minutes=minutes)
        stale = []
        for key, raw in self._data["cooldowns"].items():
            try:
                if datetime.fromisoformat(raw) < horizon:
                    stale.append(key)
            except ValueError:
                stale.append(key)
        for key in stale:
            del self._data["cooldowns"][key]
