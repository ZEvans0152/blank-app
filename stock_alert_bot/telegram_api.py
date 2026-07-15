"""Minimal Telegram Bot API client using long polling (no webhook needed)."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests

log = logging.getLogger(__name__)


class TelegramError(RuntimeError):
    pass


class TelegramAPI:
    BASE_URL = "https://api.telegram.org"

    def __init__(self, token: str, session: Optional[requests.Session] = None):
        self._url = f"{self.BASE_URL}/bot{token}"
        self._session = session or requests.Session()

    def _call(self, method: str, *, read_timeout: float = 30.0, **params) -> Any:
        url = f"{self._url}/{method}"
        last_error: Exception = TelegramError(f"{method} failed")
        for attempt in range(3):
            try:
                resp = self._session.post(url, json=params, timeout=(10, read_timeout))
                payload = resp.json()
                if payload.get("ok"):
                    return payload.get("result")
                # Respect Telegram's flood-control hint.
                retry_after = (payload.get("parameters") or {}).get("retry_after")
                description = payload.get("description", "unknown Telegram error")
                if retry_after:
                    log.warning("telegram flood control: waiting %ss", retry_after)
                    time.sleep(min(float(retry_after), 60.0))
                    continue
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    raise TelegramError(f"{method}: {description}")
                last_error = TelegramError(f"{method}: {description}")
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
            time.sleep(2**attempt)
        raise TelegramError(str(last_error))

    def get_me(self) -> dict:
        return self._call("getMe")

    def get_updates(self, offset: int, timeout: int = 10) -> list[dict]:
        return self._call(
            "getUpdates",
            read_timeout=timeout + 15,
            offset=offset,
            timeout=timeout,
            allowed_updates=["message"],
        )

    def send_message(self, chat_id: int, text: str, parse_mode: str = "HTML") -> None:
        self._call(
            "sendMessage",
            chat_id=chat_id,
            text=text[:4096],  # Telegram hard limit per message
            parse_mode=parse_mode,
            disable_web_page_preview=True,
        )
