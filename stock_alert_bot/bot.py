"""Bot orchestration: Telegram command handling + the periodic market scanner.

Single-threaded event loop: long-poll Telegram for commands (10s at a time),
and run a watchlist scan whenever the scan interval has elapsed. A scan of a
typical watchlist takes a few seconds, so command latency stays low.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from . import __version__
from .config import Config
from .formatting import format_alert, format_analysis
from .market_data import fetch_ohlcv, is_always_open, is_valid_symbol
from .signals import Analysis, analyze
from .state import State
from .telegram_api import TelegramAPI, TelegramError

log = logging.getLogger(__name__)

HELP_TEXT = """<b>Commands</b>
/watch SYMBOL [SYMBOL…] — add symbols to your watchlist
/unwatch SYMBOL [SYMBOL…] — remove symbols
/watchlist — show your watchlist
/check SYMBOL — full technical analysis right now
/scan — scan your whole watchlist right now
/mute · /unmute — pause/resume alerts
/settings — show scanner configuration
/stop — unsubscribe and delete your data
/help — this message

Symbols use Yahoo Finance notation: AAPL, SPY, BTC-USD, EURUSD=X, ^GSPC."""

WELCOME_TEXT = (
    "\U0001f44b You're subscribed to technical-analysis alerts.\n"
    "I scan your watchlist on 15-minute candles (with daily-trend context) and "
    "message you when several indicators line up on a buy or sell setup.\n\n"
    "Your starting watchlist: {watchlist}\n\n" + HELP_TEXT + "\n\n"
    "<i>Automated technical analysis — informational only, not financial advice.</i>"
)


class Bot:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tg = TelegramAPI(cfg.telegram_token)
        self.state = State(cfg.state_file)
        self._next_scan_at = 0.0  # first scan runs immediately

    # --- market hours -------------------------------------------------------
    @staticmethod
    def _us_market_open(now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(ZoneInfo("America/New_York"))
        if now.weekday() >= 5:
            return False
        minutes = now.hour * 60 + now.minute
        return 9 * 60 + 30 <= minutes <= 16 * 60

    def _should_scan(self, symbol: str) -> bool:
        if not self.cfg.market_hours_only or is_always_open(symbol):
            return True
        return self._us_market_open()

    # --- analysis ------------------------------------------------------------
    def analyze_symbol(self, symbol: str) -> Optional[Analysis]:
        trigger = fetch_ohlcv(symbol, self.cfg.trigger_interval, self.cfg.trigger_period)
        if trigger is None:
            return None
        context = fetch_ohlcv(symbol, self.cfg.context_interval, self.cfg.context_period)
        return analyze(
            symbol,
            trigger,
            context,
            min_score=self.cfg.min_score,
            strong_score=self.cfg.strong_score,
            timeframe=self.cfg.trigger_interval,
        )

    # --- scanning ------------------------------------------------------------
    def scan(self) -> None:
        chats = self.state.chats()
        symbols: set[str] = set()
        for chat in chats.values():
            if not chat.get("muted"):
                symbols.update(chat.get("watchlist", []))
        if not symbols:
            return

        log.info("scanning %d symbol(s)", len(symbols))
        for symbol in sorted(symbols):
            if not self._should_scan(symbol):
                continue
            try:
                analysis = self.analyze_symbol(symbol)
            except Exception:
                log.exception("analysis failed for %s", symbol)
                continue
            if analysis is None or analysis.signal is None:
                continue
            self._dispatch_alert(analysis, chats)

        self.state.prune_cooldowns(self.cfg.alert_cooldown_minutes * 4)
        self.state.save()

    def _dispatch_alert(self, analysis: Analysis, chats: dict[int, dict]) -> None:
        sig = analysis.signal
        assert sig is not None
        message = format_alert(analysis)
        for chat_id, chat in chats.items():
            if chat.get("muted") or analysis.symbol not in chat.get("watchlist", []):
                continue
            if self.state.cooldown_active(
                chat_id, analysis.symbol, sig.direction, self.cfg.alert_cooldown_minutes
            ):
                continue
            try:
                self.tg.send_message(chat_id, message)
                self.state.set_cooldown(chat_id, analysis.symbol, sig.direction)
                log.info(
                    "alert sent: %s %s (%s) -> chat %s",
                    sig.direction,
                    analysis.symbol,
                    sig.strength,
                    chat_id,
                )
            except TelegramError:
                log.exception("failed to send alert to chat %s", chat_id)

    # --- commands ------------------------------------------------------------
    def _reply(self, chat_id: int, text: str) -> None:
        try:
            self.tg.send_message(chat_id, text)
        except TelegramError:
            log.exception("failed to reply to chat %s", chat_id)

    def handle_update(self, update: dict) -> None:
        message = update.get("message")
        if not message:
            return
        chat_id = message.get("chat", {}).get("id")
        text = (message.get("text") or "").strip()
        if chat_id is None or not text.startswith("/"):
            return

        if self.cfg.allowed_chat_ids is not None and chat_id not in self.cfg.allowed_chat_ids:
            self._reply(chat_id, "Sorry, this bot is private.")
            return

        parts = text.split()
        command = parts[0].split("@", 1)[0].lower()
        args = [a.upper() for a in parts[1:]]

        if command == "/start":
            self.state.ensure_chat(chat_id, self.cfg.default_watchlist)
            self.state.save()
            watchlist = ", ".join(self.state.get_chat(chat_id)["watchlist"]) or "(empty)"
            self._reply(chat_id, WELCOME_TEXT.format(watchlist=watchlist))
            return
        if command == "/help":
            self._reply(chat_id, HELP_TEXT)
            return
        if command == "/stop":
            self.state.remove_chat(chat_id)
            self.state.save()
            self._reply(chat_id, "Unsubscribed. Send /start to come back any time.")
            return

        # Everything below assumes a subscribed chat; ensure_chat may create it.
        chat = self.state.ensure_chat(chat_id, self.cfg.default_watchlist)
        self.state.save()

        if command == "/watch":
            self._cmd_watch(chat_id, args)
        elif command == "/unwatch":
            self._cmd_unwatch(chat_id, args)
        elif command == "/watchlist":
            watchlist = ", ".join(chat["watchlist"]) or "(empty — add symbols with /watch)"
            self._reply(chat_id, f"<b>Watchlist</b>: {watchlist}")
        elif command == "/check":
            self._cmd_check(chat_id, args)
        elif command == "/scan":
            self._cmd_scan(chat_id, chat)
        elif command == "/mute":
            self.state.set_muted(chat_id, True)
            self.state.save()
            self._reply(chat_id, "\U0001f507 Alerts muted. /unmute to resume.")
        elif command == "/unmute":
            self.state.set_muted(chat_id, False)
            self.state.save()
            self._reply(chat_id, "\U0001f514 Alerts resumed.")
        elif command == "/settings":
            self._cmd_settings(chat_id, chat)
        else:
            self._reply(chat_id, "Unknown command. /help lists what I understand.")

    def _cmd_watch(self, chat_id: int, args: list[str]) -> None:
        if not args:
            self._reply(chat_id, "Usage: /watch SYMBOL [SYMBOL…] — e.g. /watch AAPL BTC-USD")
            return
        valid = [s for s in args if is_valid_symbol(s)]
        invalid = [s for s in args if not is_valid_symbol(s)]
        added = self.state.add_symbols(chat_id, valid, self.cfg.max_watchlist_size)
        self.state.save()
        bits = []
        if added:
            bits.append("Added: " + ", ".join(added))
        skipped = [s for s in valid if s not in added]
        if skipped:
            bits.append("Already watching (or list full): " + ", ".join(skipped))
        if invalid:
            bits.append("Not valid symbols: " + ", ".join(invalid))
        self._reply(chat_id, "\n".join(bits) or "Nothing to add.")

    def _cmd_unwatch(self, chat_id: int, args: list[str]) -> None:
        if not args:
            self._reply(chat_id, "Usage: /unwatch SYMBOL [SYMBOL…]")
            return
        removed = self.state.remove_symbols(chat_id, args)
        self.state.save()
        if removed:
            self._reply(chat_id, "Removed: " + ", ".join(removed))
        else:
            self._reply(chat_id, "None of those were on your watchlist.")

    def _cmd_check(self, chat_id: int, args: list[str]) -> None:
        if len(args) != 1 or not is_valid_symbol(args[0]):
            self._reply(chat_id, "Usage: /check SYMBOL — e.g. /check NVDA")
            return
        symbol = args[0]
        self._reply(chat_id, f"Analyzing {symbol}…")
        try:
            analysis = self.analyze_symbol(symbol)
        except Exception:
            log.exception("check failed for %s", symbol)
            analysis = None
        if analysis is None:
            self._reply(
                chat_id,
                f"Couldn't get enough data for {symbol}. Check the symbol "
                "(Yahoo Finance notation) and try again.",
            )
            return
        self._reply(chat_id, format_analysis(analysis))

    def _cmd_scan(self, chat_id: int, chat: dict) -> None:
        watchlist = chat.get("watchlist", [])
        if not watchlist:
            self._reply(chat_id, "Your watchlist is empty — add symbols with /watch first.")
            return
        self._reply(chat_id, f"Scanning {len(watchlist)} symbol(s)…")
        hits = 0
        for symbol in watchlist:
            try:
                analysis = self.analyze_symbol(symbol)
            except Exception:
                log.exception("on-demand scan failed for %s", symbol)
                continue
            if analysis and analysis.signal:
                self._reply(chat_id, format_alert(analysis))
                hits += 1
        if hits == 0:
            self._reply(
                chat_id,
                "Scan complete — no actionable signals right now. "
                "I'll keep watching and alert you when indicators line up.",
            )

    def _cmd_settings(self, chat_id: int, chat: dict) -> None:
        cfg = self.cfg
        self._reply(
            chat_id,
            "<b>Settings</b>\n"
            f"• Scan every {cfg.scan_interval_minutes} min"
            f" on {cfg.trigger_interval} candles (daily trend context)\n"
            f"• Alert threshold: score ≥ {cfg.min_score}"
            f" (STRONG at ≥ {cfg.strong_score})\n"
            f"• Cooldown: {cfg.alert_cooldown_minutes} min per symbol+direction\n"
            f"• Market-hours only for stocks: {'yes' if cfg.market_hours_only else 'no'}"
            " (crypto/forex always)\n"
            f"• Alerts muted: {'yes' if chat.get('muted') else 'no'}\n"
            f"• Data source: Yahoo Finance (can be ~15 min delayed)\n"
            f"• Bot version: {__version__}",
        )

    # --- main loop -------------------------------------------------------------
    def run(self) -> None:
        me = self.tg.get_me()
        log.info("running as @%s (scan every %d min)", me.get("username"), self.cfg.scan_interval_minutes)
        while True:
            try:
                updates = self.tg.get_updates(self.state.last_update_id + 1, timeout=10)
            except TelegramError:
                log.exception("getUpdates failed; retrying in 10s")
                time.sleep(10)
                continue

            for update in updates:
                self.state.last_update_id = max(
                    self.state.last_update_id, update.get("update_id", 0)
                )
                try:
                    self.handle_update(update)
                except Exception:
                    log.exception("failed to handle update %s", update.get("update_id"))
            if updates:
                self.state.save()

            if time.monotonic() >= self._next_scan_at:
                try:
                    self.scan()
                finally:
                    self._next_scan_at = (
                        time.monotonic() + self.cfg.scan_interval_minutes * 60
                    )
