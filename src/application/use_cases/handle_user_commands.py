"""MAIL-48 / MAIL-45 — bot właściciela: komendy z czatu → odpowiedzi z bazy, bez LLM.

Use case zna wyłącznie porty: źródło komend i kanał odpowiedzi (Telegram), offset
pollingu, wyciszenia, repozytorium (odczyty) i pozycje. Każda komenda to darmowy
odczyt albo zapis jednego wiersza — żadnego płatnego portu.

Przebieg: offset z `bot_state` → nowe wiadomości → odpowiedź na każdą → zapis
następnego offsetu. Błąd odczytu przy jednej komendzie to odpowiedź „nie udało
się”, nie przerwany przebieg; błąd wysyłki odpowiedzi jest liczony, a offset i tak
idzie dalej — komenda bez odpowiedzi nie wraca w pętli co 15 minut.

Watchlista (MAIL-45, opcja b): symbole są wyłącznie w config.toml, baza trzyma
tylko wyciszenia. `/mute` przyjmuje więc wyłącznie symbol z config.toml.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from src.application.decision_receipt import read_decision_receipt, receipt_line_pl
from src.application.errors import PortError
from src.application.ports import (
    BotStatePort,
    CommandReplyPort,
    CycleRun,
    PortfolioPositionsPort,
    RepositoryPort,
    SymbolMutePort,
    UserCommandSourcePort,
)
from src.application.report_formatting import format_local, format_local_date
from src.domain.number_format import format_number_pl
from src.domain.polish_labels import trend_forecast_label
from src.domain.user_command import (
    MAX_MUTE_DAYS,
    Command,
    Help,
    Last,
    Mute,
    Note,
    Pause,
    Pos,
    Unknown,
    Unmute,
    Watch,
    Why,
    parse_command,
)
from src.domain.watchlist import SymbolMute, effective, mute_until, watchlist_summary

logger = logging.getLogger(__name__)

# /last bez symbolu: ostatni przebieg Fast Loopa z tylu dni (`cycle_runs`, 041).
LAST_RUN_WINDOW_DAYS = 7
# Uzasadnienie LLM bywa długie — w czacie wystarczy początek, całość jest w mailu.
REASONING_PREVIEW_CHARS = 700

HELP_TEXT = "\n".join(
    [
        "🤖 Komendy StockAgenta:",
        "/why NVDA — dlaczego ostatnia predykcja (uzasadnienie i kwit decyzyjny)",
        "/last — ostatni cykl Fast Loopa; /last NVDA — ostatnia predykcja symbolu",
        "/pos — pozycje portfela (bez kwot)",
        "/watch — efektywna watchlista i wyciszenia",
        f"/mute NVDA [dni] — wycisz symbol z config.toml (domyślnie 7, max {MAX_MUTE_DAYS})",
        "/unmute NVDA — przywróć symbol do analizy",
        "/note, /pause — wkrótce",
        "Dodanie symbolu do watchlisty: tylko przez config.toml.",
    ]
)
_MUTES_DISABLED = (
    "🔇 Wyciszenia są wyłączone (symbol_mutes_enabled = false w config.toml) — "
    "watchlista to dziś cały config.toml."
)
_NOT_AVAILABLE = "⏳ Komenda {name} jest jeszcze niedostępna — wejdzie z osobnym wpisem."


@dataclass(frozen=True)
class BotOptions:
    """`config_symbols` — watchlista z config.toml (akcje i krypto) w kolejności
    configu; `mutes_enabled` — flaga `symbol_mutes_enabled`; `display_timezone` —
    strefa dat w odpowiedziach."""

    config_symbols: tuple[str, ...]
    mutes_enabled: bool
    display_timezone: str


@dataclass(frozen=True)
class BotRunOutcome:
    handled: int
    failed_replies: int


class HandleUserCommandsUseCase:
    """`positions` None = `portfolio_positions_enabled` wyłączone (`/pos` to mówi)."""

    def __init__(
        self,
        *,
        source: UserCommandSourcePort,
        reply: CommandReplyPort,
        state: BotStatePort,
        mutes: SymbolMutePort,
        repository: RepositoryPort,
        positions: PortfolioPositionsPort | None,
        options: BotOptions,
        clock: Callable[[], datetime],
    ) -> None:
        self._source = source
        self._reply = reply
        self._state = state
        self._mutes = mutes
        self._repository = repository
        self._positions = positions
        self._options = options
        self._clock = clock

    def run(self) -> BotRunOutcome:
        """Jeden przebieg pollingu. Błąd odczytu offsetu albo `getUpdates` leci
        jako `PortError` — entry point kończy się wtedy kodem 1."""
        batch = self._source.fetch_messages(self._state.get_update_offset())
        failed = 0
        for message in batch.messages:
            text = self._safe_answer(parse_command(message.text))
            try:
                self._reply.reply(message.chat_id, text)
            except PortError as exc:
                failed += 1
                logger.warning("Bot reply not delivered (%s)", type(exc).__name__)
        if batch.next_offset is not None:
            self._state.save_update_offset(batch.next_offset)
        logger.info("Bot: %d command(s) handled, %d reply(ies) failed", len(batch.messages), failed)
        return BotRunOutcome(handled=len(batch.messages), failed_replies=failed)

    def answer(self, command: Command) -> str:
        """Odpowiedź na jedną komendę (czysty tekst, po polsku)."""
        match command:
            case Help():
                return HELP_TEXT
            case Watch():
                return self._watch()
            case Mute():
                return self._mute(command)
            case Unmute():
                return self._unmute(command)
            case Pos():
                return self._pos()
            case Last():
                if command.symbol is None:
                    return self._last_run()
                return self._last_prediction(command.symbol)
            case Why():
                return self._why(command.symbol)
            case Note():
                return _NOT_AVAILABLE.format(name="/note")
            case Pause():
                return _NOT_AVAILABLE.format(name="/pause")
            case Unknown():
                return f"❔ Nie rozumiem: {command.reason}. Lista komend: /help"

    def _safe_answer(self, command: Command) -> str:
        try:
            return self.answer(command)
        except PortError as exc:
            logger.warning("Bot command failed on a port (%s)", type(exc).__name__)
            return f"⚠️ Nie udało się odczytać danych ({type(exc).__name__}) — spróbuj później."

    # --- watchlista i wyciszenia (MAIL-45) -------------------------------------

    def _watch(self) -> str:
        mutes = self._mutes.get_symbol_mutes() if self._options.mutes_enabled else []
        watchlist = effective(self._options.config_symbols, mutes, self._clock())
        lines = [f"📋 {watchlist_summary(watchlist)}"]
        if watchlist.muted:
            lines.append(
                "🔇 Wyciszone: " + ", ".join(self._until_label(mute) for mute in watchlist.muted)
            )
        lines.append(f"W analizie ({len(watchlist.symbols)}): {', '.join(watchlist.symbols)}")
        lines.append("Dodanie symbolu: tylko przez config.toml (commit), nie przez bota.")
        if not self._options.mutes_enabled:
            lines.append(_MUTES_DISABLED)
        return "\n".join(lines)

    def _mute(self, command: Mute) -> str:
        symbol = self._config_symbol(command.symbol)
        if not self._options.mutes_enabled:
            return _MUTES_DISABLED
        if symbol is None:
            return _outside_config(command.symbol)
        now = self._clock()
        until = mute_until(now, command.days)
        self._mutes.mute_symbol(symbol, until, muted_at=now)
        return (
            f"🔇 {self._until_label(SymbolMute(symbol, until))}. Fast Loop pominie płatną "
            "analizę i sekcję w mailu, ale zapisze snapshot ceny. "
            f"Przywrócenie: /unmute {symbol}"
        )

    def _unmute(self, command: Unmute) -> str:
        symbol = self._config_symbol(command.symbol)
        if not self._options.mutes_enabled:
            return _MUTES_DISABLED
        if symbol is None:
            return _outside_config(command.symbol)
        self._mutes.unmute_symbol(symbol)
        return f"🔔 {symbol} wraca do analizy od najbliższego cyklu."

    def _config_symbol(self, symbol: str) -> str | None:
        """Symbol w pisowni config.toml albo None, gdy go tam nie ma."""
        wanted = symbol.upper()
        return next((s for s in self._options.config_symbols if s.upper() == wanted), None)

    def _until_label(self, mute: SymbolMute) -> str:
        return f"{mute.symbol} do {self._local_date(mute.muted_until)}"

    # --- odczyty ---------------------------------------------------------------

    def _pos(self) -> str:
        if self._positions is None:
            return "💼 Pozycje są wyłączone (portfolio_positions_enabled = false w config.toml)."
        quantities: dict[str, Decimal] = {}
        for position in self._positions.get_positions():
            quantities[position.symbol] = quantities.get(position.symbol, Decimal(0)) + (
                position.quantity
            )
        if not quantities:
            return "💼 Brak pozycji w tabeli positions."
        as_of = self._positions.get_as_of()
        header = f"💼 Pozycje ({len(quantities)})"
        if as_of is not None:
            header += f", stan na {self._local_date(as_of)}"
        lines = [header + " — bez kwot, kwoty i P/L są tylko w mailu:"]
        lines.extend(
            f"{symbol}: {_quantity(quantity)}" for symbol, quantity in sorted(quantities.items())
        )
        return "\n".join(lines)

    def _last_run(self) -> str:
        since = self._clock() - timedelta(days=LAST_RUN_WINDOW_DAYS)
        runs = [run for run in self._repository.get_cycle_runs_since(since) if run.loop == "fast"]
        if not runs:
            return (
                f"🕐 Brak przebiegów Fast Loopa z ostatnich {LAST_RUN_WINDOW_DAYS} dni "
                "(albo cycle_runs_enabled = false)."
            )
        latest = max(runs, key=lambda run: run.started_at)
        return (
            f"🕐 Ostatni cykl Fast Loopa: {self._local(latest.started_at)} — przeanalizowane "
            f"{latest.saved} z {latest.eligible}, błędy: {latest.failures}, "
            f"mail: {_delivery_label(latest)}."
        )

    def _last_prediction(self, symbol: str) -> str:
        latest = self._repository.get_latest_prediction(symbol)
        if latest is None:
            return f"🕐 Brak predykcji {symbol}."
        price = format_number_pl(latest.price.amount, signed=False, decimals=2)
        moment = self._local(latest.timestamp)
        lines = [f"🕐 {symbol}: ostatnia predykcja {moment}, cena wejścia {price}."]
        if latest.reasoning:
            lines.append(f"Uzasadnienie: {_preview(latest.reasoning)}")
        return "\n".join(lines)

    def _why(self, symbol: str) -> str:
        """Uzasadnienie i kwit decyzyjny ostatniej predykcji — sam odczyt bazy.
        Pełny `/why` (pewność surowa i skalibrowana, głosy rady) to MAIL-34."""
        latest = self._repository.get_latest_prediction(symbol)
        if latest is None:
            return f"❓ Brak predykcji {symbol}."
        day = latest.timestamp.astimezone(UTC).date()
        receipts = self._repository.find_receipts(symbol, day)
        stored = receipts[-1] if receipts else None
        header = f"❓ {symbol} — ostatnia predykcja {self._local(latest.timestamp)}"
        if stored is not None and stored.predicted_trend:
            header += f": {trend_forecast_label(stored.predicted_trend)}"
        lines = [header]
        lines.append(f"Uzasadnienie: {_preview(latest.reasoning or 'brak w bazie')}")
        receipt = read_decision_receipt(stored.payload) if stored is not None else None
        if receipt is None:
            lines.append("Kwit decyzyjny: brak (predykcja bez receipts_enabled albo stary format).")
        else:
            lines.append(f"Kwit decyzyjny: {receipt_line_pl(receipt)}.")
        return "\n".join(lines)

    def _local(self, moment: datetime) -> str:
        return format_local(moment, self._options.display_timezone)

    def _local_date(self, moment: datetime) -> str:
        return format_local_date(moment, self._options.display_timezone)


def _outside_config(symbol: str) -> str:
    return (
        f"{symbol} nie ma na watchliście z config.toml — wyciszyć można tylko symbol "
        "z config.toml, a dodać nowy wyłącznie commitem do config.toml."
    )


def _delivery_label(run: CycleRun) -> str:
    if run.report_delivered is None:
        return "bez kanału e-mail"
    return "wysłany" if run.report_delivered else "niewysłany"


def _quantity(value: Decimal) -> str:
    """Ilość bez zbędnych zer: „4”, „12,5”; najwyżej 4 miejsca po przecinku."""
    exponent = value.normalize().as_tuple().exponent
    decimals = -exponent if isinstance(exponent, int) and exponent < 0 else 0
    return format_number_pl(value, signed=False, decimals=min(decimals, 4))


def _preview(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= REASONING_PREVIEW_CHARS:
        return text
    return text[:REASONING_PREVIEW_CHARS].rstrip() + "…"
