"""Domena: komendy właściciela z czatu bota (MAIL-48) — tekst wiadomości → `Command`.

Parser jest czysty (tylko stdlib) i nie rzuca: nieznana komenda, brakujący albo
niepoprawny argument dają `Unknown` z powodem dla odpowiedzi. Ticker przechodzi
walidację kształtu (`TICKER_PATTERN`) i normalizację do wielkich liter — tekst
z czatu nie idzie dalej jako dowolny string. Czy symbol jest na watchliście
z config.toml, rozstrzyga dopiero use case.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_MUTE_DAYS = 7
MAX_MUTE_DAYS = 90
MAX_NOTE_LENGTH = 500


@dataclass(frozen=True)
class Why:
    symbol: str


@dataclass(frozen=True)
class Mute:
    symbol: str
    days: int = DEFAULT_MUTE_DAYS


@dataclass(frozen=True)
class Unmute:
    symbol: str


@dataclass(frozen=True)
class Watch:
    pass


@dataclass(frozen=True)
class Pos:
    pass


@dataclass(frozen=True)
class Last:
    symbol: str | None = None


@dataclass(frozen=True)
class Note:
    text: str


@dataclass(frozen=True)
class Pause:
    argument: str


@dataclass(frozen=True)
class Help:
    pass


@dataclass(frozen=True)
class Unknown:
    text: str
    reason: str = ""


type Command = Why | Mute | Unmute | Watch | Pos | Last | Note | Pause | Help | Unknown

# Ticker giełdowy albo krypto z watchlisty: „NVDA”, „BTC”, „XDWD.DE”, „BRK-B”.
TICKER_PATTERN = re.compile(r"^[A-Z0-9]{1,10}(?:[.-][A-Z0-9]{1,4})?$")

_NO_ARGUMENTS = "komenda nie przyjmuje argumentów"
_ONE_TICKER = "podaj jeden ticker, np. NVDA"
_MUTE_DAYS = f"liczba dni wyciszenia to 1–{MAX_MUTE_DAYS}"


def parse_command(text: str) -> Command:
    """Tekst wiadomości → komenda. Nazwa komendy bez wielkości liter i bez sufiksu
    `@NazwaBota` (Telegram dokleja go w grupach); `/start` to pomoc."""
    tokens = text.split()
    if not tokens or not tokens[0].startswith("/"):
        return Unknown(text, "wiadomość nie jest komendą — zacznij od „/”, np. /help")
    name = tokens[0][1:].split("@", 1)[0].lower()
    args = tokens[1:]
    parser = _PARSERS.get(name)
    if parser is None:
        return Unknown(text, f"nieznana komenda /{name}")
    return parser(text, args)


def _ticker(raw: str) -> str | None:
    symbol = raw.upper()
    return symbol if TICKER_PATTERN.fullmatch(symbol) else None


def _single_ticker(text: str, args: list[str]) -> str | Unknown:
    if len(args) != 1:
        return Unknown(text, _ONE_TICKER)
    symbol = _ticker(args[0])
    return Unknown(text, f"„{args[0]}” nie wygląda na ticker") if symbol is None else symbol


def _why(text: str, args: list[str]) -> Command:
    symbol = _single_ticker(text, args)
    return symbol if isinstance(symbol, Unknown) else Why(symbol)


def _unmute(text: str, args: list[str]) -> Command:
    symbol = _single_ticker(text, args)
    return symbol if isinstance(symbol, Unknown) else Unmute(symbol)


def _mute(text: str, args: list[str]) -> Command:
    symbol = _single_ticker(text, args[:1])
    if isinstance(symbol, Unknown) or len(args) > 2:
        return Unknown(text, f"{_ONE_TICKER} i opcjonalnie liczbę dni (1–{MAX_MUTE_DAYS})")
    if len(args) == 1:
        return Mute(symbol)
    days = args[1]
    if not days.isdigit() or not 1 <= int(days) <= MAX_MUTE_DAYS:
        return Unknown(text, _MUTE_DAYS)
    return Mute(symbol, int(days))


def _last(text: str, args: list[str]) -> Command:
    if not args:
        return Last()
    symbol = _single_ticker(text, args)
    return symbol if isinstance(symbol, Unknown) else Last(symbol)


def _note(text: str, args: list[str]) -> Command:
    body = " ".join(args)
    if not body:
        return Unknown(text, "podaj treść notatki, np. /note sprzedałem połowę NVDA")
    if len(body) > MAX_NOTE_LENGTH:
        return Unknown(text, f"notatka ma najwyżej {MAX_NOTE_LENGTH} znaków")
    return Note(body)


def _pause(text: str, args: list[str]) -> Command:
    return Pause(" ".join(args))


def _without_arguments(command: Command) -> Callable[[str, list[str]], Command]:
    def parse(text: str, args: list[str]) -> Command:
        return Unknown(text, _NO_ARGUMENTS) if args else command

    return parse


_PARSERS: dict[str, Callable[[str, list[str]], Command]] = {
    "why": _why,
    "mute": _mute,
    "unmute": _unmute,
    "watch": _without_arguments(Watch()),
    "pos": _without_arguments(Pos()),
    "last": _last,
    "note": _note,
    "pause": _pause,
    "help": _without_arguments(Help()),
    "start": _without_arguments(Help()),
}
