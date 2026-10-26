"""MAIL-48 — komendy właściciela z czatu bota jako typowane `Command`."""

from __future__ import annotations

import pytest

from src.domain.user_command import (
    DEFAULT_MUTE_DAYS,
    MAX_MUTE_DAYS,
    MAX_NOTE_LENGTH,
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


def test_parses_why_with_ticker_and_rejects_unknown() -> None:
    assert parse_command("/why nvda") == Why(symbol="NVDA")
    assert isinstance(parse_command("/rm -rf"), Unknown)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/mute TSLA", Mute(symbol="TSLA", days=DEFAULT_MUTE_DAYS)),
        ("/mute tsla 14", Mute(symbol="TSLA", days=14)),
        ("/unmute xdwd.de", Unmute(symbol="XDWD.DE")),
        ("/watch", Watch()),
        ("/pos", Pos()),
        ("/last", Last(symbol=None)),
        ("/last btc", Last(symbol="BTC")),
        ("/note sprzedałem połowę NVDA", Note(text="sprzedałem połowę NVDA")),
        ("/pause 2026-11-02", Pause(argument="2026-11-02")),
        ("/pause", Pause(argument="")),
        ("/help", Help()),
        ("/start", Help()),
    ],
)
def test_parses_every_command(text: str, expected: Command) -> None:
    assert parse_command(text) == expected


def test_command_addressed_to_the_bot_and_in_upper_case_is_the_same_command() -> None:
    # W grupie Telegram dokleja nazwę bota: „/why@StockAgentBot NVDA”.
    assert parse_command("  /WHY@StockAgentBot   nvda ") == Why(symbol="NVDA")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "dlaczego NVDA?",
        "/why",
        "/why NVDA TSLA",
        "/why ../../etc/passwd",
        "/why <b>NVDA</b>",
        "/mute",
        "/mute NVDA zero",
        "/mute NVDA 0",
        "/mute NVDA 7 dni",
        f"/mute NVDA {MAX_MUTE_DAYS + 1}",
        "/unmute",
        "/watch NVDA",
        "/last NVDA TSLA",
        "/note",
        "/note " + "x" * (MAX_NOTE_LENGTH + 1),
        "/sell NVDA",
    ],
)
def test_malformed_or_unknown_input_is_unknown_never_an_exception(text: str) -> None:
    assert isinstance(parse_command(text), Unknown)


def test_unknown_keeps_a_reason_for_the_reply() -> None:
    unknown = parse_command("/mute NVDA 0")

    assert isinstance(unknown, Unknown)
    assert str(MAX_MUTE_DAYS) in unknown.reason


def test_longest_allowed_mute_is_accepted() -> None:
    assert parse_command(f"/mute NVDA {MAX_MUTE_DAYS}") == Mute(symbol="NVDA", days=MAX_MUTE_DAYS)
