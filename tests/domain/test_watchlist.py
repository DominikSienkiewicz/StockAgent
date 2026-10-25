"""MAIL-45 — efektywna watchlista: symbole z config.toml minus aktywne wyciszenia z bota.

Decyzja 2026-10-23 (opcja b): baza trzyma TYLKO wyciszenia, symbole dodaje się
wyłącznie przez config.toml — watchlista nie ma drugiego źródła prawdy.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.domain.watchlist import SymbolMute, effective, mute_until, watchlist_summary

NOW = datetime(2026, 10, 25, 8, 0, tzinfo=UTC)
CONFIG = ("AAPL", "NVDA", "TSLA", "BTC")


def test_effective_watchlist_drops_muted_until_expiry_and_brings_expired_back() -> None:
    mutes = [
        SymbolMute("NVDA", NOW + timedelta(days=1)),
        SymbolMute("TSLA", NOW - timedelta(minutes=1)),
    ]

    watchlist = effective(CONFIG, mutes, NOW)

    assert watchlist.symbols == ("AAPL", "TSLA", "BTC")
    assert watchlist.muted == (SymbolMute("NVDA", NOW + timedelta(days=1)),)
    assert watchlist.config_count == 4


def test_mute_ends_exactly_at_muted_until() -> None:
    mutes = [SymbolMute("NVDA", NOW)]

    assert effective(CONFIG, mutes, NOW).symbols == CONFIG
    assert effective(CONFIG, mutes, NOW - timedelta(seconds=1)).symbols == (
        "AAPL",
        "TSLA",
        "BTC",
    )


def test_mute_of_a_symbol_outside_config_neither_adds_nor_counts() -> None:
    watchlist = effective(CONFIG, [SymbolMute("AMD", NOW + timedelta(days=3))], NOW)

    assert watchlist.symbols == CONFIG
    assert watchlist.muted == ()


def test_muted_symbols_keep_config_order_and_ignore_letter_case() -> None:
    mutes = [
        SymbolMute("btc", NOW + timedelta(days=2)),
        SymbolMute("AAPL", NOW + timedelta(days=5)),
    ]

    watchlist = effective(CONFIG, mutes, NOW)

    assert [mute.symbol for mute in watchlist.muted] == ["AAPL", "BTC"]
    assert watchlist.symbols == ("NVDA", "TSLA")


def test_later_of_two_mutes_of_one_symbol_wins_in_any_order() -> None:
    mutes = [
        SymbolMute("NVDA", NOW - timedelta(days=1)),
        SymbolMute("NVDA", NOW + timedelta(days=4)),
    ]
    expected = (SymbolMute("NVDA", NOW + timedelta(days=4)),)

    assert effective(CONFIG, mutes, NOW).muted == expected
    assert effective(CONFIG, list(reversed(mutes)), NOW).muted == expected


def test_naive_moment_is_rejected() -> None:
    with pytest.raises(ValueError, match="strefy"):
        effective(CONFIG, [], datetime(2026, 10, 25, 8, 0))
    with pytest.raises(ValueError, match="strefy"):
        SymbolMute("NVDA", datetime(2026, 10, 26, 8, 0))


def test_mute_until_counts_whole_days_from_now() -> None:
    assert mute_until(NOW, 7) == NOW + timedelta(days=7)
    with pytest.raises(ValueError):
        mute_until(NOW, 0)


def test_summary_names_config_count_and_muted_count() -> None:
    watchlist = effective(CONFIG, [SymbolMute("NVDA", NOW + timedelta(days=1))], NOW)

    assert watchlist_summary(watchlist) == "watchlista: 4 z config − 1 wyciszone"
