"""Testy domeny Subscriber — per-recipient watchlist (U2).

Subscriber to czysty value object: e-mail + zbiór symboli, które go interesują.
Pusty zbiór = "wszystko" (subskrybent dostaje cały union symboli cyklu). Logika
jest re-slicingiem raportu — analiza i bramka volatility pozostają nietknięte,
więc cała ta domena operuje WYŁĄCZNIE na stdlib (brak płatnych portów).
"""

from __future__ import annotations

from src.domain.subscriber import Subscriber, symbols_for


class TestSubscriberWatches:
    def test_watches_symbol_in_its_set(self) -> None:
        sub = Subscriber(email="a@example.com", symbols=frozenset({"AAPL", "VOO"}))
        assert sub.watches("AAPL") is True

    def test_does_not_watch_symbol_outside_its_set(self) -> None:
        sub = Subscriber(email="a@example.com", symbols=frozenset({"AAPL"}))
        assert sub.watches("NVDA") is False

    def test_empty_set_watches_everything(self) -> None:
        # Pusty zbiór = "wszystko" — subskrybent bez zadeklarowanej watchlisty
        # dostaje cały union symboli cyklu (zachowanie jak dotychczasowy
        # pojedynczy DIGEST_TO_EMAIL).
        sub = Subscriber(email="all@example.com", symbols=frozenset())
        assert sub.watches("ANYTHING") is True


class TestSymbolsFor:
    def test_returns_intersection_with_all_symbols(self) -> None:
        sub = Subscriber(
            email="a@example.com", symbols=frozenset({"AAPL", "TSLA", "ZZZ"})
        )
        all_symbols = frozenset({"AAPL", "TSLA", "VOO"})
        # ZZZ nie ma w cyklu — wypada; VOO jest w cyklu, ale subskrybent go nie chce.
        assert symbols_for(sub, all_symbols) == frozenset({"AAPL", "TSLA"})

    def test_empty_subscriber_set_returns_all_symbols(self) -> None:
        # Pusty zbiór subskrybenta = "wszystko" → dostaje pełny union cyklu.
        sub = Subscriber(email="all@example.com", symbols=frozenset())
        all_symbols = frozenset({"AAPL", "VOO"})
        assert symbols_for(sub, all_symbols) == all_symbols

    def test_disjoint_sets_return_empty(self) -> None:
        # Subskrybent chce tylko symboli spoza bieżącego cyklu → pusty wynik
        # (use case może go wtedy pominąć / nie wysyłać pustego raportu).
        sub = Subscriber(email="a@example.com", symbols=frozenset({"GME"}))
        all_symbols = frozenset({"AAPL", "VOO"})
        assert symbols_for(sub, all_symbols) == frozenset()

    def test_result_is_subset_of_all_symbols(self) -> None:
        # symbols_for nigdy nie "wymyśla" symboli spoza cyklu — wynik to zawsze
        # podzbiór all_symbols (re-slicing, nie rozszerzanie).
        sub = Subscriber(
            email="a@example.com", symbols=frozenset({"AAPL", "OUTSIDE"})
        )
        all_symbols = frozenset({"AAPL", "VOO", "NVDA"})
        assert symbols_for(sub, all_symbols) <= all_symbols


class TestSubscriberIsFrozen:
    def test_subscriber_is_hashable_and_immutable(self) -> None:
        # frozen VO — bezpieczny jako element zbioru / klucz słownika.
        sub = Subscriber(email="a@example.com", symbols=frozenset({"AAPL"}))
        assert sub in {sub}
