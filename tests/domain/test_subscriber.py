"""Testy domeny Subscriber — per-recipient watchlist (U2).

Subscriber to czysty value object: e-mail + zbiór symboli, które go interesują.
Pusty zbiór = "wszystko" (subskrybent dostaje cały union symboli cyklu). Logika
jest re-slicingiem raportu — analiza i bramka volatility pozostają nietknięte,
więc cała ta domena operuje WYŁĄCZNIE na stdlib (brak płatnych portów).
"""

from __future__ import annotations

import pytest

from src.domain.subscriber import Subscriber, is_owner, masked_email, symbols_for


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


class TestIsOwner:
    """BEZ-01: portfel właściciela trafia wyłącznie do jego własnego maila."""

    def test_matching_address_is_the_owner(self) -> None:
        sub = Subscriber(email="owner@example.com", symbols=frozenset())
        assert is_owner(sub, "owner@example.com") is True

    def test_match_ignores_case_and_surrounding_whitespace(self) -> None:
        sub = Subscriber(email="  Owner@Example.COM ", symbols=frozenset())
        assert is_owner(sub, "owner@example.com\n") is True

    def test_other_address_is_not_the_owner(self) -> None:
        sub = Subscriber(email="friend@example.com", symbols=frozenset())
        assert is_owner(sub, "owner@example.com") is False

    @pytest.mark.parametrize("owner_email", [None, "", "   "])
    def test_unknown_owner_matches_nobody(self, owner_email: str | None) -> None:
        # Fail-closed: bez DIGEST_TO_EMAIL nikt nie jest właścicielem — także
        # subskrybent z pustym adresem nie „dopasuje się” do pustego właściciela.
        sub = Subscriber(email="", symbols=frozenset())
        assert is_owner(sub, owner_email) is False


class TestMaskedEmail:
    def test_keeps_the_first_letter_and_the_domain(self) -> None:
        assert masked_email("friend@example.com") == "f***@example.com"

    def test_address_without_at_sign_is_fully_masked(self) -> None:
        assert masked_email("not-an-email") == "***"
