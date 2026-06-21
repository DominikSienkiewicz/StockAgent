"""Testy adaptera SupabaseSubscriberRepository (U2).

Adapter czyta tabelę `subscribers` (kolumny: email, symbols) i mapuje wiersze na
domenowe `Subscriber`. `symbols` może przyjść jako lista (text[]/JSONB), CSV
string albo być puste/None → "wszystko" (pusty frozenset). Czytanie jest
graceful: błąd zapytania → []. Adapter nie wprowadza nowych płatnych wywołań —
to czysty odczyt konfiguracji odbiorców na potrzeby re-slicingu raportu.

Supabase używa builder/chain API: client.table("x").select(...).execute().
Helper `_set_chain_response` replikuje wzorzec z test_supabase_repository.py.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.domain.subscriber import Subscriber
from src.infrastructure.adapters.subscriber_repo import SupabaseSubscriberRepository


def _set_chain_response(
    client: MagicMock, methods: list[str], data: list[dict[str, Any]]
) -> MagicMock:
    """Konfiguruje łańcuch builderów kończący się na .execute() zwracającym
    `data`. Zwraca mock odpowiedzi do ewentualnych asercji.
    """
    obj = client
    for method in methods:
        obj = getattr(obj, method).return_value
    response = MagicMock()
    response.data = data
    obj.execute.return_value = response
    return response


@pytest.fixture
def mock_client(mocker: Any) -> MagicMock:
    create = mocker.patch(
        "src.infrastructure.adapters.subscriber_repo.create_client"
    )
    client = MagicMock()
    create.return_value = client
    return client


@pytest.fixture
def repo(mock_client: MagicMock) -> SupabaseSubscriberRepository:
    return SupabaseSubscriberRepository(
        url="https://test.supabase.co", key="anon-key"
    )


class TestListSubscribers:
    def test_reads_subscribers_table(
        self, repo: SupabaseSubscriberRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "select"], [])

        repo.list_subscribers()

        mock_client.table.assert_called_with("subscribers")

    def test_maps_row_with_list_symbols(
        self, repo: SupabaseSubscriberRepository, mock_client: MagicMock
    ) -> None:
        # text[] / JSONB array → frozenset
        _set_chain_response(
            mock_client,
            ["table", "select"],
            [{"email": "a@example.com", "symbols": ["AAPL", "VOO"]}],
        )

        subs = repo.list_subscribers()

        assert subs == [
            Subscriber(
                email="a@example.com", symbols=frozenset({"AAPL", "VOO"})
            )
        ]

    def test_maps_row_with_csv_symbols(
        self, repo: SupabaseSubscriberRepository, mock_client: MagicMock
    ) -> None:
        # CSV string "AAPL, VOO" → frozenset (trim spacji, pominięcie pustych).
        _set_chain_response(
            mock_client,
            ["table", "select"],
            [{"email": "b@example.com", "symbols": "AAPL, VOO , "}],
        )

        subs = repo.list_subscribers()

        assert subs == [
            Subscriber(
                email="b@example.com", symbols=frozenset({"AAPL", "VOO"})
            )
        ]

    def test_empty_symbols_means_everything(
        self, repo: SupabaseSubscriberRepository, mock_client: MagicMock
    ) -> None:
        # Brak / puste symbols → pusty frozenset = "wszystko" (subskrybent globalny).
        _set_chain_response(
            mock_client,
            ["table", "select"],
            [
                {"email": "none@example.com", "symbols": None},
                {"email": "empty-str@example.com", "symbols": ""},
                {"email": "empty-list@example.com", "symbols": []},
                {"email": "missing@example.com"},
            ],
        )

        subs = repo.list_subscribers()

        assert len(subs) == 4
        assert all(s.symbols == frozenset() for s in subs)

    def test_skips_rows_without_email(
        self, repo: SupabaseSubscriberRepository, mock_client: MagicMock
    ) -> None:
        # Wiersz bez e-maila jest bezużyteczny (nie ma gdzie wysłać) — pomijamy.
        _set_chain_response(
            mock_client,
            ["table", "select"],
            [
                {"symbols": ["AAPL"]},
                {"email": "ok@example.com", "symbols": ["VOO"]},
            ],
        )

        subs = repo.list_subscribers()

        assert subs == [
            Subscriber(email="ok@example.com", symbols=frozenset({"VOO"}))
        ]

    def test_empty_table_returns_empty_list(
        self, repo: SupabaseSubscriberRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "select"], [])
        assert repo.list_subscribers() == []

    def test_query_error_returns_empty_list(
        self, repo: SupabaseSubscriberRepository, mock_client: MagicMock
    ) -> None:
        # Graceful: tabela `subscribers` może jeszcze nie istnieć (migracja
        # niezaaplikowana) lub błąd sieci. Zwracamy [] zamiast wywalać cykl —
        # main_agent zrobi wtedy fallback do pojedynczego DIGEST_TO_EMAIL.
        mock_client.table.side_effect = Exception("relation does not exist")
        assert repo.list_subscribers() == []
