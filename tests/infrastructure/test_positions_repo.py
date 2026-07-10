"""Testy adapterów pozycji portfela (#15 — tabela `positions`, migracja 023).

`SupabasePositionsAdapter` czyta tabelę `positions` (kolumny: symbol, quantity,
avg_cost, as_of) i mapuje wiersze na domenowe `Position`. Mapowanie kolumn:
`avg_cost` → `purchase_price`, `as_of` → `purchase_date` (i źródło get_as_of).
Czytanie jest graceful: błąd zapytania / brak tabeli → [] / None (fallback na
dotychczasowe równe wagi), nigdy wyjątek. Wiersze uszkodzone (brak symbolu,
quantity <= 0, wartości niefinitne) są pomijane PRZED konstrukcją `Position`
(bo `Position.__post_init__` rzuca na quantity<=0 / NaN).

`NullPositionsAdapter` to no-op: [] / None bez żadnego I/O.

Supabase używa builder/chain API: client.table("x").select(...).execute().
Helper `_set_chain_response` replikuje wzorzec z test_supabase_repository.py.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.domain.portfolio import Position
from src.infrastructure.adapters.positions_repo import (
    NullPositionsAdapter,
    SupabasePositionsAdapter,
)


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


def _row(
    symbol: str = "OK",
    quantity: Any = "1",
    avg_cost: Any = "1",
    as_of: str = "2026-07-01T00:00:00+00:00",
) -> dict[str, Any]:
    """Buduje wiersz tabeli `positions` z sensownymi defaultami (zwięzłe testy)."""
    return {
        "symbol": symbol,
        "quantity": quantity,
        "avg_cost": avg_cost,
        "as_of": as_of,
    }


@pytest.fixture
def mock_client(mocker: Any) -> MagicMock:
    create = mocker.patch(
        "src.infrastructure.adapters.positions_repo.create_client"
    )
    client = MagicMock()
    create.return_value = client
    return client


@pytest.fixture
def repo(mock_client: MagicMock) -> SupabasePositionsAdapter:
    return SupabasePositionsAdapter(
        url="https://test.supabase.co", key="anon-key"
    )


class TestGetPositions:
    def test_reads_positions_table(
        self, repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "select", "order"], [])

        repo.get_positions()

        mock_client.table.assert_called_with("positions")

    def test_maps_row_to_position(
        self, repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        # avg_cost → purchase_price; as_of → purchase_date (data z timestampu).
        _set_chain_response(
            mock_client,
            ["table", "select", "order"],
            [
                {
                    "symbol": "NVDA",
                    "quantity": "10",
                    "avg_cost": "120.50",
                    "as_of": "2026-07-01T12:00:00+00:00",
                }
            ],
        )

        positions = repo.get_positions()

        assert positions == [
            Position(
                symbol="NVDA",
                quantity=Decimal("10"),
                purchase_price=Decimal("120.50"),
                purchase_date=date(2026, 7, 1),
            )
        ]

    def test_maps_numeric_json_values(
        self, repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        # PostgREST potrafi zwrócić NUMERIC jako float — konwersja przez str
        # zachowuje wartość bez błędu binarnego.
        _set_chain_response(
            mock_client,
            ["table", "select", "order"],
            [
                {
                    "symbol": "AAPL",
                    "quantity": 3,
                    "avg_cost": 195.25,
                    "as_of": "2026-07-05T09:30:00+00:00",
                }
            ],
        )

        positions = repo.get_positions()

        assert positions[0].quantity == Decimal("3")
        assert positions[0].purchase_price == Decimal("195.25")

    def test_skips_row_without_symbol(
        self, repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "order"],
            [
                {"quantity": "5", "avg_cost": "10", "as_of": "2026-07-01T00:00:00+00:00"},
                _row(symbol="", quantity="5", avg_cost="10"),
                _row(symbol="VOO", quantity="5", avg_cost="10"),
            ],
        )

        positions = repo.get_positions()

        assert [p.symbol for p in positions] == ["VOO"]

    def test_skips_row_with_non_positive_quantity(
        self, repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        # quantity <= 0 rozbiłoby Position.__post_init__ — filtrujemy PRZED.
        _set_chain_response(
            mock_client,
            ["table", "select", "order"],
            [
                _row(symbol="ZERO", quantity="0", avg_cost="10"),
                _row(symbol="SHORT", quantity="-3", avg_cost="10"),
                _row(symbol="OK", quantity="2", avg_cost="10"),
            ],
        )

        positions = repo.get_positions()

        assert [p.symbol for p in positions] == ["OK"]

    def test_skips_row_with_invalid_numeric(
        self, repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        # Niefinitne / nieparsowalne wartości → wiersz pominięty, nie wyjątek.
        _set_chain_response(
            mock_client,
            ["table", "select", "order"],
            [
                _row(symbol="NAN", quantity="not-a-number", avg_cost="10"),
                _row(symbol="INF", quantity="5", avg_cost="Infinity"),
                _row(symbol="NEG", quantity="5", avg_cost="-1"),
                _row(symbol="OK", quantity="5", avg_cost="10"),
            ],
        )

        positions = repo.get_positions()

        assert [p.symbol for p in positions] == ["OK"]

    def test_query_error_returns_empty_list(
        self, repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        # Graceful: tabela może nie istnieć (migracja 023 niezaaplikowana) →
        # [] zamiast wyjątku (fallback na równe wagi).
        mock_client.table.side_effect = Exception("relation does not exist")
        assert repo.get_positions() == []

    def test_query_error_logs_migration_hint(
        self,
        repo: SupabasePositionsAdapter,
        mock_client: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_client.table.side_effect = Exception("relation does not exist")

        with caplog.at_level("WARNING"):
            repo.get_positions()

        assert "023" in caplog.text


class TestGetAsOf:
    def test_returns_freshest_as_of(
        self, repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "order"],
            [
                _row(symbol="A", as_of="2026-07-01T00:00:00+00:00"),
                _row(symbol="B", as_of="2026-07-05T00:00:00+00:00"),
                _row(symbol="C", as_of="2026-07-03T00:00:00+00:00"),
            ],
        )

        assert repo.get_as_of() == datetime.fromisoformat(
            "2026-07-05T00:00:00+00:00"
        )

    def test_none_when_empty(
        self, repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "select", "order"], [])
        assert repo.get_as_of() is None

    def test_query_error_returns_none(
        self, repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        mock_client.table.side_effect = Exception("boom")
        assert repo.get_as_of() is None


class TestNullPositionsAdapter:
    def test_get_positions_returns_empty(self) -> None:
        assert NullPositionsAdapter().get_positions() == []

    def test_get_as_of_returns_none(self) -> None:
        assert NullPositionsAdapter().get_as_of() is None

    def test_does_no_io(self, mocker: Any) -> None:
        # Zero I/O: Null adapter nie może tknąć klienta Supabase.
        create = mocker.patch(
            "src.infrastructure.adapters.positions_repo.create_client"
        )

        adapter = NullPositionsAdapter()
        adapter.get_positions()
        adapter.get_as_of()

        create.assert_not_called()
