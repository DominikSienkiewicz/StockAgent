"""Testy adapterów pozycji portfela (#15 — tabela `positions`, migracja 023).

`SupabasePositionsAdapter` czyta tabelę `positions` (kolumny: symbol, quantity,
avg_cost, as_of, currency, purchase_date) i mapuje wiersze na domenowe `Position`.
Mapowanie kolumn: `avg_cost` → `purchase_price`, `purchase_date` → `purchase_date`
(`as_of` to nie data zakupu — jest tylko źródłem get_as_of).
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
        # avg_cost → purchase_price; wiersz bez `purchase_date` ma datę nieznaną —
        # `as_of` (data aktualizacji tabeli) nie jest datą zakupu (PORT-23).
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
                purchase_date=None,
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


# --- PORT-18: waluta i prawdziwa data zakupu (migracja 035) -------------------


@pytest.fixture
def currency_repo(mock_client: MagicMock) -> SupabasePositionsAdapter:
    return SupabasePositionsAdapter(url="https://test.supabase.co", key="anon-key")


class TestCurrencyColumns:
    def test_purchase_date_comes_from_its_column_not_from_as_of(
        self, currency_repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "order"],
            [
                {
                    **_row(symbol="SAP.DE", quantity="10", avg_cost="120.50",
                           as_of="2026-07-10T12:00:00+00:00"),
                    "currency": "EUR",
                    "purchase_date": "2025-03-01",
                }
            ],
        )

        positions = currency_repo.get_positions()

        assert positions == [
            Position(
                symbol="SAP.DE",
                quantity=Decimal("10"),
                purchase_price=Decimal("120.50"),
                purchase_date=date(2025, 3, 1),
                currency="EUR",
            )
        ]

    def test_null_purchase_date_stays_unknown(
        self, currency_repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        # `as_of` to data aktualizacji tabeli — nie udaje daty zakupu.
        _set_chain_response(
            mock_client,
            ["table", "select", "order"],
            [{**_row(symbol="NVDA"), "currency": "USD", "purchase_date": None}],
        )

        assert currency_repo.get_positions()[0].purchase_date is None

    def test_row_without_the_new_columns_reads_as_usd_with_unknown_date(
        self, currency_repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        # Flaga włączona przed migracją 035: `select("*")` nie zwraca kolumn —
        # odczyt łagodny, bez wyjątku.
        _set_chain_response(mock_client, ["table", "select", "order"], [_row(symbol="AAPL")])

        position = currency_repo.get_positions()[0]

        assert (position.currency, position.purchase_date) == ("USD", None)

    def test_skips_row_with_a_currency_outside_iso_format(
        self, currency_repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "order"],
            [
                {**_row(symbol="BAD"), "currency": "EURO"},
                {**_row(symbol="LOW"), "currency": " eur "},
                {**_row(symbol="OK"), "currency": "GBP"},
            ],
        )

        positions = currency_repo.get_positions()

        assert [(p.symbol, p.currency) for p in positions] == [("LOW", "EUR"), ("OK", "GBP")]

    def test_unparseable_purchase_date_is_unknown_not_a_dropped_position(
        self, currency_repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "order"],
            [{**_row(symbol="NVDA"), "purchase_date": "01.03.2025"}],
        )

        assert currency_repo.get_positions()[0].purchase_date is None


def _lot(symbol: str, currency: str = "USD", day: date | None = None) -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal("2.5"),
        purchase_price=Decimal("101.25"),
        purchase_date=day,
        currency=currency,
    )


_IMPORTED_AT = datetime.fromisoformat("2026-10-19T08:00:00+00:00")


class TestReplacePositions:
    """Import CSV zastępuje całą tabelę: najpierw INSERT nowych lotów, potem DELETE
    starych po `id` — błąd zapisu zostawia stary portfel zamiast pustej tabeli."""

    def _existing_ids(self, client: MagicMock, ids: list[int]) -> None:
        _set_chain_response(client, ["table", "select"], [{"id": i} for i in ids])

    def test_payload_carries_currency_and_purchase_date(
        self, currency_repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        self._existing_ids(mock_client, [])

        currency_repo.replace_positions(
            [_lot("SAP.DE", "EUR", date(2025, 3, 1)), _lot("NVDA")], as_of=_IMPORTED_AT
        )

        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert [(row["currency"], row["purchase_date"]) for row in payload] == [
            ("EUR", "2025-03-01"),
            ("USD", None),
        ]

    def test_inserts_new_lots_before_deleting_the_old_rows(
        self, currency_repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        self._existing_ids(mock_client, [7, 9])
        table = mock_client.table.return_value
        calls: list[str] = []

        def record(call: str) -> MagicMock:
            calls.append(call)
            return MagicMock()

        table.insert.side_effect = lambda *_: record("insert")
        table.delete.side_effect = lambda *_: record("delete")

        currency_repo.replace_positions([_lot("AAPL")], as_of=_IMPORTED_AT)

        assert calls == ["insert", "delete"]

    def test_deletes_exactly_the_rows_that_existed_before(
        self, currency_repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        self._existing_ids(mock_client, [7, 9])

        currency_repo.replace_positions([_lot("AAPL")], as_of=_IMPORTED_AT)

        mock_client.table.return_value.delete.return_value.in_.assert_called_once_with(
            "id", [7, 9]
        )

    def test_empty_table_needs_no_delete(
        self, currency_repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        self._existing_ids(mock_client, [])

        currency_repo.replace_positions([_lot("AAPL")], as_of=_IMPORTED_AT)

        mock_client.table.return_value.delete.assert_not_called()

    def test_insert_error_propagates_and_keeps_the_old_rows(
        self, currency_repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        self._existing_ids(mock_client, [7])
        table = mock_client.table.return_value
        table.insert.return_value.execute.side_effect = Exception("PGRST204")

        with pytest.raises(Exception, match="PGRST204"):
            currency_repo.replace_positions([_lot("AAPL")], as_of=_IMPORTED_AT)

        table.delete.assert_not_called()


class TestCurrentPositions:
    def test_reads_the_same_rows_as_the_report(
        self, currency_repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "order"],
            [{**_row(symbol="SAP.DE"), "currency": "EUR", "purchase_date": "2025-03-01"}],
        )

        assert [(p.symbol, p.currency) for p in currency_repo.current_positions()] == [
            ("SAP.DE", "EUR")
        ]

    def test_query_error_is_raised_not_hidden_as_an_empty_table(
        self, currency_repo: SupabasePositionsAdapter, mock_client: MagicMock
    ) -> None:
        # Dry-run importu na pustej liście pokazałby „wszystko nowe” — fałsz.
        mock_client.table.side_effect = Exception("relation does not exist")

        with pytest.raises(Exception, match="relation"):
            currency_repo.current_positions()
