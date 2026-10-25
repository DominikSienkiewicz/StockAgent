"""PORT-18: import pozycji z prostego CSV — parsowanie, walidacja, diff i dry-run.

Czysta logika (`src/tools/import_positions.py`); składanie adaptera i zapis
testuje `tests/test_main_tools.py`.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from src.domain.portfolio import Position
from src.tools.import_positions import (
    PositionsCsvError,
    format_plan,
    parse_positions_csv,
    plan_import,
)

TODAY = date(2026, 10, 19)
HEADER = "symbol,quantity,avg_cost,currency,purchase_date\n"


def _lot(
    symbol: str, qty: str, cost: str, currency: str = "USD", day: date | None = None
) -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal(qty),
        purchase_price=Decimal(cost),
        purchase_date=day,
        currency=currency,
    )


def _errors(text: str) -> tuple[str, ...]:
    with pytest.raises(PositionsCsvError) as error:
        parse_positions_csv(text, today=TODAY)
    return error.value.errors


class TestParse:
    def test_parses_lots_with_and_without_a_purchase_date(self) -> None:
        text = HEADER + "NVDA,10,118.50,USD,2025-03-01\n sap.de , 4 ,182.1, eur ,\n"

        lots = parse_positions_csv(text, today=TODAY)

        assert lots == (
            _lot("NVDA", "10", "118.50", "USD", date(2025, 3, 1)),
            _lot("SAP.DE", "4", "182.1", "EUR"),
        )

    def test_purchase_date_column_is_optional(self) -> None:
        lots = parse_positions_csv(
            "symbol,quantity,avg_cost,currency\nAAPL,3,195.25,USD\n", today=TODAY
        )

        assert lots == (_lot("AAPL", "3", "195.25"),)

    def test_several_lots_of_one_symbol_are_kept(self) -> None:
        text = HEADER + "NVDA,5,100,USD,2024-01-02\nNVDA,5,140,USD,2025-01-02\n"

        assert len(parse_positions_csv(text, today=TODAY)) == 2

    def test_collects_every_row_error_with_its_line_number(self) -> None:
        text = (
            HEADER
            + "AAPL,0,10,USD,\n"
            + "MSFT,1,-5,USD,\n"
            + "SAP.DE,1,10,EURO,\n"
            + "NVDA,1,10,USD,01.03.2025\n"
            + "AMD,1,10,USD,2026-10-20\n"
            + 'TSLA,"1,5",10,USD,\n'
            + ",1,10,USD,\n"
        )

        errors = _errors(text)

        assert errors == (
            "linia 2: quantity 0 — oczekiwana liczba dodatnia",
            "linia 3: avg_cost -5 — oczekiwana liczba dodatnia",
            "linia 4: currency EURO — oczekiwany kod ISO z 3 liter (np. USD)",
            "linia 5: purchase_date 01.03.2025 — oczekiwana data ISO RRRR-MM-DD",
            "linia 6: purchase_date 2026-10-20 — data z przyszłości",
            "linia 7: quantity 1,5 — to nie liczba (separator dziesiętny to kropka)",
            "linia 8: pusty symbol",
        )

    def test_rejects_non_finite_numbers(self) -> None:
        assert _errors(HEADER + "AAPL,NaN,Infinity,USD,\n") == (
            "linia 2: quantity NaN — oczekiwana liczba dodatnia",
            "linia 2: avg_cost Infinity — oczekiwana liczba dodatnia",
        )

    def test_rejects_a_row_with_too_many_fields(self) -> None:
        assert _errors(HEADER + "AAPL,1,10,USD,2025-01-02,extra\n") == ("linia 2: za dużo pól",)

    def test_header_must_name_the_required_columns(self) -> None:
        assert _errors("symbol,qty,avg_cost\nAAPL,1,10\n") == (
            "brak kolumn: quantity, currency",
            "nieznane kolumny: qty (dozwolone: symbol, quantity, avg_cost, currency, "
            "purchase_date)",
        )

    def test_empty_file_is_an_error_not_a_request_to_clear_the_table(self) -> None:
        assert _errors("") == (
            "pusty plik — brak nagłówka symbol,quantity,avg_cost,currency,purchase_date",
        )
        assert _errors(HEADER) == (
            "plik nie zawiera żadnej pozycji (import zastępuje całą tabelę)",
        )


class TestPlan:
    def test_diff_counts_lots_as_a_multiset(self) -> None:
        current = [
            _lot("NVDA", "8", "118"),
            _lot("AAPL", "3", "195.25"),
            _lot("AAPL", "3", "195.25"),
        ]
        incoming = [_lot("NVDA", "10", "118.50"), _lot("AAPL", "3", "195.250")]

        plan = plan_import(current, incoming)

        assert plan.added == (_lot("NVDA", "10", "118.50"),)
        assert plan.removed == (_lot("AAPL", "3", "195.25"), _lot("NVDA", "8", "118"))
        assert plan.unchanged == (_lot("AAPL", "3", "195.25"),)
        assert plan.has_changes

    def test_a_new_purchase_date_or_currency_is_a_change(self) -> None:
        current = [_lot("SAP.DE", "4", "182.1", "USD")]
        incoming = [_lot("SAP.DE", "4", "182.1", "EUR", date(2025, 3, 1))]

        plan = plan_import(current, incoming)

        assert (plan.added, plan.removed) == (tuple(incoming), tuple(current))

    def test_identical_file_has_no_changes_but_can_still_refresh_as_of(self) -> None:
        lots = [_lot("NVDA", "10", "118.50", "USD", date(2025, 3, 1))]

        plan = plan_import(lots, lots)

        assert not plan.has_changes


def test_format_plan_lists_every_lot_and_the_summary() -> None:
    plan = plan_import(
        [_lot("NVDA", "8", "118"), _lot("AAPL", "3", "195.25")],
        [_lot("NVDA", "10", "118.50", "USD", date(2025, 3, 1)), _lot("AAPL", "3", "195.25")],
    )

    text = format_plan(plan, file_lots=2)

    assert text.splitlines() == [
        "Import pozycji z CSV — 2 lot(ów) w pliku",
        "  + NVDA       10 × 118.5 USD, zakup 2025-03-01",
        "  - NVDA       8 × 118 USD, zakup nieznany",
        "  = AAPL       3 × 195.25 USD, zakup nieznany",
        "Zmiany: +1 / -1 / bez zmian 1",
    ]
