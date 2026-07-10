"""Adaptery pozycji portfela (#15 — tabela `positions`, migracja 023).

Realne pozycje użytkownika (ilość + średni koszt) napędzają całą warstwę ryzyka:
bez nich `weights`/`portfolio_pnl`/VaR liczą fikcję na RÓWNYCH wagach. Ten moduł
dostarcza port `PortfolioPositionsPort` w dwóch wariantach:

  - `SupabasePositionsAdapter` — czyta tabelę `positions` (PostgREST). Mapowanie
    kolumn DB → pola `Position`:
        symbol    → symbol
        quantity  → quantity        (Decimal, przez str — bez błędu binarnego)
        avg_cost  → purchase_price   (nazwa kolumny ≠ nazwa pola — mapujemy tu)
        as_of     → purchase_date    (data z timestampu; as_of napędza też badge STALE)
  - `NullPositionsAdapter` — no-op ([] / None), zero I/O; używany, gdy port jest
    wyłączony (fallback na dotychczasowe równe wagi).

Czytanie jest graceful: dowolny błąd zapytania (brak tabeli przy niezaaplikowanej
migracji 023, błąd sieci) → [] / None. Wtedy warstwa ryzyka robi fallback na równe
wagi zamiast wywalać cykl. Wiersze uszkodzone (brak symbolu, quantity <= 0,
wartości niefinitne/nieparsowalne) są pomijane PRZED konstrukcją `Position` —
`Position.__post_init__` rzuca na quantity<=0 i NaN, więc walidujemy wcześniej.
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, cast

from supabase import create_client

from src.application.ports import PortfolioPositionsPort
from src.domain.portfolio import Position

logger = logging.getLogger(__name__)

DEFAULT_POSITIONS_TABLE = "positions"


def _parse_decimal(raw: Any) -> Decimal | None:
    """Mapuje surową wartość NUMERIC na finitny `Decimal` albo None.

    PostgREST potrafi zwrócić NUMERIC jako float lub str — konwersja przez `str`
    zachowuje wartość bez błędu binarnego. Wartości nieparsowalne oraz NaN/Inf
    (które zatrułyby porównania w domenie) zwracają None → wiersz pominięty.
    """
    if raw is None:
        return None
    try:
        value = Decimal(str(raw))
    except (InvalidOperation, ValueError, TypeError):
        return None
    if value.is_nan() or value.is_infinite():
        return None
    return value


def _parse_position(row: dict[str, Any]) -> Position | None:
    """Mapuje wiersz tabeli `positions` na `Position` albo None (pominięcie).

    Odrzuca wiersze bez symbolu, z quantity <= 0 oraz z niefinitnym/ujemnym
    kosztem — wszystko PRZED konstrukcją `Position`, żeby nie wywołać wyjątku
    z `__post_init__`.
    """
    symbol = row.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        return None

    quantity = _parse_decimal(row.get("quantity"))
    if quantity is None or quantity <= 0:
        return None

    purchase_price = _parse_decimal(row.get("avg_cost"))
    if purchase_price is None or purchase_price < 0:
        return None

    as_of = _parse_as_of(row.get("as_of"))
    if as_of is None:
        return None

    return Position(
        symbol=symbol.strip(),
        quantity=quantity,
        purchase_price=purchase_price,
        purchase_date=as_of.date(),
    )


def _parse_as_of(raw: Any) -> datetime | None:
    """Parsuje kolumnę `as_of` (TIMESTAMPTZ, ISO 8601) na `datetime` albo None."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


class SupabasePositionsAdapter(PortfolioPositionsPort):
    """Adapter odczytu tabeli `positions` (PostgreSQL via PostgREST)."""

    def __init__(
        self,
        url: str,
        key: str,
        table_name: str = DEFAULT_POSITIONS_TABLE,
    ) -> None:
        self._client = create_client(url, key)
        self._table = table_name

    def _fetch_rows(self) -> list[dict[str, Any]]:
        """Zwraca wszystkie wiersze tabeli `positions` (najświeższe pierwsze).

        Graceful: dowolny błąd (brak tabeli przy niezaaplikowanej migracji 023,
        błąd sieci) → [] + WARNING z podpowiedzią o migracji.
        """
        try:
            response = (
                self._client.table(self._table)
                .select("*")
                .order("as_of", desc=True)
                .execute()
            )
        except Exception:
            logger.warning(
                "positions: query on table %r failed — falling back to no "
                "positions (equal weights). Apply migration 023_positions.sql "
                "to enable real portfolio weights/P&L.",
                self._table,
                exc_info=True,
            )
            return []
        return cast(list[dict[str, Any]], response.data or [])

    def get_positions(self) -> list[Position]:
        """Zwraca realne pozycje z tabeli `positions`.

        Wiersze uszkodzone (brak symbolu, quantity <= 0, wartości niefinitne)
        są pomijane. Graceful: błąd zapytania → [].
        """
        out: list[Position] = []
        for row in self._fetch_rows():
            position = _parse_position(row)
            if position is not None:
                out.append(position)
        return out

    def get_as_of(self) -> datetime | None:
        """Najświeższy `as_of` w tabeli (napędza badge STALE).

        None, gdy brak wierszy lub błąd zapytania.
        """
        stamps = [
            stamp
            for row in self._fetch_rows()
            if (stamp := _parse_as_of(row.get("as_of"))) is not None
        ]
        if not stamps:
            return None
        return max(stamps)


class NullPositionsAdapter(PortfolioPositionsPort):
    """No-op wariant portu: [] / None, zero I/O.

    Używany, gdy port pozycji jest wyłączony — warstwa ryzyka robi wtedy
    fallback na dotychczasowe równe wagi.
    """

    def get_positions(self) -> list[Position]:
        """Zawsze pusta lista — brak pozycji, brak I/O."""
        return []

    def get_as_of(self) -> datetime | None:
        """Zawsze None — brak danych o świeżości, brak I/O."""
        return None
