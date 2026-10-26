"""Adaptery pozycji portfela (#15 — tabela `positions`, migracja 023).

Realne pozycje użytkownika (ilość + średni koszt) napędzają całą warstwę ryzyka:
bez nich `weights`/`portfolio_pnl`/VaR liczą fikcję na RÓWNYCH wagach. Ten moduł
dostarcza port `PortfolioPositionsPort` w dwóch wariantach:

  - `SupabasePositionsAdapter` — czyta tabelę `positions` (PostgREST). Mapowanie
    kolumn DB → pola `Position`:
        symbol        → symbol
        quantity      → quantity       (Decimal, przez str — bez błędu binarnego)
        avg_cost      → purchase_price  (nazwa kolumny ≠ nazwa pola — mapujemy tu)
        currency      → currency       (ISO 4217, migracja 035; brak kolumny → USD)
        purchase_date → purchase_date  (prawdziwa data zakupu; NULL → None, nie as_of)
    `as_of` (TIMESTAMPTZ) NIE jest datą zakupu: napędza tylko badge STALE.
    Ten sam adapter implementuje `PositionsImportPort` (import CSV): ścisły odczyt
    i zamianę całej tabeli — INSERT nowych lotów, potem DELETE starych po `id`.
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
import re
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, cast

from src.application.ports import PortfolioPositionsPort, PositionsImportPort
from src.domain.portfolio import DEFAULT_POSITION_CURRENCY, Position, is_iso_currency
from src.infrastructure.adapters._supabase import build_supabase_client

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


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_currency(raw: Any) -> str | None:
    """Kolumna `currency` (035) → kod ISO 4217 albo None (wiersz pominięty).

    Brak kolumny w wierszu (NULL) to USD — domyślna waluta
    kolumny. Wielkość liter i spacje normalizujemy; reszta formatu jest twarda."""
    if raw is None:
        return DEFAULT_POSITION_CURRENCY
    code = str(raw).strip().upper()
    return code if is_iso_currency(code) else None


def _parse_purchase_date(raw: Any) -> date | None:
    """Kolumna `purchase_date` (DATE, ISO) → data albo None = nieznana.

    Nieparsowalna wartość też jest „nieznaną datą” — pozycja zostaje w portfelu,
    a raport mówi wprost, że kursu z dnia zakupu nie zna."""
    if not isinstance(raw, str) or not _ISO_DATE.match(raw.strip()):
        return None
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        return None


def _parse_position(row: dict[str, Any]) -> Position | None:
    """Mapuje wiersz tabeli `positions` na `Position` albo None (pominięcie).

    Odrzuca wiersze bez symbolu, z quantity <= 0, z niefinitnym/ujemnym
    kosztem oraz z walutą spoza formatu ISO — wszystko PRZED konstrukcją
    `Position`, żeby nie wywołać wyjątku z `__post_init__`.
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

    if _parse_as_of(row.get("as_of")) is None:
        return None

    currency = _parse_currency(row.get("currency"))
    if currency is None:
        return None
    return Position(
        symbol=symbol.strip(),
        quantity=quantity,
        purchase_price=purchase_price,
        purchase_date=_parse_purchase_date(row.get("purchase_date")),
        currency=currency,
    )


def _parse_as_of(raw: Any) -> datetime | None:
    """Parsuje kolumnę `as_of` (TIMESTAMPTZ, ISO 8601) na `datetime` albo None."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


class SupabasePositionsAdapter(PortfolioPositionsPort, PositionsImportPort):
    """Adapter tabeli `positions` (PostgreSQL via PostgREST).

    Czyta i zapisuje kolumny `currency`/`purchase_date` z migracji 035 (od PORT-23
    bez flagi: produkcja ma migrację). Przed nią zapis kończy się PGRST204.
    """

    def __init__(
        self,
        url: str,
        key: str,
        table_name: str = DEFAULT_POSITIONS_TABLE,
    ) -> None:
        self._client = build_supabase_client(url, key)
        self._table = table_name

    def _query_rows(self) -> list[dict[str, Any]]:
        """Wszystkie wiersze tabeli `positions` (najświeższe pierwsze); błąd → wyjątek."""
        response = self._client.table(self._table).select("*").order("as_of", desc=True).execute()
        return cast(list[dict[str, Any]], response.data or [])

    def _parse_rows(self, rows: list[dict[str, Any]]) -> list[Position]:
        return [position for row in rows if (position := _parse_position(row)) is not None]

    def _fetch_rows(self) -> list[dict[str, Any]]:
        """Zwraca wszystkie wiersze tabeli `positions` (najświeższe pierwsze).

        Graceful: dowolny błąd (brak tabeli przy niezaaplikowanej migracji 023,
        błąd sieci) → [] + WARNING z podpowiedzią o migracji.
        """
        try:
            return self._query_rows()
        except Exception:
            logger.warning(
                "positions: query on table %r failed — falling back to no "
                "positions (equal weights). Apply migration 023_positions.sql "
                "to enable real portfolio weights/P&L.",
                self._table,
                exc_info=True,
            )
            return []

    def get_positions(self) -> list[Position]:
        """Zwraca realne pozycje z tabeli `positions`.

        Wiersze uszkodzone (brak symbolu, quantity <= 0, wartości niefinitne)
        są pomijane. Graceful: błąd zapytania → [].
        """
        return self._parse_rows(self._fetch_rows())

    def current_positions(self) -> list[Position]:
        """PORT-18: te same loty co `get_positions`, ale błąd odczytu → wyjątek."""
        return self._parse_rows(self._query_rows())

    def replace_positions(self, positions: Sequence[Position], as_of: datetime) -> None:
        """PORT-18: zastępuje całą tabelę lotami z importu, wszystkie z `as_of`.

        Kolejność jest celowa: najpierw INSERT nowych lotów, potem DELETE wierszy,
        które istniały przed importem (po `id`). Błąd zapisu zostawia stary
        portfel; błąd kasowania zostawia duplikaty, które ponowny import usuwa.
        """
        payload = [self._payload(position, as_of) for position in positions]
        existing = self._client.table(self._table).select("id").execute()
        old_ids = [row["id"] for row in cast(list[dict[str, Any]], existing.data or [])]
        if payload:
            self._client.table(self._table).insert(payload).execute()
        if old_ids:
            self._client.table(self._table).delete().in_("id", old_ids).execute()

    def _payload(self, position: Position, as_of: datetime) -> dict[str, Any]:
        return {
            "symbol": position.symbol,
            "quantity": str(position.quantity),
            "avg_cost": str(position.purchase_price),
            "as_of": as_of.isoformat(),
            "currency": position.currency,
            "purchase_date": (
                None if position.purchase_date is None else position.purchase_date.isoformat()
            ),
        }

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
