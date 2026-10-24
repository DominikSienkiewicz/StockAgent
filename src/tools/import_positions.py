"""Import pozycji portfela z prostego CSV (PORT-18) — czysta logika.

Tabela `positions` była ręcznie utrzymywanym rejestrem, więc odznaka STALE
była stanem domyślnym, a data zakupu nie istniała. Ten moduł parsuje i waliduje
plik, liczy diff względem bieżącej tabeli i formatuje go do dry-runu. Składanie
`Settings` i adaptera oraz sam zapis żyją w `main_tools.py import-positions`
(ARCH-39) — tu nie ma I/O.

Format pliku (UTF-8, separator przecinek, nagłówek w pierwszym wierszu):

    symbol,quantity,avg_cost,currency,purchase_date
    NVDA,10,118.50,USD,2025-03-01
    SAP.DE,4,182.10,EUR,

  - `symbol`        — ticker jak w config.toml (wielkie litery po normalizacji),
  - `quantity`      — liczba jednostek > 0, kropka dziesiętna,
  - `avg_cost`      — średni koszt jednostki > 0, w walucie `currency`,
  - `currency`      — kod ISO 4217 (trzy litery, np. USD, EUR, GBP, PLN),
  - `purchase_date` — opcjonalna kolumna i wartość, ISO `RRRR-MM-DD`, nie
                      z przyszłości; puste = data nieznana.

Jeden wiersz = jeden lot; kilka lotów jednego symbolu (różne daty zakupu) jest
dozwolonych. Import zastępuje CAŁĄ tabelę (plik to pełny stan portfela), więc
plik bez żadnej pozycji jest błędem, a nie poleceniem „wyczyść portfel”.
"""

from __future__ import annotations

import csv
import io
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

from src.domain.portfolio import Position, is_iso_currency

CSV_COLUMNS = ("symbol", "quantity", "avg_cost", "currency", "purchase_date")
REQUIRED_COLUMNS = CSV_COLUMNS[:4]

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

LotKey = tuple[str, Decimal, Decimal, str, date | None]


class PositionsCsvError(ValueError):
    """Plik nie przeszedł walidacji — `errors` to wszystkie błędy z numerami linii."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True)
class ImportPlan:
    """Diff importu: loty dodane, usunięte i bez zmian. Brak zmian nie blokuje
    `--apply`: zapis odświeża `as_of`."""

    added: tuple[Position, ...]
    removed: tuple[Position, ...]
    unchanged: tuple[Position, ...]

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed)


def parse_positions_csv(text: str, *, today: date) -> tuple[Position, ...]:
    """Parsuje i waliduje plik. Zbiera WSZYSTKIE błędy, zanim rzuci
    `PositionsCsvError` — poprawianie pliku po jednym błędzie to strata czasu."""
    reader = csv.DictReader(io.StringIO(text))
    header_errors = _header_errors(reader.fieldnames)
    if header_errors:
        raise PositionsCsvError(header_errors)

    errors: list[str] = []
    positions: list[Position] = []
    for row in reader:
        line = reader.line_num
        if None in row:
            errors.append(f"linia {line}: za dużo pól")
            continue
        values = {
            (name or "").strip().lower(): (value or "").strip() for name, value in row.items()
        }
        position, row_errors = _parse_row(values, today=today)
        errors.extend(f"linia {line}: {error}" for error in row_errors)
        if position is not None:
            positions.append(position)

    if not errors and not positions:
        errors.append("plik nie zawiera żadnej pozycji (import zastępuje całą tabelę)")
    if errors:
        raise PositionsCsvError(errors)
    return tuple(positions)


def _header_errors(fieldnames: Sequence[str] | None) -> list[str]:
    if not fieldnames:
        return ["pusty plik — brak nagłówka " + ",".join(CSV_COLUMNS)]
    names = [name.strip().lower() for name in fieldnames]
    errors = []
    missing = [column for column in REQUIRED_COLUMNS if column not in names]
    if missing:
        errors.append(f"brak kolumn: {', '.join(missing)}")
    unknown = [name for name in names if name not in CSV_COLUMNS]
    if unknown:
        allowed = ", ".join(CSV_COLUMNS)
        errors.append(f"nieznane kolumny: {', '.join(unknown)} (dozwolone: {allowed})")
    return errors


def _parse_row(values: dict[str, str], *, today: date) -> tuple[Position | None, list[str]]:
    errors: list[str] = []
    symbol = values.get("symbol", "").upper()
    if not symbol:
        errors.append("pusty symbol")
    quantity = _positive_decimal("quantity", values.get("quantity", ""), errors)
    avg_cost = _positive_decimal("avg_cost", values.get("avg_cost", ""), errors)
    currency = values.get("currency", "").upper()
    if not is_iso_currency(currency):
        errors.append(f"currency {currency or '(puste)'} — oczekiwany kod ISO z 3 liter (np. USD)")
    purchase_date = _purchase_date(values.get("purchase_date", ""), today, errors)
    if errors or quantity is None or avg_cost is None:
        return None, errors
    return (
        Position(
            symbol=symbol,
            quantity=quantity,
            purchase_price=avg_cost,
            purchase_date=purchase_date,
            currency=currency,
        ),
        [],
    )


def _positive_decimal(column: str, raw: str, errors: list[str]) -> Decimal | None:
    try:
        value = Decimal(raw)
    except InvalidOperation:
        hint = " (separator dziesiętny to kropka)" if "," in raw else ""
        errors.append(f"{column} {raw or '(puste)'} — to nie liczba{hint}")
        return None
    if not value.is_finite() or value <= 0:
        errors.append(f"{column} {raw} — oczekiwana liczba dodatnia")
        return None
    return value


def _purchase_date(raw: str, today: date, errors: list[str]) -> date | None:
    if not raw:
        return None
    try:
        if not _ISO_DATE.match(raw):
            raise ValueError(raw)
        value = date.fromisoformat(raw)
    except ValueError:
        errors.append(f"purchase_date {raw} — oczekiwana data ISO RRRR-MM-DD")
        return None
    if value > today:
        errors.append(f"purchase_date {raw} — data z przyszłości")
        return None
    return value


def plan_import(current: Sequence[Position], incoming: Sequence[Position]) -> ImportPlan:
    """Diff lotów (multizbiór po symbolu, ilości, koszcie, walucie i dacie zakupu)."""
    before = Counter(_key(position) for position in current)
    after = Counter(_key(position) for position in incoming)
    return ImportPlan(
        added=_lots(after - before),
        removed=_lots(before - after),
        unchanged=_lots(before & after),
    )


def _key(position: Position) -> LotKey:
    return (
        position.symbol,
        position.quantity,
        position.purchase_price,
        position.currency,
        position.purchase_date,
    )


def _lots(counter: Counter[LotKey]) -> tuple[Position, ...]:
    return tuple(
        Position(
            symbol=symbol,
            quantity=quantity,
            purchase_price=price,
            currency=currency,
            purchase_date=purchase_date,
        )
        for (symbol, quantity, price, currency, purchase_date) in sorted(
            counter.elements(), key=_sort_key
        )
    )


def _sort_key(key: LotKey) -> tuple[str, str, Decimal, Decimal, str]:
    symbol, quantity, price, currency, purchase_date = key
    bought = "" if purchase_date is None else purchase_date.isoformat()
    return (symbol, bought, quantity, price, currency)


def format_plan(plan: ImportPlan, *, file_lots: int) -> str:
    """Dry-run do CLI: `+` dodane, `-` usunięte, `=` bez zmian, potem podsumowanie."""
    lines = [f"Import pozycji z CSV — {file_lots} lot(ów) w pliku"]
    for sign, lots in (("+", plan.added), ("-", plan.removed), ("=", plan.unchanged)):
        lines.extend(f"  {sign} {_describe(lot)}" for lot in lots)
    lines.append(
        f"Zmiany: +{len(plan.added)} / -{len(plan.removed)} / bez zmian {len(plan.unchanged)}"
    )
    return "\n".join(lines)


def _describe(lot: Position) -> str:
    bought = "nieznany" if lot.purchase_date is None else lot.purchase_date.isoformat()
    return (
        f"{lot.symbol:<10} {_plain(lot.quantity)} × {_plain(lot.purchase_price)} "
        f"{lot.currency}, zakup {bought}"
    )


def _plain(value: Decimal) -> str:
    return f"{value.normalize():f}"
