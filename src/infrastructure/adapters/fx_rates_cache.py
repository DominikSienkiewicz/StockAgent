"""Cache kursów z dnia w tabeli `fx_rates` (PORT-11, migracja 036).

`SupabaseCachedFxRates` to dekorator `FxRatePort`: najpierw tabela, potem źródło
(NBP), a wynik źródła trafia do tabeli. Kurs z MINIONEGO dnia się nie zmienia,
więc cache bierze wyłącznie `on_date < dziś` — kurs „na dziś” przed publikacją
tabeli NBP to kurs z wczoraj i po południu byłby nieaktualny.

Cache jest best-effort: błąd odczytu albo zapisu (brak tabeli przed migracją,
sieć) to WARNING i przejście do źródła, nigdy wyjątek ani brak kursu. Włącza go
flaga `fx_rates_cache_enabled` — bez niej DI składa samo źródło.

BEZ-08: `write_through=False` (run bez `db_writes_enabled`) czyta tabelę, ale
kursu ze źródła do niej nie dopisuje — wspólna baza należy do produkcji.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, cast

from supabase import create_client

from src.application.ports import FxRatePort
from src.domain.fx import FxRate

logger = logging.getLogger(__name__)

DEFAULT_FX_RATES_TABLE = "fx_rates"


class SupabaseCachedFxRates(FxRatePort):
    """Read-through cache kursów z minionych dni nad dowolnym `FxRatePort`."""

    def __init__(
        self,
        url: str,
        key: str,
        delegate: FxRatePort,
        *,
        today: Callable[[], date],
        table_name: str = DEFAULT_FX_RATES_TABLE,
        write_through: bool = True,
    ) -> None:
        self._client = create_client(url, key)
        self._delegate = delegate
        self._today = today
        self._table = table_name
        self._write_through = write_through
        self._skip_logged = False

    def get_rate(self, currency: str, on_date: date) -> FxRate | None:
        code = currency.strip().upper()
        cacheable = on_date < self._today()
        if cacheable:
            cached = self._read(code, on_date)
            if cached is not None:
                return cached
        rate = self._delegate.get_rate(code, on_date)
        if rate is not None and cacheable:
            self._write(rate)
        return rate

    def _read(self, currency: str, on_date: date) -> FxRate | None:
        try:
            response = (
                self._client.table(self._table)
                .select("*")
                .eq("currency", currency)
                .eq("on_date", on_date.isoformat())
                .limit(1)
                .execute()
            )
        except Exception:
            logger.warning(
                "fx_rates: cache read failed — asking the source. Apply migration "
                "036_fx_rates.sql or turn fx_rates_cache_enabled off.",
                exc_info=True,
            )
            return None
        rows = cast(list[dict[str, Any]], response.data or [])
        return _parse_row(rows[0], currency, on_date) if rows else None

    def _write(self, rate: FxRate) -> None:
        if not self._write_through:
            if not self._skip_logged:
                self._skip_logged = True
                logger.warning(
                    "DB_WRITES_ENABLED=false — fx_rates: kursy ze źródła nie trafiają "
                    "do cache (run tylko do odczytu)"
                )
            return
        record = {
            "currency": rate.currency,
            "on_date": rate.on_date.isoformat(),
            "effective_date": rate.effective_date.isoformat(),
            "rate": str(rate.rate),
        }
        try:
            self._client.table(self._table).upsert(record, on_conflict="currency,on_date").execute()
        except Exception:
            logger.warning("fx_rates: cache write failed for %s on %s", rate.currency,
                           rate.on_date, exc_info=True)


def _parse_row(row: dict[str, Any], currency: str, on_date: date) -> FxRate | None:
    """Wiersz `fx_rates` → `FxRate`; uszkodzony wiersz to brak trafienia, nie wyjątek."""
    try:
        return FxRate(
            currency=currency,
            on_date=on_date,
            effective_date=date.fromisoformat(str(row.get("effective_date"))),
            rate=Decimal(str(row.get("rate"))),
        )
    except (ValueError, InvalidOperation):
        logger.warning("fx_rates: malformed cache row for %s on %s", currency, on_date)
        return None
