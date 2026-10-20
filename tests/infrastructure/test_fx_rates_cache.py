"""PORT-11: cache kursów z dnia w `fx_rates` (migracja 036) nad `FxRatePort`.

Kurs z minionego dnia się nie zmienia, więc cache bierze tylko `on_date < dziś`.
Błąd tabeli (np. przed migracją) to przejście do źródła, nigdy brak kursu.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from src.application.ports import FxRatePort
from src.domain.fx import FxRate
from src.infrastructure.adapters.fx_rates_cache import SupabaseCachedFxRates

TODAY = date(2026, 10, 19)
BOUGHT = date(2025, 3, 2)
FRIDAY_RATE = FxRate(
    currency="USD", on_date=BOUGHT, effective_date=date(2025, 2, 28), rate=Decimal("3.9725")
)


@pytest.fixture
def client(mocker: Any) -> MagicMock:
    create = mocker.patch("src.infrastructure.adapters.fx_rates_cache.create_client")
    return create.return_value


@pytest.fixture
def source() -> Mock:
    delegate = Mock(spec=FxRatePort)
    delegate.get_rate.return_value = FRIDAY_RATE
    return delegate


def _cache(source: Mock) -> SupabaseCachedFxRates:
    return SupabaseCachedFxRates(
        url="https://test.supabase.co", key="anon", delegate=source, today=lambda: TODAY
    )


def _cached_rows(client: MagicMock, rows: list[dict[str, Any]]) -> MagicMock:
    query = client.table.return_value.select.return_value.eq.return_value.eq.return_value
    query.limit.return_value.execute.return_value.data = rows
    return query


def test_hit_returns_the_cached_rate_without_asking_the_source(
    client: MagicMock, source: Mock
) -> None:
    query = _cached_rows(
        client, [{"currency": "USD", "on_date": "2025-03-02", "effective_date": "2025-02-28",
                  "rate": "3.9725"}]
    )

    rate = _cache(source).get_rate("usd", BOUGHT)

    assert rate == FRIDAY_RATE
    source.get_rate.assert_not_called()
    client.table.assert_called_with("fx_rates")
    client.table.return_value.select.return_value.eq.assert_called_once_with("currency", "USD")
    query.limit.assert_called_once_with(1)


def test_miss_asks_the_source_and_stores_a_past_day(client: MagicMock, source: Mock) -> None:
    _cached_rows(client, [])

    rate = _cache(source).get_rate("USD", BOUGHT)

    assert rate == FRIDAY_RATE
    source.get_rate.assert_called_once_with("USD", BOUGHT)
    client.table.return_value.upsert.assert_called_once_with(
        {
            "currency": "USD",
            "on_date": "2025-03-02",
            "effective_date": "2025-02-28",
            "rate": "3.9725",
        },
        on_conflict="currency,on_date",
    )


def test_today_is_neither_read_nor_stored(client: MagicMock, source: Mock) -> None:
    # Przed publikacją tabeli „dziś” to kurs z wczoraj — nie może zostać w cache.
    source.get_rate.return_value = FxRate(
        currency="USD", on_date=TODAY, effective_date=date(2026, 10, 16), rate=Decimal("3.70")
    )

    _cache(source).get_rate("USD", TODAY)

    client.table.assert_not_called()


def test_source_without_a_rate_stores_nothing(client: MagicMock, source: Mock) -> None:
    _cached_rows(client, [])
    source.get_rate.return_value = None

    assert _cache(source).get_rate("USD", BOUGHT) is None
    client.table.return_value.upsert.assert_not_called()


def test_cache_read_error_falls_back_to_the_source(
    client: MagicMock, source: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    client.table.side_effect = Exception("relation fx_rates does not exist")

    with caplog.at_level("WARNING"):
        rate = _cache(source).get_rate("USD", BOUGHT)

    assert rate == FRIDAY_RATE
    assert "036" in caplog.text


def test_cache_write_error_still_returns_the_rate(client: MagicMock, source: Mock) -> None:
    _cached_rows(client, [])
    client.table.return_value.upsert.return_value.execute.side_effect = Exception("PGRST205")

    assert _cache(source).get_rate("USD", BOUGHT) == FRIDAY_RATE


def test_malformed_cache_row_is_a_miss(client: MagicMock, source: Mock) -> None:
    _cached_rows(client, [{"effective_date": "2025-02-28", "rate": "abc"}])

    assert _cache(source).get_rate("USD", BOUGHT) == FRIDAY_RATE
    source.get_rate.assert_called_once()


def test_read_only_cache_reads_but_never_stores(
    client: MagicMock, source: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    # BEZ-08: lokalny run (DB_WRITES_ENABLED=false) czyta wspólny cache, ale nie
    # dopisuje do niego kursów — tabela należy do produkcyjnych przebiegów.
    query = _cached_rows(client, [])
    cache = SupabaseCachedFxRates(
        url="https://test.supabase.co",
        key="anon",
        delegate=source,
        today=lambda: TODAY,
        write_through=False,
    )

    with caplog.at_level("WARNING"):
        first = cache.get_rate("USD", BOUGHT)
        second = cache.get_rate("USD", date(2025, 3, 3))

    assert first == second == FRIDAY_RATE
    assert query.limit.call_count == 2
    client.table.return_value.upsert.assert_not_called()
    assert caplog.text.count("DB_WRITES_ENABLED=false") == 1
