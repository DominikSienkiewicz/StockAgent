"""ML-31 — faza cech rynkowych cyklu: kurs raz na cykl i cechy z historii snapshotów."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application.market_features import (
    PRICE_HISTORY_DAYS,
    CycleQuoteCache,
    collect_market_features,
)
from src.application.ports import MarketDataPort, RepositoryPort
from src.domain.price_features import PRICE_FEATURE_NAMES
from src.domain.value_objects import Money

_NOW = datetime(2026, 10, 21, 20, 0, tzinfo=UTC)


def _history(prices: list[float]) -> list[tuple[datetime, Money]]:
    start = _NOW - timedelta(days=len(prices))
    return [
        (start + timedelta(days=i), Money(Decimal(str(price)))) for i, price in enumerate(prices)
    ]


class TestCycleQuoteCache:
    def test_second_read_of_a_symbol_does_not_hit_the_delegate(self) -> None:
        delegate = Mock(spec=MarketDataPort)
        delegate.get_current_price.return_value = Money(Decimal("101.5"))
        cache = CycleQuoteCache(delegate)

        first = cache.get_current_price("AAPL")
        second = cache.get_current_price("AAPL")

        assert first == second == Money(Decimal("101.5"))
        delegate.get_current_price.assert_called_once_with("AAPL")

    def test_a_failed_read_is_not_cached(self) -> None:
        delegate = Mock(spec=MarketDataPort)
        delegate.get_current_price.side_effect = [ValueError("no quote"), Money(Decimal("7"))]
        cache = CycleQuoteCache(delegate)

        with pytest.raises(ValueError):
            cache.get_current_price("AAPL")

        assert cache.get_current_price("AAPL") == Money(Decimal("7"))


class TestCollectMarketFeatures:
    def test_features_come_from_the_snapshot_history_and_the_cycle_quote(self) -> None:
        market = Mock(spec=MarketDataPort)
        market.get_current_price.return_value = Money(Decimal("110"))
        repository = Mock(spec=RepositoryPort)
        repository.get_price_history.return_value = _history([100.0, 104.0])

        features = collect_market_features(
            ["AAPL"], market_port=market, repository=repository, as_of=_NOW
        )

        assert set(features["AAPL"]) == set(PRICE_FEATURE_NAMES)
        assert features["AAPL"]["ret_1"] == pytest.approx(110.0 / 104.0 - 1.0)
        assert features["AAPL"]["ret_20"] is None
        repository.get_price_history.assert_called_once_with("AAPL", PRICE_HISTORY_DAYS)

    def test_a_symbol_without_a_quote_gets_empty_features_and_the_phase_goes_on(self) -> None:
        market = Mock(spec=MarketDataPort)
        market.get_current_price.side_effect = [ValueError("no quote"), Money(Decimal("50"))]
        repository = Mock(spec=RepositoryPort)
        repository.get_price_history.return_value = _history([49.0])

        features = collect_market_features(
            ["SSNLF", "MSFT"], market_port=market, repository=repository, as_of=_NOW
        )

        assert all(value is None for value in features["SSNLF"].values())
        assert features["MSFT"]["ret_1"] == pytest.approx(50.0 / 49.0 - 1.0)

    def test_a_failed_history_read_leaves_the_features_empty(self) -> None:
        market = Mock(spec=MarketDataPort)
        market.get_current_price.return_value = Money(Decimal("10"))
        repository = Mock(spec=RepositoryPort)
        repository.get_price_history.side_effect = RuntimeError("PGRST")

        features = collect_market_features(
            ["AAPL"], market_port=market, repository=repository, as_of=_NOW
        )

        assert all(value is None for value in features["AAPL"].values())

    def test_the_phase_reads_only_free_sources(self) -> None:
        market = Mock(spec=MarketDataPort)
        market.get_current_price.return_value = Money(Decimal("10"))
        repository = Mock(spec=RepositoryPort)
        repository.get_price_history.return_value = []

        collect_market_features(
            ["AAPL", "AAPL"], market_port=market, repository=repository, as_of=_NOW
        )

        # Jeden kurs i jeden odczyt historii na symbol, zero zapisów.
        market.get_current_price.assert_called_once_with("AAPL")
        assert [c[0] for c in repository.method_calls] == ["get_price_history"]
