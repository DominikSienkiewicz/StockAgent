"""ML-31 — faza cech rynkowych cyklu: kurs raz na cykl i cechy z historii snapshotów."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import Mock

import pytest

from src.application.market_features import (
    PRICE_HISTORY_DAYS,
    CycleQuoteCache,
    collect_market_features,
)
from src.application.ports import MarketDataPort, RepositoryPort
from src.domain.cross_section import CROSS_SECTION_FEATURE_NAMES
from src.domain.price_features import PRICE_FEATURE_NAMES
from src.domain.value_objects import AssetType, Money

_NOW = datetime(2026, 10, 21, 20, 0, tzinfo=UTC)
_THRESHOLDS = {AssetType.STOCK: 0.02, AssetType.ETF: 0.02, AssetType.CRYPTO: 0.05}
_UNIVERSE: dict[str, Any] = {"asset_types": {}, "gate_thresholds": _THRESHOLDS}


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
            ["AAPL"], market_port=market, repository=repository, as_of=_NOW, **_UNIVERSE
        )

        assert set(features["AAPL"]) == set(PRICE_FEATURE_NAMES) | set(
            CROSS_SECTION_FEATURE_NAMES
        )
        assert features["AAPL"]["ret_1"] == pytest.approx(110.0 / 104.0 - 1.0)
        assert features["AAPL"]["ret_20"] is None
        repository.get_price_history.assert_called_once_with("AAPL", PRICE_HISTORY_DAYS)

    def test_a_symbol_without_a_quote_gets_empty_features_and_the_phase_goes_on(self) -> None:
        market = Mock(spec=MarketDataPort)
        market.get_current_price.side_effect = [ValueError("no quote"), Money(Decimal("50"))]
        repository = Mock(spec=RepositoryPort)
        repository.get_price_history.return_value = _history([49.0])

        features = collect_market_features(
            ["SSNLF", "MSFT"], market_port=market, repository=repository, as_of=_NOW,
            **_UNIVERSE,
        )

        assert all(value is None for value in features["SSNLF"].values())
        assert features["MSFT"]["ret_1"] == pytest.approx(50.0 / 49.0 - 1.0)

    def test_a_failed_history_read_leaves_the_features_empty(self) -> None:
        market = Mock(spec=MarketDataPort)
        market.get_current_price.return_value = Money(Decimal("10"))
        repository = Mock(spec=RepositoryPort)
        repository.get_price_history.side_effect = RuntimeError("PGRST")

        features = collect_market_features(
            ["AAPL"], market_port=market, repository=repository, as_of=_NOW, **_UNIVERSE
        )

        assert all(value is None for value in features["AAPL"].values())

    def test_the_phase_reads_only_free_sources(self) -> None:
        market = Mock(spec=MarketDataPort)
        market.get_current_price.return_value = Money(Decimal("10"))
        repository = Mock(spec=RepositoryPort)
        repository.get_price_history.return_value = []

        collect_market_features(
            ["AAPL", "AAPL"], market_port=market, repository=repository, as_of=_NOW,
            **_UNIVERSE,
        )

        # Jeden kurs i jeden odczyt historii na symbol, zero zapisów.
        market.get_current_price.assert_called_once_with("AAPL")
        assert [c[0] for c in repository.method_calls] == ["get_price_history"]


class TestCrossSectionPhase:
    """ML-32 — cechy cross-sectional liczone po zebraniu kursów WSZYSTKICH symboli;
    symbol bez kursu nie wchodzi do uniwersum i ma None."""

    def test_rank_and_excess_vs_ivv_use_the_whole_cycle(self) -> None:
        quotes = {"AAPL": "97", "MSFT": "101", "IVV": "99", "BTC": "104", "SSNLF": None}
        market = Mock(spec=MarketDataPort)

        def quote(symbol: str) -> Money:
            price = quotes[symbol]
            if price is None:
                raise ValueError("no quote")
            return Money(Decimal(price))

        market.get_current_price.side_effect = quote
        repository = Mock(spec=RepositoryPort)
        repository.get_price_history.return_value = _history([100.0])

        features = collect_market_features(
            list(quotes),
            market_port=market,
            repository=repository,
            as_of=_NOW,
            asset_types={"IVV": AssetType.ETF, "BTC": AssetType.CRYPTO},
            gate_thresholds=_THRESHOLDS,
        )

        assert features["AAPL"]["ret_1_rank"] == 0.0
        assert features["BTC"]["ret_1_rank"] == 1.0
        assert features["AAPL"]["ret_1_excess_vs_benchmark"] == pytest.approx(-0.02)
        assert features["MSFT"]["ret_1_excess_vs_class_median"] == pytest.approx(0.02)
        # |Δ| ≥ próg klasy: AAPL (−3%) tak, MSFT (+1%) i IVV (−1%) nie, BTC (+4% < 5%) nie.
        assert features["IVV"]["breadth"] == pytest.approx(0.25)
        assert all(features["SSNLF"][name] is None for name in CROSS_SECTION_FEATURE_NAMES)


def test_macro_features_of_the_cycle_reach_every_symbol() -> None:
    # DAT-22: cechy makro są wspólne dla cyklu — ta sama wartość u każdego symbolu.
    market = Mock(spec=MarketDataPort)
    market.get_current_price.return_value = Money(Decimal("10"))
    repository = Mock(spec=RepositoryPort)
    repository.get_price_history.return_value = []
    macro = {"vix_percentile_1y": 93.0, "macro_observed_on": {"VIXCLS": "2026-10-20"}}

    features = collect_market_features(
        ["AAPL", "BTC"], market_port=market, repository=repository, as_of=_NOW,
        macro_features=macro, **_UNIVERSE,
    )

    assert features["AAPL"]["vix_percentile_1y"] == features["BTC"]["vix_percentile_1y"] == 93.0
    assert features["BTC"]["macro_observed_on"] == {"VIXCLS": "2026-10-20"}
