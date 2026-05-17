from datetime import UTC, datetime
from unittest.mock import Mock

from src.application.ports import FundamentalsPort, RepositoryPort
from src.domain.value_objects import Fundamentals
from src.infrastructure.adapters.cached_fundamentals import (
    CachedFundamentalsAdapter,
    NullFundamentalsAdapter,
)


def _sample_fundamentals() -> Fundamentals:
    return Fundamentals(
        trailing_pe=25.0, forward_pe=20.0, peg_ratio=1.2,
        eps_growth_yoy=0.15,
        fetched_at=datetime(2026, 5, 17, tzinfo=UTC),
    )


def test_null_adapter_returns_none() -> None:
    assert NullFundamentalsAdapter().get_fundamentals("AAPL") is None


def test_cached_returns_from_cache_when_fresh() -> None:
    repo = Mock(spec=RepositoryPort)
    delegate = Mock(spec=FundamentalsPort)
    cached = _sample_fundamentals()
    repo.get_cached_fundamentals.return_value = cached

    adapter = CachedFundamentalsAdapter(repo=repo, delegate=delegate)
    result = adapter.get_fundamentals("AAPL")

    assert result is cached
    delegate.get_fundamentals.assert_not_called()
    repo.save_fundamentals.assert_not_called()


def test_cached_falls_back_to_delegate_when_stale() -> None:
    repo = Mock(spec=RepositoryPort)
    delegate = Mock(spec=FundamentalsPort)
    repo.get_cached_fundamentals.return_value = None
    fresh = _sample_fundamentals()
    delegate.get_fundamentals.return_value = fresh

    adapter = CachedFundamentalsAdapter(repo=repo, delegate=delegate)
    result = adapter.get_fundamentals("AAPL")

    assert result is fresh
    delegate.get_fundamentals.assert_called_once_with("AAPL")
    repo.save_fundamentals.assert_called_once_with("AAPL", fresh)


def test_cached_does_not_save_when_delegate_returns_none() -> None:
    repo = Mock(spec=RepositoryPort)
    delegate = Mock(spec=FundamentalsPort)
    repo.get_cached_fundamentals.return_value = None
    delegate.get_fundamentals.return_value = None

    adapter = CachedFundamentalsAdapter(repo=repo, delegate=delegate)
    assert adapter.get_fundamentals("VOO") is None
    repo.save_fundamentals.assert_not_called()
