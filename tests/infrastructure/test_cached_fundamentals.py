import threading
import time
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


class TestSingleFlight:
    """Per-symbol single-flight: dwa nakładające się cykle dla tego samego
    symbolu nie powinny obie spalić płatnego requestu AV — drugi wołający
    dostaje świeżo wypełniony cache zamiast ponownie iść do delegata."""

    def test_concurrent_same_symbol_calls_delegate_once(self) -> None:
        repo = Mock(spec=RepositoryPort)
        delegate = Mock(spec=FundamentalsPort)
        fresh = _sample_fundamentals()

        # Cache pusty na starcie; po pierwszym save'ie zwraca już dane —
        # symuluje read-through repo widziane przez drugiego wołającego.
        cache_state: dict[str, Fundamentals] = {}
        repo.get_cached_fundamentals.side_effect = lambda sym: cache_state.get(sym)
        repo.save_fundamentals.side_effect = (
            lambda sym, val: cache_state.__setitem__(sym, val)
        )

        # Wolny delegat → okno na wyścig: drugi wątek wchodzi w czasie
        # gdy pierwszy jeszcze „płaci” za request.
        def slow_fetch(sym: str) -> Fundamentals:
            time.sleep(0.05)
            return fresh

        delegate.get_fundamentals.side_effect = slow_fetch

        adapter = CachedFundamentalsAdapter(repo=repo, delegate=delegate)
        results: list[Fundamentals | None] = []

        def worker() -> None:
            results.append(adapter.get_fundamentals("AAPL"))

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Płatny delegat ruszył DOKŁADNIE raz — drugi wątek dostał cache.
        assert delegate.get_fundamentals.call_count == 1
        assert results == [fresh, fresh]

    def test_different_symbols_not_serialized(self) -> None:
        # Lock jest per-symbol: AAPL i MSFT nie blokują się nawzajem.
        repo = Mock(spec=RepositoryPort)
        delegate = Mock(spec=FundamentalsPort)
        repo.get_cached_fundamentals.return_value = None
        fresh = _sample_fundamentals()

        in_flight = threading.Event()
        proceed = threading.Event()
        seen_overlap = threading.Event()

        def fetch(sym: str) -> Fundamentals:
            if sym == "AAPL":
                in_flight.set()          # AAPL trzyma swój lock
                proceed.wait(timeout=1)  # i czeka
            else:
                # MSFT wejdzie tylko jeśli NIE jest serializowany za AAPL.
                if in_flight.wait(timeout=1):
                    seen_overlap.set()
                proceed.set()            # zwalnia AAPL
            return fresh

        delegate.get_fundamentals.side_effect = fetch
        adapter = CachedFundamentalsAdapter(repo=repo, delegate=delegate)

        t_aapl = threading.Thread(target=lambda: adapter.get_fundamentals("AAPL"))
        t_msft = threading.Thread(target=lambda: adapter.get_fundamentals("MSFT"))
        t_aapl.start()
        t_msft.start()
        t_aapl.join()
        t_msft.join()

        # MSFT pracował RÓWNOLEGLE z AAPL — różne symbole się nie blokują.
        assert seen_overlap.is_set()
