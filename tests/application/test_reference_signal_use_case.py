"""Testy use case'u sygnału referencyjnego WIA-24 (`reference-v1`).

Raz na przebieg Slow Loopa: darmowy odczyt ceny, rozliczenie zeszłotygodniowego
wywołania tym samym odczytem i nowe wywołanie albo abstynencja. Jedyne porty to
darmowa cena i repozytorium (tabela `reference_predictions`, migracja 042) —
ani `prediction_logs`, ani `price_snapshots`, ani żadnego płatnego portu.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application.ports import MarketDataPort, OpenReferenceCall, RepositoryPort
from src.application.use_cases.reference_signal import (
    ReferenceSignalSummary,
    RunReferenceSignalUseCase,
)
from src.domain.prediction import TrendDirection
from src.domain.reference_signal import ReferenceCall, ReferenceOutcome, ReferenceRead
from src.domain.value_objects import Money

SUNDAY = datetime(2026, 10, 25, 3, 0, tzinfo=UTC)
REFERENCE_METHODS = {
    "get_reference_reads",
    "get_open_reference_calls",
    "save_reference_read",
    "settle_reference_call",
}


def _market(*prices: str) -> Mock:
    market = Mock(spec=MarketDataPort)
    market.get_current_price.side_effect = [Money(Decimal(p)) for p in prices]
    return market


def _repository(
    reads: list[ReferenceRead] | None = None, open_calls: list[OpenReferenceCall] | None = None
) -> Mock:
    repository = Mock(spec=RepositoryPort)
    repository.get_reference_reads.return_value = reads or []
    repository.get_open_reference_calls.return_value = open_calls or []
    return repository


def _open_call(days_ago: float, price: str = "100", record_id: str = "7") -> OpenReferenceCall:
    return OpenReferenceCall(
        id=record_id,
        call=ReferenceCall(
            symbol="AAPL",
            trend=TrendDirection.BULLISH,
            price_at_call=Decimal(price),
            called_at=SUNDAY - timedelta(days=days_ago),
        ),
    )


def _last_week_read() -> ReferenceRead:
    return ReferenceRead(symbol="AAPL", price=Decimal("100"), read_at=SUNDAY - timedelta(days=7))


def _run(market: Mock, repository: Mock, symbols: list[str]) -> ReferenceSignalSummary:
    use_case = RunReferenceSignalUseCase(market=market, repository=repository)
    return use_case.run(symbols, now=SUNDAY)


def test_first_run_abstains_and_stores_the_read_for_next_week() -> None:
    repository = _repository()

    summary = _run(_market("103", "250"), repository, ["AAPL", "BTC"])

    assert [c.args for c in repository.save_reference_read.call_args_list] == [
        (ReferenceRead(symbol="AAPL", price=Decimal("103"), read_at=SUNDAY), None),
        (ReferenceRead(symbol="BTC", price=Decimal("250"), read_at=SUNDAY), None),
    ]
    assert summary == ReferenceSignalSummary(
        reads=2, calls=0, abstained=2, hits=0, misses=0, expired=0, failures=0, total=2
    )


def test_calls_the_sign_of_the_change_since_last_weeks_read() -> None:
    last_week = _last_week_read()
    repository = _repository(reads=[last_week])

    summary = _run(_market("103"), repository, ["AAPL"])

    read, call = repository.save_reference_read.call_args.args
    assert call == ReferenceCall(
        symbol="AAPL", trend=TrendDirection.BULLISH, price_at_call=Decimal("103"), called_at=SUNDAY
    )
    assert read.price == Decimal("103")
    assert summary.calls == 1 and summary.abstained == 0


def test_reads_the_window_of_last_week() -> None:
    repository = _repository()

    _run(_market("103"), repository, ["AAPL"])

    repository.get_reference_reads.assert_called_once_with(SUNDAY - timedelta(days=8))


def test_settles_last_weeks_call_with_this_runs_read() -> None:
    repository = _repository(open_calls=[_open_call(days_ago=7)])

    summary = _run(_market("104"), repository, ["AAPL"])

    repository.settle_reference_call.assert_called_once_with(
        "7", ReferenceOutcome.HIT, Decimal("104"), SUNDAY
    )
    assert summary.hits == 1


def test_mid_week_run_leaves_the_call_open() -> None:
    repository = _repository(open_calls=[_open_call(days_ago=3)])

    _run(_market("104"), repository, ["AAPL"])

    repository.settle_reference_call.assert_not_called()


def test_call_without_a_read_in_its_horizon_expires_without_a_price() -> None:
    repository = _repository(open_calls=[_open_call(days_ago=14)])

    summary = _run(_market("104"), repository, ["AAPL"])

    repository.settle_reference_call.assert_called_once_with(
        "7", ReferenceOutcome.EXPIRED, None, SUNDAY
    )
    assert summary.expired == 1


def test_per_symbol_failure_does_not_break_the_run() -> None:
    market = Mock(spec=MarketDataPort)
    market.get_current_price.side_effect = [RuntimeError("429"), Money(Decimal("250"))]
    repository = _repository()

    summary = _run(market, repository, ["AAPL", "BTC"])

    (read, _), = [c.args for c in repository.save_reference_read.call_args_list]
    assert read.symbol == "BTC"
    assert summary.failures == 1 and summary.total == 2


def test_failed_history_read_stops_the_run_before_any_write() -> None:
    # Bez historii każde wywołanie byłoby abstynencją, a odczyt i tak by się zapisał —
    # tydzień z dziurą zamiast jawnego błędu przebiegu.
    repository = _repository()
    repository.get_reference_reads.side_effect = RuntimeError("Supabase 503")

    with pytest.raises(RuntimeError):
        _run(_market("103"), repository, ["AAPL"])

    repository.save_reference_read.assert_not_called()


def test_never_calls_paid_ports() -> None:
    # Use case nie przyjmuje płatnego portu, a z dwóch, które dostaje, woła wyłącznie
    # darmową cenę i metody tabeli `reference_predictions`.
    market = _market("104")
    repository = _repository(open_calls=[_open_call(days_ago=7)])

    _run(market, repository, ["AAPL"])

    assert set(inspect.signature(RunReferenceSignalUseCase).parameters) == {
        "market",
        "repository",
    }
    assert {name for name, *_ in market.method_calls} == {"get_current_price"}
    assert {name for name, *_ in repository.method_calls} <= REFERENCE_METHODS


def test_never_writes_price_snapshots_or_llm_predictions() -> None:
    last_week = _last_week_read()
    repository = _repository(reads=[last_week], open_calls=[_open_call(days_ago=7)])

    _run(_market("104"), repository, ["AAPL"])

    repository.save_price_snapshot.assert_not_called()
    repository.save_prediction.assert_not_called()
    repository.update_prediction_accuracy.assert_not_called()


def test_rerun_a_day_later_stores_the_read_without_a_new_call() -> None:
    # Poprawka v1: najwyżej jedno wywołanie na symbol w 6 dni; ponowny przebieg tylko
    # rozlicza dojrzałe wywołania i zapisuje odczyt.
    monday = SUNDAY + timedelta(days=1)
    week_before = ReferenceRead(
        symbol="AAPL", price=Decimal("100"), read_at=monday - timedelta(days=7)
    )
    repository = _repository(reads=[week_before], open_calls=[_open_call(days_ago=0)])
    use_case = RunReferenceSignalUseCase(market=_market("105"), repository=repository)

    summary = use_case.run(["AAPL"], now=monday)

    read, call = repository.save_reference_read.call_args.args
    assert read == ReferenceRead(symbol="AAPL", price=Decimal("105"), read_at=monday)
    assert call is None
    repository.settle_reference_call.assert_not_called()
    assert summary.calls == 0 and summary.held == 1 and summary.reads == 1


def test_call_returns_six_days_after_the_previous_one() -> None:
    six_days_ago = SUNDAY - timedelta(days=6)
    baseline = ReferenceRead(symbol="AAPL", price=Decimal("100"), read_at=six_days_ago)
    repository = _repository(reads=[baseline], open_calls=[_open_call(days_ago=6)])

    summary = _run(_market("104"), repository, ["AAPL"])

    repository.settle_reference_call.assert_called_once_with(
        "7", ReferenceOutcome.HIT, Decimal("104"), SUNDAY
    )
    _, call = repository.save_reference_read.call_args.args
    assert call is not None and call.trend is TrendDirection.BULLISH
    assert summary.calls == 1 and summary.held == 0
