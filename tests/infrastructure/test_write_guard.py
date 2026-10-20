"""BEZ-08: strażnik zapisu — lokalny run tylko do odczytu wobec wspólnej bazy.

Lokalny run i GitHub Actions piszą do tej samej bazy Supabase, więc każdy
lokalny przebieg zanieczyszczał produkcyjny track record. Wrappery z
`write_guard` delegują odczyty, a zapisy pomijają z jednym WARNING-iem na metodę.

Strażnik klasyfikacji: każda abstrakcyjna metoda portu jest jawnie odczytem albo
zapisem. Nowa metoda portu bez klasyfikacji = czerwony test, zanim po cichu
zacznie zapisywać z lokalnego runu albo przestanie czytać.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from abc import ABC
from typing import Any
from unittest.mock import MagicMock, create_autospec, sentinel

import pytest

from src.application.ports import (
    AttestationPublisherPort,
    PositionsImportPort,
    RepositoryPort,
)
from src.infrastructure.adapters.write_guard import (
    DRY_RUN_PREDICTION_PREFIX,
    ReadOnlyAttestationPublisher,
    ReadOnlyPositionsImport,
    ReadOnlyRepository,
)

GUARDS: list[tuple[type[ABC], Any]] = [
    (RepositoryPort, ReadOnlyRepository),
    (PositionsImportPort, ReadOnlyPositionsImport),
    (AttestationPublisherPort, ReadOnlyAttestationPublisher),
]
GUARD_IDS = [guard.__name__ for _, guard in GUARDS]


def _cases(kind: str) -> list[tuple[type[ABC], Any, str]]:
    return [
        (port, guard, method)
        for port, guard in GUARDS
        for method in sorted(getattr(guard, kind))
    ]


READ_CASES = _cases("READ_METHODS")
WRITE_CASES = _cases("WRITE_METHODS")


def _ids(cases: list[tuple[type[ABC], Any, str]]) -> list[str]:
    return [f"{guard.__name__}.{method}" for _, guard, method in cases]


def _every_argument(port: type[ABC], method: str) -> dict[str, Any]:
    """Każdy parametr metody portu (także opcjonalny) jako unikalny sentinel."""
    signature = inspect.signature(getattr(port, method))
    return {name: getattr(sentinel, name) for name in list(signature.parameters)[1:]}


def _bound_arguments(port: type[ABC], method: str, call: Any) -> dict[str, Any]:
    """Argumenty wywołania delegata po związaniu z sygnaturą portu — niezależnie
    od tego, czy wrapper podał je pozycyjnie, czy po nazwie."""
    signature = inspect.signature(getattr(port, method))
    bound = signature.bind(None, *call.args, **call.kwargs)
    return {name: value for name, value in bound.arguments.items() if name != "self"}


@pytest.mark.parametrize(("port", "guard"), GUARDS, ids=GUARD_IDS)
def test_every_port_method_is_classified_exactly_once(port: type[ABC], guard: Any) -> None:
    reads, writes = guard.READ_METHODS, guard.WRITE_METHODS

    assert not reads & writes, f"metoda i odczytem, i zapisem: {sorted(reads & writes)}"
    unclassified = set(port.__abstractmethods__) - reads - writes
    assert not unclassified, (
        f"{port.__name__}: {sorted(unclassified)} bez klasyfikacji w {guard.__name__} — "
        "dopisz do READ_METHODS albo WRITE_METHODS"
    )
    assert reads | writes == set(port.__abstractmethods__)


@pytest.mark.parametrize(("port", "guard", "method"), WRITE_CASES, ids=_ids(WRITE_CASES))
def test_write_never_reaches_the_delegate(port: type[ABC], guard: Any, method: str) -> None:
    delegate = create_autospec(port, instance=True)

    getattr(guard(delegate), method)(**_every_argument(port, method))

    assert delegate.mock_calls == []


@pytest.mark.parametrize(("port", "guard", "method"), READ_CASES, ids=_ids(READ_CASES))
def test_read_is_delegated_with_every_argument(port: type[ABC], guard: Any, method: str) -> None:
    delegate = create_autospec(port, instance=True)
    arguments = _every_argument(port, method)

    result = getattr(guard(delegate), method)(**arguments)

    read = getattr(delegate, method)
    read.assert_called_once()
    assert _bound_arguments(port, method, read.call_args) == arguments
    # `probe_database` niczego nie zwraca — liczy się samo wywołanie (wyjątek = brak bazy).
    if inspect.signature(getattr(port, method)).return_annotation != "None":
        assert result is read.return_value


def test_save_prediction_returns_a_fresh_dry_run_id() -> None:
    repository = ReadOnlyRepository(create_autospec(RepositoryPort, instance=True))

    first = repository.save_prediction({"symbol": "AAPL"})
    second = repository.save_prediction({"symbol": "AAPL"})

    assert first.startswith(DRY_RUN_PREDICTION_PREFIX)
    assert DRY_RUN_PREDICTION_PREFIX == "dry-run-"
    uuid.UUID(first.removeprefix(DRY_RUN_PREDICTION_PREFIX))
    assert first != second


def test_skipped_write_warns_once_per_method(caplog: pytest.LogCaptureFixture) -> None:
    repository = ReadOnlyRepository(create_autospec(RepositoryPort, instance=True))

    with caplog.at_level(logging.DEBUG, logger="src.infrastructure.adapters.write_guard"):
        for _ in range(3):
            repository.save_price_snapshot("AAPL", MagicMock())
        repository.save_quota_alert(MagicMock())

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "save_price_snapshot" in warnings[0]
    assert "save_quota_alert" in warnings[1]
    assert all("DB_WRITES_ENABLED=false" in message for message in warnings)


def test_attestation_writes_report_not_published() -> None:
    publisher = ReadOnlyAttestationPublisher(
        create_autospec(AttestationPublisherPort, instance=True)
    )

    assert publisher.publish_commitment({"symbol": "AAPL"}) == ""
    assert publisher.publish_reveal({"symbol": "AAPL"}) == ""


def test_llm_cost_ledger_write_is_skipped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # FIN-13: ledger kosztów `llm_calls` to zapis jak każdy inny — lokalny run go pomija.
    delegate = create_autospec(RepositoryPort, instance=True)
    repository = ReadOnlyRepository(delegate)

    with caplog.at_level(logging.WARNING, logger="src.infrastructure.adapters.write_guard"):
        repository.save_llm_calls([MagicMock()])

    assert delegate.mock_calls == []
    assert "save_llm_calls" in caplog.text
    assert "save_llm_calls" in ReadOnlyRepository.WRITE_METHODS
