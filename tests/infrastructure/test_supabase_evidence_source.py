"""NIEZ-32: `SupabaseRepository` jako źródło kopii tabel dowodowych.

Paginacja po `.range()` (PostgREST tnie SELECT do ~1000 wierszy bez błędu) i
tolerancja braku migracji: kolumna, której baza nie ma (42703), wypada z selecta,
a reszta kopii powstaje.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from postgrest.exceptions import APIError

from src.application.ports import EvidenceSourcePort
from src.infrastructure.adapters.supabase_repo import SupabaseRepository


@pytest.fixture
def mock_client(mocker: Any) -> MagicMock:
    create = mocker.patch("src.infrastructure.adapters.supabase_repo.build_supabase_client")
    client = MagicMock()
    create.return_value = client
    return client


@pytest.fixture
def repo(mock_client: MagicMock) -> SupabaseRepository:
    return SupabaseRepository(url="https://test.supabase.co", key="sb_secret_test")


def _response(rows: list[dict[str, Any]]) -> MagicMock:
    response = MagicMock()
    response.data = rows
    return response


def _order(mock_client: MagicMock) -> MagicMock:
    return cast(MagicMock, mock_client.table.return_value.select.return_value.order.return_value)


def test_repository_is_an_evidence_source(repo: SupabaseRepository) -> None:
    assert isinstance(repo, EvidenceSourcePort)


def test_reads_requested_columns_in_stable_order(
    repo: SupabaseRepository, mock_client: MagicMock
) -> None:
    _order(mock_client).range.return_value.execute.return_value = _response([{"id": 1}])

    rows = repo.read_rows("council_votes", ("id", "investor_name"))

    assert rows == [{"id": 1}]
    mock_client.table.assert_called_with("council_votes")
    mock_client.table.return_value.select.assert_called_with("id, investor_name")
    mock_client.table.return_value.select.return_value.order.assert_called_with("id")


def test_paginates_past_the_postgrest_cap(repo: SupabaseRepository, mock_client: MagicMock) -> None:
    full = [{"id": i} for i in range(1000)]
    _order(mock_client).range.return_value.execute.side_effect = [
        _response(full),
        _response([{"id": 1000}]),
    ]

    rows = repo.read_rows("model_scorecards", ("id",))

    assert len(rows) == 1001


def test_column_missing_before_its_migration_is_dropped(
    repo: SupabaseRepository, mock_client: MagicMock
) -> None:
    _order(mock_client).range.return_value.execute.side_effect = [
        APIError({"code": "42703", "message": "column prediction_logs.gate_delta does not exist"}),
        _response([{"id": "p1"}]),
    ]

    rows = repo.read_rows("prediction_logs", ("id", "gate_delta", "interval_low"))

    assert rows == [{"id": "p1"}]
    selects = [c.args[0] for c in mock_client.table.return_value.select.call_args_list]
    assert selects == ["id, gate_delta, interval_low", "id, interval_low"]


def test_other_api_errors_propagate(repo: SupabaseRepository, mock_client: MagicMock) -> None:
    _order(mock_client).range.return_value.execute.side_effect = APIError(
        {"code": "57014", "message": "canceling statement due to statement timeout"}
    )

    with pytest.raises(APIError):
        repo.read_rows("prediction_logs", ("id",))
