"""ARCH-16: odczyty okna track recordu — jeden odczyt na dispatch, ten sam błąd
dla każdej sekcji."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.application.ports import RepositoryPort
from src.application.track_record_window import TrackRecordReads

_ROW = {
    "symbol": "AAPL",
    "predicted_trend": "BULLISH",
    "is_trend_correct": True,
    "confidence_score": 0.8,
}


def _repo() -> Mock:
    repo = Mock(spec=RepositoryPort)
    repo.get_resolved_predictions.return_value = [_ROW]
    repo.get_resolved_for_calibration.return_value = [_ROW]
    return repo


def test_nothing_is_read_until_a_section_asks() -> None:
    repo = _repo()

    TrackRecordReads(repo, 30)

    repo.get_resolved_predictions.assert_not_called()
    repo.get_resolved_for_calibration.assert_not_called()


def test_every_section_shares_one_read_of_the_window() -> None:
    repo = _repo()
    reads = TrackRecordReads(repo, 30)

    reads.resolved()
    reads.resolved_rows()
    reads.resolved()
    reads.calibration()
    reads.calibration()

    repo.get_resolved_predictions.assert_called_once_with(30)
    repo.get_resolved_for_calibration.assert_called_once_with(30)
    assert [p.symbol for p in reads.resolved()] == ["AAPL"]
    assert reads.calibration().directional == [(0.8, True)]


def test_a_failed_read_fails_every_section_and_is_retried() -> None:
    repo = _repo()
    repo.get_resolved_for_calibration.side_effect = RuntimeError("supabase down")
    reads = TrackRecordReads(repo, 30)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="supabase down"):
            reads.calibration()

    assert repo.get_resolved_for_calibration.call_count == 2
    assert reads.resolved_rows() == [_ROW]
