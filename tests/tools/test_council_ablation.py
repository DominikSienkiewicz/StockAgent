"""WIA-43 — raport CLI leave-one-out person rady (czysta funkcja + podkomenda)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

import main_tools
from src.domain.council_ablation import leave_one_out
from src.domain.persona_track_record import PersonaVoteOutcome
from src.tools.council_ablation import format_council_ablation

_T0 = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)


def _vote(
    prediction_id: str, investor: str, recommendation: str, confidence: float, day: int = 0
) -> PersonaVoteOutcome:
    at = _T0 + timedelta(days=day)
    return PersonaVoteOutcome(
        prediction_id=prediction_id,
        investor_name=investor,
        recommendation=recommendation,
        confidence=confidence,
        symbol="AAPL",
        price_at_prediction=Decimal("100"),
        actual_price=Decimal("110"),
        predicted_at=at,
        settled_at=at + timedelta(days=1),
    )


def _council(prediction_id: str = "p-1", day: int = 0) -> list[PersonaVoteOutcome]:
    return [
        _vote(prediction_id, "Alfa", "BUY", 0.9, day),
        _vote(prediction_id, "Beta", "BUY", 0.5, day),
        _vote(prediction_id, "Gamma", "SELL", 0.6, day),
    ]


class TestFormat:
    def test_lists_every_persona_with_deltas_and_flip_rate(self) -> None:
        text = format_council_ablation(leave_one_out(_council()), days=90)

        assert "leave-one-out" in text
        for name in ("Alfa", "Beta", "Gamma"):
            assert name in text
        assert "Δ trafności" in text
        assert "Δ Brier" in text
        assert "odwraca" in text

    def test_marks_a_removal_candidate(self) -> None:
        votes = [
            _vote("p-1", "Alfa", "BUY", 0.5),
            _vote("p-1", "Beta", "SELL", 0.9),
            _vote("p-1", "Gamma", "BUY", 0.3),
        ]

        text = format_council_ablation(leave_one_out(votes), days=90)

        beta_line = next(line for line in text.splitlines() if "Beta" in line)
        assert "kandydat do wycięcia" in beta_line

    def test_states_the_interaction_blind_spot(self) -> None:
        text = format_council_ablation(leave_one_out(_council()), days=90)

        assert "interakcje" in text

    def test_says_when_there_is_nothing_to_score(self) -> None:
        text = format_council_ablation(leave_one_out([]), days=30)

        assert "Brak rozliczonych predykcji" in text


class TestCli:
    def test_reads_the_vote_history_of_the_window(
        self, mocker: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        settings = mocker.patch("src.config.Settings.from_env").return_value
        settings.crypto_symbols = ["BTC"]
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_persona_vote_outcomes.return_value = _council()

        code = main_tools.main(["council-ablation", "--days", "45"])

        assert code == 0
        repo_cls.return_value.get_persona_vote_outcomes.assert_called_once_with(window_days=45)
        assert "Alfa" in capsys.readouterr().out

    def test_without_migration_045_says_so(
        self, mocker: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mocker.patch("src.config.Settings.from_env")
        repo_cls = mocker.patch("src.infrastructure.adapters.supabase_repo.SupabaseRepository")
        repo_cls.return_value.get_persona_vote_outcomes.side_effect = RuntimeError("PGRST202")

        code = main_tools.main(["council-ablation"])

        assert code == 1
        assert "045" in capsys.readouterr().err
