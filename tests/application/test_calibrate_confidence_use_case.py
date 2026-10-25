from unittest.mock import Mock

from src.application.ports import RepositoryPort
from src.application.use_cases.calibrate_confidence import CalibrateConfidenceUseCase


def _use_case(repo: Mock) -> CalibrateConfidenceUseCase:
    return CalibrateConfidenceUseCase(repository_port=repo)


def _writes(repo: Mock) -> list[str]:
    return [call[0] for call in repo.method_calls if call[0].startswith("save_")]


class TestCalibrateConfidence:
    def test_skips_when_no_resolved_predictions(self) -> None:
        repo = Mock(spec=RepositoryPort)
        repo.get_resolved_for_calibration.return_value = []

        result = _use_case(repo).run()

        assert result["status"] == "skipped"
        assert _writes(repo) == []

    def test_rows_without_a_confidence_or_an_outcome_are_not_scored(self) -> None:
        repo = Mock(spec=RepositoryPort)
        repo.get_resolved_for_calibration.return_value = [
            {"id": "p1", "confidence_score": 0.9, "is_trend_correct": False},
            {"id": "u1", "confidence_score": None, "is_trend_correct": True},
            {"id": "u2", "confidence_score": 0.7, "is_trend_correct": None},
        ]

        result = _use_case(repo).run(days=30)

        assert result["n_samples"] == 1
        assert set(repo.save_calibration_scores.call_args.args[0]) == {"p1"}

    def test_reports_the_calibration_gap(self) -> None:
        repo = Mock(spec=RepositoryPort)
        repo.get_resolved_for_calibration.return_value = [
            {"id": "p1", "confidence_score": 0.9, "is_trend_correct": False},
            {"id": "p2", "confidence_score": 0.6, "is_trend_correct": True},
        ]

        result = _use_case(repo).run(days=30)

        assert result["status"] == "calibrated"
        assert result["n_samples"] == 2
        # mean_conf 0.75 vs accuracy 0.5 → gap 0.25
        assert abs(result["calibration_gap"] - 0.25) < 1e-9
        assert "insight" not in result


class TestArithmeticScoresWithoutJudge:
    """WIA-65 (decyzja 2026-10-23, opcja b): nic nie czyta wniosku sędziego, więc
    kalibracja zostaje arytmetyczna — bez wywołania LLM i jednym zapisem zbiorczym."""

    def test_25_usable_rows_cost_no_llm_call_and_one_bulk_write(self) -> None:
        repo = Mock(spec=RepositoryPort)
        repo.get_resolved_for_calibration.return_value = [
            {"id": f"p{i}", "confidence_score": 0.8, "is_trend_correct": i % 2 == 0}
            for i in range(25)
        ]

        result = _use_case(repo).run(days=30)

        # Use case nie dostaje portu LLM: jedyny zapis to zbiorczy zapis ocen.
        assert _writes(repo) == ["save_calibration_scores"]
        scores = repo.save_calibration_scores.call_args.args[0]
        # p0: trafiona przy 0.8 → 1 − |0.8 − 1| = 0.8; p1: chybiona → 1 − 0.8 = 0.2.
        assert len(scores) == 25
        assert abs(scores["p0"] - 0.8) < 1e-9
        assert abs(scores["p1"] - 0.2) < 1e-9
        assert result["status"] == "calibrated"
        assert result["n_samples"] == 25


class TestBrierCalibrationScore:
    """WIA-37: z flagą wersji zapis to właściwa reguła Briera z wersją 2; bez flagi
    reguła liniowa z wersją 1 (kolumna z 044 jest przed RPC z 077)."""

    @staticmethod
    def _repo() -> Mock:
        repo = Mock(spec=RepositoryPort)
        repo.get_resolved_for_calibration.return_value = [
            {"id": "p1", "confidence_score": 0.9, "is_trend_correct": False},
            {"id": "p2", "confidence_score": 0.8, "is_trend_correct": True},
        ]
        return repo

    def test_persists_brier_based_score_with_version_two(self) -> None:
        repo = self._repo()

        CalibrateConfidenceUseCase(repository_port=repo, brier_score_enabled=True).run()

        scores = repo.save_calibration_scores.call_args.args[0]
        # p1: 1 − (0.9 − 0)² = 0.19; p2: 1 − (0.8 − 1)² = 0.96.
        assert abs(scores["p1"] - 0.19) < 1e-9
        assert abs(scores["p2"] - 0.96) < 1e-9
        assert repo.save_calibration_scores.call_args.kwargs == {"score_version": 2}

    def test_flag_off_keeps_the_linear_score_with_version_one(self) -> None:
        repo = self._repo()

        _use_case(repo).run()

        scores = repo.save_calibration_scores.call_args.args[0]
        assert abs(scores["p1"] - 0.1) < 1e-9
        assert repo.save_calibration_scores.call_args.kwargs == {"score_version": 1}
