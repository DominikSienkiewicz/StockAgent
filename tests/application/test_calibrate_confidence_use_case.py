from unittest.mock import Mock

from src.application.ports import LLMPort, RepositoryPort
from src.application.use_cases.calibrate_confidence import CalibrateConfidenceUseCase


def _use_case(repo: Mock, llm: Mock) -> CalibrateConfidenceUseCase:
    return CalibrateConfidenceUseCase(repository_port=repo, llm_port=llm)


class TestCalibrateConfidence:
    def test_skips_when_no_resolved_predictions(self):
        repo = Mock(spec=RepositoryPort)
        repo.get_resolved_for_calibration.return_value = []
        llm = Mock(spec=LLMPort)

        result = _use_case(repo, llm).run()

        assert result["status"] == "skipped"
        llm.analyze_mistake.assert_not_called()
        repo.save_calibration.assert_not_called()

    def test_persists_per_prediction_and_one_judge_call(self):
        repo = Mock(spec=RepositoryPort)
        repo.get_resolved_for_calibration.return_value = [
            {"id": "p1", "confidence_score": 0.9, "is_trend_correct": False},
            {"id": "p2", "confidence_score": 0.6, "is_trend_correct": True},
        ]
        llm = Mock(spec=LLMPort)
        llm.analyze_mistake.return_value = "Przepewna na spadkach."

        result = _use_case(repo, llm).run(days=30)

        assert result["status"] == "calibrated"
        assert result["n_samples"] == 2
        # mean_conf 0.75 vs accuracy 0.5 → gap 0.25
        assert abs(result["calibration_gap"] - 0.25) < 1e-9
        # JEDNO wywołanie sędziego, zapis per predykcja
        llm.analyze_mistake.assert_called_once()
        assert repo.save_calibration.call_count == 2
        # p1: confidence 0.9, błędna → score 1-0.9 = 0.1
        first = repo.save_calibration.call_args_list[0]
        assert first.args[0] == "p1"
        assert abs(first.args[1] - 0.1) < 1e-9

    def test_judge_failure_still_persists_scores(self):
        repo = Mock(spec=RepositoryPort)
        repo.get_resolved_for_calibration.return_value = [
            {"id": "p1", "confidence_score": 0.8, "is_trend_correct": True},
        ]
        llm = Mock(spec=LLMPort)
        llm.analyze_mistake.side_effect = RuntimeError("LLM down")

        result = _use_case(repo, llm).run()

        assert result["status"] == "calibrated"
        repo.save_calibration.assert_called_once()
        assert result["insight"] == ""


class TestBrierCalibrationScore:
    """WIA-37: z flagą wersji sędzia zapisuje właściwą regułę Briera razem z jej
    wersją; bez flagi zapis jest bajt w bajt jak dotąd (v1, bez nowego klucza)."""

    @staticmethod
    def _repo() -> Mock:
        repo = Mock(spec=RepositoryPort)
        repo.get_resolved_for_calibration.return_value = [
            {"id": "p1", "confidence_score": 0.9, "is_trend_correct": False},
            {"id": "p2", "confidence_score": 0.8, "is_trend_correct": True},
        ]
        return repo

    def test_judge_persists_brier_based_score(self):
        repo = self._repo()
        llm = Mock(spec=LLMPort)
        llm.analyze_mistake.return_value = "Przepewna."

        CalibrateConfidenceUseCase(
            repository_port=repo, llm_port=llm, brier_score_enabled=True
        ).run()

        first, second = repo.save_calibration.call_args_list
        # p1: 1 − (0.9 − 0)² = 0.19; p2: 1 − (0.8 − 1)² = 0.96.
        assert first.args[0] == "p1"
        assert abs(first.args[1] - 0.19) < 1e-9
        assert abs(second.args[1] - 0.96) < 1e-9
        assert first.kwargs == {"score_version": 2}

    def test_flag_off_keeps_the_linear_score_without_a_version(self):
        repo = self._repo()
        llm = Mock(spec=LLMPort)
        llm.analyze_mistake.return_value = "Przepewna."

        _use_case(repo, llm).run()

        first = repo.save_calibration.call_args_list[0]
        assert abs(first.args[1] - 0.1) < 1e-9
        assert first.kwargs == {}
