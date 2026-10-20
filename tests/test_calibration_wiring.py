"""Wiring statystyki i kalibracji track recordu (W2-1) w entry pointach.

Osobny plik zamiast dopisywania do `test_main_entrypoints.py`: te same
mocki I/O, ale bez dotykania wspólnego, długiego modułu testów.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import main_agent
import main_trainer
from src.application.ports import LLMPort, RepositoryPort
from src.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="sk-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1", "av2"],
        supabase_url="https://test.supabase.co",
        supabase_key="anon",
        symbols=["AAPL"],
        volatility_threshold=Decimal("0.02"),
        ml_model_path="data/models/missing.ubj",
        db_writes_enabled=True,
        paid_calls_enabled=True,
    )


class TestCalibrationJudgeScoreVersion:
    """WIA-37: flaga `calibration_score_version_enabled` dociera do sędziego."""

    @staticmethod
    def _trainer(mocker: MagicMock) -> MagicMock:
        mocker.patch("src.infrastructure.adapters.supabase_repo.create_client")
        mocker.patch("src.infrastructure.llm.openai_client.OpenAI")
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "trained_successfully"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        mocker.patch("main_trainer.SupabaseRepository", return_value=MagicMock(spec=RepositoryPort))
        mocker.patch("main_trainer.build_notifier", return_value=MagicMock())
        mocker.patch("main_agent.build_llm_adapter", return_value=MagicMock(spec=LLMPort))
        judge_cls = mocker.patch(
            "src.application.use_cases.calibrate_confidence.CalibrateConfidenceUseCase"
        )
        judge_cls.return_value.run.return_value = {"status": "calibrated"}
        return judge_cls

    @pytest.mark.parametrize("enabled", [True, False])
    def test_flag_reaches_the_judge(
        self, settings: Settings, mocker: MagicMock, enabled: bool
    ) -> None:
        judge_cls = self._trainer(mocker)
        settings.calibration_judge_enabled = True
        settings.weekly_recap_enabled = False
        settings.calibration_score_version_enabled = enabled

        main_trainer.main(settings)

        assert judge_cls.call_args.kwargs["brier_score_enabled"] is enabled

    def test_flag_ships_off(self) -> None:
        assert Settings.model_fields["calibration_score_version_enabled"].default is False


_MIXED_CALIBRATION_ROWS = [
    {"confidence_score": 0.9, "is_trend_correct": True, "predicted_trend": "SIDEWAYS"},
    {"confidence_score": 0.9, "is_trend_correct": False, "predicted_trend": "BULLISH"},
    {"confidence_score": 0.6, "is_trend_correct": True, "predicted_trend": "BEARISH"},
]


class TestDirectionalCalibrationWiring:
    """WIA-39: ranking sygnałów i Track Record kalibrują krzywą kierunkową,
    a SIDEWAYS dostaje własne n i Brier."""

    @staticmethod
    def _repo() -> MagicMock:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_resolved_predictions.return_value = []
        repo.get_resolved_for_calibration.return_value = _MIXED_CALIBRATION_ROWS
        return repo

    def test_signal_ranking_bucket_ignores_sideways(self) -> None:
        settings = MagicMock()
        settings.calibrated_confidence_enabled = True
        settings.track_record_days = 30

        buckets = main_agent._build_signal_calibration(settings, self._repo())

        assert buckets is not None
        top = next(b for b in buckets if b.lower <= 0.9 < b.upper)
        assert top.count == 1
        assert top.hit_rate == 0.0

    def test_track_record_scores_split_directional_and_sideways(self) -> None:
        settings = MagicMock()
        settings.equity_curve_enabled = False
        settings.calibration_curve_enabled = True
        settings.interval_coverage_enabled = False
        settings.track_record_days = 30

        stats = main_agent._build_track_record_stats(settings, self._repo())

        assert stats is not None
        assert stats.scores is not None and stats.scores.n == 2
        assert stats.sideways_scores is not None and stats.sideways_scores.n == 1

    def test_track_record_curve_is_directional(self) -> None:
        settings = MagicMock()
        settings.equity_curve_enabled = False
        settings.lessons_enabled = False
        settings.calibration_curve_enabled = True
        settings.track_record_days = 30

        _, buckets, _ = main_agent._build_track_record(settings, self._repo())

        assert buckets is not None
        assert sum(b.count for b in buckets) == 2


class TestScorecardCalibrationCycles:
    """WIA-58: scorecard dostaje próbki kierunkowe pogrupowane po dniu UTC."""

    def test_scorecard_receives_directional_samples_by_cycle(self, mocker: MagicMock) -> None:
        settings = MagicMock()
        settings.public_scorecard_enabled = True
        settings.track_record_days = 30
        repo = MagicMock(spec=RepositoryPort)
        repo.get_resolved_predictions.return_value = []
        repo.get_resolved_for_calibration.return_value = [
            {**row, "timestamp": stamp}
            for row, stamp in zip(
                _MIXED_CALIBRATION_ROWS,
                [
                    "2026-10-19T05:30:00+00:00",
                    "2026-10-19T05:31:00+00:00",
                    "2026-10-20T05:30:00+00:00",
                ],
                strict=True,
            )
        ]
        render = mocker.patch("main_agent.render_public_scorecard_html", return_value="")

        main_agent._publish_public_scorecard(settings, repo, MagicMock())

        assert render.call_args.args[2] == [(0.9, False), (0.6, True)]
        assert render.call_args.kwargs["calibration_by_cycle"] == (
            ((0.9, False),),
            ((0.6, True),),
        )
