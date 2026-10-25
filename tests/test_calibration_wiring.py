"""Wiring statystyki i kalibracji track recordu (W2-1) w entry pointach.

Osobny plik zamiast dopisywania do `test_main_agent.py`: te same
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
from tests.settings_kwargs import NO_ENV_FILE


@pytest.fixture
def settings() -> Settings:
    return Settings(
        **NO_ENV_FILE,
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
        judge_cls: MagicMock = mocker.patch(
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


class TestCalibrationWithoutLlm:
    """WIA-65 (decyzja 2026-10-23, opcja b): krok kalibracji Slow Loopa nie płaci
    za LLM — liczy arytmetyczny `calibration_score` i zapisuje go jednym zapisem."""

    def test_calibration_step_makes_no_llm_call(
        self, settings: Settings, mocker: MagicMock
    ) -> None:
        mocker.patch("src.infrastructure.adapters.supabase_repo.create_client")
        mocker.patch("src.infrastructure.llm.openai_client.OpenAI")
        fake_uc = MagicMock()
        fake_uc.run.return_value = {"status": "trained_successfully"}
        mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
        repo = MagicMock(spec=RepositoryPort)
        repo.get_resolved_for_calibration.return_value = [
            {"id": f"p{i}", "confidence_score": 0.7, "is_trend_correct": i % 3 == 0}
            for i in range(25)
        ]
        mocker.patch("main_trainer.SupabaseRepository", return_value=repo)
        provider = MagicMock(spec=LLMPort)
        mocker.patch("main_agent.build_llm_adapter", return_value=provider)
        settings.calibration_judge_enabled = True
        settings.weekly_recap_enabled = False

        main_trainer.main(settings)

        assert provider.analyze_mistake.call_count == 0
        calibration_writes = [
            call[0] for call in repo.method_calls if call[0].startswith("save_calibration")
        ]
        assert calibration_writes == ["save_calibration_scores"]


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
        render = mocker.patch(
            "src.application.use_cases.publish_public_scorecard.render_public_scorecard_html",
            return_value="",
        )

        main_agent._publish_public_scorecard(settings, repo, MagicMock())

        assert render.call_args.args[2] == [(0.9, False), (0.6, True)]
        assert render.call_args.kwargs["calibration_by_cycle"] == (
            ((0.9, False),),
            ((0.6, True),),
        )


def _rows_by_day(days: int, per_day: int, hit: object) -> list[dict[str, object]]:
    return [
        {
            "confidence_score": 0.6 + 0.3 * (i % 2),
            "is_trend_correct": hit(day, i),  # type: ignore[operator]
            "predicted_trend": "BULLISH",
            "timestamp": f"2026-09-{day + 1:02d}T05:30:00+00:00",
        }
        for day in range(days)
        for i in range(per_day)
    ]


class TestUninformativeConfidenceGate:
    """WIA-38: AUC < 0.55 przy n_eff ≥ 50 → skalibrowana pewność nie przestawia
    rankingu (surowa pewność, jak bez historii)."""

    @staticmethod
    def _settings() -> MagicMock:
        settings = MagicMock()
        settings.calibrated_confidence_enabled = True
        settings.track_record_days = 30
        return settings

    def test_uninformative_confidence_leaves_the_ranking_raw(self) -> None:
        # Trafienie zależy od dnia, nie od pewności: AUC = 0.5 na 60 cyklach.
        repo = MagicMock(spec=RepositoryPort)
        repo.get_resolved_for_calibration.return_value = _rows_by_day(
            30, 4, lambda day, i: (day + i // 2) % 2 == 0
        ) + _rows_by_day(30, 4, lambda day, i: (day + i // 2) % 2 == 1)

        assert main_agent._build_signal_calibration(self._settings(), repo) is None

    def test_informative_confidence_keeps_the_calibration(self) -> None:
        # Pewność 0.9 trafia, 0.6 pudłuje: AUC = 1.
        repo = MagicMock(spec=RepositoryPort)
        repo.get_resolved_for_calibration.return_value = _rows_by_day(
            60, 2, lambda day, i: i % 2 == 1
        )

        assert main_agent._build_signal_calibration(self._settings(), repo)


class TestPersonaLeaderboardWiring:
    """WIA-30: ranking person po dolnej granicy Wilsona; WIA-22: werdykt na
    baseline'ie i n_eff z okna scorecardu."""

    @staticmethod
    def _repo(stats: dict[str, tuple[float, int]]) -> MagicMock:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_accuracy_stats.return_value = None
        repo.get_recently_resolved_predictions.return_value = []
        repo.get_recent_quota_alerts.return_value = []
        repo.get_council_vote_history.return_value = []
        repo.get_persona_track_record.return_value = stats
        return repo

    def test_report_context_ranks_personas_by_the_wilson_lower_bound(self) -> None:
        monitor = MagicMock()
        monitor.alerts = []
        repo = self._repo({"A": (1.0, 5), "B": (0.8, 50)})

        *_, ranking = main_agent._fetch_report_context(repo, monitor, ["AAPL"])

        assert [r.investor_name for r in ranking] == ["B", "A"]

    @staticmethod
    def _stats() -> object:
        from src.application.report_track_record import TrackRecordStats
        from src.domain.track_record import ScoredCall, assess_directional_hit_rate

        calls = [
            ScoredCall(
                cycle=f"2026-09-{day + 1:02d}",
                predicted_trend="BULLISH",
                is_correct=(day + i) % 2 == 0,
                realized_change=0.01 if (day + i) % 2 == 0 else -0.01,
            )
            for day in range(20)
            for i in range(2)
        ]
        return TrackRecordStats(hit_rate=assess_directional_hit_rate(calls, n_boot=0))

    def test_personas_get_a_verdict_from_the_track_record_window(self) -> None:
        from src.domain.persona_track_record import rank_personas

        records = rank_personas({"Buffett": (0.9, 60), "Wood": (0.5, 60)}, rank_by="wilson")

        assessed = main_agent._assess_personas(records, self._stats())  # type: ignore[arg-type]

        assert [r.investor_name for r in assessed] == ["Buffett", "Wood"]
        assert all(r.verdict is not None for r in assessed)
        assert assessed[0].significant is True

    def test_without_track_record_evidence_personas_stay_unassessed(self) -> None:
        from src.domain.persona_track_record import rank_personas

        records = rank_personas({"Buffett": (0.9, 60)})

        assert main_agent._assess_personas(records, None) == records


class TestIsotonicCalibrationFlag:
    """ML-28: flaga `isotonic_calibration_enabled` dociera do flag raportu."""

    @pytest.mark.parametrize("enabled", [True, False])
    def test_flag_reaches_the_report_flags(self, settings: Settings, enabled: bool) -> None:
        settings = settings.model_copy(update={"isotonic_calibration_enabled": enabled})

        assert main_agent._report_flags(settings).isotonic_calibration_enabled is enabled


class TestIsotonicClassCalibrationFlag:
    """ML-54: flaga `isotonic_class_calibration_enabled` dociera do flag raportu
    razem z listą krypto, z której assembler wyznacza klasę aktywa."""

    @pytest.mark.parametrize("enabled", [True, False])
    def test_flag_reaches_the_report_flags(self, settings: Settings, enabled: bool) -> None:
        settings = settings.model_copy(
            update={"isotonic_class_calibration_enabled": enabled, "crypto_symbols": ["BTC"]}
        )

        flags = main_agent._report_flags(settings)

        assert flags.isotonic_class_calibration_enabled is enabled
        assert list(flags.crypto_symbols) == ["BTC"]
