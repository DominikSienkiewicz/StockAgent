"""Wiring `main_agent` dla uczciwego track recordu (W2-2a).

- WIA-35: chip trafności per symbol z darmowego odczytu zamkniętych predykcji
  okna track recordu, za tą samą flagą co krzywa kapitału.
- WIA-49: wygaszanie predykcji, których reflect już nigdy nie oceni, i ich
  liczba w raporcie.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

import main_agent
from src.application.ports import RepositoryPort
from src.config import Settings
from tests.settings_kwargs import NO_ENV_FILE


@pytest.fixture
def cycle_settings() -> Settings:
    # Jak smoke testy entry pointa: bez `.env`, zapis i płatne porty jak w GHA.
    return Settings(
        **NO_ENV_FILE,
        openai_api_key="sk-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1"],
        supabase_url="https://test.supabase.co",
        supabase_key="anon",
        symbols=["AAPL", "VOO"],
        volatility_threshold=Decimal("0.02"),
        ml_model_path="data/models/missing.ubj",
        db_writes_enabled=True,
        paid_calls_enabled=True,
    )


def _settings(**overrides: object) -> MagicMock:
    settings = MagicMock()
    settings.track_record_days = 30
    settings.equity_curve_enabled = True
    for name, value in overrides.items():
        setattr(settings, name, value)
    return settings


def _row(symbol: str, trend: str, correct: bool) -> dict[str, object]:
    return {"symbol": symbol, "predicted_trend": trend, "is_trend_correct": correct}


class TestSymbolTrackRecordWiring:
    def test_records_come_from_the_track_record_window(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_resolved_predictions.return_value = [
            _row("NVDA", "BULLISH", True),
            _row("NVDA", "BEARISH", False),
            _row("NVDA", "SIDEWAYS", True),
        ]

        view = main_agent._build_symbol_track_record(_settings(), repo)

        assert view is not None
        assert view.window_days == 30
        assert (view.records["NVDA"].hits, view.records["NVDA"].directional) == (1, 2)
        repo.get_resolved_predictions.assert_called_once_with(30)

    def test_flag_off_reads_nothing(self) -> None:
        repo = MagicMock(spec=RepositoryPort)

        view = main_agent._build_symbol_track_record(_settings(equity_curve_enabled=False), repo)

        assert view is None
        repo.get_resolved_predictions.assert_not_called()

    def test_read_error_leaves_the_mail_without_chips(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_resolved_predictions.side_effect = RuntimeError("supabase down")

        assert main_agent._build_symbol_track_record(_settings(), repo) is None


class TestExpireStalePredictions:
    """WIA-49: predykcje starsze niż okno oceny (`reflection_max_age_hours`)
    dostają jawną flagę wygaśnięcia, a mail podaje ich liczbę."""

    def test_every_symbol_is_expired_past_the_reflection_window(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.expire_unverified_predictions.side_effect = [2, 0, 1]

        expired = main_agent._expire_stale_predictions(
            _settings(reflection_max_age_hours=96), repo, ["AAPL", "VOO", "BTC"]
        )

        assert expired == 3
        assert [c.args for c in repo.expire_unverified_predictions.call_args_list] == [
            ("AAPL", 96),
            ("VOO", 96),
            ("BTC", 96),
        ]

    def test_without_an_age_bound_nothing_expires(self) -> None:
        # 0 wyłącza granicę wieku w reflect — każda predykcja może jeszcze zostać oceniona.
        repo = MagicMock(spec=RepositoryPort)

        expired = main_agent._expire_stale_predictions(
            _settings(reflection_max_age_hours=0), repo, ["AAPL"]
        )

        assert expired == 0
        repo.expire_unverified_predictions.assert_not_called()

    def test_one_failing_symbol_does_not_hide_the_others(self) -> None:
        repo = MagicMock(spec=RepositoryPort)
        repo.expire_unverified_predictions.side_effect = [RuntimeError("PGRST"), 4]

        expired = main_agent._expire_stale_predictions(
            _settings(reflection_max_age_hours=96), repo, ["AAPL", "VOO"]
        )

        assert expired == 4

    def test_main_passes_the_expired_count_to_the_report(
        self, cycle_settings: Settings, mocker: MockerFixture
    ) -> None:
        mocker.patch("src.infrastructure.adapters.supabase_repo.build_supabase_client")
        mocker.patch("src.infrastructure.llm.openai_client.OpenAI")
        repository = MagicMock(spec=RepositoryPort)
        repository.expire_unverified_predictions.return_value = 2
        mocker.patch("main_agent.build_repository", return_value=repository)
        use_case = MagicMock()
        use_case.run.return_value = {"status": "ignored", "delta": Decimal("0")}
        mocker.patch("main_agent.build_use_case", return_value=use_case)
        dispatch = mocker.patch("main_agent._dispatch_reports", return_value=True)

        main_agent.main(cycle_settings, clock=lambda: datetime(2026, 10, 21, 5, 30, tzinfo=UTC))

        assert dispatch.call_args.kwargs["context"].expired_unverified == 4
        assert {c.args[0] for c in repository.expire_unverified_predictions.call_args_list} == {
            "AAPL",
            "VOO",
        }
