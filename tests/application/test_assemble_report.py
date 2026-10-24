"""ARCH-06: `ReportAssembler` składa kontekst raportu z portów i `ReportFlags`.

Wiring z `Settings` (aliasy w `main_agent`) pilnują testy w
`tests/test_main_entrypoints.py`; tu use case bez composition root.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

from src.application.errors import ProviderUnavailable
from src.application.ports import RepositoryPort, StoredShockAlert, WebPublisherPort
from src.application.quota_monitor import QuotaMonitor
from src.application.report_builder import ReportContext, SymbolResult
from src.application.report_evidence import PaperCosts
from src.application.use_cases.assemble_report import ReportAssembler, ReportFlags
from src.application.use_cases.publish_public_scorecard import (
    PublicScorecardOptions,
    PublishPublicScorecardUseCase,
)
from src.domain.cycle_maturity import CycleMaturity
from src.domain.evaluation_protocol import EvaluationProtocol

STARTED_AT = datetime(2026, 10, 18, 5, 30, tzinfo=UTC)
_SETTLED = {
    "symbol": "AAPL",
    "predicted_trend": "BULLISH",
    "is_trend_correct": True,
    "confidence_score": 0.8,
    "price_at_prediction": "100",
    "actual_price_after_12h": "101",
    "timestamp": "2026-10-17T05:30:00+00:00",
}


def _repo() -> Mock:
    repo = Mock(spec=RepositoryPort)
    repo.get_accuracy_stats.return_value = None
    repo.get_recently_resolved_predictions.return_value = []
    repo.get_recent_quota_alerts.return_value = []
    repo.get_council_vote_history.return_value = []
    repo.get_persona_track_record.return_value = {}
    repo.get_resolved_predictions.return_value = [_SETTLED]
    repo.get_resolved_for_calibration.return_value = [_SETTLED]
    repo.get_price_history.return_value = []
    repo.get_settled_moves.return_value = []
    return repo


def _assemble(repo: Mock, flags: ReportFlags) -> ReportContext:
    return (
        ReportAssembler(repo, flags, PaperCosts(cost_bps=10.0))
        .assemble(
            results=[SymbolResult(symbol="AAPL", status="saved")],
            started_at=STARTED_AT,
            quota_monitor=QuotaMonitor(),
            all_symbols=["AAPL"],
            context=ReportContext(sector_map={"AAPL": "Tech"}),
        )
        .context
    )


def _protocol() -> EvaluationProtocol:
    from src.config import load_evaluation_protocol

    return load_evaluation_protocol()


def test_calibration_rows_fetched_once_per_cycle() -> None:
    repo = _repo()
    flags = ReportFlags(
        track_record_days=30,
        reflection_min_age_hours=12,
        symbols=["AAPL"],
        equity_curve_enabled=True,
        calibration_curve_enabled=True,
        calibrated_confidence_enabled=True,
    )
    costs = PaperCosts(cost_bps=10.0)

    assembled = ReportAssembler(repo, flags, costs).assemble(
        results=[],
        started_at=STARTED_AT,
        quota_monitor=QuotaMonitor(),
        all_symbols=["AAPL"],
        context=ReportContext(),
    )
    PublishPublicScorecardUseCase(
        repo,
        Mock(spec=WebPublisherPort),
        PublicScorecardOptions(
            window_days=30,
            settle_after_hours=12,
            symbols=["AAPL"],
            crypto_symbols=[],
            load_protocol=_protocol,
        ),
        reads=assembled.reads,
    ).run(costs)

    assert repo.get_resolved_for_calibration.call_count == 1
    assert repo.get_resolved_predictions.call_count == 1
    assert assembled.context.calibration_buckets is not None
    assert assembled.context.signal_calibration_buckets is not None


def test_sections_behind_flags_off_read_nothing_from_the_window() -> None:
    repo = _repo()

    context = _assemble(repo, ReportFlags(track_record_days=30, reflection_min_age_hours=12))

    repo.get_resolved_predictions.assert_not_called()
    repo.get_resolved_for_calibration.assert_not_called()
    assert context.equity_curve is None and context.track_record_stats is None
    assert context.cycle_maturity is CycleMaturity.STEADY_STATE
    assert context.portfolio_sectors == {"Tech": 1}


def test_settled_window_flag_reads_settlements_since_the_cycle_start() -> None:
    repo = _repo()

    context = _assemble(
        repo,
        ReportFlags(
            track_record_days=30, reflection_min_age_hours=12, settled_at_windows_enabled=True
        ),
    )

    assert repo.get_recently_resolved_predictions.call_args.kwargs == {"since": STARTED_AT}
    assert context.resolved_this_cycle is True


class TestShocksSinceLastMail:
    """ALR-06: szoki z watcha od ostatniego dostarczonego maila do sekcji ⚡."""

    _SHOCK = StoredShockAlert(
        symbol="NVDA",
        delta=-0.062,
        direction="DOWN",
        created_at=datetime(2026, 10, 17, 22, 7, tzinfo=UTC),
    )

    @staticmethod
    def _flags(*, enabled: bool) -> ReportFlags:
        return ReportFlags(
            track_record_days=30, reflection_min_age_hours=12, shock_alerts_enabled=enabled
        )

    def test_reads_alerts_since_the_last_delivered_mail(self) -> None:
        repo = _repo()
        last_mail = datetime(2026, 10, 17, 5, 31, tzinfo=UTC)
        repo.get_last_report_at.return_value = last_mail
        repo.get_shock_alerts_since.return_value = [self._SHOCK]

        context = _assemble(repo, self._flags(enabled=True))

        repo.get_last_report_at.assert_called_once_with(STARTED_AT)
        repo.get_shock_alerts_since.assert_called_once_with(last_mail)
        assert context.shock_alerts == [self._SHOCK]

    def test_without_a_delivered_mail_falls_back_to_the_last_day(self) -> None:
        repo = _repo()
        repo.get_last_report_at.return_value = None
        repo.get_shock_alerts_since.return_value = []

        context = _assemble(repo, self._flags(enabled=True))

        repo.get_shock_alerts_since.assert_called_once_with(STARTED_AT - timedelta(hours=24))
        assert context.shock_alerts == []

    def test_failed_read_hides_the_section(self) -> None:
        repo = _repo()
        repo.get_last_report_at.side_effect = ProviderUnavailable("cycle_runs down")
        repo.get_shock_alerts_since.side_effect = ProviderUnavailable("shock_alerts down")

        context = _assemble(repo, self._flags(enabled=True))

        repo.get_shock_alerts_since.assert_called_once_with(STARTED_AT - timedelta(hours=24))
        assert context.shock_alerts is None

    def test_paused_watch_reads_nothing(self) -> None:
        repo = _repo()

        context = _assemble(repo, self._flags(enabled=False))

        repo.get_shock_alerts_since.assert_not_called()
        repo.get_last_report_at.assert_not_called()
        assert context.shock_alerts is None
