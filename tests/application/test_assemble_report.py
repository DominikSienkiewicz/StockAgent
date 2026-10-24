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
from src.application.report_models import ResolvedPrediction
from src.application.report_track_record import TrackRecordStats
from src.application.use_cases.assemble_report import (
    ReportAssembler,
    ReportFlags,
    assess_personas,
    fetch_report_context,
)
from src.application.use_cases.publish_public_scorecard import (
    PublicScorecardOptions,
    PublishPublicScorecardUseCase,
)
from src.domain.council_redundancy import CouncilVoteSample
from src.domain.cycle_maturity import CycleMaturity
from src.domain.evaluation_protocol import EvaluationProtocol
from src.domain.persona_track_record import PersonaScoring, PersonaTrackRecord

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


def test_model_scorecard_carries_the_snapshot_model_cards() -> None:
    # ML-52 (decyzja 2026-10-25): karty ALL_FREE i ALL_LLM z tego samego odczytu
    # okna karty trafiają do maila jedną linią obok werdyktu modelu wspólnego.
    repo = _repo()
    repo.get_recent_model_scorecards.return_value = [
        {"symbol": "ALL", "status": "trained_successfully",
         "timestamp": "2026-10-17T03:35:00+00:00"},
        {"symbol": "ALL_FREE", "status": "skipped", "n_samples": 12,
         "timestamp": "2026-10-17T03:36:00+00:00"},
    ]

    context = _assemble(
        repo,
        ReportFlags(
            track_record_days=30, reflection_min_age_hours=12, model_scorecard_enabled=True
        ),
    )

    assert context.model_scorecard is not None
    assert [card.symbol for card in context.model_scorecard.snapshot_models] == ["ALL_FREE"]


class TestPersonaBrierRanking:
    """WIA-44: leaderboard po Brier Skill pewności tylko za flagą
    `persona_brier_ranking_enabled`; bez niej zero nowych odczytów."""

    @staticmethod
    def _scoring() -> dict[str, PersonaScoring]:
        return {
            "Alfa": PersonaScoring(hit_rate=0.6, votes=20, brier=0.40, log_loss=0.95),
            "Zeta": PersonaScoring(hit_rate=0.6, votes=20, brier=0.24, log_loss=0.67),
        }

    def _fetch(self, repo: Mock, *, enabled: bool) -> list[PersonaTrackRecord]:
        monitor = QuotaMonitor()
        *_, ranking = fetch_report_context(
            repo, monitor, ["AAPL"], persona_brier_ranking=enabled
        )
        return ranking

    def test_flag_orders_the_leaderboard_by_brier_skill(self) -> None:
        repo = _repo()
        repo.get_persona_scoring.return_value = self._scoring()

        ranking = self._fetch(repo, enabled=True)

        assert [r.investor_name for r in ranking] == ["Zeta", "Alfa"]
        assert ranking[0].brier == 0.24
        repo.get_persona_track_record.assert_not_called()

    def test_flag_off_never_reads_the_scoring(self) -> None:
        repo = _repo()
        repo.get_persona_track_record.return_value = {"Alfa": (0.6, 20), "Zeta": (0.6, 20)}

        ranking = self._fetch(repo, enabled=False)

        assert [r.investor_name for r in ranking] == ["Alfa", "Zeta"]
        assert all(r.brier is None for r in ranking)
        repo.get_persona_scoring.assert_not_called()

    def test_flag_without_scoring_falls_back_to_the_hit_rate_ranking(self) -> None:
        repo = _repo()
        repo.get_persona_scoring.return_value = {}
        repo.get_persona_track_record.return_value = {"Alfa": (0.6, 20), "Zeta": (0.5, 20)}

        ranking = self._fetch(repo, enabled=True)

        assert [r.investor_name for r in ranking] == ["Alfa", "Zeta"]
        assert all(r.brier is None for r in ranking)

    def test_significance_assessment_keeps_the_brier_order(self) -> None:
        from src.domain.persona_track_record import rank_scored_personas
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
        stats = TrackRecordStats(hit_rate=assess_directional_hit_rate(calls, n_boot=0))
        ranked = rank_scored_personas(self._scoring())

        assessed = assess_personas(ranked, stats)

        assert [r.investor_name for r in assessed] == ["Zeta", "Alfa"]
        assert all(r.verdict is not None for r in assessed)

    def test_the_report_flag_reaches_the_assembler(self) -> None:
        repo = _repo()
        repo.get_persona_scoring.return_value = self._scoring()
        flags = ReportFlags(
            track_record_days=30,
            reflection_min_age_hours=12,
            symbols=["AAPL"],
            persona_brier_ranking_enabled=True,
        )

        context = _assemble(repo, flags)

        assert [r.investor_name for r in context.persona_track_record or []] == ["Zeta", "Alfa"]


class TestCouncilRedundancyContext:
    """WIA-45: redundancja rady z zapisanych głosów 30 dni — tylko za flagą."""

    _CYCLE = ("BUY", "SELL", "HOLD", "BUY", "SELL", "HOLD", "BUY")

    def _samples(self) -> list[CouncilVoteSample]:
        return [
            CouncilVoteSample(f"p-{index}", name, self._CYCLE[index % len(self._CYCLE)])  # type: ignore[arg-type]
            for index in range(30)
            for name in ("Graham", "Buffett", "Marks")
        ]

    @staticmethod
    def _flags(*, enabled: bool) -> ReportFlags:
        return ReportFlags(
            track_record_days=30,
            reflection_min_age_hours=12,
            symbols=["AAPL"],
            council_redundancy_enabled=enabled,
        )

    def test_flag_puts_the_measured_redundancy_into_the_context(self) -> None:
        repo = _repo()
        repo.get_council_vote_samples.return_value = self._samples()

        context = _assemble(repo, self._flags(enabled=True))

        assert context.council_redundancy is not None
        assert context.council_redundancy.is_effectively_single_voter is True
        repo.get_council_vote_samples.assert_called_once_with(30)

    def test_flag_off_never_reads_the_votes(self) -> None:
        repo = _repo()

        context = _assemble(repo, self._flags(enabled=False))

        assert context.council_redundancy is None
        repo.get_council_vote_samples.assert_not_called()

    def test_no_votes_leave_the_context_without_redundancy(self) -> None:
        repo = _repo()
        repo.get_council_vote_samples.return_value = []

        assert _assemble(repo, self._flags(enabled=True)).council_redundancy is None


class TestExpiredCountFromTheWindow:
    """WIA-49: „N predykcji wygasło bez oceny” liczy okno track recordu, nie tylko
    wiersze oznaczone w bieżącym przebiegu."""

    @staticmethod
    def _assemble_with(repo: Mock, this_run: int) -> ReportContext:
        flags = ReportFlags(track_record_days=30, reflection_min_age_hours=12, symbols=["AAPL"])
        return (
            ReportAssembler(repo, flags, PaperCosts(cost_bps=10.0))
            .assemble(
                results=[SymbolResult(symbol="AAPL", status="saved")],
                started_at=STARTED_AT,
                quota_monitor=QuotaMonitor(),
                all_symbols=["AAPL"],
                context=ReportContext(expired_unverified=this_run),
            )
            .context
        )

    def test_window_count_replaces_the_count_of_this_run(self) -> None:
        repo = _repo()
        repo.count_expired_unverified.return_value = 7

        context = self._assemble_with(repo, this_run=2)

        assert context.expired_unverified == 7
        repo.count_expired_unverified.assert_called_once_with(30)

    def test_this_run_is_never_reported_below_what_it_just_flagged(self) -> None:
        repo = _repo()
        repo.count_expired_unverified.return_value = 0

        assert self._assemble_with(repo, this_run=3).expired_unverified == 3


class TestReceiptLineFollowsTheReceiptsFlag:
    """WIA-56: linia „stała na analogach” pod chybieniem tylko przy `receipts_enabled`."""

    _MISS = {
        "symbol": "NVDA",
        "predicted_trend": "BULLISH",
        "is_trend_correct": False,
        "price_at_prediction": "100",
        "actual_price_after_12h": "97",
        "decision_receipts": {
            "schema_version": 1,
            "effective_threshold": "0.03",
            "precedents": [{"id": "3f2a1b9c-0d4e-4f5a-9b8c-7d6e5f4a3b2c", "similarity": 0.91}],
        },
    }

    def _fetch(self, *, enabled: bool) -> list[ResolvedPrediction]:
        repo = _repo()
        repo.get_recently_resolved_predictions.return_value = [self._MISS]
        _, resolved, *_ = fetch_report_context(
            repo, QuotaMonitor(), ["NVDA"], receipts_enabled=enabled
        )
        assert resolved is not None
        return resolved

    def test_receipts_off_drops_the_receipt_from_the_mail_rows(self) -> None:
        [miss] = self._fetch(enabled=False)

        assert miss.receipt is None

    def test_receipts_on_keeps_the_receipt_line(self) -> None:
        [miss] = self._fetch(enabled=True)

        assert miss.receipt is not None

    def test_the_report_flag_reaches_the_assembler(self) -> None:
        repo = _repo()
        repo.get_recently_resolved_predictions.return_value = [self._MISS]
        flags = ReportFlags(track_record_days=30, reflection_min_age_hours=12, symbols=["NVDA"])

        context = _assemble(repo, flags)

        assert [r.receipt for r in context.resolved_predictions or []] == [None]


# -- ML-28: kalibracja izotoniczna w rankingu sygnałów (flaga) ---------------------


def _calibration_row(days_ago: int, confidence: float, correct: bool, trend: str = "BULLISH"):
    from datetime import timedelta

    return {
        "confidence_score": confidence,
        "is_trend_correct": correct,
        "predicted_trend": trend,
        "timestamp": (STARTED_AT - timedelta(days=days_ago)).isoformat(),
    }


def _overconfident_history() -> list[dict[str, object]]:
    """0.9 trafia rzadziej niż 0.6 w obu oknach (45 i 10 dni temu)."""
    rows: list[dict[str, object]] = []
    for days_ago in (45, 10):
        for confidence, hits, misses in ((0.6, 12, 8), (0.9, 11, 9)):
            rows += [_calibration_row(days_ago, confidence, True)] * hits
            rows += [_calibration_row(days_ago, confidence, False)] * misses
    return rows


def _isotonic_flags(**overrides: bool) -> ReportFlags:
    values = {"calibrated_confidence_enabled": True, "isotonic_calibration_enabled": True}
    values.update(overrides)
    return ReportFlags(track_record_days=30, reflection_min_age_hours=12, **values)


def _history_repo(history: list[dict[str, object]]) -> Mock:
    repo = _repo()
    repo.get_resolved_for_calibration.side_effect = lambda days: (
        history if days == 60 else [_SETTLED]
    )
    return repo


def test_isotonic_map_enters_the_ranking_when_it_wins_out_of_sample() -> None:
    from src.domain.isotonic_calibration import fit_isotonic

    history = _overconfident_history()
    repo = _history_repo(history)

    context = _assemble(repo, _isotonic_flags())

    older = [
        (row["confidence_score"], row["is_trend_correct"])
        for row in history
        if row["timestamp"] < (STARTED_AT.replace(day=1)).isoformat()
    ]
    assert context.signal_isotonic_calibration == fit_isotonic(older)
    assert context.signal_calibration_buckets is not None


def test_isotonic_flag_off_reads_no_extra_history() -> None:
    repo = _history_repo(_overconfident_history())

    context = _assemble(repo, _isotonic_flags(isotonic_calibration_enabled=False))

    assert [c.args for c in repo.get_resolved_for_calibration.call_args_list] == [(30,)]
    assert context.signal_isotonic_calibration is None


def test_isotonic_needs_the_calibrated_ranking() -> None:
    repo = _history_repo(_overconfident_history())

    context = _assemble(repo, _isotonic_flags(calibrated_confidence_enabled=False))

    repo.get_resolved_for_calibration.assert_not_called()
    assert context.signal_isotonic_calibration is None


def test_sideways_rows_stay_out_of_the_isotonic_history() -> None:
    # WIA-39: SIDEWAYS trafia za darmo — tylko kierunkowe próbki kalibrują ranking.
    history = [
        _calibration_row(days_ago, 0.9, True, trend="SIDEWAYS")
        for days_ago in (45, 10)
        for _ in range(40)
    ]

    context = _assemble(_history_repo(history), _isotonic_flags())

    assert context.signal_isotonic_calibration is None


def test_failed_isotonic_history_read_keeps_the_buckets(caplog) -> None:
    from src.application.errors import ProviderUnavailable

    repo = _repo()

    def read(days: int) -> list[dict[str, object]]:
        if days == 60:
            raise ProviderUnavailable("supabase down")
        return [_SETTLED]

    repo.get_resolved_for_calibration.side_effect = read

    context = _assemble(repo, _isotonic_flags())

    assert context.signal_isotonic_calibration is None
    assert context.signal_calibration_buckets is not None
    assert "isotonic" in caplog.text.lower()
