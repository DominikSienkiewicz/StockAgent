"""BEZ-08 — strażnik zapisu: lokalny run tylko do odczytu wobec wspólnej bazy.

Lokalny run i GitHub Actions używają TEJ SAMEJ bazy Supabase. Bez strażnika każdy
lokalny przebieg dopisywał snapshoty, nadpisywał predykcje z tej samej godziny
(upsert po `symbol,timestamp_hour`) i rozliczał cudze predykcje — produkcyjny
track record mieszał się z eksperymentami z laptopa.

Wrappery z tego modułu owijają porty zapisu, gdy `Settings.db_writes_enabled`
jest false (domyślnie; workflowy ustawiają `DB_WRITES_ENABLED: "true"`):
odczyty idą do delegata bez zmian, a zapisy są pomijane z jednym WARNING-iem na
metodę. Każda abstrakcyjna metoda portu jest jawnie sklasyfikowana
w `READ_METHODS` albo `WRITE_METHODS` — test strażnik pilnuje, że nowa metoda
portu nie przejdzie bez klasyfikacji.

Wrappery składa wyłącznie DI (`main_agent.build_repository`,
`main_trainer.build_repository`, fabryki publishera i importu pozycji).
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any, ClassVar

from src.application.ports import (
    AttestationPublisherPort,
    CycleRun,
    LatestPrediction,
    OpenReferenceCall,
    PositionsImportPort,
    PreviousVerdict,
    RepositoryPort,
    StoredReceipt,
)
from src.application.report_council_history import InvestorHistory
from src.domain.council import InvestorOpinion
from src.domain.earnings import CachedEarningsDate
from src.domain.finops import LLMCallEntry
from src.domain.mistake_taxonomy import MistakeCategory
from src.domain.persona_track_record import PersonaVoteOutcome
from src.domain.portfolio import Position
from src.domain.prediction import Prediction
from src.domain.quota import QuotaAlert
from src.domain.reference_signal import ReferenceCall, ReferenceOutcome, ReferenceRead
from src.domain.regime import MarketRegime, RegimeState
from src.domain.value_objects import AssetType, Fundamentals, Money

logger = logging.getLogger(__name__)

DRY_RUN_PREDICTION_PREFIX = "dry-run-"


class _SkippedWrites:
    """Jeden WARNING na metodę zapisu na instancję strażnika. Fast Loop robi
    zapis per symbol (kilkadziesiąt na cykl), więc kolejne pominięcia idą na
    DEBUG — inaczej log tonie w powtórzeniach."""

    def __init__(self, port: str) -> None:
        self._port = port
        self._warned: set[str] = set()
        self._lock = threading.Lock()

    def skip(self, method: str) -> None:
        with self._lock:
            first = method not in self._warned
            self._warned.add(method)
        if first:
            logger.warning(
                "DB_WRITES_ENABLED=false — %s.%s pominięty (run tylko do odczytu; "
                "kolejne wywołania tej metody bez WARNING-u)",
                self._port,
                method,
            )
        else:
            logger.debug("DB_WRITES_ENABLED=false — %s.%s pominięty", self._port, method)


class ReadOnlyRepository(RepositoryPort):
    """`RepositoryPort` tylko do odczytu: odczyty delegowane, zapisy pomijane.

    `save_prediction` zwraca `dry-run-<uuid>`, żeby kroki zależne od id (głosy
    rady, commitment) dostały wartość, której nie da się pomylić z wierszem bazy.
    `refresh_feature_store` też jest zapisem: odświeża widok we wspólnej bazie."""

    READ_METHODS: ClassVar[frozenset[str]] = frozenset({
        "get_last_price",
        "get_price_history",
        "get_last_prediction_price",
        "get_latest_prediction",
        "get_latest_regime_state",
        "get_unverified_prediction",
        "get_feature_store_data",
        "get_accuracy_stats",
        "get_recently_resolved_predictions",
        "get_resolved_predictions",
        "get_cached_fundamentals",
        "get_fundamentals_fetched_at",
        "get_cached_earnings",
        "get_recent_quota_alerts",
        "find_similar_predictions",
        "get_council_votes_for_prediction",
        "get_receipt",
        "find_receipts",
        "get_resolved_for_calibration",
        "get_persona_accuracy",
        "get_persona_track_record",
        "get_persona_vote_outcomes",
        "get_resolved_predictions_detailed",
        "get_settled_moves",
        "get_unrevealed_commitments",
        "attestation_columns_available",
        "get_current_hour_commitment_salt",
        "get_latest_snapshot_at",
        "get_last_scorecard_at",
        "probe_database",
        "missing_prediction_columns",
        "get_last_price_snapshot",
        "get_sent_shock_alerts",
        "get_shock_alert_times",
        "get_previous_verdict",
        "get_recent_model_scorecards",
        "get_council_vote_history",
        "get_reference_reads",
        "get_open_reference_calls",
    })
    WRITE_METHODS: ClassVar[frozenset[str]] = frozenset({
        "save_price_snapshot",
        "expire_unverified_predictions",
        "save_feature_snapshot",
        "save_regime_state",
        "save_prediction",
        "update_prediction_accuracy",
        "refresh_feature_store",
        "save_fundamentals",
        "save_earnings",
        "save_quota_alert",
        "save_calibration",
        "mark_commitment_revealed",
        "save_shock_alert",
        "save_model_scorecard",
        "save_council_votes",
        "save_llm_calls",
        "save_cycle_run",
        "save_reference_read",
        "settle_reference_call",
    })

    def __init__(self, delegate: RepositoryPort) -> None:
        self._delegate = delegate
        self._skipped = _SkippedWrites(type(delegate).__name__)

    # ----- odczyty -----

    def get_last_price(self, symbol: str) -> Money | None:
        return self._delegate.get_last_price(symbol)

    def get_price_history(self, symbol: str, days: int) -> list[tuple[datetime, Money]]:
        return self._delegate.get_price_history(symbol, days)

    def get_last_prediction_price(
        self, symbol: str, *, before: datetime | None = None
    ) -> Money | None:
        return self._delegate.get_last_prediction_price(symbol, before=before)

    def get_latest_prediction(self, symbol: str) -> LatestPrediction | None:
        return self._delegate.get_latest_prediction(symbol)

    def get_latest_regime_state(self) -> RegimeState | None:
        return self._delegate.get_latest_regime_state()

    def get_unverified_prediction(
        self, symbol: str, min_age_hours: int = 0, max_age_hours: int | None = None
    ) -> Prediction | None:
        return self._delegate.get_unverified_prediction(
            symbol, min_age_hours=min_age_hours, max_age_hours=max_age_hours
        )

    def get_feature_store_data(
        self, symbol: str, feature_set_version: int | None = None
    ) -> list[dict[str, Any]]:
        return self._delegate.get_feature_store_data(
            symbol, feature_set_version=feature_set_version
        )

    def get_accuracy_stats(self, days: int) -> dict[str, Any]:
        return self._delegate.get_accuracy_stats(days)

    def get_recently_resolved_predictions(
        self, hours: int, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        return self._delegate.get_recently_resolved_predictions(hours, since=since)

    def get_resolved_predictions(self, days: int) -> list[dict[str, Any]]:
        return self._delegate.get_resolved_predictions(days)

    def get_cached_fundamentals(self, symbol: str) -> Fundamentals | None:
        return self._delegate.get_cached_fundamentals(symbol)

    def get_fundamentals_fetched_at(self, symbols: Sequence[str]) -> dict[str, datetime]:
        return self._delegate.get_fundamentals_fetched_at(symbols)

    def get_cached_earnings(self, symbol: str) -> CachedEarningsDate | None:
        return self._delegate.get_cached_earnings(symbol)

    def get_recent_quota_alerts(self, hours: int) -> list[QuotaAlert]:
        return self._delegate.get_recent_quota_alerts(hours)

    def find_similar_predictions(
        self, embedding: list[float], limit: int = 3, regime: str | None = None
    ) -> list[dict[str, Any]]:
        return self._delegate.find_similar_predictions(embedding, limit=limit, regime=regime)

    def get_council_votes_for_prediction(self, prediction_id: str) -> list[InvestorOpinion]:
        return self._delegate.get_council_votes_for_prediction(prediction_id)

    def get_receipt(self, prediction_id: str) -> StoredReceipt | None:
        return self._delegate.get_receipt(prediction_id)

    def find_receipts(self, symbol: str, day: date) -> list[StoredReceipt]:
        return self._delegate.find_receipts(symbol, day)

    def get_resolved_for_calibration(self, days: int) -> list[dict[str, Any]]:
        return self._delegate.get_resolved_for_calibration(days)

    def get_persona_accuracy(
        self, asset_type: AssetType, window_days: int = 90
    ) -> dict[str, float]:
        return self._delegate.get_persona_accuracy(asset_type, window_days=window_days)

    def get_persona_track_record(self, window_days: int = 90) -> dict[str, tuple[float, int]]:
        return self._delegate.get_persona_track_record(window_days=window_days)

    def get_persona_vote_outcomes(self, window_days: int = 90) -> list[PersonaVoteOutcome]:
        return self._delegate.get_persona_vote_outcomes(window_days=window_days)

    def get_resolved_predictions_detailed(
        self, days: int, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        return self._delegate.get_resolved_predictions_detailed(days, since=since)

    def get_settled_moves(
        self, days: int, *, with_intervals: bool = False
    ) -> list[dict[str, Any]]:
        return self._delegate.get_settled_moves(days, with_intervals=with_intervals)

    def get_unrevealed_commitments(self, days: int | None = None) -> list[dict[str, Any]]:
        return self._delegate.get_unrevealed_commitments(days=days)

    def attestation_columns_available(self) -> bool:
        return self._delegate.attestation_columns_available()

    def get_current_hour_commitment_salt(self, symbol: str) -> str | None:
        return self._delegate.get_current_hour_commitment_salt(symbol)

    def get_latest_snapshot_at(self) -> datetime | None:
        return self._delegate.get_latest_snapshot_at()

    def get_last_scorecard_at(self) -> datetime | None:
        return self._delegate.get_last_scorecard_at()

    def probe_database(self) -> None:
        self._delegate.probe_database()

    def missing_prediction_columns(self, columns: frozenset[str]) -> frozenset[str]:
        return self._delegate.missing_prediction_columns(columns)

    def get_last_price_snapshot(self, symbol: str) -> tuple[Money, datetime] | None:
        return self._delegate.get_last_price_snapshot(symbol)

    def get_sent_shock_alerts(self, since: date) -> set[tuple[str, date]]:
        return self._delegate.get_sent_shock_alerts(since)

    def get_shock_alert_times(self, since: datetime) -> dict[str, datetime]:
        return self._delegate.get_shock_alert_times(since)

    def get_previous_verdict(self, symbol: str) -> PreviousVerdict | None:
        return self._delegate.get_previous_verdict(symbol)

    def get_recent_model_scorecards(self, days: int = 30) -> list[dict[str, Any]]:
        return self._delegate.get_recent_model_scorecards(days=days)

    def get_council_vote_history(
        self,
        symbols: list[str],
        *,
        window_days: int = 30,
        investors: list[str] | None = None,
    ) -> list[InvestorHistory]:
        return self._delegate.get_council_vote_history(
            symbols, window_days=window_days, investors=investors
        )

    def get_reference_reads(self, since: datetime) -> list[ReferenceRead]:
        return self._delegate.get_reference_reads(since)

    def get_open_reference_calls(self) -> list[OpenReferenceCall]:
        return self._delegate.get_open_reference_calls()

    # ----- zapisy (pomijane) -----

    def save_price_snapshot(self, symbol: str, price: Money) -> None:
        self._skipped.skip("save_price_snapshot")

    def save_feature_snapshot(
        self,
        symbol: str,
        features: Mapping[str, Any],
        *,
        gated: bool,
        feature_set_version: int,
    ) -> None:
        self._skipped.skip("save_feature_snapshot")

    def save_regime_state(
        self, cycle_at: datetime, raw: MarketRegime, state: RegimeState
    ) -> None:
        self._skipped.skip("save_regime_state")

    def save_prediction(self, prediction: dict[str, Any]) -> str:
        self._skipped.skip("save_prediction")
        return f"{DRY_RUN_PREDICTION_PREFIX}{uuid.uuid4()}"

    def expire_unverified_predictions(self, symbol: str, older_than_hours: int) -> int:
        self._skipped.skip("expire_unverified_predictions")
        return 0

    def update_prediction_accuracy(
        self,
        prediction_id: str,
        actual_price: Decimal,
        accuracy_score: float,
        is_trend_correct: bool,
        insight: str,
        correction_category: MistakeCategory | None = None,
    ) -> None:
        self._skipped.skip("update_prediction_accuracy")

    def refresh_feature_store(self) -> None:
        self._skipped.skip("refresh_feature_store")

    def save_fundamentals(self, symbol: str, fundamentals: Fundamentals) -> None:
        self._skipped.skip("save_fundamentals")

    def save_earnings(self, entry: CachedEarningsDate) -> None:
        self._skipped.skip("save_earnings")

    def save_quota_alert(self, alert: QuotaAlert) -> None:
        self._skipped.skip("save_quota_alert")

    def save_calibration(
        self,
        prediction_id: str,
        score: float,
        insight: str,
        *,
        score_version: int | None = None,
    ) -> None:
        self._skipped.skip("save_calibration")

    def mark_commitment_revealed(self, prediction_id: str) -> None:
        self._skipped.skip("mark_commitment_revealed")

    def save_shock_alert(
        self, symbol: str, alert_date: date, delta: float, direction: str
    ) -> None:
        self._skipped.skip("save_shock_alert")

    def save_model_scorecard(
        self, symbol: str, result: Mapping[str, Any], *, run_id: str | None = None
    ) -> None:
        self._skipped.skip("save_model_scorecard")

    def save_council_votes(
        self, prediction_id: str, symbol: str, votes: list[InvestorOpinion]
    ) -> None:
        self._skipped.skip("save_council_votes")

    def save_llm_calls(self, calls: Sequence[LLMCallEntry]) -> None:
        self._skipped.skip("save_llm_calls")

    def save_cycle_run(self, run: CycleRun) -> None:
        self._skipped.skip("save_cycle_run")

    def save_reference_read(self, read: ReferenceRead, call: ReferenceCall | None) -> None:
        self._skipped.skip("save_reference_read")

    def settle_reference_call(
        self,
        record_id: str,
        outcome: ReferenceOutcome,
        settled_price: Decimal | None,
        settled_at: datetime,
    ) -> None:
        self._skipped.skip("settle_reference_call")


class ReadOnlyPositionsImport(PositionsImportPort):
    """Import pozycji tylko do odczytu: diff z tabelą działa, zamiana tabeli nie."""

    READ_METHODS: ClassVar[frozenset[str]] = frozenset({"current_positions"})
    WRITE_METHODS: ClassVar[frozenset[str]] = frozenset({"replace_positions"})

    def __init__(self, delegate: PositionsImportPort) -> None:
        self._delegate = delegate
        self._skipped = _SkippedWrites(type(delegate).__name__)

    def current_positions(self) -> list[Position]:
        return self._delegate.current_positions()

    def replace_positions(self, positions: Sequence[Position], as_of: datetime) -> None:
        self._skipped.skip("replace_positions")


class ReadOnlyAttestationPublisher(AttestationPublisherPort):
    """Publikacja atestacji pomijana: commitment predykcji `dry-run-…`, której
    nie ma w bazie, byłby fałszywym wpisem w publicznym logu. "" = nie
    opublikowano (kontrakt portu), więc sweep nie stempluje `revealed_at`."""

    READ_METHODS: ClassVar[frozenset[str]] = frozenset()
    WRITE_METHODS: ClassVar[frozenset[str]] = frozenset({"publish_commitment", "publish_reveal"})

    def __init__(self, delegate: AttestationPublisherPort) -> None:
        self._skipped = _SkippedWrites(type(delegate).__name__)

    def publish_commitment(self, record: Mapping[str, Any]) -> str:
        self._skipped.skip("publish_commitment")
        return ""

    def publish_reveal(self, record: Mapping[str, Any]) -> str:
        self._skipped.skip("publish_reveal")
        return ""
