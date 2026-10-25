"""DX-04: `RepositoryPort` na słownikach — baza trybu offline i testów wielocyklowych.

Fake podmienia warstwę zapytań, nie parsowanie: wiersze żyją w kształcie,
w jakim zwraca je PostgREST (JSON po serializacji `_serialize_record`, chwile
w ISO 8601), a na obiekty domeny zamieniają je te same funkcje co w adapterze
Supabase. Semantyka, na której stoją graf i raport:

- predykcja i snapshot ceny to upsert po (symbol, godzina UTC) — ponowny zapis
  w tej samej godzinie scala kolumny i zostawia id oraz chwilę pierwszego zapisu;
- kolejka nierozliczonych idzie od najstarszej, z progami wieku i bez wygasłych;
- rozliczenie zapisuje się raz (strażnik `actual_price_after_12h IS NULL`)
  i stempluje `settled_at` (trigger migracji 029);
- okna odczytów liczy wstrzyknięty zegar, a trafność pomija atrapy LLM
  i rozliczenia z nieaktualnego notowania (`_countable`).

Odczyty liczone w bazie przez RPC albo pgvector (wagi, ranking i Brier person,
analogi RAG, widok `ml_feature_store`) zwracają pusty wynik — tak jak adapter
przy braku migracji, więc sekcje maila same się chowają. Fake nie ma schematu:
każda kolumna „istnieje”, a numer migracji jest nieznany (`schema_version` None).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from itertools import count
from typing import Any

from src.application.ports import (
    CycleRun,
    DeferredDiagnosis,
    LatestPrediction,
    OpenInvalidation,
    OpenReferenceCall,
    PreviousVerdict,
    RepositoryPort,
    StoredReceipt,
    StoredShockAlert,
)
from src.application.report_council_history import InvestorHistory
from src.domain.attestation import StoredCommitment
from src.domain.corporate_actions import SplitSuspicion
from src.domain.council import InvestorOpinion, parse_recommendation
from src.domain.council_redundancy import CouncilVoteSample
from src.domain.earnings import EARNINGS_CACHE_TTL_HOURS, CachedEarningsDate
from src.domain.equity_curve import directional_hit_rate
from src.domain.finops import LLMCallEntry
from src.domain.mistake_taxonomy import MistakeCategory
from src.domain.paper_ledger import (
    GENESIS_HASH,
    LedgerHead,
    LedgerRow,
    LedgerTrade,
    PaperTradeClose,
    PaperTradeEntry,
    chain_entry,
    utc_timestamp,
)
from src.domain.persona_track_record import PersonaScoring, PersonaVoteOutcome
from src.domain.prediction import UNDIAGNOSED_MISTAKE_PLACEHOLDERS, Prediction, normalize_trend
from src.domain.price_levels import PriceLevel, PriceLevelAlert
from src.domain.promotion import SettledShadowForecast, ShadowForecast
from src.domain.quota import QuotaAlert
from src.domain.reference_signal import ReferenceCall, ReferenceOutcome, ReferenceRead
from src.domain.regime import MarketRegime, RegimeState
from src.domain.session_close import SessionCloseSettlement
from src.domain.significance import effective_n
from src.domain.track_record import group_by_cycle
from src.domain.value_objects import (
    FUNDAMENTALS_CACHE_TTL_HOURS,
    AssetType,
    Fundamentals,
    Money,
)

# Parsowanie wierszy wspólne z adapterem Supabase — fake zastępuje bazę, nie mapowanie.
from src.infrastructure.adapters.supabase_repo import (
    _PREDICTION_PRICE_COLUMNS,
    _SESSION_CLOSE_CORRECT_COLUMN,
    EXPIRED_UNVERIFIED_FLAG,
    _countable,
    _deferred_diagnosis,
    _deserialize_verdict,
    _ledger_trade,
    _paper_trade_record,
    _serialize_record,
    _session_close_payload,
    _session_close_stats,
    _settled_shadow,
    _stored_receipt,
    _utc_day_key,
    _utc_hour_start,
    _without_llm_fallback,
    _without_stale_quote,
)

Row = dict[str, Any]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _json_row(record: Mapping[str, Any]) -> Row:
    """Wiersz po drodze przez JSON, jak w PostgREST: krotki stają się listami,
    a wartość, której JSON nie zapisze, wywraca zapis tak jak u dostawcy."""
    row: Row = json.loads(json.dumps(dict(record)))
    return row


def _stamp(value: Any) -> datetime | None:
    """Chwila z wiersza (ISO 8601) albo None."""
    return datetime.fromisoformat(value) if isinstance(value, str) and value else None


def _newest_first(rows: Iterable[Row], column: str = "timestamp") -> list[Row]:
    return sorted(rows, key=lambda row: str(row.get(column) or ""), reverse=True)


def _oldest_first(rows: Iterable[Row], column: str = "timestamp") -> list[Row]:
    return sorted(rows, key=lambda row: str(row.get(column) or ""))


def _since(rows: Iterable[Row], column: str, start: datetime) -> list[Row]:
    return [row for row in rows if (moment := _stamp(row.get(column))) and moment >= start]


def _not_expired(row: Row) -> bool:
    return EXPIRED_UNVERIFIED_FLAG not in (row.get("data_quality_flags") or [])


class InMemoryRepository(RepositoryPort):
    """`RepositoryPort` w pamięci procesu z zegarem wstrzykiwanym przez DI.

    Bezpieczny dla wątków na tyle, na ile potrzebuje go równoległa ścieżka
    cyklu: każdy zapis idzie pod jednym lockiem. Id wierszy to kolejne liczby
    (deterministyczny przebieg offline daje te same id za każdym razem)."""

    def __init__(self, *, clock: Callable[[], datetime] = _utc_now) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._ids = count(1)
        self._predictions: dict[str, Row] = {}
        self._snapshots: list[Row] = []
        self._feature_snapshots: dict[tuple[str, str], Row] = {}
        self._shadow: dict[tuple[str, str], Row] = {}
        self._council_votes: dict[tuple[str, str], Row] = {}
        self._quota_alerts: list[QuotaAlert] = []
        self._fundamentals: dict[str, Fundamentals] = {}
        self._earnings: dict[str, CachedEarningsDate] = {}
        self._regime: list[tuple[datetime, MarketRegime, RegimeState]] = []
        self._shock_alerts: dict[tuple[str, date], StoredShockAlert] = {}
        self._price_levels: dict[str, Row] = {}
        self._scorecards: list[Row] = []
        self._llm_calls: list[LLMCallEntry] = []
        self._cycle_runs: list[CycleRun] = []
        self._reference: dict[str, Row] = {}
        self._corporate_actions: dict[tuple[str, str, date], SplitSuspicion] = {}
        self._paper_trades: list[Row] = []

    # ------------------------------------------------------------------
    # Pomocnicze
    # ------------------------------------------------------------------

    def _now(self) -> datetime:
        return self._clock()

    def _next_id(self) -> str:
        return str(next(self._ids))

    def _rows(self, symbol: str | None = None) -> list[Row]:
        rows = list(self._predictions.values())
        return rows if symbol is None else [row for row in rows if row.get("symbol") == symbol]

    def _resolved(self, start: datetime, since: datetime | None = None) -> list[Row]:
        """Okno zamkniętych predykcji jak `_resolved_window` adaptera."""
        column, moment = ("timestamp", start) if since is None else ("settled_at", since)
        rows = [r for r in self._rows() if r.get("is_trend_correct") is not None]
        return _newest_first(_since(rows, column, moment), column)

    def _current_hour_row(self, symbol: str) -> Row | None:
        hour = _utc_hour_start(self._now()).isoformat()
        return next(
            (r for r in self._rows(symbol) if r.get("timestamp_hour") == hour), None
        )

    # ------------------------------------------------------------------
    # price_snapshots
    # ------------------------------------------------------------------

    def get_last_price(self, symbol: str) -> Money | None:
        snapshot = self.get_last_price_snapshot(symbol)
        return snapshot[0] if snapshot is not None else None

    def get_price_history(self, symbol: str, days: int) -> list[tuple[datetime, Money]]:
        start = self._now() - timedelta(days=days)
        rows = _oldest_first(
            _since((r for r in self._snapshots if r["symbol"] == symbol), "timestamp", start)
        )
        return [(datetime.fromisoformat(r["timestamp"]), Money(Decimal(r["price"]))) for r in rows]

    def get_last_price_snapshot(self, symbol: str) -> tuple[Money, datetime] | None:
        rows = _newest_first(r for r in self._snapshots if r["symbol"] == symbol)
        if not rows:
            return None
        return Money(Decimal(rows[0]["price"])), datetime.fromisoformat(rows[0]["timestamp"])

    def get_latest_snapshot_at(self) -> datetime | None:
        rows = _newest_first(self._snapshots)
        return datetime.fromisoformat(rows[0]["timestamp"]) if rows else None

    def save_price_snapshot(self, symbol: str, price: Money) -> None:
        now = self._now()
        hour = _utc_hour_start(now).isoformat()
        price_text = str(_serialize_record({"price": price.amount}, frozenset({"price"}))["price"])
        with self._lock:
            for row in self._snapshots:
                if row["symbol"] == symbol and row["timestamp_hour"] == hour:
                    row["price"] = price_text
                    return
            self._snapshots.append(
                {
                    "symbol": symbol,
                    "price": price_text,
                    "timestamp": now.isoformat(),
                    "timestamp_hour": hour,
                }
            )

    # ------------------------------------------------------------------
    # prediction_logs — zapis i kolejka rozliczeń
    # ------------------------------------------------------------------

    def save_prediction(self, prediction: dict[str, Any]) -> str:
        record = _json_row(_serialize_record(prediction, _PREDICTION_PRICE_COLUMNS))
        moment = _stamp(record.get("timestamp")) or self._now()
        hour = _utc_hour_start(moment).isoformat()
        with self._lock:
            existing = next(
                (
                    row
                    for row in self._predictions.values()
                    if row.get("symbol") == record.get("symbol")
                    and row.get("timestamp_hour") == hour
                ),
                None,
            )
            if existing is not None:
                existing.update(record)
                return str(existing["id"])
            prediction_id = self._next_id()
            self._predictions[prediction_id] = {
                **record,
                "id": prediction_id,
                "timestamp": moment.isoformat(),
                "timestamp_hour": hour,
                "actual_price_after_12h": None,
                "is_trend_correct": None,
            }
            return prediction_id

    def get_last_prediction_price(
        self, symbol: str, *, before: datetime | None = None
    ) -> Money | None:
        rows = self._rows(symbol)
        if before is not None:
            cutoff = _utc_hour_start(before).isoformat()
            rows = [r for r in rows if str(r.get("timestamp_hour")) < cutoff]
        newest = _newest_first(rows)
        if not newest or newest[0].get("price_at_prediction") is None:
            return None
        return Money(Decimal(str(newest[0]["price_at_prediction"])))

    def get_latest_prediction(self, symbol: str) -> LatestPrediction | None:
        newest = _newest_first(self._rows(symbol))
        if not newest:
            return None
        row = newest[0]
        stamp = _stamp(row.get("timestamp"))
        if row.get("price_at_prediction") is None or stamp is None:
            return None
        reasoning = str(row.get("reasoning_text") or "").strip()
        return LatestPrediction(
            price=Money(Decimal(str(row["price_at_prediction"]))),
            timestamp=stamp,
            reasoning=reasoning or None,
        )

    def get_unverified_prediction(
        self, symbol: str, min_age_hours: int = 0, max_age_hours: int | None = None
    ) -> Prediction | None:
        now = self._now()
        rows = [
            r
            for r in self._rows(symbol)
            if r.get("actual_price_after_12h") is None and _not_expired(r)
        ]
        if min_age_hours > 0:
            cutoff = now - timedelta(hours=min_age_hours)
            rows = [r for r in rows if (stamp := _stamp(r.get("timestamp"))) and stamp <= cutoff]
        if max_age_hours is not None and max_age_hours > 0:
            rows = _since(rows, "timestamp", now - timedelta(hours=max_age_hours))
        oldest = _oldest_first(rows)
        if not oldest:
            return None
        row = oldest[0]
        return Prediction(
            id=str(row["id"]),
            symbol=str(row["symbol"]),
            predicted_trend=normalize_trend(row.get("predicted_trend"))[0],
            price_at_prediction=Decimal(str(row["price_at_prediction"])),
            predicted_target_price=Decimal(str(row["predicted_target_price"])),
            news_summary=None if row.get("news_summary") is None else str(row["news_summary"]),
            reasoning=None if row.get("reasoning_text") is None else str(row["reasoning_text"]),
            created_at=_stamp(row.get("timestamp")),
        )

    def expire_unverified_predictions(self, symbol: str, older_than_hours: int) -> int:
        cutoff = self._now() - timedelta(hours=older_than_hours)
        expired = 0
        with self._lock:
            for row in _oldest_first(self._rows(symbol)):
                stamp = _stamp(row.get("timestamp"))
                if row.get("actual_price_after_12h") is not None or stamp is None:
                    continue
                if stamp >= cutoff or not _not_expired(row):
                    continue
                row["data_quality_flags"] = [
                    *(row.get("data_quality_flags") or []),
                    EXPIRED_UNVERIFIED_FLAG,
                ]
                expired += 1
        return expired

    def count_expired_unverified(self, days: int) -> int:
        rows = _since(self._rows(), "timestamp", self._now() - timedelta(days=days))
        return sum(
            1
            for row in rows
            if row.get("actual_price_after_12h") is None and not _not_expired(row)
        )

    def update_prediction_accuracy(
        self,
        prediction_id: str,
        actual_price: Decimal,
        accuracy_score: float,
        is_trend_correct: bool,
        insight: str,
        correction_category: MistakeCategory | None = None,
        *,
        stale_quote: bool | None = None,
        session_close: SessionCloseSettlement | None = None,
        entry_price: Decimal | None = None,
    ) -> None:
        settlement: dict[str, Any] = {
            "actual_price_after_12h": actual_price,
            "accuracy_score": accuracy_score,
            "is_trend_correct": is_trend_correct,
            "correction_insights": insight,
        }
        if correction_category is not None:
            settlement["correction_category"] = correction_category.value
        if stale_quote is not None:
            settlement["stale_quote"] = stale_quote
        if session_close is not None:
            settlement.update(_session_close_payload(session_close))
        if entry_price is not None:
            settlement["entry_price"] = entry_price
        payload = _json_row(_serialize_record(settlement, _PREDICTION_PRICE_COLUMNS))
        with self._lock:
            row = self._predictions.get(prediction_id)
            if row is None or row.get("actual_price_after_12h") is not None:
                return
            row.update(payload)
            row["settled_at"] = self._now().isoformat()

    # ------------------------------------------------------------------
    # prediction_logs — odczyty track recordu
    # ------------------------------------------------------------------

    def get_accuracy_stats(self, days: int) -> dict[str, Any]:
        start = self._now() - timedelta(days=days)
        rows = _countable(self._resolved(start))
        calls = [(str(r.get("predicted_trend") or ""), bool(r["is_trend_correct"])) for r in rows]
        has_close = any(r.get(_SESSION_CLOSE_CORRECT_COLUMN) is not None for r in rows)
        session_close = _session_close_stats(rows) if has_close else {}
        if not calls:
            return {
                "mean_accuracy": None,
                "sample_count": 0,
                "correct_count": 0,
                "days_window": days,
                "directional_count": 0,
                "directional_correct_count": 0,
                "n_eff": 0.0,
                **session_close,
            }
        directional = directional_hit_rate(calls)
        correct = sum(1 for _, hit in calls if hit)
        n_eff = effective_n(
            group_by_cycle(
                (_utc_day_key(r.get("timestamp")), bool(r["is_trend_correct"])) for r in rows
            )
        )
        return {
            "mean_accuracy": correct / len(calls),
            "sample_count": len(calls),
            "correct_count": correct,
            "days_window": days,
            "directional_count": directional.directional,
            "directional_correct_count": directional.hits,
            "n_eff": n_eff,
            **session_close,
        }

    def get_recently_resolved_predictions(
        self, hours: int, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        return _countable(self._resolved(self._now() - timedelta(hours=hours), since))

    def get_resolved_predictions(self, days: int) -> list[dict[str, Any]]:
        return _countable(self._resolved(self._now() - timedelta(days=days)))

    def get_resolved_predictions_detailed(
        self, days: int, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        return _countable(self._resolved(self._now() - timedelta(days=days), since))

    def get_resolved_for_calibration(self, days: int) -> list[dict[str, Any]]:
        rows = self._resolved(self._now() - timedelta(days=days))
        return _countable([r for r in rows if r.get("confidence_score") is not None])

    def get_settled_moves(
        self, days: int, *, with_intervals: bool = False
    ) -> list[dict[str, Any]]:
        start = self._now() - timedelta(days=days)
        rows = [r for r in self._rows() if r.get("actual_price_after_12h") is not None]
        return _without_stale_quote(_oldest_first(_since(rows, "timestamp", start)))

    def get_previous_verdict(self, symbol: str) -> PreviousVerdict | None:
        rows = _newest_first(r for r in self._rows(symbol) if r.get("council_verdict"))
        if not rows:
            return None
        row = rows[0]
        stamp = _stamp(row.get("timestamp"))
        if not isinstance(row["council_verdict"], dict) or stamp is None:
            return None
        return PreviousVerdict(
            verdict=_deserialize_verdict(row["council_verdict"]),
            sentiment_score=float(row.get("sentiment_score") or 0.0),
            timestamp=stamp,
        )

    def get_receipt(self, prediction_id: str) -> StoredReceipt | None:
        row = self._predictions.get(prediction_id)
        return _stored_receipt(row) if row is not None else None

    def find_receipts(self, symbol: str, day: date) -> list[StoredReceipt]:
        start = datetime(day.year, day.month, day.day, tzinfo=UTC)
        end = start + timedelta(days=1)
        rows = [
            r
            for r in _since(self._rows(symbol), "timestamp", start)
            if (stamp := _stamp(r.get("timestamp"))) and stamp < end
        ]
        return [_stored_receipt(row) for row in _oldest_first(rows)]

    def save_calibration(
        self,
        prediction_id: str,
        score: float,
        insight: str,
        *,
        score_version: int | None = None,
    ) -> None:
        with self._lock:
            row = self._predictions.get(prediction_id)
            if row is None:
                return
            row["confidence_calibration"] = score
            row["calibration_insight"] = insight
            if score_version is not None:
                row["calibration_score_version"] = score_version

    def get_deferred_diagnoses(self, limit: int) -> list[DeferredDiagnosis]:
        rows = _newest_first(
            r
            for r in self._rows()
            if r.get("is_trend_correct") is False
            and r.get("correction_insights") in UNDIAGNOSED_MISTAKE_PLACEHOLDERS
        )
        deferred = [
            item for row in _without_llm_fallback(rows) if (item := _deferred_diagnosis(row))
        ]
        return deferred[: max(0, limit)]

    def save_backfilled_diagnosis(
        self,
        prediction_id: str,
        insight: str,
        *,
        replaces: str,
        correction_category: MistakeCategory | None = None,
    ) -> None:
        with self._lock:
            row = self._predictions.get(prediction_id)
            if row is None or row.get("correction_insights") != replaces:
                return
            row["correction_insights"] = insight
            if correction_category is not None:
                row["correction_category"] = correction_category.value

    # ------------------------------------------------------------------
    # Atestacja (#16, WIA-01) — kolumny w wierszu predykcji
    # ------------------------------------------------------------------

    def get_unrevealed_commitments(self, days: int | None = None) -> list[dict[str, Any]]:
        rows = self._rows()
        if days is not None:
            rows = _since(rows, "timestamp", self._now() - timedelta(days=days))
        return _oldest_first(
            r
            for r in rows
            if r.get("commitment_hash") is not None
            and r.get("actual_price_after_12h") is not None
            and r.get("revealed_at") is None
        )

    def get_current_hour_commitment_salt(self, symbol: str) -> str | None:
        row = self._current_hour_row(symbol)
        salt = None if row is None else row.get("commitment_salt")
        return None if salt is None else str(salt)

    def get_current_hour_commitment(self, symbol: str) -> StoredCommitment | None:
        row = self._current_hour_row(symbol)
        if row is None or not row.get("commitment_hash") or not row.get("commitment_salt"):
            return None
        payload = row.get("commitment_payload")
        return StoredCommitment(
            commitment=str(row["commitment_hash"]),
            salt=str(row["commitment_salt"]),
            payload=None if payload is None else str(payload),
        )

    def attestation_columns_available(self) -> bool:
        return True

    def mark_commitment_revealed(self, prediction_id: str) -> None:
        with self._lock:
            row = self._predictions.get(prediction_id)
            if row is not None:
                row["revealed_at"] = self._now().isoformat()

    # ------------------------------------------------------------------
    # ML-27: poziom unieważnienia tezy
    # ------------------------------------------------------------------

    def get_open_invalidations(self, since: datetime) -> list[OpenInvalidation]:
        rows = _newest_first(
            r
            for r in _since(self._rows(), "timestamp", since)
            if r.get("invalidation_price") is not None
            and r.get("actual_price_after_12h") is None
            and r.get("invalidation_alerted_at") is None
        )
        return [
            OpenInvalidation(
                prediction_id=str(row["id"]),
                symbol=str(row["symbol"]),
                predicted_trend=str(row["predicted_trend"]),
                price_at_prediction=Decimal(str(row["price_at_prediction"])),
                invalidation_price=Decimal(str(row["invalidation_price"])),
                predicted_at=datetime.fromisoformat(str(row["timestamp"])),
            )
            for row in rows
        ]

    def mark_invalidation_alerted(self, prediction_id: str, at: datetime) -> None:
        with self._lock:
            row = self._predictions.get(prediction_id)
            if row is not None and row.get("invalidation_alerted_at") is None:
                row["invalidation_alerted_at"] = at.isoformat()

    # ------------------------------------------------------------------
    # Feature store, cienie modeli, reżim
    # ------------------------------------------------------------------

    def save_feature_snapshot(
        self,
        symbol: str,
        features: Mapping[str, Any],
        *,
        gated: bool,
        feature_set_version: int,
    ) -> None:
        now = self._now()
        key = (symbol, _utc_hour_start(now).isoformat())
        row = _json_row(
            {
                "symbol": symbol,
                "features": dict(features),
                "gated": gated,
                "feature_set_version": feature_set_version,
            }
        )
        with self._lock:
            previous = self._feature_snapshots.get(key)
            as_of = previous["as_of"] if previous is not None else now.isoformat()
            self._feature_snapshots[key] = {**row, "as_of": as_of}

    def get_feature_store_data(
        self, symbol: str, feature_set_version: int | None = None
    ) -> list[dict[str, Any]]:
        return []

    def refresh_feature_store(self) -> None:
        return None

    def save_shadow_predictions(
        self, prediction_id: str, symbol: str, forecasts: Sequence[ShadowForecast]
    ) -> None:
        with self._lock:
            for forecast in forecasts:
                key = (prediction_id, forecast.model_name)
                previous = self._shadow.get(key)
                self._shadow[key] = {
                    "prediction_id": prediction_id,
                    "symbol": symbol,
                    "model_name": forecast.model_name,
                    "model_version": forecast.model_version,
                    "predicted_return": forecast.predicted_return,
                    "proba": forecast.proba,
                    "created_at": (
                        previous["created_at"] if previous else self._now().isoformat()
                    ),
                }

    def get_settled_shadow_predictions(self, since: datetime) -> list[SettledShadowForecast]:
        settled: list[SettledShadowForecast] = []
        for row in _oldest_first(_since(self._shadow.values(), "created_at", since), "created_at"):
            prediction = self._predictions.get(str(row["prediction_id"]))
            if prediction is None or prediction.get("actual_price_after_12h") is None:
                continue
            joined = {**row, "prediction_logs": prediction}
            if (item := _settled_shadow(joined)) is not None:
                settled.append(item)
        return settled

    def get_latest_regime_state(self) -> RegimeState | None:
        if not self._regime:
            return None
        return max(enumerate(self._regime), key=lambda item: (item[1][0], item[0]))[1][2]

    def save_regime_state(
        self, cycle_at: datetime, raw: MarketRegime, state: RegimeState
    ) -> None:
        with self._lock:
            self._regime.append((cycle_at, raw, state))

    # ------------------------------------------------------------------
    # Cache fundamentów i kalendarza wyników
    # ------------------------------------------------------------------

    def get_cached_fundamentals(self, symbol: str) -> Fundamentals | None:
        cached = self._fundamentals.get(symbol)
        cutoff = self._now() - timedelta(hours=FUNDAMENTALS_CACHE_TTL_HOURS)
        return cached if cached is not None and cached.fetched_at >= cutoff else None

    def save_fundamentals(self, symbol: str, fundamentals: Fundamentals) -> None:
        with self._lock:
            self._fundamentals[symbol] = fundamentals

    def get_fundamentals_fetched_at(self, symbols: Sequence[str]) -> dict[str, datetime]:
        return {s: self._fundamentals[s].fetched_at for s in symbols if s in self._fundamentals}

    def get_cached_earnings(self, symbol: str) -> CachedEarningsDate | None:
        cached = self._earnings.get(symbol)
        cutoff = self._now() - timedelta(hours=EARNINGS_CACHE_TTL_HOURS)
        return cached if cached is not None and cached.fetched_at >= cutoff else None

    def save_earnings(self, entry: CachedEarningsDate) -> None:
        with self._lock:
            self._earnings[entry.symbol] = entry

    # ------------------------------------------------------------------
    # Alerty kwot i szoków
    # ------------------------------------------------------------------

    def save_quota_alert(self, alert: QuotaAlert) -> None:
        with self._lock:
            self._quota_alerts.append(alert)

    def get_recent_quota_alerts(self, hours: int) -> list[QuotaAlert]:
        cutoff = self._now() - timedelta(hours=hours)
        recent = [alert for alert in self._quota_alerts if alert.occurred_at >= cutoff]
        return sorted(recent, key=lambda alert: alert.occurred_at, reverse=True)

    def save_shock_alert(
        self, symbol: str, alert_date: date, delta: float, direction: str
    ) -> None:
        with self._lock:
            # UNIQUE(symbol, alert_date) — drugi alert tego dnia odbija się jak w bazie.
            key = (symbol, alert_date)
            if key in self._shock_alerts:
                raise ValueError(f"shock alert for {symbol} on {alert_date} already exists")
            self._shock_alerts[key] = StoredShockAlert(
                symbol=symbol, delta=delta, direction=direction, created_at=self._now()
            )

    def get_sent_shock_alerts(self, since: date) -> set[tuple[str, date]]:
        return {key for key in self._shock_alerts if key[1] >= since}

    def get_shock_alert_times(self, since: datetime) -> dict[str, datetime]:
        latest: dict[str, datetime] = {}
        for alert in self._shock_alerts.values():
            if alert.created_at >= since and alert.created_at > latest.get(
                alert.symbol, datetime.min.replace(tzinfo=UTC)
            ):
                latest[alert.symbol] = alert.created_at
        return latest

    def get_shock_alerts_since(self, since: datetime) -> list[StoredShockAlert]:
        alerts = [alert for alert in self._shock_alerts.values() if alert.created_at >= since]
        return sorted(alerts, key=lambda alert: alert.created_at)

    # ------------------------------------------------------------------
    # Rada: głosy person i odczyty liczone w bazie
    # ------------------------------------------------------------------

    def save_council_votes(
        self, prediction_id: str, symbol: str, votes: list[InvestorOpinion]
    ) -> None:
        now = self._now().isoformat()
        with self._lock:
            for vote in votes:
                self._council_votes[(prediction_id, vote.investor_name)] = _json_row(
                    {
                        "id": self._next_id(),
                        "prediction_id": prediction_id,
                        "symbol": symbol,
                        "investor_name": vote.investor_name,
                        "recommendation": vote.recommendation,
                        "confidence": vote.confidence,
                        "reasoning": vote.reasoning,
                        "key_factors": list(vote.key_factors),
                        "timestamp": now,
                    }
                )

    def get_council_votes_for_prediction(self, prediction_id: str) -> list[InvestorOpinion]:
        opinions: list[InvestorOpinion] = []
        for row in self._council_votes.values():
            recommendation = parse_recommendation(row.get("recommendation"))
            if row["prediction_id"] != prediction_id or recommendation is None:
                continue
            opinions.append(
                InvestorOpinion(
                    investor_name=str(row["investor_name"]),
                    recommendation=recommendation,
                    confidence=float(row.get("confidence") or 0.0),
                    reasoning=str(row.get("reasoning") or ""),
                    key_factors=tuple(row.get("key_factors") or ()),
                )
            )
        return opinions

    def _recent_votes(self, window_days: int) -> list[Row]:
        start = self._now() - timedelta(days=window_days)
        return sorted(
            _since(self._council_votes.values(), "timestamp", start),
            key=lambda row: (str(row["timestamp"]), int(row["id"])),
            reverse=True,
        )

    def get_council_vote_history(
        self,
        symbols: list[str],
        *,
        window_days: int = 30,
        investors: list[str] | None = None,
    ) -> list[InvestorHistory]:
        grouped: dict[tuple[str, str], list[Any]] = {}
        for row in self._recent_votes(window_days):
            name, symbol = str(row["investor_name"]), str(row["symbol"])
            recommendation = parse_recommendation(row.get("recommendation"))
            if symbol not in symbols or recommendation is None:
                continue
            if investors and name not in investors:
                continue
            grouped.setdefault((name, symbol), []).append(recommendation)
        return [
            InvestorHistory(
                investor_name=name,
                symbol=symbol,
                recommendations=tuple(recommendations),
                window_days=window_days,
            )
            for (name, symbol), recommendations in grouped.items()
        ]

    def get_council_vote_samples(self, window_days: int = 30) -> list[CouncilVoteSample]:
        samples: list[CouncilVoteSample] = []
        for row in self._recent_votes(window_days):
            recommendation = parse_recommendation(row.get("recommendation"))
            if recommendation is not None:
                samples.append(
                    CouncilVoteSample(
                        str(row["prediction_id"]), str(row["investor_name"]), recommendation
                    )
                )
        return samples

    def get_persona_accuracy(
        self, asset_type: AssetType, window_days: int = 90
    ) -> dict[str, float]:
        return {}

    def get_persona_track_record(self, window_days: int = 90) -> dict[str, tuple[float, int]]:
        return {}

    def get_persona_scoring(self, window_days: int = 90) -> dict[str, PersonaScoring]:
        return {}

    def get_persona_vote_outcomes(self, window_days: int = 90) -> list[PersonaVoteOutcome]:
        return []

    def find_similar_predictions(
        self,
        embedding: list[float],
        limit: int = 3,
        regime: str | None = None,
        *,
        symbol: str | None = None,
        before: datetime | None = None,
    ) -> list[dict[str, Any]]:
        return []

    # ------------------------------------------------------------------
    # Przebiegi, ledger kosztów, scorecardy, zdrowie bazy
    # ------------------------------------------------------------------

    def save_llm_calls(self, calls: Sequence[LLMCallEntry]) -> None:
        with self._lock:
            self._llm_calls.extend(calls)

    def save_cycle_run(self, run: CycleRun) -> None:
        with self._lock:
            self._cycle_runs.append(run)

    def get_cycle_runs_since(self, since: datetime) -> list[CycleRun]:
        return sorted(
            (run for run in self._cycle_runs if run.started_at >= since),
            key=lambda run: run.started_at,
        )

    def get_last_report_at(self, before: datetime) -> datetime | None:
        delivered = [
            run.started_at
            for run in self._cycle_runs
            if run.loop == "fast" and run.report_delivered is True and run.started_at < before
        ]
        return max(delivered, default=None)

    def save_model_scorecard(
        self, symbol: str, result: Mapping[str, Any], *, run_id: str | None = None
    ) -> None:
        row = {**dict(result), "symbol": symbol, "timestamp": self._now().isoformat()}
        if run_id is not None:
            row["run_id"] = run_id
        with self._lock:
            self._scorecards.append(_json_row(row))

    def get_recent_model_scorecards(self, days: int = 30) -> list[dict[str, Any]]:
        start = self._now() - timedelta(days=days)
        return _newest_first(_since(self._scorecards, "timestamp", start))

    def get_last_scorecard_at(self) -> datetime | None:
        rows = _newest_first(self._scorecards)
        return datetime.fromisoformat(rows[0]["timestamp"]) if rows else None

    def probe_database(self) -> None:
        return None

    def schema_version(self) -> int | None:
        return None

    def missing_prediction_columns(self, columns: frozenset[str]) -> frozenset[str]:
        return frozenset()

    # ------------------------------------------------------------------
    # ALR-11: poziomy cenowe
    # ------------------------------------------------------------------

    def get_active_price_levels(self) -> list[PriceLevelAlert]:
        return [
            PriceLevelAlert(id=alert_id, level=row["level"], last_seen_price=row["last_seen"])
            for alert_id, row in sorted(self._price_levels.items(), key=lambda i: int(i[0]))
            if row["active"]
        ]

    def add_price_levels(self, levels: Sequence[PriceLevel]) -> None:
        with self._lock:
            known = {
                (row["level"].symbol, row["level"].level, row["level"].direction)
                for row in self._price_levels.values()
            }
            for level in levels:
                if (level.symbol, level.level, level.direction) in known:
                    continue
                known.add((level.symbol, level.level, level.direction))
                self._price_levels[self._next_id()] = {
                    "level": level,
                    "last_seen": None,
                    "active": True,
                }

    def record_price_level_reading(self, alert_id: str, price: Decimal) -> None:
        with self._lock:
            row = self._price_levels.get(alert_id)
            if row is not None and row["active"]:
                row["last_seen"] = price

    def mark_price_level_triggered(
        self, alert_id: str, price: Decimal, triggered_at: datetime
    ) -> None:
        with self._lock:
            row = self._price_levels.get(alert_id)
            if row is not None:
                row.update(active=False, last_seen=price, triggered_at=triggered_at)

    # ------------------------------------------------------------------
    # WIA-24: sygnał referencyjny
    # ------------------------------------------------------------------

    def get_reference_reads(self, since: datetime) -> list[ReferenceRead]:
        rows = sorted(
            (row for row in self._reference.values() if row["read"].read_at >= since),
            key=lambda row: (row["read"].read_at, int(row["id"])),
        )
        return [row["read"] for row in rows]

    def get_open_reference_calls(self) -> list[OpenReferenceCall]:
        rows = sorted(
            (
                row
                for row in self._reference.values()
                if row["call"] is not None and row["outcome"] is None
            ),
            key=lambda row: (row["read"].read_at, int(row["id"])),
        )
        return [OpenReferenceCall(id=str(row["id"]), call=row["call"]) for row in rows]

    def save_reference_read(self, read: ReferenceRead, call: ReferenceCall | None) -> None:
        with self._lock:
            record_id = self._next_id()
            self._reference[record_id] = {
                "id": record_id,
                "read": read,
                "call": call,
                "outcome": None,
            }

    def settle_reference_call(
        self,
        record_id: str,
        outcome: ReferenceOutcome,
        settled_price: Decimal | None,
        settled_at: datetime,
    ) -> None:
        with self._lock:
            row = self._reference.get(record_id)
            if row is not None:
                row.update(outcome=outcome, settled_price=settled_price, settled_at=settled_at)

    # ------------------------------------------------------------------
    # DAT-23 i WIA-59: zdarzenia korporacyjne, paper ledger
    # ------------------------------------------------------------------

    def save_corporate_action(self, action: SplitSuspicion) -> None:
        with self._lock:
            key = (action.symbol, action.kind.value, action.detected_on)
            self._corporate_actions.setdefault(key, action)

    def append_paper_trade(self, entry: PaperTradeEntry) -> LedgerRow | None:
        with self._lock:
            if any(row["prediction_id"] == entry.prediction_id for row in self._paper_trades):
                return None
            head = self._paper_trades[-1]["row_hash"] if self._paper_trades else GENESIS_HASH
            ledger_row = chain_entry(entry, head)
            self._paper_trades.append(
                {
                    **_paper_trade_record(ledger_row),
                    "seq": len(self._paper_trades) + 1,
                    "exit_price": None,
                }
            )
            return ledger_row

    def close_paper_trade(self, prediction_id: str, close: PaperTradeClose) -> None:
        with self._lock:
            for row in self._paper_trades:
                if row["prediction_id"] == prediction_id and row["exit_price"] is None:
                    row.update(
                        exit_price=str(Money(close.exit_price).for_storage().amount),
                        exit_ts=utc_timestamp(close.exit_ts),
                        pnl=str(close.pnl),
                        exit_reason=close.exit_reason.value,
                    )

    def get_paper_trades(self, days: int) -> list[LedgerTrade]:
        start = self._now() - timedelta(days=days)
        return [_ledger_trade(row) for row in _since(self._paper_trades, "entry_ts", start)]

    def get_paper_ledger_head(self) -> LedgerHead | None:
        if not self._paper_trades:
            return None
        last = self._paper_trades[-1]
        return LedgerHead(seq=int(last["seq"]), row_hash=str(last["row_hash"]))
