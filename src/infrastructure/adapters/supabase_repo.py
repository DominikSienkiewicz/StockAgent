from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal, cast

from supabase import create_client

from src.application.ports import RepositoryPort
from src.application.report_council_history import InvestorHistory
from src.domain.council import InvestorOpinion
from src.domain.prediction import Prediction, TrendDirection
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.regime_key import canonical_regime_key
from src.domain.value_objects import (
    FUNDAMENTALS_CACHE_TTL_HOURS,
    AssetType,
    Fundamentals,
    Money,
)

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "prediction_logs"
DEFAULT_FEATURE_VIEW = "ml_feature_store"
DEFAULT_SNAPSHOT_TABLE = "price_snapshots"
DEFAULT_COUNCIL_VOTES_TABLE = "council_votes"
DEFAULT_QUOTA_ALERTS_TABLE = "quota_alerts"

# Retry parametry dla refresh_feature_store — Slow Loop wywołuje RPC przed
# treningiem; transient błąd sieci nie powinien wywalać całego treningu.
_REFRESH_MAX_ATTEMPTS = 3
_REFRESH_BACKOFF_S = 2.0

# Hosted Supabase / PostgREST tnie SELECT do ~1000 wierszy BEZ błędu (#8).
# Agregaty (get_accuracy_stats itd.) muszą paginować po .range(), inaczej
# trafność liczona jest na obciętym, niedeterministycznym podzbiorze.
_PAGE_SIZE = 1000


def _paginate(query_builder: Any) -> list[dict[str, Any]]:
    """Iteruje .range(offset, offset+_PAGE_SIZE-1) po `query_builder` aż do
    zwrócenia krótkiej strony (<_PAGE_SIZE), kumulując wszystkie wiersze.

    `query_builder` to gotowy builder (po .order/.gte/...), na którym wywołanie
    .range(...).execute() zwraca kolejną stronę. Dla zbiorów <_PAGE_SIZE
    zachowanie jest identyczne jak pojedynczy select (jedna iteracja)."""
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = query_builder.range(offset, offset + _PAGE_SIZE - 1).execute()
        page = _rows(response)
        rows.extend(page)
        if len(page) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    return rows


def _serialize(value: Any) -> Any:
    """Decimal → str (JSON-safe, zachowuje precyzję) — pozostałe typy bez zmian."""
    if isinstance(value, Decimal):
        return str(value)
    return value


def _serialize_record(record: dict[str, Any]) -> dict[str, Any]:
    return {k: _serialize(v) for k, v in record.items()}


def _rows(response: Any) -> list[dict[str, Any]]:
    """Narrow supabase response.data (JSON union) do list[dict]."""
    data = response.data or []
    return cast(list[dict[str, Any]], data)


class SupabaseRepository(RepositoryPort):
    """Adapter dla Supabase (PostgreSQL via PostgREST).

    Tabela `prediction_logs` + zmaterializowany widok `ml_feature_store`
    — schema patrz docs/08-database-schema.md.
    """

    def __init__(
        self,
        url: str,
        key: str,
        table_name: str = DEFAULT_TABLE,
        feature_view: str = DEFAULT_FEATURE_VIEW,
        snapshot_table: str = DEFAULT_SNAPSHOT_TABLE,
        council_votes_table: str = DEFAULT_COUNCIL_VOTES_TABLE,
        quota_alerts_table: str = DEFAULT_QUOTA_ALERTS_TABLE,
    ) -> None:
        self._client = create_client(url, key)
        self._table = table_name
        self._view = feature_view
        self._snapshot_table = snapshot_table
        self._council_votes_table = council_votes_table
        self._quota_alerts_table = quota_alerts_table

    # -----------------------------------------------------------------------
    # Reads
    # -----------------------------------------------------------------------

    def get_last_price(self, symbol: str) -> Money | None:
        # Czytamy z price_snapshots — zapisywane w KAŻDYM cyklu, więc nawet
        # gdy bot dotąd nie zrobił żadnej predykcji, ma punkt odniesienia.
        response = (
            self._client.table(self._snapshot_table)
            .select("price")
            .eq("symbol", symbol)
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )
        rows = _rows(response)
        if not rows:
            return None
        return Money(Decimal(str(rows[0]["price"])))

    def get_price_history(
        self, symbol: str, days: int
    ) -> list[tuple[datetime, Money]]:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        response = (
            self._client.table(self._snapshot_table)
            .select("timestamp, price")
            .eq("symbol", symbol)
            .gte("timestamp", cutoff.isoformat())
            .order("timestamp", desc=False)
            .execute()
        )
        out: list[tuple[datetime, Money]] = []
        for row in _rows(response):
            ts = row.get("timestamp")
            price = row.get("price")
            if not isinstance(ts, str) or price is None:
                continue
            out.append(
                (datetime.fromisoformat(ts), Money(Decimal(str(price))))
            )
        return out

    def get_last_prediction_price(self, symbol: str) -> Money | None:
        # Cena ostatniej zalogowanej predykcji — referencja dla cechy price_delta
        # w inference (musi pokrywać się z LAG(price_at_prediction) w widoku
        # ml_feature_store, inaczej cecha ma inny rozkład w treningu i predykcji).
        response = (
            self._client.table(self._table)
            .select("price_at_prediction")
            .eq("symbol", symbol)
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )
        rows = _rows(response)
        if not rows or rows[0].get("price_at_prediction") is None:
            return None
        return Money(Decimal(str(rows[0]["price_at_prediction"])))

    def get_unverified_prediction(
        self, symbol: str, min_age_hours: int = 0
    ) -> Prediction | None:
        query = (
            self._client.table(self._table)
            .select("*")
            .eq("symbol", symbol)
            .is_("actual_price_after_12h", "null")
        )
        # Górne ograniczenie wieku — pomija predykcje zbyt świeże, by je
        # rzetelnie ocenić (patrz docstring portu). Filtr dokładamy tylko gdy
        # próg > 0, by nie zmieniać zachowania domyślnego.
        if min_age_hours > 0:
            cutoff = datetime.now(UTC) - timedelta(hours=min_age_hours)
            query = query.lte("timestamp", cutoff.isoformat())
        # ASC (oldest-first) — #19: opróżniamy backlog, rozliczając NAJSTARSZĄ
        # nierozliczoną predykcję, która przekroczyła próg min_age_hours. Przy
        # DESC starsze nierozliczone predykcje nigdy nie stałyby się znów
        # "najnowsze" i utknęłyby na zawsze (brak actual_price_after_12h →
        # nie wchodzą do ml_feature_store, nie liczą się do hit-rate).
        response = query.order("timestamp", desc=False).limit(1).execute()
        rows = _rows(response)
        if not rows:
            return None
        row = rows[0]
        return Prediction(
            id=str(row["id"]),
            symbol=str(row["symbol"]),
            predicted_trend=TrendDirection(str(row["predicted_trend"])),
            price_at_prediction=Decimal(str(row["price_at_prediction"])),
            predicted_target_price=Decimal(str(row["predicted_target_price"])),
        )

    def get_feature_store_data(self, symbol: str) -> list[dict[str, Any]]:
        response = (
            self._client.table(self._view)
            .select("*")
            .eq("symbol", symbol)
            .order("timestamp", desc=False)
            .execute()
        )
        return _rows(response)

    def refresh_feature_store(self) -> None:
        """Wywołuje RPC `refresh_ml_feature_store` (zdef. w migrations/001_init.sql).

        Bez tego widok zmaterializowany pokazuje stan z momentu ostatniego
        REFRESH — co psuje Continual Learning (model uczy się na nieaktualnym
        snapshocie). Wywoływane przez Slow Loop przed treningiem.

        Retry z exponential backoff — transient błąd sieci/Supabase
        (gateway 5xx) nie powinien wywalać całego Slow Loop.
        """
        last_exc: Exception | None = None
        for attempt in range(1, _REFRESH_MAX_ATTEMPTS + 1):
            try:
                self._client.rpc("refresh_ml_feature_store").execute()
                return
            except Exception as exc:
                last_exc = exc
                if attempt < _REFRESH_MAX_ATTEMPTS:
                    delay = _REFRESH_BACKOFF_S * (2 ** (attempt - 1))
                    logger.warning(
                        "refresh_feature_store attempt %d/%d failed (%s) — retrying in %.1fs",
                        attempt,
                        _REFRESH_MAX_ATTEMPTS,
                        type(exc).__name__,
                        delay,
                    )
                    time.sleep(delay)
        assert last_exc is not None  # for type narrowing
        raise last_exc

    def get_recently_resolved_predictions(self, hours: int) -> list[dict[str, Any]]:
        """Predykcje zamknięte (z oceną kierunku) w ostatnich `hours` godzinach.

        Filtr po `is_trend_correct` (nie `accuracy_score`) — to ono napędza
        trafność raportu i naturalnie odsiewa wiersze sprzed migracji 009."""
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        # Pełen post-mortem w raporcie: oryginalne uzasadnienie + diagnoza + ceny
        # (a nie tylko kierunek), żeby pokazać "dlaczego wzrosło/spadło i czy
        # prognoza się potwierdziła".
        # Paginacja (#8) — bez niej raport pokazałby max ~1000 zamkniętych
        # predykcji (cichy cap PostgREST).
        query = (
            self._client.table(self._table)
            .select(
                "symbol, predicted_trend, is_trend_correct, timestamp, "
                "reasoning_text, correction_insights, price_at_prediction, "
                "actual_price_after_12h"
            )
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("is_trend_correct", "null")
            .order("timestamp", desc=True)
        )
        return _paginate(query)

    def get_resolved_predictions(self, days: int) -> list[dict[str, Any]]:
        """Zamknięte predykcje (oceniony kierunek) z ostatnich `days` dni.

        Ten sam zestaw kolumn co `get_recently_resolved_predictions` (godziny),
        ale szersze okno dniowe — zasila track-record (krzywa kapitału, panel
        lessons). `parse_resolved_predictions` parsuje oba kształty. Paginacja
        (#8) — pełen zbiór, nie obcięte ~1000 wierszy PostgREST."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        query = (
            self._client.table(self._table)
            .select(
                "symbol, predicted_trend, is_trend_correct, timestamp, "
                "reasoning_text, correction_insights, price_at_prediction, "
                "actual_price_after_12h"
            )
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("is_trend_correct", "null")
            .order("timestamp", desc=True)
        )
        return _paginate(query)

    def get_resolved_predictions_for_eval(self, days: int) -> list[dict[str, Any]]:
        """Zamknięte predykcje z ostatnich `days` dni z polami do ewaluacji
        (ceny + kierunek). Read-only; używane przez `src.tools.evaluate`.

        Metoda concrete-only (nie część `RepositoryPort`) — to narzędzie
        offline-analizy, nie część runtime'owego kontraktu use case'ów."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        # Paginacja (#8) — ewaluacja musi widzieć pełen zbiór, nie obcięte 1000.
        query = (
            self._client.table(self._table)
            .select(
                "predicted_trend, is_trend_correct, price_at_prediction, "
                "predicted_target_price, actual_price_after_12h, timestamp"
            )
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("actual_price_after_12h", "null")
            .order("timestamp", desc=True)
        )
        return _paginate(query)

    def get_accuracy_stats(self, days: int) -> dict[str, Any]:
        """Trafność kierunkowa predykcji z ostatnich `days` dni.

        `mean_accuracy` to hit-rate (udział trafionych kierunkowo predykcji),
        liczony z `is_trend_correct` — NIE z `accuracy_score` (bliskość ceny do
        celu, przy cold-starcie zawsze ~100% niezależnie od kierunku).

        Zwraca dict z kluczami: `mean_accuracy`, `sample_count`, `correct_count`,
        `days_window`. Gdy brak danych → mean_accuracy=None.
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)
        # Paginacja (#8): hit-rate liczony na PEŁNYM zbiorze, nie obciętych 1000
        # wierszy. .order("timestamp") czyni podzbiór deterministycznym —
        # bez niego obcięta próbka była losowa przy każdym wywołaniu.
        query = (
            self._client.table(self._table)
            .select("is_trend_correct")
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("is_trend_correct", "null")
            .order("timestamp", desc=True)
        )
        rows = _paginate(query)
        flags = [
            bool(r["is_trend_correct"])
            for r in rows
            if r.get("is_trend_correct") is not None
        ]

        if not flags:
            return {
                "mean_accuracy": None,
                "sample_count": 0,
                "correct_count": 0,
                "days_window": days,
            }
        correct_count = sum(1 for f in flags if f)
        return {
            "mean_accuracy": correct_count / len(flags),
            "sample_count": len(flags),
            "correct_count": correct_count,
            "days_window": days,
        }

    def get_cached_fundamentals(self, symbol: str) -> Fundamentals | None:
        """Zwraca fundamentale z cache jeśli `fetched_at` mieści się w
        FUNDAMENTALS_CACHE_TTL_HOURS. Filtr TTL po stronie repo —
        adapter nie zna polityki świeżości."""
        cutoff = datetime.now(UTC) - timedelta(hours=FUNDAMENTALS_CACHE_TTL_HOURS)
        response = (
            self._client.table("fundamentals_cache")
            .select("*")
            .eq("symbol", symbol)
            .gte("fetched_at", cutoff.isoformat())
            .execute()
        )
        rows = _rows(response)
        if not rows:
            return None
        row = rows[0]
        return Fundamentals(
            trailing_pe=row.get("trailing_pe"),
            forward_pe=row.get("forward_pe"),
            peg_ratio=row.get("peg_ratio"),
            eps_growth_yoy=row.get("eps_growth_yoy"),
            fetched_at=datetime.fromisoformat(row["fetched_at"]),
        )

    def save_fundamentals(self, symbol: str, fundamentals: Fundamentals) -> None:
        """Upsert jednego wiersza per symbol (nadpisuje poprzedni snapshot)."""
        payload: dict[str, Any] = {
            "symbol": symbol,
            "trailing_pe": fundamentals.trailing_pe,
            "forward_pe": fundamentals.forward_pe,
            "peg_ratio": fundamentals.peg_ratio,
            "eps_growth_yoy": fundamentals.eps_growth_yoy,
            "fetched_at": fundamentals.fetched_at.isoformat(),
        }
        self._client.table("fundamentals_cache").upsert(payload).execute()

    # -----------------------------------------------------------------------
    # Writes
    # -----------------------------------------------------------------------

    def save_price_snapshot(self, symbol: str, price: Money) -> None:
        # Upsert na (symbol, timestamp_hour) (#7) — re-run GHA / nakładający się
        # workflow_dispatch / retry LangGraph aktualizuje istniejący wiersz w tej
        # samej godzinie zamiast duplikować (duplikaty zaburzały trafność).
        record = _serialize_record({"symbol": symbol, "price": price.amount})
        (
            self._client.table(self._snapshot_table)
            .upsert(record, on_conflict="symbol,timestamp_hour")
            .execute()
        )

    def save_prediction(self, prediction: dict[str, Any]) -> str:
        # Upsert na (symbol, timestamp_hour) (#7) — idempotentny zapis predykcji.
        # Domyślny merge upsertu zwraca wiersz, więc nadal mamy z czego odczytać
        # id (graf używa go jako prediction_id).
        record = _serialize_record(prediction)
        response = (
            self._client.table(self._table)
            .upsert(record, on_conflict="symbol,timestamp_hour")
            .execute()
        )
        rows = _rows(response)
        if not rows:
            raise RuntimeError(
                f"Supabase upsert returned no data for {record.get('symbol')}"
            )
        return str(rows[0]["id"])

    def save_quota_alert(self, alert: QuotaAlert) -> None:
        row = {
            "source": alert.source,
            "severity": alert.severity.value,
            "message": alert.message,
            "action": alert.action,
            "occurred_at": alert.occurred_at.isoformat(),
        }
        self._client.table(self._quota_alerts_table).insert(row).execute()

    def get_recent_quota_alerts(self, hours: int) -> list[QuotaAlert]:
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        response = (
            self._client.table(self._quota_alerts_table)
            .select("*")
            .gte("occurred_at", cutoff.isoformat())
            .order("occurred_at", desc=True)
            .execute()
        )
        rows = _rows(response)
        out: list[QuotaAlert] = []
        for row in rows:
            try:
                severity = QuotaSeverity(row["severity"])
            except (KeyError, ValueError):
                continue
            occurred = row.get("occurred_at")
            if not isinstance(occurred, str):
                continue
            out.append(
                QuotaAlert(
                    source=str(row.get("source", "?")),
                    severity=severity,
                    message=str(row.get("message", "")),
                    action=str(row.get("action", "")),
                    occurred_at=datetime.fromisoformat(occurred),
                )
            )
        return out

    def find_similar_predictions(
        self, embedding: list[float], limit: int = 3, regime: str | None = None
    ) -> list[dict[str, Any]]:
        """Similarity search nad `prediction_logs.embedding` (pgvector) przez
        RPC `match_news_embeddings` (migracja 011; filtr reżimu z migracji 017).

        Graceful: gdy RPC nie istnieje (migracja niezaaplikowana) lub błąd
        sieci/Supabase — zwraca `[]`, żeby RAG był czysto opcjonalnym
        wzbogaceniem, a nie twardą zależnością predykcji.

        `regime`: gdy podany, dokłada parametr `filter_regime` do RPC (migracja
        017 filtruje analogi po reżimie; NULL/UNKNOWN w bazie przepuszcza)."""
        params: dict[str, Any] = {
            "query_embedding": embedding,
            "match_count": limit,
        }
        if regime is not None:
            params["filter_regime"] = canonical_regime_key(regime)
        try:
            response = self._client.rpc("match_news_embeddings", params).execute()
        except Exception:
            logger.warning(
                "match_news_embeddings RPC unavailable — RAG retrieval skipped "
                "(apply migration 011 to enable).",
                exc_info=True,
            )
            return []
        return _rows(response)

    def get_resolved_for_calibration(self, days: int) -> list[dict[str, Any]]:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        response = (
            self._client.table(self._table)
            .select("id, confidence_score, is_trend_correct, reasoning_text")
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("is_trend_correct", "null")
            .not_.is_("confidence_score", "null")
            .order("timestamp", desc=True)
            .execute()
        )
        return _rows(response)

    def save_calibration(
        self, prediction_id: str, score: float, insight: str
    ) -> None:
        (
            self._client.table(self._table)
            .update(
                {"confidence_calibration": score, "calibration_insight": insight}
            )
            .eq("id", prediction_id)
            .execute()
        )

    def get_persona_accuracy(
        self, asset_type: AssetType, window_days: int = 90
    ) -> dict[str, float]:
        # Liczenie trafności per persona (rekomendacja vs realny ruch ceny) robi
        # RPC `persona_accuracy_weights` w SQL (migracja 015) — Python tylko mapuje
        # wynik. `asset_type` zarezerwowany na przyszłe ważenie per klasa aktywa
        # (dziś council_votes nie nosi typu → waga globalna). Graceful: brak RPC
        # / błąd → {} (ważenie wyłączone, każda persona waży 1.0).
        _ = asset_type
        try:
            response = self._client.rpc(
                "persona_accuracy_weights", {"window_days": window_days}
            ).execute()
        except Exception:
            logger.warning(
                "persona_accuracy_weights RPC unavailable — ważenie person pominięte "
                "(zaaplikuj migrację 015).",
                exc_info=True,
            )
            return {}
        out: dict[str, float] = {}
        for row in _rows(response):
            name = row.get("investor_name")
            weight = row.get("weight")
            if name is None or weight is None:
                continue
            out[str(name)] = float(weight)
        return out

    def get_persona_track_record(
        self, window_days: int = 90
    ) -> dict[str, tuple[float, int]]:
        # RPC `persona_accuracy_stats` (migracja 018) — ten sam JOIN co 015, ale
        # zwraca surowy hit-rate ORAZ COUNT(*) głosów. RPC z 015 niesie tylko
        # przeskalowaną wagę, z której nie da się odtworzyć wielkości próbki.
        # Graceful: brak RPC / błąd → {} (leaderboard sam się chowa).
        try:
            response = self._client.rpc(
                "persona_accuracy_stats", {"window_days": window_days}
            ).execute()
        except Exception:
            logger.warning(
                "persona_accuracy_stats RPC unavailable — leaderboard person "
                "pominięty (zaaplikuj migrację 018).",
                exc_info=True,
            )
            return {}
        out: dict[str, tuple[float, int]] = {}
        for row in _rows(response):
            name = row.get("investor_name")
            hit_rate = row.get("hit_rate")
            votes = row.get("vote_count")
            if name is None or hit_rate is None or votes is None:
                continue
            out[str(name)] = (float(hit_rate), int(votes))
        return out

    def save_model_scorecard(self, symbol: str, result: Mapping[str, Any]) -> None:
        # Jeden wiersz na przebieg treningu (migracja 019). Kolumna `symbol` jest
        # kluczowa: trening leci per-symbol, więc bez niej nie wiadomo, który
        # kandydat przeszedł bramkę, a który został odrzucony.
        payload = {
            "symbol": symbol,
            "status": result.get("status"),
            "candidate_holdout_rmse": result.get("candidate_holdout_rmse"),
            "baseline_rmse": result.get("baseline_rmse"),
            "candidate_holdout_directional_hit_rate": result.get(
                "candidate_holdout_directional_hit_rate"
            ),
            "n_folds": result.get("n_folds"),
            "n_samples": result.get("n_samples"),
        }
        self._client.table("model_scorecards").insert(payload).execute()

    def get_recent_model_scorecards(self, days: int = 30) -> list[dict[str, Any]]:
        # Graceful: brak migracji 019 / błąd → [] (sekcja raportu się chowa).
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        try:
            response = (
                self._client.table("model_scorecards")
                .select("*")
                .gte("timestamp", cutoff)
                .order("timestamp", desc=True)
                .execute()
            )
        except Exception:
            logger.warning(
                "model_scorecards niedostępne — karta kondycji modelu pominięta "
                "(zaaplikuj migrację 019).",
                exc_info=True,
            )
            return []
        return list(_rows(response))

    def get_council_vote_history(
        self,
        symbols: list[str],
        *,
        window_days: int = 30,
        investors: list[str] | None = None,
    ) -> list[InvestorHistory]:
        if not symbols:
            return []
        cutoff = datetime.now(UTC) - timedelta(days=window_days)
        query = (
            self._client.table(self._council_votes_table)
            .select("investor_name, symbol, recommendation, timestamp")
            .in_("symbol", symbols)
            .gte("timestamp", cutoff.isoformat())
        )
        if investors:
            query = query.in_("investor_name", investors)
        response = query.order("timestamp", desc=True).execute()
        # Grupowanie per (inwestor, symbol); kolejność malejąca po czasie
        # zachowana (rekomendacje od najnowszej).
        grouped: dict[tuple[str, str], list[Literal["BUY", "SELL", "HOLD"]]] = {}
        for row in _rows(response):
            name = str(row.get("investor_name") or "")
            sym = str(row.get("symbol") or "")
            rec_raw = row.get("recommendation")
            if not name or not sym or rec_raw not in ("BUY", "SELL", "HOLD"):
                continue
            rec = cast("Literal['BUY', 'SELL', 'HOLD']", rec_raw)
            grouped.setdefault((name, sym), []).append(rec)
        return [
            InvestorHistory(
                investor_name=name,
                symbol=sym,
                recommendations=tuple(recs),
                window_days=window_days,
            )
            for (name, sym), recs in grouped.items()
        ]

    def get_council_votes_for_prediction(
        self, prediction_id: str
    ) -> list[InvestorOpinion]:
        response = (
            self._client.table(self._council_votes_table)
            .select("investor_name, recommendation, confidence, reasoning, key_factors")
            .eq("prediction_id", prediction_id)
            .execute()
        )
        out: list[InvestorOpinion] = []
        for row in _rows(response):
            rec = row.get("recommendation")
            if rec not in ("BUY", "SELL", "HOLD"):
                continue
            out.append(
                InvestorOpinion(
                    investor_name=str(row.get("investor_name", "?")),
                    recommendation=rec,
                    confidence=float(row.get("confidence") or 0.0),
                    reasoning=str(row.get("reasoning") or ""),
                    key_factors=tuple(row.get("key_factors") or []),
                )
            )
        return out

    def save_council_votes(
        self,
        prediction_id: str,
        symbol: str,
        votes: list[InvestorOpinion],
    ) -> None:
        if not votes:
            return
        rows = [
            _serialize_record(
                {
                    "prediction_id": prediction_id,
                    "symbol": symbol,
                    "investor_name": vote.investor_name,
                    "recommendation": vote.recommendation,
                    "confidence": vote.confidence,
                    "reasoning": vote.reasoning,
                    "key_factors": vote.key_factors,
                }
            )
            for vote in votes
        ]
        self._client.table(self._council_votes_table).insert(rows).execute()

    def update_prediction_accuracy(
        self,
        prediction_id: str,
        actual_price: Decimal,
        accuracy_score: float,
        is_trend_correct: bool,
        insight: str,
    ) -> None:
        payload = _serialize_record(
            {
                "actual_price_after_12h": actual_price,
                "accuracy_score": accuracy_score,
                "is_trend_correct": is_trend_correct,
                "correction_insights": insight,
            }
        )
        # Idempotency guard: aktualizujemy tylko gdy actual_price_after_12h jest
        # NULL. Bez tego dwa równoczesne cykle (np. ręczny workflow_dispatch
        # nakładający się na scheduled run) mogłyby nadpisać już ocenioną
        # predykcję drugą oceną z lekko innej ceny.
        (
            self._client.table(self._table)
            .update(payload)
            .eq("id", prediction_id)
            .is_("actual_price_after_12h", "null")
            .execute()
        )
