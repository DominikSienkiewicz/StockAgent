from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, cast

from supabase import create_client

from src.application.ports import RepositoryPort
from src.domain.council import InvestorOpinion
from src.domain.prediction import Prediction, TrendDirection
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.value_objects import FUNDAMENTALS_CACHE_TTL_HOURS, Fundamentals, Money

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
        response = query.order("timestamp", desc=True).limit(1).execute()
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
        response = (
            self._client.table(self._table)
            .select(
                "symbol, predicted_trend, is_trend_correct, timestamp, "
                "reasoning_text, correction_insights, price_at_prediction, "
                "actual_price_after_12h"
            )
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("is_trend_correct", "null")
            .order("timestamp", desc=True)
            .execute()
        )
        return _rows(response)

    def get_resolved_predictions_for_eval(self, days: int) -> list[dict[str, Any]]:
        """Zamknięte predykcje z ostatnich `days` dni z polami do ewaluacji
        (ceny + kierunek). Read-only; używane przez `src.tools.evaluate`.

        Metoda concrete-only (nie część `RepositoryPort`) — to narzędzie
        offline-analizy, nie część runtime'owego kontraktu use case'ów."""
        cutoff = datetime.now(UTC) - timedelta(days=days)
        response = (
            self._client.table(self._table)
            .select(
                "predicted_trend, is_trend_correct, price_at_prediction, "
                "predicted_target_price, actual_price_after_12h, timestamp"
            )
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("actual_price_after_12h", "null")
            .order("timestamp", desc=True)
            .execute()
        )
        return _rows(response)

    def get_accuracy_stats(self, days: int) -> dict[str, Any]:
        """Trafność kierunkowa predykcji z ostatnich `days` dni.

        `mean_accuracy` to hit-rate (udział trafionych kierunkowo predykcji),
        liczony z `is_trend_correct` — NIE z `accuracy_score` (bliskość ceny do
        celu, przy cold-starcie zawsze ~100% niezależnie od kierunku).

        Zwraca dict z kluczami: `mean_accuracy`, `sample_count`, `correct_count`,
        `days_window`. Gdy brak danych → mean_accuracy=None.
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)
        response = (
            self._client.table(self._table)
            .select("is_trend_correct")
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("is_trend_correct", "null")
            .execute()
        )
        rows = _rows(response)
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
        record = _serialize_record({"symbol": symbol, "price": price.amount})
        self._client.table(self._snapshot_table).insert(record).execute()

    def save_prediction(self, prediction: dict[str, Any]) -> str:
        record = _serialize_record(prediction)
        response = self._client.table(self._table).insert(record).execute()
        rows = _rows(response)
        if not rows:
            raise RuntimeError(
                f"Supabase insert returned no data for {record.get('symbol')}"
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
        self, embedding: list[float], limit: int = 3
    ) -> list[dict[str, Any]]:
        """Similarity search nad `prediction_logs.embedding` (pgvector) przez
        RPC `match_news_embeddings` (migracja 011).

        Graceful: gdy RPC nie istnieje (migracja niezaaplikowana) lub błąd
        sieci/Supabase — zwraca `[]`, żeby RAG był czysto opcjonalnym
        wzbogaceniem, a nie twardą zależnością predykcji."""
        try:
            response = self._client.rpc(
                "match_news_embeddings",
                {"query_embedding": embedding, "match_count": limit},
            ).execute()
        except Exception:
            logger.warning(
                "match_news_embeddings RPC unavailable — RAG retrieval skipped "
                "(apply migration 011 to enable).",
                exc_info=True,
            )
            return []
        return _rows(response)

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
