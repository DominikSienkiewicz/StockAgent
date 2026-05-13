from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, cast

from supabase import create_client

from src.application.ports import RepositoryPort
from src.domain.prediction import Prediction, TrendDirection
from src.domain.value_objects import Money

DEFAULT_TABLE = "prediction_logs"
DEFAULT_FEATURE_VIEW = "ml_feature_store"


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
    ) -> None:
        self._client = create_client(url, key)
        self._table = table_name
        self._view = feature_view

    # -----------------------------------------------------------------------
    # Reads
    # -----------------------------------------------------------------------

    def get_last_price(self, symbol: str) -> Money | None:
        response = (
            self._client.table(self._table)
            .select("price_at_prediction")
            .eq("symbol", symbol)
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )
        rows = _rows(response)
        if not rows:
            return None
        return Money(Decimal(str(rows[0]["price_at_prediction"])))

    def get_unverified_prediction(self, symbol: str) -> Prediction | None:
        response = (
            self._client.table(self._table)
            .select("*")
            .eq("symbol", symbol)
            .is_("actual_price_after_12h", "null")
            .order("timestamp", desc=True)
            .limit(1)
            .execute()
        )
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
        """
        self._client.rpc("refresh_ml_feature_store").execute()

    def get_recently_resolved_predictions(self, hours: int) -> list[dict[str, Any]]:
        """Predykcje zamknięte (z accuracy_score) w ostatnich `hours` godzinach."""
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        response = (
            self._client.table(self._table)
            .select("symbol, predicted_trend, accuracy_score, timestamp")
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("accuracy_score", "null")
            .order("timestamp", desc=True)
            .execute()
        )
        return _rows(response)

    def get_accuracy_stats(self, days: int) -> dict[str, Any]:
        """Średnia accuracy_score predykcji z ostatnich `days` dni
        (tylko te z wypełnionym `actual_price_after_12h`).

        Zwraca dict z kluczami: `mean_accuracy`, `sample_count`, `correct_count`,
        `days_window`. Gdy brak danych → mean_accuracy=None.
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)
        response = (
            self._client.table(self._table)
            .select("accuracy_score")
            .gte("timestamp", cutoff.isoformat())
            .not_.is_("accuracy_score", "null")
            .execute()
        )
        rows = _rows(response)
        scores = [float(r["accuracy_score"]) for r in rows if r.get("accuracy_score") is not None]

        if not scores:
            return {
                "mean_accuracy": None,
                "sample_count": 0,
                "correct_count": 0,
                "days_window": days,
            }
        return {
            "mean_accuracy": sum(scores) / len(scores),
            "sample_count": len(scores),
            "correct_count": sum(1 for s in scores if s > 0.5),
            "days_window": days,
        }

    # -----------------------------------------------------------------------
    # Writes
    # -----------------------------------------------------------------------

    def save_prediction(self, prediction: dict[str, Any]) -> str:
        record = _serialize_record(prediction)
        response = self._client.table(self._table).insert(record).execute()
        rows = _rows(response)
        if not rows:
            raise RuntimeError(
                f"Supabase insert returned no data for {record.get('symbol')}"
            )
        return str(rows[0]["id"])

    def update_prediction_accuracy(
        self,
        prediction_id: str,
        actual_price: Decimal,
        insight: str,
    ) -> None:
        payload = _serialize_record(
            {
                "actual_price_after_12h": actual_price,
                "correction_insights": insight,
            }
        )
        self._client.table(self._table).update(payload).eq("id", prediction_id).execute()
