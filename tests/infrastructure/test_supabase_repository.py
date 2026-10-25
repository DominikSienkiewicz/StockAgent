import logging
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, cast
from unittest.mock import MagicMock

import httpx
import pytest

from src.application.errors import NotFound, ProviderUnavailable
from src.application.ports import RepositoryPort, StoredShockAlert
from src.domain.paper_ledger import PaperTradeEntry
from src.domain.prediction import TrendDirection
from src.domain.session_close import SessionCloseSettlement
from src.domain.value_objects import FUNDAMENTALS_CACHE_TTL_HOURS, Fundamentals, Money
from src.infrastructure.adapters.supabase_repo import SupabaseRepository

# ---------------------------------------------------------------------------
# Helpers — Supabase client uses a builder/chain API:
#   client.table("x").select(...).eq(...).order(...).limit(...).execute()
# ---------------------------------------------------------------------------


def _set_chain_response(
    client: MagicMock, methods: list[str], data: list[dict[str, Any]]
) -> MagicMock:
    """Configures a chained call ending in .execute() returning `data`.

    Returns the response mock for further assertions if needed.
    """
    obj = client
    for method in methods:
        obj = getattr(obj, method).return_value
    response = MagicMock()
    response.data = data
    obj.execute.return_value = response
    return response


@pytest.fixture
def mock_client(mocker: Any) -> MagicMock:
    create = mocker.patch("src.infrastructure.adapters.supabase_repo.create_client")
    client = MagicMock()
    create.return_value = client
    return client


@pytest.fixture
def repo(mock_client: MagicMock) -> SupabaseRepository:
    return SupabaseRepository(url="https://test.supabase.co", key="anon-key")


# ---------------------------------------------------------------------------
# get_last_price / save_price_snapshot — price_snapshots table
# ---------------------------------------------------------------------------


class TestGetLastPrice:
    def test_returns_money_from_price_snapshots(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "order", "limit"],
            [{"price": 192.5}],
        )

        price = repo.get_last_price("AAPL")

        assert isinstance(price, Money)
        assert price.amount == Decimal("192.5")

    def test_returns_none_when_no_history(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client, ["table", "select", "eq", "order", "limit"], data=[]
        )
        assert repo.get_last_price("UNKNOWN") is None

    def test_queries_price_snapshots_table_with_symbol_filter(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client, ["table", "select", "eq", "order", "limit"], data=[]
        )

        repo.get_last_price("VOO")

        # Czyta z price_snapshots, NIE z prediction_logs
        mock_client.table.assert_called_with("price_snapshots")
        mock_client.table.return_value.select.return_value.eq.assert_called_with(
            "symbol", "VOO"
        )


class TestSavePriceSnapshot:
    def test_upserts_symbol_and_price_to_snapshots_table(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "upsert"], [{"id": "snap-1"}])

        repo.save_price_snapshot("AAPL", Money(Decimal("298.87")))

        mock_client.table.assert_called_with("price_snapshots")
        inserted = mock_client.table.return_value.upsert.call_args.args[0]
        assert inserted["symbol"] == "AAPL"
        # Decimal serializowany do str (JSON-safe)
        assert str(inserted["price"]) == "298.87"

    def test_upsert_uses_symbol_timestamp_hour_conflict_key(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """Idempotency (#7): re-run GHA / nakładający się workflow_dispatch /
        retry LangGraph nie może duplikować snapshotu. Upsert na konflikt
        (symbol, timestamp_hour) aktualizuje istniejący logiczny wiersz zamiast
        wstawiać duplikat (duplikaty podwójnie liczyłyby się w widoku
        ml_feature_store i w get_accuracy_stats)."""
        _set_chain_response(mock_client, ["table", "upsert"], [{"id": "snap-1"}])

        repo.save_price_snapshot("AAPL", Money(Decimal("298.87")))

        upsert = mock_client.table.return_value.upsert
        upsert.assert_called_once()
        assert upsert.call_args.kwargs["on_conflict"] == "symbol,timestamp_hour"

    def test_price_serialized_with_four_decimal_places(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # ARCH-38: kolumna DECIMAL(14,4) — połówka idzie od zera (ROUND_HALF_UP)
        # jak w Postgresie, a nie do parzystej, jak domyślnie w Pythonie.
        _set_chain_response(mock_client, ["table", "upsert"], [{"id": "snap-1"}])

        repo.save_price_snapshot("AAPL", Money(Decimal("90.11115")))

        inserted = mock_client.table.return_value.upsert.call_args.args[0]
        assert inserted["price"] == "90.1112"


# ---------------------------------------------------------------------------
# get_last_prediction_price — referencja dla cechy price_delta (P0: train/serve
# skew). Czyta cenę OSTATNIEJ zalogowanej predykcji z prediction_logs, bo widok
# ml_feature_store liczy price_delta jako LAG(price_at_prediction) — inference
# musi użyć tej samej referencji, nie ostatniego snapshotu.
# ---------------------------------------------------------------------------


class TestGetLastPredictionPrice:
    def test_returns_money_from_latest_prediction(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "order", "limit"],
            [{"price_at_prediction": 187.25}],
        )

        price = repo.get_last_prediction_price("AAPL")

        assert isinstance(price, Money)
        assert price.amount == Decimal("187.25")

    def test_returns_none_when_no_predictions(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client, ["table", "select", "eq", "order", "limit"], data=[]
        )
        assert repo.get_last_prediction_price("UNKNOWN") is None

    def test_reads_prediction_logs_not_snapshots(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client, ["table", "select", "eq", "order", "limit"], data=[]
        )

        repo.get_last_prediction_price("VOO")

        # Referencja = poprzednia zalogowana predykcja (prediction_logs),
        # NIE snapshot ceny (price_snapshots).
        mock_client.table.assert_called_with("prediction_logs")
        mock_client.table.return_value.select.return_value.eq.assert_called_with(
            "symbol", "VOO"
        )

    def test_excludes_rows_from_the_current_hour(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # ML-18: re-run w tej samej godzinie UTC nadpisze upsertem wiersz z tej
        # godziny — delta liczona od niego byłaby deltą od samej siebie.
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "lt", "order", "limit"],
            [{"price_at_prediction": 150.0}],
        )

        price = repo.get_last_prediction_price(
            "AAPL", before=datetime(2026, 10, 21, 14, 37, 12, tzinfo=UTC)
        )

        assert price == Money(Decimal("150.0"))
        lt = mock_client.table.return_value.select.return_value.eq.return_value.lt
        lt.assert_called_once_with("timestamp_hour", "2026-10-21T14:00:00+00:00")

    def test_before_in_another_zone_is_cut_to_the_utc_hour(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # `timestamp_hour` to date_trunc('hour', timestamp, 'UTC') (migracja 013).
        _set_chain_response(
            mock_client, ["table", "select", "eq", "lt", "order", "limit"], data=[]
        )
        warsaw = timezone(timedelta(hours=2))

        repo.get_last_prediction_price(
            "AAPL", before=datetime(2026, 10, 21, 16, 5, tzinfo=warsaw)
        )

        lt = mock_client.table.return_value.select.return_value.eq.return_value.lt
        lt.assert_called_once_with("timestamp_hour", "2026-10-21T14:00:00+00:00")


class TestGetLatestPrediction:
    """ALR-08/ALR-10: ostatnia predykcja symbolu z chwilą i uzasadnieniem — kontekst
    alertu szoku i referencja dryfu bramki volatility."""

    def test_returns_price_time_and_reasoning(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "order", "limit"],
            [
                {
                    "price_at_prediction": 187.25,
                    "timestamp": "2026-10-15T13:30:00+00:00",
                    "reasoning_text": "  Popyt na AI.  ",
                }
            ],
        )

        latest = repo.get_latest_prediction("AAPL")

        assert latest is not None
        assert latest.price == Money(Decimal("187.25"))
        assert latest.timestamp == datetime(2026, 10, 15, 13, 30, tzinfo=UTC)
        assert latest.reasoning == "Popyt na AI."
        mock_client.table.assert_called_with("prediction_logs")
        mock_client.table.return_value.select.return_value.eq.assert_called_with(
            "symbol", "AAPL"
        )
        mock_client.table.return_value.select.return_value.eq.return_value.order.assert_called_with(
            "timestamp", desc=True
        )

    def test_none_without_predictions(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "select", "eq", "order", "limit"], [])

        assert repo.get_latest_prediction("AAPL") is None

    def test_row_without_price_or_time_is_no_prediction(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "order", "limit"],
            [{"price_at_prediction": None, "timestamp": "2026-10-15T13:30:00+00:00"}],
        )

        assert repo.get_latest_prediction("AAPL") is None


# ---------------------------------------------------------------------------
# save_prediction
# ---------------------------------------------------------------------------


class TestSavePrediction:
    def test_returns_upserted_uuid(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # Upsert z domyślnym merge zwraca wiersz (z id) — save_prediction MUSI
        # nadal zwracać to id (graf używa go jako prediction_id).
        _set_chain_response(
            mock_client,
            ["table", "upsert"],
            [{"id": "uuid-789", "symbol": "AAPL"}],
        )

        record_id = repo.save_prediction(
            {
                "symbol": "AAPL",
                "price_at_prediction": Decimal("100.0"),
                "predicted_trend": "BULLISH",
                "predicted_target_price": Decimal("105.0"),
                "reasoning_text": "macro tailwinds",
            }
        )

        assert record_id == "uuid-789"

    def test_upsert_uses_symbol_timestamp_hour_conflict_key(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """Idempotency (#7): same-hour re-run aktualizuje istniejący wiersz
        zamiast duplikować. Bez on_conflict duplikaty podwójnie liczyłyby się
        w get_accuracy_stats i w widoku ml_feature_store."""
        _set_chain_response(
            mock_client, ["table", "upsert"], [{"id": "uuid-1", "symbol": "AAPL"}]
        )

        repo.save_prediction(
            {
                "symbol": "AAPL",
                "price_at_prediction": Decimal("100.0"),
                "predicted_trend": "BULLISH",
                "predicted_target_price": Decimal("105.0"),
            }
        )

        upsert = mock_client.table.return_value.upsert
        upsert.assert_called_once()
        assert upsert.call_args.kwargs["on_conflict"] == "symbol,timestamp_hour"

    def test_raises_when_upsert_returns_empty_rows(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # Supabase czasem zwraca puste response.data (np. przy konflikcie RLS).
        # Bez tego guardu prediction_id propagowałby się jako None / KeyError.
        _set_chain_response(mock_client, ["table", "upsert"], data=[])
        prediction = {
            "symbol": "AAPL",
            "price_at_prediction": Decimal("100.0"),
            "predicted_trend": "BULLISH",
            "predicted_target_price": Decimal("105.0"),
        }

        with pytest.raises(RuntimeError, match="AAPL"):
            repo.save_prediction(prediction)

    def test_serializes_decimal_values_for_json(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "upsert"], [{"id": "uuid-1"}])

        repo.save_prediction(
            {
                "symbol": "AAPL",
                "price_at_prediction": Decimal("100.55"),
                "predicted_target_price": Decimal("105.0"),
            }
        )

        inserted = mock_client.table.return_value.upsert.call_args.args[0]
        # Decimal nie jest JSON-serializowalny natywnie — adapter musi zamieniać na str/float
        assert isinstance(inserted["price_at_prediction"], str | float | int)
        assert str(inserted["price_at_prediction"]) == "100.55"

    def test_price_columns_are_quantized_to_decimal_14_4(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # ARCH-38: tylko kolumny DECIMAL(14,4); inne Decimale (np. pole spoza
        # tej skali) zachowują pełną precyzję.
        _set_chain_response(mock_client, ["table", "upsert"], [{"id": "uuid-1"}])

        repo.save_prediction(
            {
                "symbol": "AAPL",
                "price_at_prediction": Decimal("90.11115"),
                "predicted_target_price": Decimal("90.11111111101"),
                "interval_low": Decimal("-0.00005"),
                "interval_high": Decimal("91.00005"),
                "gate_delta_raw": Decimal("0.123456789"),
            }
        )

        inserted = mock_client.table.return_value.upsert.call_args.args[0]
        assert inserted["price_at_prediction"] == "90.1112"
        assert inserted["predicted_target_price"] == "90.1111"
        assert inserted["interval_low"] == "-0.0001"
        assert inserted["interval_high"] == "91.0001"
        assert inserted["gate_delta_raw"] == "0.123456789"


# ---------------------------------------------------------------------------
# get_unverified_prediction
# ---------------------------------------------------------------------------


class TestGetUnverifiedPrediction:
    def test_maps_row_to_prediction_domain_object(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "is_", "or_", "order", "limit"],
            [
                {
                    "id": "uuid-1",
                    "symbol": "AAPL",
                    "predicted_trend": "BULLISH",
                    "price_at_prediction": 100.0,
                    "predicted_target_price": 105.0,
                }
            ],
        )

        prediction = repo.get_unverified_prediction("AAPL")

        assert prediction is not None
        assert prediction.id == "uuid-1"
        assert prediction.symbol == "AAPL"
        assert prediction.predicted_trend == TrendDirection.BULLISH
        assert prediction.price_at_prediction == Decimal("100.0")
        assert prediction.predicted_target_price == Decimal("105.0")

    def test_maps_news_summary_and_reasoning_for_mistake_diagnosis(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """WIA-18: diagnoza błędu ma widzieć newsy i uzasadnienie, na których
        oparto prognozę — `select("*")` je zwraca, mapowanie je gubiło."""
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "is_", "or_", "order", "limit"],
            [
                {
                    "id": "uuid-1",
                    "symbol": "AAPL",
                    "predicted_trend": "BULLISH",
                    "price_at_prediction": 100.0,
                    "predicted_target_price": 105.0,
                    "news_summary": "Fed holds | Chips rally",
                    "reasoning_text": "Popyt na GPU rośnie.",
                }
            ],
        )

        prediction = repo.get_unverified_prediction("AAPL")

        assert prediction is not None
        assert prediction.news_summary == "Fed holds | Chips rally"
        assert prediction.reasoning == "Popyt na GPU rośnie."

    def test_missing_news_and_reasoning_map_to_none(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "is_", "or_", "order", "limit"],
            [
                {
                    "id": "uuid-1",
                    "symbol": "AAPL",
                    "predicted_trend": "BULLISH",
                    "price_at_prediction": 100.0,
                    "predicted_target_price": 105.0,
                    "news_summary": None,
                    "reasoning_text": None,
                }
            ],
        )

        prediction = repo.get_unverified_prediction("AAPL")

        assert prediction is not None
        assert prediction.news_summary is None
        assert prediction.reasoning is None

    @pytest.mark.parametrize(
        ("stored", "expected"),
        [
            ("bullish", TrendDirection.BULLISH),
            (None, TrendDirection.SIDEWAYS),
            ("UP", TrendDirection.SIDEWAYS),
        ],
    )
    def test_legacy_trend_values_do_not_block_reflection(
        self,
        repo: SupabaseRepository,
        mock_client: MagicMock,
        stored: str | None,
        expected: TrendDirection,
    ) -> None:
        """A2 (2026-10-19): wiersze sprzed ML-07 mają trend lowercase, NULL albo
        śmieć. `TrendDirection(...)` rzucał, a kolejka oldest-first podawała ten
        sam wiersz w każdym cyklu — reflect symbolu stał na zawsze."""
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "is_", "or_", "order", "limit"],
            [
                {
                    "id": "uuid-1",
                    "symbol": "AAPL",
                    "predicted_trend": stored,
                    "price_at_prediction": 100.0,
                    "predicted_target_price": 105.0,
                }
            ],
        )

        prediction = repo.get_unverified_prediction("AAPL")

        assert prediction is not None
        assert prediction.predicted_trend == expected

    def test_returns_none_when_no_unverified_prediction(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client, ["table", "select", "eq", "is_", "or_", "order", "limit"], []
        )
        assert repo.get_unverified_prediction("AAPL") is None

    def test_filters_by_null_actual_price(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client, ["table", "select", "eq", "is_", "or_", "order", "limit"], []
        )

        repo.get_unverified_prediction("AAPL")

        # is_("actual_price_after_12h", "null") — postgresT składnia dla IS NULL
        is_call = mock_client.table.return_value.select.return_value.eq.return_value.is_
        is_call.assert_called_with("actual_price_after_12h", "null")

    _FLAG_FILTER = (
        'data_quality_flags.is.null,data_quality_flags.not.cs.["expired_unverified"]'
    )

    def test_rows_flagged_as_expired_are_not_served_for_reflection(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """WIA-49: wiersz oznaczony `expired_unverified` jest już poza oceną — odczyt
        nierozliczonych go pomija (rzędy bez flag, `NULL` sprzed migracji 008, zostają)."""
        _set_chain_response(
            mock_client, ["table", "select", "eq", "is_", "or_", "order", "limit"], []
        )

        repo.get_unverified_prediction("AAPL")

        or_call = (
            mock_client.table.return_value.select.return_value.eq.return_value
            .is_.return_value.or_
        )
        or_call.assert_called_once_with(self._FLAG_FILTER)

    def test_the_flag_filter_comes_with_the_age_bounds(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "is_", "lte", "gte", "or_", "order", "limit"],
            [],
        )

        repo.get_unverified_prediction("AAPL", min_age_hours=6, max_age_hours=96)

        or_call = (
            mock_client.table.return_value.select.return_value.eq.return_value
            .is_.return_value.lte.return_value.gte.return_value.or_
        )
        or_call.assert_called_once_with(self._FLAG_FILTER)


    def test_orders_oldest_first_to_drain_backlog(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """#19: self-reflection musi opróżniać backlog — bierzemy NAJSTARSZĄ
        nierozliczoną predykcję (timestamp ASC), nie najnowszą. Przy DESC
        starsze nierozliczone predykcje nigdy nie stałyby się znów "najnowsze"
        i utknęłyby na zawsze (brak actual_price_after_12h → nie wchodzą do
        ml_feature_store, nie liczą się do hit-rate)."""
        _set_chain_response(
            mock_client, ["table", "select", "eq", "is_", "or_", "order", "limit"], []
        )

        repo.get_unverified_prediction("AAPL")

        order_call = (
            mock_client.table.return_value.select.return_value.eq.return_value
            .is_.return_value.or_.return_value.order
        )
        order_call.assert_called_with("timestamp", desc=False)

    def test_min_age_hours_adds_timestamp_upper_bound(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """min_age_hours>0 ogranicza do predykcji starszych niż próg —
        chroni przed przedwczesną oceną świeżo zapisanej predykcji przy
        nakładających się cyklach (manualny dispatch + scheduled)."""
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "is_", "lte", "or_", "order", "limit"],
            [],
        )

        repo.get_unverified_prediction("AAPL", min_age_hours=6)

        lte_call = (
            mock_client.table.return_value.select.return_value.eq.return_value
            .is_.return_value.lte
        )
        assert lte_call.call_args.args[0] == "timestamp"

    def test_no_timestamp_filter_when_min_age_zero(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """Domyślnie (min_age_hours=0) zachowanie bez zmian — brak filtra .lte,
        żeby nie złamać wstecznej kompatybilności."""
        _set_chain_response(
            mock_client, ["table", "select", "eq", "is_", "or_", "order", "limit"], []
        )

        repo.get_unverified_prediction("AAPL")

        lte = (
            mock_client.table.return_value.select.return_value.eq.return_value
            .is_.return_value.lte
        )
        lte.assert_not_called()

    def test_max_age_hours_adds_timestamp_lower_bound(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """WIA-04 (G09): po przerwie reflect oceniał predykcję sprzed tygodni
        bieżącą ceną, jakby to był horyzont „12h”. `max_age_hours` dokłada dolną
        granicę znacznika czasu — starsze predykcje zostają nierozliczone."""
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "is_", "gte", "or_", "order", "limit"],
            [],
        )
        before = datetime.now(UTC)

        repo.get_unverified_prediction("AAPL", max_age_hours=96)

        gte_call = (
            mock_client.table.return_value.select.return_value.eq.return_value
            .is_.return_value.gte
        )
        column, cutoff = gte_call.call_args.args
        assert column == "timestamp"
        expected = before - timedelta(hours=96)
        assert abs(datetime.fromisoformat(cutoff) - expected) < timedelta(minutes=1)

    def test_min_and_max_age_bound_the_window_from_both_sides(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "is_", "lte", "gte", "or_", "order", "limit"],
            [],
        )

        repo.get_unverified_prediction("AAPL", min_age_hours=6, max_age_hours=96)

        lte_builder = (
            mock_client.table.return_value.select.return_value.eq.return_value
            .is_.return_value.lte
        )
        assert lte_builder.call_args.args[0] == "timestamp"
        assert lte_builder.return_value.gte.call_args.args[0] == "timestamp"

    def test_no_lower_bound_without_max_age(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client, ["table", "select", "eq", "is_", "or_", "order", "limit"], []
        )

        repo.get_unverified_prediction("AAPL")

        gte = (
            mock_client.table.return_value.select.return_value.eq.return_value
            .is_.return_value.gte
        )
        gte.assert_not_called()


# ---------------------------------------------------------------------------
# update_prediction_accuracy
# ---------------------------------------------------------------------------


class TestCorrectionCategory:
    """WIA-57: kolumna `correction_category` istnieje od migracji 046 — zapis
    i odczyt tylko za flagą, inaczej PGRST204 zgubiłby rozliczenie predykcji."""

    @staticmethod
    def _update(repo: SupabaseRepository, mock_client: MagicMock) -> dict[str, Any]:
        from src.domain.mistake_taxonomy import MistakeCategory

        _set_chain_response(mock_client, ["table", "update", "eq", "is_"], [{"id": "u"}])
        repo.update_prediction_accuracy(
            prediction_id="u",
            actual_price=Decimal("99.0"),
            accuracy_score=0.5,
            is_trend_correct=False,
            insight="Zignorowałem Fed.",
            correction_category=MistakeCategory.MACRO,
        )
        payload: dict[str, Any] = mock_client.table.return_value.update.call_args.args[0]
        return payload

    def test_flag_off_keeps_the_column_out_of_the_payload(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        assert "correction_category" not in self._update(repo, mock_client)

    def test_flag_on_writes_the_category(self, mock_client: MagicMock) -> None:
        repo = SupabaseRepository(
            url="https://test.supabase.co", key="anon-key", correction_category_enabled=True
        )

        assert self._update(repo, mock_client)["correction_category"] == "MACRO"

    def test_lessons_read_selects_the_column_only_with_the_flag(
        self, mock_client: MagicMock
    ) -> None:
        select = mock_client.table.return_value.select
        _set_chain_response(
            mock_client, ["table", "select", "gte", "not_", "is_", "order", "range"], []
        )
        plain = SupabaseRepository(url="https://test.supabase.co", key="anon-key")
        flagged = SupabaseRepository(
            url="https://test.supabase.co", key="anon-key", correction_category_enabled=True
        )

        plain.get_resolved_predictions(30)
        assert "correction_category" not in select.call_args.args[0]
        flagged.get_resolved_predictions(30)
        assert "correction_category" in select.call_args.args[0]


class TestUpdatePredictionAccuracy:
    def test_updates_record_with_price_accuracy_and_insight(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client, ["table", "update", "eq", "is_"], [{"id": "uuid-1"}]
        )

        repo.update_prediction_accuracy(
            prediction_id="uuid-1",
            actual_price=Decimal("99.0"),
            accuracy_score=0.87,
            is_trend_correct=False,
            insight="Zignorowałem makro.",
        )

        payload = mock_client.table.return_value.update.call_args.args[0]
        # Decimal musi być serializowany
        assert str(payload["actual_price_after_12h"]) == "99.0"
        # accuracy_score zamyka pętlę feedback — bez niego get_accuracy_stats() pusty
        assert payload["accuracy_score"] == pytest.approx(0.87)
        # is_trend_correct napędza trafność raportu (kierunek, nie bliskość ceny)
        assert payload["is_trend_correct"] is False
        assert payload["correction_insights"] == "Zignorowałem makro."
        mock_client.table.return_value.update.return_value.eq.assert_called_with(
            "id", "uuid-1"
        )

    def test_idempotency_guard_filters_already_resolved_predictions(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """Idempotency: update musi mieć WHERE actual_price_after_12h IS NULL.

        Bez tego dwa równoczesne cykle (np. ręczny workflow_dispatch nakładający
        się na scheduled run) mogłyby nadpisać już ocenioną predykcję drugą
        oceną z lekko innej ceny."""
        _set_chain_response(
            mock_client, ["table", "update", "eq", "is_"], [{"id": "uuid-1"}]
        )

        repo.update_prediction_accuracy(
            prediction_id="uuid-1",
            actual_price=Decimal("99.0"),
            accuracy_score=0.87,
            is_trend_correct=True,
            insight="x",
        )

        is_call = (
            mock_client.table.return_value.update.return_value.eq.return_value.is_
        )
        is_call.assert_called_with("actual_price_after_12h", "null")

    def test_actual_price_is_quantized_to_the_column_scale(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # ARCH-38: `actual_price_after_12h` to DECIMAL(14,4).
        _set_chain_response(
            mock_client, ["table", "update", "eq", "is_"], [{"id": "uuid-1"}]
        )

        repo.update_prediction_accuracy(
            prediction_id="uuid-1",
            actual_price=Decimal("99.00005"),
            accuracy_score=0.87,
            is_trend_correct=True,
            insight="x",
        )

        payload = mock_client.table.return_value.update.call_args.args[0]
        assert payload["actual_price_after_12h"] == "99.0001"


# ---------------------------------------------------------------------------
# get_feature_store_data
# ---------------------------------------------------------------------------


class TestGetFeatureStoreData:
    def test_returns_rows_from_materialized_view(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        rows: list[dict[str, Any]] = [
            {"price_current": 100.0, "sentiment_score": 75.0, "llm_trend_signal": 1},
            {"price_current": 102.0, "sentiment_score": 80.0, "llm_trend_signal": 1},
        ]
        _set_chain_response(
            mock_client, ["table", "select", "eq", "order", "order", "range"], rows
        )

        result = repo.get_feature_store_data("AAPL")

        assert result == rows
        mock_client.table.assert_any_call("ml_feature_store")

    def test_returns_empty_list_when_no_data(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "select", "eq", "order"], [])
        assert repo.get_feature_store_data("UNKNOWN") == []


# ---------------------------------------------------------------------------
# refresh_feature_store — retry z exponential backoff
# ---------------------------------------------------------------------------


class TestRefreshFeatureStoreRetry:
    def test_retries_on_transient_error_then_succeeds(
        self, repo: SupabaseRepository, mock_client: MagicMock, mocker: Any
    ) -> None:
        # Pierwsze 2 wywołania rzucają, trzecie OK.
        rpc = mock_client.rpc.return_value
        rpc.execute.side_effect = [
            RuntimeError("503 Service Unavailable"),
            RuntimeError("503 Service Unavailable"),
            MagicMock(data=[]),
        ]
        mocker.patch("src.infrastructure.adapters.supabase_repo.time.sleep")

        repo.refresh_feature_store()  # nie powinno rzucić

        assert rpc.execute.call_count == 3

    def test_raises_after_exhausting_retries(
        self, repo: SupabaseRepository, mock_client: MagicMock, mocker: Any
    ) -> None:
        rpc = mock_client.rpc.return_value
        rpc.execute.side_effect = RuntimeError("persistent failure")
        mocker.patch("src.infrastructure.adapters.supabase_repo.time.sleep")

        with pytest.raises(RuntimeError, match="persistent failure"):
            repo.refresh_feature_store()

        # 3 próby = MAX_ATTEMPTS
        assert rpc.execute.call_count == 3

    def test_succeeds_on_first_attempt_does_not_sleep(
        self, repo: SupabaseRepository, mock_client: MagicMock, mocker: Any
    ) -> None:
        rpc = mock_client.rpc.return_value
        rpc.execute.return_value = MagicMock(data=[])
        sleep_mock = mocker.patch("src.infrastructure.adapters.supabase_repo.time.sleep")

        repo.refresh_feature_store()

        assert rpc.execute.call_count == 1
        sleep_mock.assert_not_called()


# ---------------------------------------------------------------------------
# get_cached_fundamentals / save_fundamentals — fundamentals_cache table
# ---------------------------------------------------------------------------


class TestSaveFundamentals:
    def test_save_fundamentals_upserts_row(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """save_fundamentals musi wywołać upsert na fundamentals_cache z pełnym payloadem."""
        _set_chain_response(mock_client, ["table", "upsert"], [])
        fetched = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)
        fund = Fundamentals(
            trailing_pe=25.4,
            forward_pe=22.1,
            peg_ratio=1.5,
            eps_growth_yoy=0.12,
            fetched_at=fetched,
        )

        repo.save_fundamentals("AAPL", fund)

        mock_client.table.assert_called_with("fundamentals_cache")
        payload = mock_client.table.return_value.upsert.call_args.args[0]
        assert payload["symbol"] == "AAPL"
        assert payload["trailing_pe"] == pytest.approx(25.4)
        assert payload["forward_pe"] == pytest.approx(22.1)
        assert payload["peg_ratio"] == pytest.approx(1.5)
        assert payload["eps_growth_yoy"] == pytest.approx(0.12)
        # fetched_at musi być ISO string
        assert payload["fetched_at"] == fetched.isoformat()


class TestGetCachedFundamentals:
    def test_get_cached_fundamentals_returns_none_when_empty(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """Gdy Supabase zwraca puste data, metoda zwraca None."""
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "gte"],
            [],
        )

        result = repo.get_cached_fundamentals("AAPL")

        assert result is None

    def test_get_cached_fundamentals_returns_fundamentals_when_fresh(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """Gdy Supabase zwraca wiersz, metoda zwraca wypełniony obiekt Fundamentals."""
        fetched_iso = "2026-05-18T12:00:00+00:00"
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "gte"],
            [
                {
                    "symbol": "AAPL",
                    "trailing_pe": 25.4,
                    "forward_pe": 22.1,
                    "peg_ratio": 1.5,
                    "eps_growth_yoy": 0.12,
                    "fetched_at": fetched_iso,
                }
            ],
        )

        result = repo.get_cached_fundamentals("AAPL")

        assert result is not None
        assert isinstance(result, Fundamentals)
        assert result.trailing_pe == pytest.approx(25.4)
        assert result.forward_pe == pytest.approx(22.1)
        assert result.peg_ratio == pytest.approx(1.5)
        assert result.eps_growth_yoy == pytest.approx(0.12)
        assert result.fetched_at == datetime.fromisoformat(fetched_iso)

    def test_get_cached_fundamentals_uses_ttl_filter(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """Metoda filtruje wyniki przez .gte("fetched_at", cutoff) z tolerancją 1 minuty."""
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "gte"],
            [],
        )

        before_call = datetime.now(UTC)
        repo.get_cached_fundamentals("AAPL")
        after_call = datetime.now(UTC)

        # Pobierz argument cutoff przekazany do .gte(...)
        gte_call = (
            mock_client.table.return_value.select.return_value.eq.return_value.gte
        )
        gte_call.assert_called_once()
        field_arg, cutoff_arg = gte_call.call_args.args
        assert field_arg == "fetched_at"

        cutoff_dt = datetime.fromisoformat(cutoff_arg)
        ttl = timedelta(hours=FUNDAMENTALS_CACHE_TTL_HOURS)
        expected_min = before_call - ttl - timedelta(minutes=1)
        expected_max = after_call - ttl + timedelta(minutes=1)
        assert expected_min <= cutoff_dt <= expected_max


# ---------------------------------------------------------------------------
# Port conformance
# ---------------------------------------------------------------------------


class TestAdapterImplementsPort:
    def test_is_repository_port(self) -> None:
        assert issubclass(SupabaseRepository, RepositoryPort)


# ---------------------------------------------------------------------------
# save_council_votes — strukturalny audit trail rady
# ---------------------------------------------------------------------------


class TestSaveCalibration:
    def test_updates_prediction_with_calibration(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client, ["table", "update", "eq"], [{"id": "pred-1"}]
        )

        repo.save_calibration("pred-1", 0.72, "Przepewna na krypto.")

        mock_client.table.assert_called_with("prediction_logs")
        payload = mock_client.table.return_value.update.call_args.args[0]
        assert payload["confidence_calibration"] == pytest.approx(0.72)
        assert payload["calibration_insight"] == "Przepewna na krypto."


class TestSaveCalibrationScores:
    """WIA-65: arytmetyczne oceny kalibracji jednym wywołaniem RPC (migracja 077)."""

    @staticmethod
    def _execute(client: MagicMock) -> MagicMock:
        return cast(MagicMock, client.rpc.return_value.execute)

    def test_sends_every_score_in_one_rpc_call(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        self._execute(mock_client).return_value = MagicMock(data=2)

        updated = repo.save_calibration_scores({"p1": 0.19, "p2": 0.96}, score_version=2)

        assert updated == 2
        mock_client.rpc.assert_called_once_with(
            "save_calibration_scores",
            {
                "p_scores": [{"id": "p1", "score": 0.19}, {"id": "p2", "score": 0.96}],
                "p_score_version": 2,
            },
        )
        mock_client.table.assert_not_called()

    def test_no_scores_make_no_call(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        assert repo.save_calibration_scores({}, score_version=1) == 0
        mock_client.rpc.assert_not_called()

    def test_a_payload_that_is_not_a_count_is_malformed(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.application.errors import MalformedPayload

        self._execute(mock_client).return_value = MagicMock(data=[{"updated": 2}])

        with pytest.raises(MalformedPayload):
            repo.save_calibration_scores({"p1": 0.5}, score_version=1)

    def test_missing_function_leaves_as_not_found(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from postgrest.exceptions import APIError

        self._execute(mock_client).side_effect = APIError(
            {"code": "PGRST202", "message": "no function"}
        )

        with pytest.raises(NotFound):
            repo.save_calibration_scores({"p1": 0.5}, score_version=1)


class TestDbSizeReport:
    """NIEZ-37: RPC `db_size_report` (migracja 077) — relacja i bajty, brak funkcji
    przed migracją jako `NotFound`."""

    @staticmethod
    def _execute(client: MagicMock) -> MagicMock:
        return cast(MagicMock, client.rpc.return_value.execute)

    def test_maps_rows_and_drops_malformed_ones(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.db_size import RelationSize

        self._execute(mock_client).return_value = MagicMock(
            data=[
                {"relation_name": "prediction_logs", "total_bytes": 4096},
                {"relation_name": "flag", "total_bytes": True},
                {"relation_name": None, "total_bytes": 1},
                "not a row",
            ]
        )

        assert repo.get_db_size_report() == [RelationSize("prediction_logs", 4096)]
        mock_client.rpc.assert_called_once_with("db_size_report")

    def test_a_payload_that_is_not_a_list_is_malformed(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.application.errors import MalformedPayload

        self._execute(mock_client).return_value = MagicMock(data={"size": 1})

        with pytest.raises(MalformedPayload):
            repo.get_db_size_report()

    def test_missing_function_leaves_as_not_found(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from postgrest.exceptions import APIError

        self._execute(mock_client).side_effect = APIError(
            {"code": "PGRST202", "message": "no function"}
        )

        with pytest.raises(NotFound):
            repo.get_db_size_report()


class TestDiagnosisJudgement:
    """EXP-10 — próbka diagnoz bez oceny i zapis oceny (odrzucenie podmienia tylko
    przeczytaną diagnozę)."""

    @staticmethod
    def _query(client: MagicMock) -> MagicMock:
        select = client.table.return_value.select.return_value
        return cast(
            MagicMock,
            select.eq.return_value.is_.return_value.not_.is_.return_value.gte.return_value
            .order.return_value.order.return_value,
        )

    def test_reads_unjudged_wrong_predictions_of_the_window(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        rows = [
            _deferred_row(id="fallback", data_quality_flags=["llm_analysis_failed"]),
            _deferred_row(correction_insights="Zignorowałem Fed."),
        ]
        self._query(mock_client).range.return_value.execute.return_value = MagicMock(data=rows)

        (row,) = repo.get_diagnoses_to_judge(7)

        select = mock_client.table.return_value.select.return_value
        select.eq.assert_called_once_with("is_trend_correct", False)
        select.eq.return_value.is_.assert_called_once_with("diagnosis_judge_score", "null")
        assert row.prediction.id == "p-1"
        assert row.insight == "Zignorowałem Fed."
        assert row.actual_price == Decimal("97.0000")

    def test_a_pass_writes_only_the_score(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "update", "eq"], [])

        repo.save_diagnosis_judgement("p-1", score=3)

        update = mock_client.table.return_value.update
        assert update.call_args.args[0] == {"diagnosis_judge_score": 3}
        update.return_value.eq.assert_called_once_with("id", "p-1")

    def test_a_rejection_moves_the_text_it_read(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.prediction import DIAGNOSIS_REJECTED_INSIGHT

        _set_chain_response(mock_client, ["table", "update", "eq", "eq"], [])

        repo.save_diagnosis_judgement("p-1", score=1, rejected_insight="Rynek był zmienny.")

        update = mock_client.table.return_value.update
        assert update.call_args.args[0] == {
            "diagnosis_judge_score": 1,
            "correction_insights": DIAGNOSIS_REJECTED_INSIGHT,
            "rejected_diagnosis": "Rynek był zmienny.",
        }
        update.return_value.eq.return_value.eq.assert_called_once_with(
            "correction_insights", "Rynek był zmienny."
        )


class TestGetPersonaAccuracy:
    def test_maps_rpc_rows_to_weights(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.value_objects import AssetType

        _set_chain_response(
            mock_client,
            ["rpc"],
            [
                {"investor_name": "Wood", "weight": 1.4},
                {"investor_name": "Burry", "weight": 0.7},
            ],
        )

        result = repo.get_persona_accuracy(AssetType.CRYPTO)

        assert result == {"Wood": 1.4, "Burry": 0.7}

    def test_returns_empty_on_rpc_error(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.value_objects import AssetType

        mock_client.rpc.side_effect = Exception("RPC missing")
        assert repo.get_persona_accuracy(AssetType.STOCK) == {}


class TestGetPersonaTrackRecord:
    """#3 — RPC `persona_accuracy_stats` (migracja 018) niesie hit-rate ORAZ
    liczbę rozliczonych głosów; RPC z migracji 015 zwracało samą wagę."""

    def test_maps_rpc_rows_to_hit_rate_and_vote_count(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["rpc"],
            [
                {"investor_name": "Buffett", "hit_rate": 0.68, "vote_count": 22},
                {"investor_name": "Wood", "hit_rate": 0.41, "vote_count": 18},
            ],
        )

        result = repo.get_persona_track_record()

        assert result == {"Buffett": (0.68, 22), "Wood": (0.41, 18)}
        assert mock_client.rpc.call_args.args[0] == "persona_accuracy_stats"

    def test_passes_window_days_to_rpc(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["rpc"], [])

        repo.get_persona_track_record(window_days=30)

        assert mock_client.rpc.call_args.args[1] == {"window_days": 30}

    def test_skips_rows_with_missing_fields(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["rpc"],
            [
                {"investor_name": "Buffett", "hit_rate": 0.68, "vote_count": 22},
                {"investor_name": "Wood", "hit_rate": None, "vote_count": 18},
                {"investor_name": "Lynch", "hit_rate": 0.5},
                {"hit_rate": 0.9, "vote_count": 30},
            ],
        )

        assert repo.get_persona_track_record() == {"Buffett": (0.68, 22)}

    def test_returns_empty_on_rpc_error(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # Graceful: brak migracji 018 → brak sekcji w raporcie, nie wyjątek.
        mock_client.rpc.side_effect = Exception("RPC missing")

        assert repo.get_persona_track_record() == {}


class TestGetPersonaScoring:
    """WIA-44 — RPC `persona_scoring_stats` (migracja 065): hit-rate, liczba głosów,
    średni Brier i log loss pewności; pasmo HOLD krypto idzie parametrem RPC."""

    @staticmethod
    def _repo(crypto: list[str] | None = None) -> SupabaseRepository:
        return SupabaseRepository(
            url="https://test.supabase.co", key="anon-key", crypto_symbols=crypto or []
        )

    def test_maps_rpc_rows_to_scoring(self, mock_client: MagicMock) -> None:
        from src.domain.persona_track_record import PersonaScoring

        _set_chain_response(
            mock_client,
            ["rpc"],
            [
                {"investor_name": "Buffett", "hit_rate": 0.68, "vote_count": 22,
                 "brier": 0.21, "log_loss": 0.61},
                {"investor_name": "Wood", "hit_rate": 0.41, "vote_count": 18,
                 "brier": 0.33, "log_loss": 0.9},
            ],
        )

        result = self._repo().get_persona_scoring(window_days=30)

        assert result == {
            "Buffett": PersonaScoring(hit_rate=0.68, votes=22, brier=0.21, log_loss=0.61),
            "Wood": PersonaScoring(hit_rate=0.41, votes=18, brier=0.33, log_loss=0.9),
        }

    def test_passes_the_window_and_the_crypto_symbols_to_the_rpc(
        self, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["rpc"], [])

        self._repo(["btc", "eth"]).get_persona_scoring(window_days=45)

        assert mock_client.rpc.call_args.args == (
            "persona_scoring_stats",
            {"window_days": 45, "crypto_symbols": ["BTC", "ETH"]},
        )

    def test_skips_rows_with_missing_fields(self, mock_client: MagicMock) -> None:
        _set_chain_response(
            mock_client,
            ["rpc"],
            [
                {"investor_name": "Buffett", "hit_rate": 0.68, "vote_count": 22,
                 "brier": 0.21, "log_loss": 0.61},
                {"investor_name": "Wood", "hit_rate": 0.4, "vote_count": 18,
                 "brier": None, "log_loss": 0.9},
                {"investor_name": "Lynch", "hit_rate": 0.5, "vote_count": 9, "brier": 0.2},
                {"hit_rate": 0.9, "vote_count": 30, "brier": 0.1, "log_loss": 0.3},
            ],
        )

        assert list(self._repo().get_persona_scoring()) == ["Buffett"]

    def test_without_the_migration_it_degrades_to_empty_and_names_065(
        self, mock_client: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        from postgrest.exceptions import APIError

        mock_client.rpc.side_effect = APIError(
            {"code": "PGRST202", "message": "function persona_scoring_stats not found"}
        )

        with caplog.at_level(logging.WARNING):
            assert self._repo().get_persona_scoring() == {}

        assert "apply migration 065" in caplog.text


class TestGetCouncilVoteSamples:
    """WIA-45 — głosy person z `council_votes` (prediction_id, persona, rekomendacja)
    z ostatnich N dni; bez RPC i bez migracji, paginowane po `id`."""

    _CHAIN = ["table", "select", "gte", "order", "order", "range"]

    def test_maps_rows_to_vote_samples(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.council_redundancy import CouncilVoteSample

        _set_chain_response(
            mock_client,
            self._CHAIN,
            [
                {"prediction_id": "p-1", "investor_name": "Buffett", "recommendation": "buy"},
                {"prediction_id": "p-1", "investor_name": "Wood", "recommendation": "SELL"},
            ],
        )

        samples = repo.get_council_vote_samples(window_days=30)

        assert samples == [
            CouncilVoteSample("p-1", "Buffett", "BUY"),
            CouncilVoteSample("p-1", "Wood", "SELL"),
        ]
        assert mock_client.table.call_args.args[0] == "council_votes"

    def test_skips_rows_without_an_id_a_persona_or_a_valid_recommendation(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            self._CHAIN,
            [
                {"prediction_id": None, "investor_name": "Buffett", "recommendation": "BUY"},
                {"prediction_id": "p-1", "investor_name": "", "recommendation": "BUY"},
                {"prediction_id": "p-1", "investor_name": "Wood", "recommendation": "STRONG_BUY"},
                {"prediction_id": "p-1", "investor_name": "Lynch", "recommendation": "HOLD"},
            ],
        )

        assert [s.investor_name for s in repo.get_council_vote_samples()] == ["Lynch"]

    def test_database_error_degrades_to_no_samples(
        self, repo: SupabaseRepository, mock_client: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        from postgrest.exceptions import APIError

        chain = mock_client
        for step in self._CHAIN[:-1]:
            chain = getattr(chain, step).return_value
        chain.range.return_value.execute.side_effect = APIError(
            {"code": "PGRST000", "message": "down"}
        )

        with caplog.at_level(logging.WARNING):
            assert repo.get_council_vote_samples() == []

        assert "council_votes" in caplog.text


_VOTE_OUTCOME_CHAIN = ["rpc", "order", "order", "order", "range"]


def _vote_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "prediction_id": "p-1",
        "investor_name": "Buffett",
        "recommendation": "HOLD",
        "confidence": 0.8,
        "symbol": "BTC",
        "price_at_prediction": "100.0000",
        "actual_price_after_12h": "102.0000",
        "predicted_at": "2026-09-01T10:00:00+00:00",
        "settled_at": "2026-09-02T05:30:00+00:00",
    }
    row.update(overrides)
    return row


class TestPersonaVoteOutcomes:
    """WIA-51 — RPC `persona_vote_outcomes` (migracja 045) zwraca surowe głosy;
    trafność i wagi liczy domena z pasmem HOLD klasy aktywa."""

    @staticmethod
    def _flagged_repo() -> SupabaseRepository:
        return SupabaseRepository(
            url="https://test.supabase.co",
            key="anon-key",
            persona_vote_outcomes_enabled=True,
            crypto_symbols=["btc"],
        )

    def test_maps_rpc_rows_to_domain_outcomes(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.persona_track_record import PersonaVoteOutcome

        _set_chain_response(mock_client, _VOTE_OUTCOME_CHAIN, [_vote_row()])

        (outcome,) = repo.get_persona_vote_outcomes(window_days=30)

        assert outcome == PersonaVoteOutcome(
            prediction_id="p-1",
            investor_name="Buffett",
            recommendation="HOLD",
            confidence=0.8,
            symbol="BTC",
            price_at_prediction=Decimal("100.0000"),
            actual_price=Decimal("102.0000"),
            predicted_at=datetime(2026, 9, 1, 10, 0, tzinfo=UTC),
            settled_at=datetime(2026, 9, 2, 5, 30, tzinfo=UTC),
        )
        assert mock_client.rpc.call_args.args == ("persona_vote_outcomes", {"window_days": 30})

    def test_skips_rows_without_prices_or_persona(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            _VOTE_OUTCOME_CHAIN,
            [
                _vote_row(actual_price_after_12h=None),
                _vote_row(investor_name=None),
                _vote_row(settled_at=None),
            ],
        )

        (outcome,) = repo.get_persona_vote_outcomes()

        assert outcome.settled_at is None

    def test_flag_scores_hold_on_crypto_with_the_crypto_tolerance(
        self, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            _VOTE_OUTCOME_CHAIN,
            [_vote_row(), _vote_row(prediction_id="p-2", symbol="AAPL")],
        )

        stats = self._flagged_repo().get_persona_track_record(window_days=90)

        # BTC +2% mieści się w paśmie krypto, AAPL +2% nie — RPC 018 dałoby 0/2.
        assert stats == {"Buffett": (0.5, 2)}
        assert mock_client.rpc.call_args.args[0] == "persona_vote_outcomes"

    def test_flag_derives_weights_from_the_same_rule(self, mock_client: MagicMock) -> None:
        from src.domain.value_objects import AssetType

        _set_chain_response(
            mock_client,
            _VOTE_OUTCOME_CHAIN,
            [_vote_row(), _vote_row(prediction_id="p-2", symbol="AAPL")],
        )

        weights = self._flagged_repo().get_persona_accuracy(AssetType.STOCK)

        assert weights == {"Buffett": pytest.approx(1.0)}
        assert mock_client.rpc.call_args.args[0] == "persona_vote_outcomes"

    def test_flag_without_migration_degrades_to_empty(
        self, mock_client: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        from postgrest.exceptions import APIError

        from src.domain.value_objects import AssetType

        mock_client.rpc.side_effect = APIError(
            {"code": "PGRST202", "message": "function persona_vote_outcomes not found"}
        )
        repo = self._flagged_repo()

        with caplog.at_level(logging.WARNING):
            assert repo.get_persona_track_record() == {}
            assert repo.get_persona_accuracy(AssetType.STOCK) == {}

        assert "apply migration 045" in caplog.text

    @pytest.mark.parametrize(
        ("code", "hinted"),
        [
            ("PGRST202", True),
            ("PGRST205", True),
            ("42P01", True),
            ("42883", True),
            ("42703", False),
            ("PGRST204", False),
        ],
    )
    def test_migration_hint_only_for_a_missing_function_or_table(
        self, mock_client: MagicMock, caplog: pytest.LogCaptureFixture, code: str, hinted: bool
    ) -> None:
        # NIEZ-16 (decyzja 2026-10-23): podpowiedź „apply migration” wyłącznie dla
        # brakującej funkcji albo tabeli; brak kolumny ma osobną, tolerancyjną ścieżkę.
        from postgrest.exceptions import APIError

        mock_client.rpc.side_effect = APIError({"code": code, "message": "missing"})

        with caplog.at_level(logging.WARNING):
            assert self._flagged_repo().get_persona_track_record() == {}

        assert ("apply migration 045" in caplog.text) is hinted
        if not hinted:
            assert "APIError" in caplog.text

    def test_without_flag_the_old_rpcs_stay(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["rpc"], [])

        repo.get_persona_track_record()

        assert mock_client.rpc.call_args.args[0] == "persona_accuracy_stats"


_DEFERRED_CHAIN = ["table", "select", "in_", "eq", "order", "range"]


def _deferred_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": "p-1",
        "symbol": "AAPL",
        "timestamp": "2026-08-01T10:00:00+00:00",
        "predicted_trend": "BULLISH",
        "price_at_prediction": "100.0000",
        "predicted_target_price": "104.0000",
        "actual_price_after_12h": "97.0000",
        "news_summary": "Nagłówek | Drugi",
        "reasoning_text": "Popyt rośnie.",
        "correction_insights": "Niska zmienność — diagnoza LLM pominięta (bez ponowienia).",
        "is_trend_correct": False,
        "data_quality_flags": [],
        "decision_receipts": {"schema_version": 1, "effective_threshold": "0.03"},
    }
    row.update(overrides)
    return row


class TestDeferredDiagnoses:
    """WIA-53 — błędne predykcje z placeholderem zamiast diagnozy i zapis
    diagnozy, który podmienia tylko przeczytany placeholder."""

    def test_reads_wrong_predictions_with_a_placeholder_newest_first(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.prediction import UNDIAGNOSED_MISTAKE_PLACEHOLDERS

        _set_chain_response(mock_client, _DEFERRED_CHAIN, [_deferred_row()])

        (row,) = repo.get_deferred_diagnoses(limit=5)

        select = mock_client.table.return_value.select
        select.return_value.in_.assert_called_once_with(
            "correction_insights", list(UNDIAGNOSED_MISTAKE_PLACEHOLDERS)
        )
        select.return_value.in_.return_value.eq.assert_called_once_with(
            "is_trend_correct", False
        )
        select.return_value.in_.return_value.eq.return_value.order.assert_called_once_with(
            "timestamp", desc=True
        )
        assert row.prediction.id == "p-1"
        assert row.prediction.news_summary == "Nagłówek | Drugi"
        assert row.prediction.reasoning == "Popyt rośnie."
        assert row.actual_price == Decimal("97.0000")
        assert row.is_trend_correct is False
        assert row.placeholder.startswith("Niska zmienność")
        assert row.receipt == {"schema_version": 1, "effective_threshold": "0.03"}

    def test_skips_llm_fallback_rows_and_caps_at_the_limit(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        rows = [
            _deferred_row(id="fallback", data_quality_flags=["llm_analysis_failed"]),
            _deferred_row(id="p-1"),
            _deferred_row(id="p-2"),
            _deferred_row(id="no-price", actual_price_after_12h=None),
        ]
        _set_chain_response(mock_client, _DEFERRED_CHAIN, rows)

        out = repo.get_deferred_diagnoses(limit=1)

        assert [row.prediction.id for row in out] == ["p-1"]

    def test_without_migration_020_reads_without_the_receipt(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from postgrest.exceptions import APIError

        ok = MagicMock()
        row_without_receipt = _deferred_row()
        del row_without_receipt["decision_receipts"]
        ok.data = [row_without_receipt]
        missing = APIError(
            {"code": "42703", "message": "column prediction_logs.decision_receipts does not exist"}
        )
        range_ = mock_client.table.return_value.select.return_value.in_.return_value
        range_.eq.return_value.order.return_value.range.return_value.execute.side_effect = [
            missing,
            ok,
        ]

        (row,) = repo.get_deferred_diagnoses(limit=5)

        assert row.receipt is None
        second_select = mock_client.table.return_value.select.call_args.args[0]
        assert "decision_receipts" not in second_select

    def test_save_replaces_only_the_placeholder_it_read(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.mistake_taxonomy import MistakeCategory

        _set_chain_response(mock_client, ["table", "update", "eq", "eq"], [])

        repo.save_backfilled_diagnosis(
            "p-1", "Zignorowałem Fed.", replaces="placeholder",
            correction_category=MistakeCategory.MACRO,
        )

        update = mock_client.table.return_value.update
        # Flaga 046 off — bez klucza kategorii.
        assert update.call_args.args[0] == {"correction_insights": "Zignorowałem Fed."}
        update.return_value.eq.assert_called_once_with("id", "p-1")
        update.return_value.eq.return_value.eq.assert_called_once_with(
            "correction_insights", "placeholder"
        )

    def test_save_writes_the_category_with_the_flag(self, mock_client: MagicMock) -> None:
        from src.domain.mistake_taxonomy import MistakeCategory

        _set_chain_response(mock_client, ["table", "update", "eq", "eq"], [])
        repo = SupabaseRepository(
            url="https://test.supabase.co", key="anon-key", correction_category_enabled=True
        )

        repo.save_backfilled_diagnosis(
            "p-1", "Zignorowałem Fed.", replaces="placeholder",
            correction_category=MistakeCategory.MACRO,
        )

        payload = mock_client.table.return_value.update.call_args.args[0]
        assert payload["correction_category"] == "MACRO"


class TestGetCouncilVoteHistory:
    def test_groups_rows_into_investor_histories(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "in_", "gte", "order", "order", "range"],
            [
                {"investor_name": "Soros", "symbol": "NVDA",
                 "recommendation": "SELL", "timestamp": "2026-06-10T00:00:00+00:00"},
                {"investor_name": "Soros", "symbol": "NVDA",
                 "recommendation": "HOLD", "timestamp": "2026-06-09T00:00:00+00:00"},
                {"investor_name": "Wood", "symbol": "NVDA",
                 "recommendation": "BUY", "timestamp": "2026-06-10T00:00:00+00:00"},
            ],
        )

        result = repo.get_council_vote_history(["NVDA"], window_days=30)

        soros = next(h for h in result if h.investor_name == "Soros")
        assert soros.symbol == "NVDA"
        assert soros.recommendations == ("SELL", "HOLD")  # newest-first
        assert soros.window_days == 30

    def test_lowercase_vote_counts_and_unknown_vote_is_skipped(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # ARCH-03: ta sama reguła co przy werdykcie — „sell” to SELL, a głos
        # spoza BUY/SELL/HOLD nie wchodzi do historii.
        _set_chain_response(
            mock_client,
            ["table", "select", "in_", "gte", "order", "order", "range"],
            [
                {"investor_name": "Soros", "symbol": "NVDA",
                 "recommendation": "sell", "timestamp": "2026-06-10T00:00:00+00:00"},
                {"investor_name": "Soros", "symbol": "NVDA",
                 "recommendation": "STRONG_BUY", "timestamp": "2026-06-09T00:00:00+00:00"},
            ],
        )

        result = repo.get_council_vote_history(["NVDA"], window_days=30)

        assert [h.recommendations for h in result] == [("SELL",)]

    def test_returns_empty_when_no_votes(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client, ["table", "select", "in_", "gte", "order"], []
        )
        assert repo.get_council_vote_history(["X"]) == []

    def test_queries_council_votes_table(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client, ["table", "select", "in_", "gte", "order"], []
        )
        repo.get_council_vote_history(["NVDA"])
        mock_client.table.assert_called_with("council_votes")


class TestGetPriceHistory:
    def test_maps_rows_to_timestamp_price_pairs(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        chain = mock_client.table.return_value
        _set_paginated_response(
            chain.select.return_value.eq.return_value.gte.return_value.order.return_value,
            [[
                {"timestamp": "2026-06-01T00:00:00+00:00", "price": 100.0},
                {"timestamp": "2026-06-02T00:00:00+00:00", "price": 110.0},
            ]],
        )

        result = repo.get_price_history("AAPL", days=30)

        assert len(result) == 2
        ts, money = result[0]
        assert isinstance(money, Money)
        assert money.amount == Decimal("100.0")
        assert ts.year == 2026

    def test_returns_empty_when_no_history(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client, ["table", "select", "eq", "gte", "order"], []
        )
        assert repo.get_price_history("X", days=30) == []

    def test_queries_price_snapshots_chronologically(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client, ["table", "select", "eq", "gte", "order"], []
        )
        repo.get_price_history("VOO", days=7)
        mock_client.table.assert_called_with("price_snapshots")

    def test_history_longer_than_one_page_is_read_in_full(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # PORT-22: radar czyta 90 dni; PostgREST tnie SELECT do ~1000 wierszy bez
        # błędu, więc bez paginacji najstarsze snapshoty znikały po cichu.
        base = datetime(2026, 6, 1, tzinfo=UTC)
        rows = [
            {"timestamp": (base + timedelta(hours=i)).isoformat(), "price": 100.0 + i}
            for i in range(1005)
        ]
        chain = mock_client.table.return_value
        order_builder = (
            chain.select.return_value.eq.return_value.gte.return_value.order.return_value
        )
        _set_paginated_response(order_builder, [rows[:1000], rows[1000:]])

        result = repo.get_price_history("AAPL", days=90)

        assert len(result) == 1005
        assert result[-1][1].amount == Decimal("1104.0")
        assert order_builder.range.call_count == 2


class TestGetCouncilVotesForPrediction:
    def test_maps_rows_to_investor_opinions(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "eq"],
            [
                {
                    "investor_name": "Burry",
                    "recommendation": "SELL",
                    "confidence": 0.8,
                    "reasoning": "Zadłużenie",
                    "key_factors": ["leverage"],
                }
            ],
        )

        result = repo.get_council_votes_for_prediction("pred-1")

        assert len(result) == 1
        assert result[0].investor_name == "Burry"
        assert result[0].recommendation == "SELL"
        assert result[0].key_factors == ("leverage",)

    def test_returns_empty_when_no_votes(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "select", "eq"], [])
        assert repo.get_council_votes_for_prediction("pred-x") == []

    def test_queries_council_votes_by_prediction_id(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "select", "eq"], [])
        repo.get_council_votes_for_prediction("pred-9")
        mock_client.table.assert_called_with("council_votes")


class TestSaveCouncilVotes:
    def test_batch_inserts_one_row_per_investor(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.council import InvestorOpinion

        _set_chain_response(mock_client, ["table", "insert"], [{"id": 1}, {"id": 2}])

        votes = [
            InvestorOpinion(
                investor_name="Warren Buffett",
                recommendation="BUY",
                confidence=0.9,
                reasoning="Silna fosa.",
                key_factors=("moat",),
            ),
            InvestorOpinion(
                investor_name="Michael Burry",
                recommendation="SELL",
                confidence=0.7,
                reasoning="Zawyżone P/E.",
                key_factors=("bubble",),
            ),
        ]

        repo.save_council_votes(
            prediction_id="uuid-pred-1", symbol="AAPL", votes=votes
        )

        # Sprawdzamy że tabela to council_votes
        mock_client.table.assert_any_call("council_votes")
        inserted = mock_client.table.return_value.insert.call_args.args[0]
        assert isinstance(inserted, list)
        assert len(inserted) == 2
        names = {row["investor_name"] for row in inserted}
        assert names == {"Warren Buffett", "Michael Burry"}
        for row in inserted:
            assert row["prediction_id"] == "uuid-pred-1"
            assert row["symbol"] == "AAPL"
            assert row["recommendation"] in {"BUY", "SELL", "HOLD"}
            assert 0.0 <= row["confidence"] <= 1.0

    def test_upserts_on_prediction_and_investor_key(self, mock_client: MagicMock) -> None:
        # NIEZ-14 (migracja 054): re-run w tej samej godzinie trafia w ten sam wiersz
        # predykcji, więc głosy nadpisują się zamiast dublować wagę w rankingu person.
        from src.domain.council import InvestorOpinion

        repo = SupabaseRepository(
            url="https://test.supabase.co", key="anon-key", council_votes_upsert_enabled=True
        )
        _set_chain_response(mock_client, ["table", "upsert"], [{"id": 1}])
        vote = InvestorOpinion(
            investor_name="Warren Buffett",
            recommendation="BUY",
            confidence=0.9,
            reasoning="Silna fosa.",
            key_factors=("moat",),
        )

        repo.save_council_votes(prediction_id="uuid-pred-1", symbol="AAPL", votes=[vote])

        mock_client.table.assert_any_call("council_votes")
        upsert = mock_client.table.return_value.upsert
        rows = upsert.call_args.args[0]
        assert [(row["prediction_id"], row["investor_name"]) for row in rows] == [
            ("uuid-pred-1", "Warren Buffett")
        ]
        assert upsert.call_args.kwargs == {"on_conflict": "prediction_id,investor_name"}
        mock_client.table.return_value.insert.assert_not_called()

    def test_noop_for_empty_votes(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # Brak głosów (np. rada padła całkowicie) — nie wołamy insert na pusto.
        repo.save_council_votes(prediction_id="uuid-1", symbol="AAPL", votes=[])
        # Żadne wywołanie .table("council_votes") nie powinno się odbyć
        for call in mock_client.table.call_args_list:
            assert call.args[0] != "council_votes"


class TestQuotaAlerts:
    def test_save_quota_alert_inserts_to_quota_alerts_table(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from datetime import UTC, datetime

        from src.domain.quota import QuotaAlert, QuotaSeverity

        alert = QuotaAlert(
            source="Alpha Vantage",
            severity=QuotaSeverity.CRITICAL,
            message="All 5 keys exhausted",
            action="Add new key in .env",
            occurred_at=datetime(2026, 6, 1, 14, 32, tzinfo=UTC),
        )

        repo.save_quota_alert(alert)

        mock_client.table.assert_any_call("quota_alerts")
        inserted = mock_client.table.return_value.insert.call_args.args[0]
        assert inserted["source"] == "Alpha Vantage"
        assert inserted["severity"] == "CRITICAL"
        assert inserted["message"] == "All 5 keys exhausted"
        assert inserted["action"] == "Add new key in .env"
        assert inserted["occurred_at"].startswith("2026-06-01")

    def test_get_recent_quota_alerts_parses_rows(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.quota import QuotaSeverity

        mock_response = MagicMock()
        mock_response.data = [
            {
                "source": "OpenAI",
                "severity": "WARNING",
                "message": "TPM hit",
                "action": "tier-up",
                "occurred_at": "2026-06-01T14:30:00+00:00",
            },
            {
                "source": "Alpha Vantage",
                "severity": "CRITICAL",
                "message": "exhausted",
                "action": "add key",
                "occurred_at": "2026-06-01T13:00:00+00:00",
            },
        ]
        # Łańcuch: table().select().gte().order().execute()
        chain = mock_client.table.return_value
        chain.select.return_value.gte.return_value.order.return_value.execute.return_value = (
            mock_response
        )

        alerts = repo.get_recent_quota_alerts(hours=24)

        assert len(alerts) == 2
        assert alerts[0].source == "OpenAI"
        assert alerts[0].severity is QuotaSeverity.WARNING
        assert alerts[1].source == "Alpha Vantage"
        assert alerts[1].severity is QuotaSeverity.CRITICAL

    def test_get_recent_quota_alerts_skips_malformed_rows(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        mock_response = MagicMock()
        mock_response.data = [
            {"severity": "WARNING", "occurred_at": "2026-06-01T14:30:00+00:00"},
            # nieznane severity
            {
                "source": "X",
                "severity": "UNKNOWN_LEVEL",
                "occurred_at": "2026-06-01T13:00:00+00:00",
            },
            # bez occurred_at
            {"source": "X", "severity": "WARNING"},
        ]
        chain = mock_client.table.return_value
        chain.select.return_value.gte.return_value.order.return_value.execute.return_value = (
            mock_response
        )

        # Tylko pierwszy ma wszystkie wymagane pola.
        alerts = repo.get_recent_quota_alerts(hours=24)

        assert len(alerts) == 1


class TestGetRecentlyResolvedSelectsPostMortemColumns:
    def test_select_includes_reasoning_price_and_insight(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """Post-mortem w raporcie potrzebuje oryginalnego uzasadnienia, diagnozy
        i cen — muszą być w SELECT, inaczej sekcja 'Zamknięte predykcje' nie ma
        z czego pokazać 'dlaczego wzrosło/spadło i czy się potwierdziło'."""
        chain = mock_client.table.return_value
        order_builder = (
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value
        )
        _set_paginated_response(order_builder, [[]])

        repo.get_recently_resolved_predictions(24)
        select_arg = mock_client.table.return_value.select.call_args.args[0]
        for col in (
            "reasoning_text",
            "correction_insights",
            "price_at_prediction",
            "actual_price_after_12h",
        ):
            assert col in select_arg

    def test_aggregates_all_rows_across_pages(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """#8: get_recently_resolved_predictions też paginuje — bez tego raport
        pokazywałby max 1000 zamkniętych predykcji."""
        full_page = [{"symbol": "AAPL"} for _ in range(1000)]
        short_page = [{"symbol": "VOO"}]
        chain = mock_client.table.return_value
        order_builder = (
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value
        )
        _set_paginated_response(order_builder, [full_page, short_page])

        out = repo.get_recently_resolved_predictions(24)

        assert len(out) == 1001
        assert order_builder.range.call_count == 2


def _set_paginated_response(
    range_builder: MagicMock, pages: list[list[dict[str, Any]]]
) -> None:
    """Konfiguruje terminalny `.range(...).execute()` tak, by kolejne wywołania
    zwracały kolejne strony (#8: paginacja wokół PostgREST cap ~1000).

    `range_builder` to obiekt, na którym wołane jest `.range(offset, end)`
    (np. `chain...order.return_value`). Każde wejście listy `pages` to jedna
    strona; helper zwraca odpowiedź per wywołanie `.range(...)`."""
    responses = []
    for page in pages:
        resp = MagicMock()
        resp.data = page
        responses.append(resp)
    # .range(...) zwraca builder, którego .execute() daje kolejną stronę.
    range_results = []
    for resp in responses:
        builder = MagicMock()
        builder.execute.return_value = resp
        range_results.append(builder)
    range_builder.range.side_effect = range_results


class TestGetAccuracyStats:
    def test_small_result_set_computes_hit_rate(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """Zachowanie dla <PAGE_SIZE wierszy bez zmian: jedna krótka strona."""
        rows = [
            {"is_trend_correct": True},
            {"is_trend_correct": False},
            {"is_trend_correct": True},
        ]
        chain = mock_client.table.return_value
        order_builder = (
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value
        )
        _set_paginated_response(order_builder, [rows])

        stats = repo.get_accuracy_stats(days=30)

        assert stats["sample_count"] == 3
        assert stats["correct_count"] == 2
        assert stats["mean_accuracy"] == 2 / 3
        assert stats["days_window"] == 30

    def test_adds_deterministic_order_by_timestamp(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """#8: bez .order() obcięty podzbiór (~1000) był niedeterministyczny."""
        chain = mock_client.table.return_value
        order_builder = (
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value
        )
        _set_paginated_response(order_builder, [[]])

        repo.get_accuracy_stats(days=30)

        order_call = chain.select.return_value.gte.return_value.not_.is_.return_value.order
        assert order_call.call_args.args[0] == "timestamp"
        assert order_call.call_args.kwargs.get("desc") is True

    def test_mean_accuracy_computed_over_all_pages(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """#8: cap ~1000 biasował hit-rate. mean_accuracy musi obejmować pełen
        zbiór z wszystkich stron paginacji, nie tylko pierwsze 1000 wierszy."""
        # Strona 1: 1000 wierszy, wszystkie True. Strona 2: 1000 wierszy False.
        page1 = [{"is_trend_correct": True} for _ in range(1000)]
        page2 = [{"is_trend_correct": False} for _ in range(1000)]
        short_page = [{"is_trend_correct": True}]
        chain = mock_client.table.return_value
        order_builder = (
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value
        )
        _set_paginated_response(order_builder, [page1, page2, short_page])

        stats = repo.get_accuracy_stats(days=30)

        # 1001 True z 2001 → gdyby obcięto do 1000, byłoby 100%.
        assert stats["sample_count"] == 2001
        assert stats["correct_count"] == 1001
        assert stats["mean_accuracy"] == 1001 / 2001
        assert order_builder.range.call_count == 3

    def test_directional_calls_are_counted_beside_the_total(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """WIA-29 (c): mail pokazuje trafność kierunkową obok łącznej. Liczby
        łączne (sizing, kalibracja) bez zmian; kierunkowe pomijają SIDEWAYS
        i nieznany trend — ta sama reguła co `directional_hit_rate`."""
        rows: list[dict[str, Any]] = [
            {"is_trend_correct": True, "predicted_trend": "BULLISH"},
            {"is_trend_correct": False, "predicted_trend": "BEARISH"},
            {"is_trend_correct": True, "predicted_trend": "SIDEWAYS"},
            {"is_trend_correct": True, "predicted_trend": None},
        ]
        chain = mock_client.table.return_value
        order_builder = (
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value
        )
        _set_paginated_response(order_builder, [rows])

        stats = repo.get_accuracy_stats(days=30)

        assert stats["sample_count"] == 4
        assert stats["correct_count"] == 3
        assert stats["mean_accuracy"] == 0.75
        assert stats["directional_count"] == 2
        assert stats["directional_correct_count"] == 1
        assert "predicted_trend" in chain.select.call_args.args[0]

    def test_empty_window_has_no_directional_calls(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        chain = mock_client.table.return_value
        order_builder = (
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value
        )
        _set_paginated_response(order_builder, [[]])

        stats = repo.get_accuracy_stats(days=30)

        assert stats["mean_accuracy"] is None
        assert stats["directional_count"] == 0
        assert stats["directional_correct_count"] == 0


    def test_effective_sample_groups_rows_by_utc_day(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """PORT-01: sizing potrzebuje liczby NIEZALEŻNYCH zakładów. Trzy dni,
        w których wszystkie predykcje dnia trafiły albo chybiły razem, to trzy
        zakłady (n_eff = 3), nie dziewięć. Pozostałe pola bez zmian."""
        rows = [
            {"is_trend_correct": day_ok, "predicted_trend": "BULLISH",
             "timestamp": f"2026-10-{day:02d}T{hour:02d}:00:00+00:00"}
            for day, day_ok in ((14, True), (15, False), (16, True))
            for hour in (9, 13, 17)
        ]
        chain = mock_client.table.return_value
        order_builder = (
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value
        )
        _set_paginated_response(order_builder, [rows])

        stats = repo.get_accuracy_stats(days=30)

        assert stats["n_eff"] == pytest.approx(3.0)
        assert stats["sample_count"] == 9
        assert stats["correct_count"] == 6
        assert "timestamp" in chain.select.call_args.args[0]

    def test_effective_sample_of_independent_days_equals_the_count(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        rows = [
            {"is_trend_correct": i % 2 == 0, "predicted_trend": "BEARISH",
             "timestamp": f"2026-10-{i + 1:02d}T12:00:00+00:00"}
            for i in range(5)
        ]
        chain = mock_client.table.return_value
        order_builder = (
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value
        )
        _set_paginated_response(order_builder, [rows])

        assert repo.get_accuracy_stats(days=30)["n_eff"] == pytest.approx(5.0)

    def test_empty_window_has_zero_effective_sample(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        chain = mock_client.table.return_value
        order_builder = (
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value
        )
        _set_paginated_response(order_builder, [[]])

        assert repo.get_accuracy_stats(days=30)["n_eff"] == 0.0


class TestGetResolvedPredictionsForEval:
    def test_returns_rows_and_filters_by_window(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        rows = [
            {
                "predicted_trend": "BULLISH",
                "is_trend_correct": True,
                "price_at_prediction": 100.0,
                "predicted_target_price": 104.0,
                "actual_price_after_12h": 105.0,
                "timestamp": "2026-06-01T12:00:00+00:00",
            }
        ]
        # Łańcuch z paginacją: table().select().gte().not_.is_().order().range().execute()
        chain = mock_client.table.return_value
        order_builder = (
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value
        )
        # Jedna krótka strona (<PAGE_SIZE) → koniec paginacji.
        _set_paginated_response(order_builder, [rows])

        out = repo.get_resolved_predictions_for_eval(days=30)

        assert out == rows
        gte_call = chain.select.return_value.gte
        assert gte_call.call_args.args[0] == "timestamp"

    def test_aggregates_all_rows_across_pages(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """#8: hosted Supabase cap ~1000 wierszy bez błędu. Read musi paginować
        po .range(...) aż do krótkiej strony, inaczej trafność jest obcinana."""
        full_page = [{"timestamp": f"2026-06-01T00:{i:02d}:00+00:00"} for i in range(1000)]
        short_page = [{"timestamp": "2026-06-02T00:00:00+00:00"}]
        chain = mock_client.table.return_value
        order_builder = (
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value
        )
        _set_paginated_response(order_builder, [full_page, short_page])

        out = repo.get_resolved_predictions_for_eval(days=30)

        assert len(out) == 1001
        assert order_builder.range.call_count == 2
        # Pierwsza strona: offset 0; druga: offset 1000.
        assert order_builder.range.call_args_list[0].args == (0, 999)
        assert order_builder.range.call_args_list[1].args == (1000, 1999)


class TestFindSimilarPredictions:
    def test_calls_rpc_and_returns_rows(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        rows = [{"news_summary": "Fed hawkish", "predicted_trend": "BEARISH"}]
        mock_response = MagicMock()
        mock_response.data = rows
        mock_client.rpc.return_value.execute.return_value = mock_response

        out = repo.find_similar_predictions([0.1, 0.2, 0.3], limit=3)

        assert out == rows
        name, params = mock_client.rpc.call_args.args
        assert name == "match_news_embeddings"
        assert params["query_embedding"] == [0.1, 0.2, 0.3]
        assert params["match_count"] == 3

    def test_returns_empty_list_when_rpc_unavailable(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # RPC nieobecne (migracja 011 niezaaplikowana) → graceful [], nie crash.
        mock_client.rpc.side_effect = Exception("function does not exist")

        out = repo.find_similar_predictions([0.1, 0.2], limit=3)

        assert out == []


class TestModelScorecards:
    """#12 — scorecard walk-forward per symbol (migracja 019)."""

    def test_save_writes_symbol_and_gate_metrics(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "insert"], [{"id": 1}])

        repo.save_model_scorecard(
            "AAPL",
            {
                "status": "trained_successfully",
                "candidate_holdout_rmse": 0.021,
                "baseline_rmse": 0.024,
                "candidate_holdout_directional_hit_rate": 0.58,
                "n_folds": 7,
                "n_samples": 1240,
            },
        )

        mock_client.table.assert_called_with("model_scorecards")
        payload = mock_client.table.return_value.insert.call_args.args[0]
        # Bez `symbol` ~43 przebiegi treningu byłyby nierozróżnialne.
        assert payload["symbol"] == "AAPL"
        assert payload["status"] == "trained_successfully"
        assert payload["n_folds"] == 7

    def test_get_recent_returns_rows(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "gte", "order"],
            [{"symbol": "AAPL", "status": "trained_successfully"}],
        )

        rows = repo.get_recent_model_scorecards(30)

        assert rows
        assert rows[0]["symbol"] == "AAPL"

    def test_get_recent_is_graceful_on_error(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        mock_client.table.side_effect = Exception("no table")

        assert repo.get_recent_model_scorecards(30) == []


class TestGetPreviousVerdict:
    """#8 — poprzedni werdykt rady (JSONB z migracji 005) do detektora flipów.
    Adapter jest SUROWY: graceful degradation robi caller (styl repo)."""

    @staticmethod
    def _row() -> dict[str, Any]:
        return {
            "council_verdict": {
                "final_recommendation": "BUY",
                "consensus_strength": 0.8,
                "summary": "Kupujemy.",
                "dissenting_views": ["Soros: SELL"],
                "investor_opinions": [
                    {
                        "investor_name": "Soros",
                        "recommendation": "SELL",
                        "confidence": 0.7,
                        "reasoning": "...",
                        "key_factors": [],
                    }
                ],
            },
            "sentiment_score": 0.31,
            "timestamp": "2026-07-09T10:00:00+00:00",
        }

    @staticmethod
    def _set_verdict_chain(client: MagicMock, data: list[dict[str, Any]]) -> None:
        # `.not_` to PROPERTY, nie wywołanie — generyczny helper łańcucha
        # (_set_chain_response) tego nie odwzoruje.
        chain = client.table.return_value.select.return_value.eq.return_value
        chain = chain.not_.is_.return_value.order.return_value.limit.return_value
        response = MagicMock()
        response.data = data
        chain.execute.return_value = response

    def test_maps_row_to_previous_verdict(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        self._set_verdict_chain(mock_client, [self._row()])

        previous = repo.get_previous_verdict("NVDA")

        assert previous is not None
        assert previous.verdict.final_recommendation == "BUY"
        assert previous.sentiment_score == pytest.approx(0.31)
        assert previous.verdict.dissenting_views == ("Soros: SELL",)
        assert len(previous.verdict.investor_opinions) == 1

    def test_returns_none_when_no_rows(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        self._set_verdict_chain(mock_client, [])

        assert repo.get_previous_verdict("NVDA") is None

    def test_previous_verdict_sanitizes_unknown_recommendation(
        self,
        repo: SupabaseRepository,
        mock_client: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # ARCH-03: JSONB przechodził przez `cast(Any)` bez walidacji, więc
        # „STRONG_BUY” wchodził do `CouncilVerdict` jako rekomendacja rady.
        row = self._row()
        row["council_verdict"]["final_recommendation"] = "STRONG_BUY"
        row["council_verdict"]["investor_opinions"][0]["recommendation"] = "sell"
        self._set_verdict_chain(mock_client, [row])

        with caplog.at_level(logging.WARNING):
            previous = repo.get_previous_verdict("NVDA")

        assert previous is not None
        assert previous.verdict.final_recommendation == "HOLD"
        assert previous.verdict.investor_opinions[0].recommendation == "SELL"
        assert "STRONG_BUY" in caplog.text

    def test_verdict_without_failed_personas_reads_as_full_council(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # MAIL-12: JSONB sprzed pola nie zgaduje awarii z treści opinii — 0.
        self._set_verdict_chain(mock_client, [self._row()])

        previous = repo.get_previous_verdict("NVDA")

        assert previous is not None
        assert previous.verdict.failed_personas == 0

    @pytest.mark.parametrize(("raw", "expected"), [(3, 3), ("2", 2), (-1, 0), ("x", 0)])
    def test_verdict_reads_failed_personas_from_jsonb(
        self, repo: SupabaseRepository, mock_client: MagicMock, raw: object, expected: int
    ) -> None:
        row = self._row()
        row["council_verdict"]["failed_personas"] = raw
        self._set_verdict_chain(mock_client, [row])

        previous = repo.get_previous_verdict("NVDA")

        assert previous is not None
        assert previous.verdict.failed_personas == expected


class TestShockAlerts:
    """#11 — dedup alertów szoku (migracja 021, UNIQUE(symbol, alert_date))."""

    def test_save_writes_symbol_date_and_direction(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "insert"], [{"id": 1}])

        repo.save_shock_alert("BTC", date(2026, 7, 10), -0.082, "DOWN")

        mock_client.table.assert_called_with("shock_alerts")
        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert payload["symbol"] == "BTC"
        assert payload["alert_date"] == "2026-07-10"
        assert payload["direction"] == "DOWN"

    def test_first_level_payload_has_no_level_key(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # ALR-16: szczebel 1 to DEFAULT kolumny z 081 — zapis jak przed migracją,
        # więc flaga drabiny off nie dotyka bazy bez 081.
        _set_chain_response(mock_client, ["table", "insert"], [{"id": 1}])

        repo.save_shock_alert("BTC", date(2026, 7, 10), -0.082, "DOWN")

        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert "level" not in payload

    def test_escalation_writes_its_level(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "insert"], [{"id": 2}])

        repo.save_shock_alert("NVDA", date(2026, 10, 24), -0.095, "DOWN", level=2)

        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert payload["level"] == 2
        assert payload["symbol"] == "NVDA"

    def test_levels_of_the_day_group_rungs_per_symbol(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.shock import SentShockLevel, ShockDirection

        _set_chain_response(
            mock_client,
            ["table", "select", "eq"],
            [
                {"symbol": "NVDA", "level": 2, "direction": "DOWN"},
                {"symbol": "NVDA", "level": 1, "direction": "DOWN"},
                {"symbol": "BTC", "level": 1, "direction": "UP"},
            ],
        )

        levels = repo.get_shock_alert_levels(date(2026, 10, 24))

        mock_client.table.assert_called_with("shock_alerts")
        mock_client.table.return_value.select.return_value.eq.assert_called_with(
            "alert_date", "2026-10-24"
        )
        assert levels == {
            "NVDA": (
                SentShockLevel(level=1, direction=ShockDirection.DOWN),
                SentShockLevel(level=2, direction=ShockDirection.DOWN),
            ),
            "BTC": (SentShockLevel(level=1, direction=ShockDirection.UP),),
        }

    def test_unreadable_level_row_fails_the_whole_read(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # Pominięty szczebel mógłby wyjść drugi raz — watch na ten przebieg wraca do
        # „jeden alert dziennie”, zamiast zgadywać.
        from src.application.errors import MalformedPayload

        _set_chain_response(
            mock_client,
            ["table", "select", "eq"],
            [{"symbol": "NVDA", "level": 1, "direction": "SIDEWAYS"}],
        )

        with pytest.raises(MalformedPayload):
            repo.get_shock_alert_levels(date(2026, 10, 24))

    def test_get_sent_returns_symbol_date_pairs(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "gte"],
            [{"symbol": "BTC", "alert_date": "2026-07-10"}],
        )

        assert repo.get_sent_shock_alerts(date(2026, 7, 10)) == {
            ("BTC", date(2026, 7, 10))
        }

    def test_get_sent_is_graceful_without_migration(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        mock_client.table.side_effect = Exception("no table")

        assert repo.get_sent_shock_alerts(date(2026, 7, 10)) == set()

    def test_last_price_snapshot_returns_price_and_timestamp(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "order", "limit"],
            [{"price": "65000.0", "timestamp": "2026-07-10T09:00:00+00:00"}],
        )

        snapshot = repo.get_last_price_snapshot("BTC")

        assert snapshot is not None
        price, stamp = snapshot
        assert price.amount == Decimal("65000.0")
        assert stamp.hour == 9


class TestGetResolvedPredictionsDetailed:
    """#17 — recap tygodniowy potrzebuje pól, których `get_resolved_predictions`
    NIE zwraca: id, predicted_target_price, confidence_score, council_verdict."""

    def test_selects_the_fields_the_weekly_recap_needs(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # `.not_` to PROPERTY, nie wywołanie — generyczny helper go nie odwzoruje.
        chain = mock_client.table.return_value.select.return_value.gte.return_value
        chain = chain.not_.is_.return_value.order.return_value
        response = MagicMock()
        response.data = [{"id": "p-1", "symbol": "AAPL"}]
        chain.range.return_value.execute.return_value = response

        rows = repo.get_resolved_predictions_detailed(7)

        selected = mock_client.table.return_value.select.call_args.args[0]
        for field in (
            "id",
            "predicted_target_price",
            "confidence_score",
            "council_verdict",
            "actual_price_after_12h",
            "price_at_prediction",
        ):
            assert field in selected
        assert rows
        assert rows[0]["id"] == "p-1"


class TestLivenessHeartbeats:
    """NIEZ-11 — ślady życia jobów dla deadmana: najnowszy snapshot ceny (Fast Loop)
    i najnowszy wiersz karty modelu (Slow Loop)."""

    def test_last_scorecard_at_reads_the_newest_row(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "order", "limit"],
            [{"timestamp": "2026-10-11T09:35:05+00:00"}],
        )

        stamp = repo.get_last_scorecard_at()

        assert stamp == datetime(2026, 10, 11, 9, 35, 5, tzinfo=UTC)
        mock_client.table.assert_called_with("model_scorecards")

    def test_latest_snapshot_at_reads_price_snapshots(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "order", "limit"],
            [{"timestamp": "2026-08-17T05:57:17+00:00"}],
        )

        assert repo.get_latest_snapshot_at() == datetime(2026, 8, 17, 5, 57, 17, tzinfo=UTC)
        mock_client.table.assert_called_with("price_snapshots")

    def test_no_rows_means_no_heartbeat(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "select", "order", "limit"], [])

        assert repo.get_last_scorecard_at() is None


class TestDatabaseProbe:
    """NIEZ-28 — jawny monitor bazy: deadman czyta raz dziennie jeden wiersz
    `price_snapshots` przez PostgREST, tylko do odczytu. Błąd musi wyjść na zewnątrz,
    bo to on oznacza „baza nie odpowiada” (np. wstrzymany projekt Supabase Free)."""

    def test_probe_reads_one_price_snapshot_row(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "limit"],
            [{"timestamp": "2026-08-17T05:57:17+00:00"}],
        )

        repo.probe_database()

        mock_client.table.assert_called_with("price_snapshots")
        mock_client.table.return_value.select.return_value.limit.assert_called_with(1)

    def test_probe_accepts_an_empty_table(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "select", "limit"], [])

        repo.probe_database()

    def test_probe_never_writes(self, repo: SupabaseRepository, mock_client: MagicMock) -> None:
        _set_chain_response(mock_client, ["table", "select", "limit"], [])

        repo.probe_database()

        table = mock_client.table.return_value
        for write in ("insert", "upsert", "update", "delete"):
            getattr(table, write).assert_not_called()
        mock_client.rpc.assert_not_called()

    def test_probe_propagates_an_unreachable_database(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        chain = mock_client.table.return_value.select.return_value.limit.return_value
        chain.execute.side_effect = RuntimeError("Supabase 540: project paused")

        with pytest.raises(RuntimeError):
            repo.probe_database()

    def test_probe_rejects_a_response_that_is_not_a_row_list(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        response = _set_chain_response(mock_client, ["table", "select", "limit"], [])
        response.data = "<html>Project paused</html>"

        with pytest.raises(RuntimeError):
            repo.probe_database()


def _fluent_query(mock_client: MagicMock, pages: list[list[dict[str, Any]]]) -> MagicMock:
    """Builder, którego każda metoda filtra/sortowania zwraca SAM SIEBIE, a kolejne
    `.range(...).execute()` dają kolejne strony. Test sprawdza paginację, nie
    dokładny kształt łańcucha filtrów."""
    builder = MagicMock()
    for method in ("select", "eq", "gte", "is_", "in_", "order"):
        getattr(builder, method).return_value = builder
    builder.not_ = builder
    mock_client.table.return_value = builder
    _set_paginated_response(builder, pages)
    return builder


class TestPaginatedSlowLoopReads:
    """NIEZ-15: PostgREST tnie SELECT do ~1000 wierszy BEZ błędu. Kalibracja i sweep
    ujawnień liczyły na obciętym, niedeterministycznym podzbiorze."""

    def test_resolved_for_calibration_paginates_past_1000(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        full_page = [{"id": f"p-{i}"} for i in range(1000)]
        builder = _fluent_query(mock_client, [full_page, [{"id": f"q-{i}"} for i in range(5)]])

        rows = repo.get_resolved_for_calibration(30)

        assert len(rows) == 1005
        assert builder.range.call_count == 2

    def test_resolved_for_calibration_breaks_timestamp_ties_by_id(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # Bez drugiego klucza sortowania granica strony przy równych timestampach
        # jest niedeterministyczna — ten sam wiersz może wypaść albo wejść dwa razy.
        builder = _fluent_query(mock_client, [[]])

        repo.get_resolved_for_calibration(30)

        ordered_by = [c.args[0] for c in builder.order.call_args_list]
        assert ordered_by == ["timestamp", "id"]

    def test_unrevealed_commitments_paginates_past_1000(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        full_page = [{"id": f"p-{i}"} for i in range(1000)]
        builder = _fluent_query(mock_client, [full_page, [{"id": f"q-{i}"} for i in range(5)]])

        rows = repo.get_unrevealed_commitments()

        assert len(rows) == 1005
        assert builder.range.call_count == 2
        assert [c.args[0] for c in builder.order.call_args_list] == ["timestamp", "id"]

    def test_unrevealed_commitments_stays_graceful_without_migration(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        builder = _fluent_query(mock_client, [[]])
        builder.range.side_effect = Exception("column commitment_hash does not exist")

        assert repo.get_unrevealed_commitments() == []

    def test_get_feature_store_data_paginates_past_1000(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # Trening czytał max ~1000 wierszy widoku — reszta historii znikała po cichu.
        full_page = [{"id": f"p-{i}"} for i in range(1000)]
        builder = _fluent_query(mock_client, [full_page, [{"id": f"q-{i}"} for i in range(5)]])

        rows = repo.get_feature_store_data("AAPL")

        assert len(rows) == 1005
        assert builder.range.call_count == 2
        assert [c.args[0] for c in builder.order.call_args_list] == ["timestamp", "id"]

    def test_council_vote_history_paginates_past_1000(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        vote = {"investor_name": "Soros", "symbol": "NVDA", "recommendation": "BUY"}
        builder = _fluent_query(mock_client, [[vote] * 1000, [vote] * 5])

        history = repo.get_council_vote_history(["NVDA"])

        assert len(history[0].recommendations) == 1005
        assert builder.range.call_count == 2
        assert [c.args[0] for c in builder.order.call_args_list] == ["timestamp", "id"]


class TestLlmFallbackRowsExcludedFromTrackRecord:
    """WIA-09: po awarii głównego LLM predict_node zapisuje neutralną atrapę
    („SIDEWAYS 50%”) z flagą `llm_analysis_failed`. Hit-rate i ECE liczyły ją
    jak prawdziwą predykcję. Każdy odczyt track-recordu musi ją odsiać — i musi
    pobierać kolumnę flag, inaczej filtr w produkcji nie ma czego sprawdzić."""

    _FALLBACK = {"data_quality_flags": ["llm_analysis_failed"]}

    @staticmethod
    def _order_builder(mock_client: MagicMock) -> MagicMock:
        chain = mock_client.table.return_value
        return cast(
            MagicMock,
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value,
        )

    def test_accuracy_stats_skip_llm_fallback_rows(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        rows: list[dict[str, Any]] = [
            {"is_trend_correct": True, "data_quality_flags": []},
            {"is_trend_correct": True, **self._FALLBACK},
            {"is_trend_correct": False, "data_quality_flags": None},
        ]
        _set_paginated_response(self._order_builder(mock_client), [rows])

        stats = repo.get_accuracy_stats(days=30)

        assert stats["sample_count"] == 2
        assert stats["correct_count"] == 1
        assert "data_quality_flags" in mock_client.table.return_value.select.call_args.args[0]

    def test_resolved_predictions_skip_llm_fallback_rows(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        rows = [
            {"symbol": "AAPL", "data_quality_flags": ["av_sentiment_score_missing"]},
            {"symbol": "NVDA", **self._FALLBACK},
        ]
        _set_paginated_response(self._order_builder(mock_client), [rows])

        out = repo.get_resolved_predictions(30)

        assert [r["symbol"] for r in out] == ["AAPL"]
        assert "data_quality_flags" in mock_client.table.return_value.select.call_args.args[0]

    def test_recently_resolved_predictions_skip_llm_fallback_rows(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        rows: list[dict[str, Any]] = [{"symbol": "AAPL"}, {"symbol": "NVDA", **self._FALLBACK}]
        _set_paginated_response(self._order_builder(mock_client), [rows])

        out = repo.get_recently_resolved_predictions(24)

        assert [r["symbol"] for r in out] == ["AAPL"]
        assert "data_quality_flags" in mock_client.table.return_value.select.call_args.args[0]

    def test_calibration_samples_skip_llm_fallback_rows(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        builder = _fluent_query(mock_client, [[{"id": "p-1"}, {"id": "p-2", **self._FALLBACK}]])

        rows = repo.get_resolved_for_calibration(30)

        assert [r["id"] for r in rows] == ["p-1"]
        assert "data_quality_flags" in builder.select.call_args.args[0]


# ---------------------------------------------------------------------------
# NIEZ-15: każda strona paginacji = jedna para offset/limit. Prawdziwy builder
# postgrest-py, mock wyłącznie na warstwie HTTP (httpx) — MagicMock-owy łańcuch
# nie odwzoruje tego, że `.range()` DOKLEJA parametry do tego samego buildera.
# ---------------------------------------------------------------------------

_PAGINATED_READS: dict[str, Callable[[SupabaseRepository], object]] = {
    "recently_resolved": lambda repo: repo.get_recently_resolved_predictions(24),
    "resolved": lambda repo: repo.get_resolved_predictions(30),
    "resolved_detailed": lambda repo: repo.get_resolved_predictions_detailed(7),
    "resolved_for_eval": lambda repo: repo.get_resolved_predictions_for_eval(30),
    "accuracy_stats": lambda repo: repo.get_accuracy_stats(30),
    "resolved_for_calibration": lambda repo: repo.get_resolved_for_calibration(30),
    "unrevealed_commitments": lambda repo: repo.get_unrevealed_commitments(30),
    "feature_store": lambda repo: repo.get_feature_store_data("NVDA"),
    "council_vote_history": lambda repo: repo.get_council_vote_history(["NVDA"]),
    "price_history": lambda repo: repo.get_price_history("NVDA", 90),
}


class TestPaginationSendsOneRangePerPage:
    @staticmethod
    def _serve_rows(mocker: Any, total: int) -> list[httpx.QueryParams]:
        """Udaje PostgREST: zwraca wycinek `total` wierszy wg OSTATNIEGO offsetu
        i limitu z zapytania, a każde zapytanie zapisuje do listy."""
        rows = [
            {
                "id": f"p-{i}",
                "symbol": "NVDA",
                "is_trend_correct": True,
                "data_quality_flags": None,
                "investor_name": "Soros",
                "recommendation": "BUY",
            }
            for i in range(total)
        ]
        sent: list[httpx.QueryParams] = []

        def fake_request(
            _client: httpx.Client, method: str, url: str, **kwargs: Any
        ) -> httpx.Response:
            params = httpx.QueryParams(kwargs.get("params"))
            sent.append(params)
            offset = int(params.get_list("offset")[-1])
            limit = int(params.get_list("limit")[-1])
            page = rows[offset : offset + limit]
            return httpx.Response(200, json=page, request=httpx.Request(method, url))

        mocker.patch.object(httpx.Client, "request", autospec=True, side_effect=fake_request)
        return sent

    @pytest.mark.parametrize("read", _PAGINATED_READS.values(), ids=_PAGINATED_READS.keys())
    def test_second_page_replaces_offset_instead_of_appending(
        self, mocker: Any, read: Callable[[SupabaseRepository], object]
    ) -> None:
        sent = self._serve_rows(mocker, total=1005)
        repo = SupabaseRepository(url="https://test.supabase.co", key="anon-key")

        read(repo)

        # Bez naprawy druga strona szła jako offset=0&offset=1000&limit=1000&limit=1000.
        assert [p.get_list("offset") for p in sent] == [["0"], ["1000"]]
        assert [p.get_list("limit") for p in sent] == [["1000"], ["1000"]]

    @pytest.mark.parametrize("read", _PAGINATED_READS.values(), ids=_PAGINATED_READS.keys())
    def test_every_page_keeps_the_same_filters_and_order(
        self, mocker: Any, read: Callable[[SupabaseRepository], object]
    ) -> None:
        sent = self._serve_rows(mocker, total=1005)
        repo = SupabaseRepository(url="https://test.supabase.co", key="anon-key")

        read(repo)

        def without_range(params: httpx.QueryParams) -> list[tuple[str, str]]:
            return [(k, v) for k, v in params.multi_items() if k not in ("offset", "limit")]

        assert len(sent) == 2
        assert without_range(sent[0]) == without_range(sent[1])
        assert any(key == "order" for key, _ in without_range(sent[1]))


class TestFeatureStoreFeatureSetVersion:
    """DAT-09: trening może czytać tylko bieżącą wersję zestawu cech (migracja
    027). Bez wersji zapytanie nie dotyka kolumny, której przed migracją nie ma."""

    @staticmethod
    def _serve_view(mocker: Any, rows: list[dict[str, Any]]) -> list[httpx.QueryParams]:
        """Udaje PostgREST nad widokiem: filtr `feature_set_version=eq.N` działa
        jak w bazie (NULL i inne wersje odpadają), a każde zapytanie trafia do listy."""
        sent: list[httpx.QueryParams] = []

        def fake_request(
            _client: httpx.Client, method: str, url: str, **kwargs: Any
        ) -> httpx.Response:
            params = httpx.QueryParams(kwargs.get("params"))
            sent.append(params)
            wanted = params.get("feature_set_version")
            page = [
                row
                for row in rows
                if wanted is None or f"eq.{row['feature_set_version']}" == wanted
            ]
            return httpx.Response(200, json=page, request=httpx.Request(method, url))

        mocker.patch.object(httpx.Client, "request", autospec=True, side_effect=fake_request)
        return sent

    _ROWS: list[dict[str, Any]] = [
        {"id": "old", "symbol": "NVDA", "feature_set_version": 1},
        {"id": "unknown", "symbol": "NVDA", "feature_set_version": None},
        {"id": "current", "symbol": "NVDA", "feature_set_version": 2},
    ]

    def test_read_skips_rows_of_an_older_feature_set_version(self, mocker: Any) -> None:
        sent = self._serve_view(mocker, self._ROWS)
        repo = SupabaseRepository(url="https://test.supabase.co", key="anon-key")

        rows = repo.get_feature_store_data("NVDA", feature_set_version=2)

        assert [row["id"] for row in rows] == ["current"]
        assert sent[0].get("feature_set_version") == "eq.2"

    def test_read_without_a_version_sends_no_version_filter(self, mocker: Any) -> None:
        # Przed migracją 027 filtr po nieistniejącej kolumnie to 400 i padły trening.
        sent = self._serve_view(mocker, self._ROWS)
        repo = SupabaseRepository(url="https://test.supabase.co", key="anon-key")

        rows = repo.get_feature_store_data("NVDA")

        assert len(rows) == 3
        assert "feature_set_version" not in sent[0]


class TestModelScorecardBaselines:
    """ML-10: baseline'y „dryf średniej" i „sam LLM" trafiają do karty tylko za
    flagą — kolumny istnieją od migracji 028, a nowy klucz przed nią to PGRST204."""

    _RESULT: dict[str, Any] = {
        "status": "skipped_validation_failed",
        "candidate_holdout_rmse": 0.021,
        "baseline_rmse": 0.024,
        "drift_baseline_rmse": 0.023,
        "llm_baseline_rmse": 0.020,
        "n_folds": 4,
        "n_samples": 225,
    }

    def test_flag_off_payload_has_no_new_columns(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        repo.save_model_scorecard("ALL", self._RESULT)

        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert "drift_baseline_rmse" not in payload
        assert "llm_baseline_rmse" not in payload

    def test_flag_on_payload_carries_both_baselines(self, mock_client: MagicMock) -> None:
        repo = SupabaseRepository(
            url="https://test.supabase.co", key="anon-key", scorecard_baselines_enabled=True
        )

        repo.save_model_scorecard("ALL", self._RESULT)

        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert payload["drift_baseline_rmse"] == 0.023
        assert payload["llm_baseline_rmse"] == 0.020
        assert payload["baseline_rmse"] == 0.024


class TestModelScorecardRunId:
    """ML-24: `run_id` przebiegu trafia do karty tylko za flagą — kolumna istnieje
    od migracji 049, a nowy klucz przed nią to PGRST204 i przepadły wiersz karty."""

    _RUN_ID = "5f0c6c1e-8a43-4d3a-9b0c-2f1d3e4a5b6c"

    def test_flag_off_payload_has_no_run_id(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        repo.save_model_scorecard("ALL", {"status": "skipped"}, run_id=self._RUN_ID)

        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert "run_id" not in payload

    def test_flag_on_payload_carries_the_run_id(self, mock_client: MagicMock) -> None:
        repo = SupabaseRepository(
            url="https://test.supabase.co", key="anon-key", scorecard_run_id_enabled=True
        )

        repo.save_model_scorecard("AAPL", {"status": "skipped"}, run_id=self._RUN_ID)

        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert payload["run_id"] == self._RUN_ID
        assert payload["symbol"] == "AAPL"

    def test_flag_on_without_a_run_id_sends_no_key(self, mock_client: MagicMock) -> None:
        repo = SupabaseRepository(
            url="https://test.supabase.co", key="anon-key", scorecard_run_id_enabled=True
        )

        repo.save_model_scorecard("AAPL", {"status": "skipped"})

        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert "run_id" not in payload


class TestModelScorecardFeatureStats:
    """ML-21: `feature_stats` z treningu (biny kwantylowe + PSI) trafia do karty
    tylko za flagą — kolumna JSONB istnieje od migracji 052, a nowy klucz przed nią
    to PGRST204 i przepadły wiersz karty."""

    _STATS: dict[str, Any] = {
        "price_delta": {
            "mean": 0.001,
            "std": 0.02,
            "edges": [-0.01, 0.0, 0.01],
            "fractions": [0.25, 0.25, 0.25, 0.25],
            "psi_recent_30d": 0.31,
        }
    }

    def test_flag_off_payload_has_no_feature_stats(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        repo.save_model_scorecard("ALL", {"status": "skipped", "feature_stats": self._STATS})

        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert "feature_stats" not in payload

    def test_save_model_scorecard_persists_feature_stats(self, mock_client: MagicMock) -> None:
        repo = SupabaseRepository(
            url="https://test.supabase.co",
            key="anon-key",
            scorecard_feature_stats_enabled=True,
        )

        repo.save_model_scorecard(
            "ALL", {"status": "trained_successfully", "feature_stats": self._STATS}
        )

        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert payload["feature_stats"] == self._STATS

    def test_flag_on_without_stats_sends_no_key(self, mock_client: MagicMock) -> None:
        repo = SupabaseRepository(
            url="https://test.supabase.co",
            key="anon-key",
            scorecard_feature_stats_enabled=True,
        )

        repo.save_model_scorecard("ALL", {"status": "skipped"})

        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert "feature_stats" not in payload


class TestModelScorecardWeightsSha256:
    """CI-13: sha256 pliku wag trafia do karty tylko za flagą — kolumna istnieje od
    migracji 060, a nowy klucz przed nią to PGRST204 i przepadły wiersz karty."""

    _SHA = "0f" * 32

    def test_flag_off_payload_has_no_weights_hash(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        repo.save_model_scorecard(
            "ALL", {"status": "trained_successfully", "weights_sha256": self._SHA}
        )

        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert "weights_sha256" not in payload

    def test_flag_on_payload_carries_the_weights_hash(self, mock_client: MagicMock) -> None:
        repo = SupabaseRepository(
            url="https://test.supabase.co",
            key="anon-key",
            scorecard_weights_sha256_enabled=True,
        )

        repo.save_model_scorecard(
            "ALL", {"status": "trained_successfully", "weights_sha256": self._SHA}
        )

        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert payload["weights_sha256"] == self._SHA

    def test_flag_on_without_weights_sends_no_key(self, mock_client: MagicMock) -> None:
        repo = SupabaseRepository(
            url="https://test.supabase.co",
            key="anon-key",
            scorecard_weights_sha256_enabled=True,
        )

        repo.save_model_scorecard("ALL", {"status": "skipped_validation_failed"})

        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert "weights_sha256" not in payload


class TestModelScorecardProvenance:
    """ML-22: proweniencja treningu (odcisk zbioru, zakres dat, liczności, parametry,
    wersje, SHA kodu) trafia do karty tylko za flagą — kolumny istnieją od migracji
    064, a nowy klucz przed nią to PGRST204 i przepadły wiersz karty."""

    _PROVENANCE: dict[str, Any] = {
        "dataset_sha256": "ab" * 32,
        "data_start": "2026-10-01T07:30:00+00:00",
        "data_end": "2026-10-20T07:30:00+00:00",
        "n_per_symbol": {"AAPL": 40, "BTC": 20},
        "training_params": {"max_depth": 4, "random_state": 0},
        "library_version": "xgboost 3.2.0",
        "feature_set_version": 4,
        "git_sha": "abc1234",
        "label_overlap_count": 10,
        "label_overlap_measured": 49,
    }

    def test_flag_off_payload_has_no_provenance(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        repo.save_model_scorecard("ALL", {"status": "trained_successfully", **self._PROVENANCE})

        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert not set(self._PROVENANCE) & set(payload)

    def test_flag_on_payload_carries_the_provenance(self, mock_client: MagicMock) -> None:
        repo = SupabaseRepository(
            url="https://test.supabase.co",
            key="anon-key",
            scorecard_provenance_enabled=True,
        )

        repo.save_model_scorecard("ALL", {"status": "trained_successfully", **self._PROVENANCE})

        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert {key: payload[key] for key in self._PROVENANCE} == self._PROVENANCE

    def test_flag_on_skipped_run_sends_no_provenance_keys(self, mock_client: MagicMock) -> None:
        repo = SupabaseRepository(
            url="https://test.supabase.co",
            key="anon-key",
            scorecard_provenance_enabled=True,
        )

        repo.save_model_scorecard("ALL", {"status": "skipped", "feature_set_version": 4})

        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert not set(self._PROVENANCE) & set(payload)


class TestSettledAtWindows:
    """WIA-19: okno „rozliczone od `since`” filtruje i sortuje po `settled_at`
    (migracja 029). Bez `since` zapytanie nie dotyka kolumny, której przed
    migracją nie ma — filtr po niej to 400 i raport bez zamkniętych predykcji."""

    _SINCE = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)

    @staticmethod
    def _capture(mocker: Any) -> list[httpx.QueryParams]:
        sent: list[httpx.QueryParams] = []

        def fake_request(
            _client: httpx.Client, method: str, url: str, **kwargs: Any
        ) -> httpx.Response:
            sent.append(httpx.QueryParams(kwargs.get("params")))
            return httpx.Response(200, json=[], request=httpx.Request(method, url))

        mocker.patch.object(httpx.Client, "request", autospec=True, side_effect=fake_request)
        return sent

    _READS: dict[str, Callable[[SupabaseRepository, datetime | None], object]] = {
        "recently_resolved": lambda repo, since: repo.get_recently_resolved_predictions(
            24, since=since
        ),
        "resolved_detailed": lambda repo, since: repo.get_resolved_predictions_detailed(
            7, since=since
        ),
    }

    @pytest.mark.parametrize("read", _READS.values(), ids=_READS.keys())
    def test_since_filters_and_orders_by_settlement_moment(
        self, mocker: Any, read: Callable[[SupabaseRepository, datetime | None], object]
    ) -> None:
        sent = self._capture(mocker)
        repo = SupabaseRepository(url="https://test.supabase.co", key="anon-key")

        read(repo, self._SINCE)

        params = sent[0]
        assert params.get("settled_at") == f"gte.{self._SINCE.isoformat()}"
        assert params.get("order") == "settled_at.desc"
        # Data UTWORZENIA nie tnie okna — piątkowa predykcja rozliczona w poniedziałek wchodzi.
        assert "timestamp" not in params
        assert params.get("is_trend_correct") == "not.is.null"

    @pytest.mark.parametrize("read", _READS.values(), ids=_READS.keys())
    def test_without_since_the_query_never_touches_settled_at(
        self, mocker: Any, read: Callable[[SupabaseRepository, datetime | None], object]
    ) -> None:
        sent = self._capture(mocker)
        repo = SupabaseRepository(url="https://test.supabase.co", key="anon-key")

        read(repo, None)

        assert all("settled_at" not in str(params) for params in sent)
        assert sent[0].get("timestamp", "").startswith("gte.")
        assert sent[0].get("order") == "timestamp.desc"


class TestResolvedForEvalRecipeColumn:
    """WIA-20: kolumna `recipe_version` (migracja 030) trafia do selecta TYLKO na
    żądanie — przed migracją select po niej to 400 i pusta ewaluacja."""

    def _select_columns(
        self, repo: SupabaseRepository, mock_client: MagicMock, **kwargs: Any
    ) -> str:
        chain = mock_client.table.return_value
        order_builder = (
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value
        )
        _set_paginated_response(order_builder, [[]])
        repo.get_resolved_predictions_for_eval(30, **kwargs)
        return cast(str, chain.select.call_args.args[0])

    def test_default_select_has_no_recipe_column(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        assert "recipe_version" not in self._select_columns(repo, mock_client)

    def test_with_recipe_selects_the_recipe_column(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        assert "recipe_version" in self._select_columns(repo, mock_client, with_recipe=True)


class TestResolvedForEvalUniverse:
    """WIA-48: wersja uniwersum siedzi w kwicie v2 (JSONB z migracji 020), więc
    odczyt idzie ścieżką JSON — bez nowej kolumny i bez nowej migracji."""

    def _select_columns(
        self, repo: SupabaseRepository, mock_client: MagicMock, **kwargs: Any
    ) -> str:
        chain = mock_client.table.return_value
        order_builder = (
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value
        )
        _set_paginated_response(order_builder, [[]])
        repo.get_resolved_predictions_for_eval(30, **kwargs)
        return cast(str, chain.select.call_args.args[0])

    def test_default_select_has_no_universe(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        assert "universe_version" not in self._select_columns(repo, mock_client)

    def test_with_universe_reads_the_receipt_path(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        columns = self._select_columns(repo, mock_client, with_universe=True)

        assert "universe_version:decision_receipts->>universe_version" in columns


class TestSaveFeatureSnapshot:
    """ML-14: wektor cech point-in-time (migracja 031). Idempotentny jak snapshot
    ceny — re-run w tej samej godzinie aktualizuje wiersz, nie duplikuje go."""

    def test_upserts_the_vector_on_symbol_and_hour(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "upsert"], [{"id": 1}])

        repo.save_feature_snapshot(
            "AAPL",
            {"price": 99.0, "regime": "NEUTRAL"},
            gated=False,
            feature_set_version=2,
        )

        mock_client.table.assert_called_with("feature_snapshots")
        upsert = mock_client.table.return_value.upsert
        upsert.assert_called_once()
        assert upsert.call_args.args[0] == {
            "symbol": "AAPL",
            "features": {"price": 99.0, "regime": "NEUTRAL"},
            "gated": False,
            "feature_set_version": 2,
        }
        assert upsert.call_args.kwargs["on_conflict"] == "symbol,as_of_hour"

    def test_payload_carries_no_label(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # Etykietę (zwrot 12h) dołącza się później z price_snapshots — w chwili
        # zapisu przyszłości jeszcze nie ma, a `as_of` stawia baza (DEFAULT NOW()).
        _set_chain_response(mock_client, ["table", "upsert"], [{"id": 1}])

        repo.save_feature_snapshot("AAPL", {}, gated=True, feature_set_version=2)

        payload = mock_client.table.return_value.upsert.call_args.args[0]
        assert set(payload) == {"symbol", "features", "gated", "feature_set_version"}


class TestGetSettledMoves:
    """ML-34: okno kalibracyjne przedziału ruchu z ROZLICZONYCH predykcji (kolumny
    istniejące od 001) i — na żądanie — przedziały do pokrycia (migracja 032)."""

    def _call(
        self, repo: SupabaseRepository, mock_client: MagicMock, pages: list[list[dict[str, Any]]],
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], MagicMock]:
        chain = mock_client.table.return_value
        order_builder = (
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value
        )
        _set_paginated_response(order_builder, pages)
        rows = repo.get_settled_moves(90, **kwargs)
        return rows, chain

    def test_reads_settled_rows_oldest_first(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        row = {
            "symbol": "AAPL",
            "timestamp": "2026-10-16T17:30:00+00:00",
            "price_at_prediction": 100.0,
            "predicted_target_price": 100.0,
            "actual_price_after_12h": 101.0,
        }

        rows, chain = self._call(repo, mock_client, [[row]])

        assert rows == [row]
        mock_client.table.assert_called_with("prediction_logs")
        columns = chain.select.call_args.args[0]
        assert {c.strip() for c in columns.split(",")} == {
            "symbol", "timestamp", "price_at_prediction",
            "predicted_target_price", "actual_price_after_12h",
        }
        chain.select.return_value.gte.return_value.not_.is_.assert_called_with(
            "actual_price_after_12h", "null"
        )
        order = chain.select.return_value.gte.return_value.not_.is_.return_value.order
        order.assert_called_with("timestamp", desc=False)

    def test_interval_columns_only_on_request(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _, chain = self._call(repo, mock_client, [[]], with_intervals=True)

        columns = chain.select.call_args.args[0]
        assert "interval_low" in columns
        assert "interval_high" in columns

    def test_paginates_past_the_postgrest_cap(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        full = [{"symbol": "AAPL"}] * 1000

        rows, _ = self._call(repo, mock_client, [full, [{"symbol": "MSFT"}]])

        assert len(rows) == 1001


class TestRegimeHistory:
    """ALR-03: stan histerezy reżimu w `regime_history` (migracja 033) — jeden
    wiersz na cykl; odczyt bierze najnowszy po `cycle_at`."""

    _CHAIN = ["table", "select", "order", "order", "limit"]

    def test_no_history_reads_as_none(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, self._CHAIN, [])

        assert repo.get_latest_regime_state() is None
        mock_client.table.assert_called_with("regime_history")

    def test_reads_the_newest_row_by_cycle_time(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.regime import MarketRegime, RegimeState

        _set_chain_response(
            mock_client,
            self._CHAIN,
            [{"raw_regime": "RISK_ON", "regime": "RISK_OFF", "streak": 1}],
        )

        state = repo.get_latest_regime_state()

        assert state == RegimeState(
            regime=MarketRegime.RISK_OFF, pending=MarketRegime.RISK_ON, streak=1
        )
        by_cycle = mock_client.table.return_value.select.return_value.order
        by_cycle.assert_called_once_with("cycle_at", desc=True)
        by_cycle.return_value.order.assert_called_once_with("id", desc=True)
        by_cycle.return_value.order.return_value.limit.assert_called_once_with(1)

    def test_settled_row_has_no_pending_exit(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.regime import MarketRegime, RegimeState

        _set_chain_response(
            mock_client,
            self._CHAIN,
            [{"raw_regime": "NEUTRAL", "regime": "NEUTRAL", "streak": 0}],
        )

        assert repo.get_latest_regime_state() == RegimeState(regime=MarketRegime.NEUTRAL)

    def test_saves_one_row_per_cycle(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.regime import MarketRegime, RegimeState

        _set_chain_response(mock_client, ["table", "insert"], [{"id": 1}])
        cycle_at = datetime(2026, 10, 19, 13, 30, tzinfo=UTC)

        repo.save_regime_state(
            cycle_at,
            MarketRegime.RISK_ON,
            RegimeState(regime=MarketRegime.RISK_OFF, pending=MarketRegime.RISK_ON, streak=1),
        )

        mock_client.table.assert_called_with("regime_history")
        insert = mock_client.table.return_value.insert
        insert.assert_called_once_with({
            "cycle_at": "2026-10-19T13:30:00+00:00",
            "raw_regime": "RISK_ON",
            "regime": "RISK_OFF",
            "streak": 1,
        })


class TestResolvedForEvalExtraFields:
    """WIA-55: kolumny segmentów (rada 005, pewność 016, reżim 017, Alpha Fusion
    022, edge 025) trafiają do selecta TYLKO na żądanie `--by` — domyślny
    przebieg ewaluacji ma działać na każdej bazie."""

    def _select_columns(
        self, repo: SupabaseRepository, mock_client: MagicMock, **kwargs: Any
    ) -> str:
        chain = mock_client.table.return_value
        order_builder = (
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value
        )
        _set_paginated_response(order_builder, [[]])
        repo.get_resolved_predictions_for_eval(30, **kwargs)
        return cast(str, chain.select.call_args.args[0])

    def test_default_select_reads_no_segment_column(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        columns = self._select_columns(repo, mock_client)

        for column in ("council_verdict", "regime", "confidence_score", "edge_sigma", "symbol"):
            assert column not in columns

    def test_council_recommendation_is_read_from_the_verdict_json(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        columns = self._select_columns(
            repo, mock_client, extra_fields=("council_recommendation", "regime")
        )

        assert (
            "council_recommendation:council_verdict->>final_recommendation" in columns
        )
        assert ", regime" in columns

    def test_unknown_field_is_rejected(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        with pytest.raises(ValueError, match="persona"):
            repo.get_resolved_predictions_for_eval(30, extra_fields=("persona",))


class TestResolvedForEvalGateDelta:
    """EXP-01: `gate_delta` (migracja 034) czytane tylko na żądanie `--echo`;
    przed migracją PostgREST zwraca 42703, a repozytorium tłumaczy to na
    `MissingColumnError`, żeby CLI wypisało brakującą migrację zamiast wyjątku."""

    def _order_builder(self, mock_client: MagicMock) -> MagicMock:
        chain = mock_client.table.return_value
        return cast(
            MagicMock,
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value,
        )

    def test_gate_delta_is_an_eval_field(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_paginated_response(self._order_builder(mock_client), [[]])

        repo.get_resolved_predictions_for_eval(30, extra_fields=("gate_delta",))

        assert ", gate_delta" in mock_client.table.return_value.select.call_args.args[0]

    def test_undefined_column_becomes_missing_column_error(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from postgrest.exceptions import APIError

        from src.infrastructure.adapters.supabase_repo import MissingColumnError

        self._order_builder(mock_client).range.return_value.execute.side_effect = APIError(
            {"code": "42703", "message": "column prediction_logs.gate_delta does not exist"}
        )

        with pytest.raises(MissingColumnError) as raised:
            repo.get_resolved_predictions_for_eval(30, extra_fields=("regime", "gate_delta"))

        assert raised.value.column == "gate_delta"

    def test_other_api_errors_propagate_unchanged(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from postgrest.exceptions import APIError

        self._order_builder(mock_client).range.return_value.execute.side_effect = APIError(
            {"code": "57014", "message": "canceling statement due to statement timeout"}
        )

        with pytest.raises(APIError):
            repo.get_resolved_predictions_for_eval(30, extra_fields=("gate_delta",))


class TestResolvedForEvalExperimentArms:
    """EXP-05: `evaluate --experiment` czyta ramiona z kolumny (migracja 037)
    i z kwitu v2 (020) oraz konsensus rady (005) — tylko na żądanie."""

    def _order_builder(self, mock_client: MagicMock) -> MagicMock:
        chain = mock_client.table.return_value
        return cast(
            MagicMock,
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value,
        )

    def test_arms_and_consensus_are_eval_fields(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_paginated_response(self._order_builder(mock_client), [[]])

        repo.get_resolved_predictions_for_eval(
            30,
            extra_fields=("council_consensus", "experiment_arms", "receipt_experiment_arms"),
        )

        columns = mock_client.table.return_value.select.call_args.args[0]
        assert ", experiment_arms" in columns
        assert "receipt_experiment_arms:decision_receipts->experiment_arms" in columns
        assert "council_consensus:council_verdict->>consensus_strength" in columns

    def test_missing_arms_column_becomes_missing_column_error(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from postgrest.exceptions import APIError

        from src.infrastructure.adapters.supabase_repo import MissingColumnError

        self._order_builder(mock_client).range.return_value.execute.side_effect = APIError(
            {"code": "42703", "message": "column prediction_logs.experiment_arms does not exist"}
        )

        with pytest.raises(MissingColumnError) as raised:
            repo.get_resolved_predictions_for_eval(
                30, extra_fields=("council_consensus", "experiment_arms", "receipt_experiment_arms")
            )

        assert raised.value.column == "experiment_arms"

class TestResolvedForEvalSkipsLlmFallback:
    """Atrapa po awarii LLM („SIDEWAYS 50%”, flaga `llm_analysis_failed`) nie jest
    predykcją — ewaluacja pomija ją tym samym filtrem co mail i Track Record."""

    def _order_builder(self, mock_client: MagicMock) -> MagicMock:
        chain = mock_client.table.return_value
        return cast(
            MagicMock,
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value,
        )

    def test_select_reads_the_quality_flags(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_paginated_response(self._order_builder(mock_client), [[]])

        repo.get_resolved_predictions_for_eval(30)

        assert "data_quality_flags" in mock_client.table.return_value.select.call_args.args[0]

    def test_llm_fallback_rows_are_dropped(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        real = {"predicted_trend": "BULLISH", "data_quality_flags": ["av_keys_exhausted"]}
        unflagged = {"predicted_trend": "BEARISH", "data_quality_flags": None}
        fallback = {"predicted_trend": "SIDEWAYS", "data_quality_flags": ["llm_analysis_failed"]}
        _set_paginated_response(self._order_builder(mock_client), [[real, fallback, unflagged]])

        out = repo.get_resolved_predictions_for_eval(30)

        assert out == [real, unflagged]


# ---------------------------------------------------------------------------
# FIN-13: save_llm_calls — ledger kosztów `llm_calls` (migracja 040)
# ---------------------------------------------------------------------------


class TestSaveLLMCalls:
    @staticmethod
    def _entries() -> list[Any]:
        from src.domain.finops import LLMCallEntry, LLMUsage

        return [
            LLMCallEntry(
                run_id="18324",
                node="predict",
                symbol="NVDA",
                usage=LLMUsage(
                    provider="anthropic",
                    model="claude-sonnet-4-6",
                    input_tokens=1200,
                    output_tokens=300,
                    cache_read_tokens=4000,
                    cache_write_tokens=0,
                ),
                cost_usd=0.0093,
                price_as_of=date(2026, 10, 20),
            ),
            LLMCallEntry(
                run_id="18324",
                node="weekly_recap",
                symbol=None,
                usage=LLMUsage(
                    provider="openai", model="gpt-9-ultra", input_tokens=10, output_tokens=5
                ),
                cost_usd=None,
                price_as_of=None,
            ),
        ]

    def test_inserts_one_row_per_call_in_a_single_batch(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        repo.save_llm_calls(self._entries())

        mock_client.table.assert_called_once_with("llm_calls")
        insert = mock_client.table.return_value.insert
        insert.assert_called_once()
        rows = insert.call_args.args[0]
        assert rows == [
            {
                "run_id": "18324",
                "symbol": "NVDA",
                "node": "predict",
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "input_tokens": 1200,
                "output_tokens": 300,
                "cache_read_tokens": 4000,
                "cache_write_tokens": 0,
                "cost_usd": 0.0093,
                "price_as_of": "2026-10-20",
            },
            {
                "run_id": "18324",
                "symbol": None,
                "node": "weekly_recap",
                "provider": "openai",
                "model": "gpt-9-ultra",
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "cost_usd": None,
                "price_as_of": None,
            },
        ]

    def test_no_calls_means_no_insert(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        repo.save_llm_calls([])

        mock_client.table.assert_not_called()


# ---------------------------------------------------------------------------
# NIEZ-29: save_cycle_run — historia przebiegów `cycle_runs` (migracja 041)
# ---------------------------------------------------------------------------


class TestSaveCycleRun:
    @staticmethod
    def _run(**overrides: Any) -> Any:
        from src.application.ports import CycleRun

        fields: dict[str, Any] = {
            "loop": "fast",
            "run_id": "18324",
            "run_attempt": 2,
            "started_at": datetime(2026, 10, 20, 5, 30, tzinfo=UTC),
            "duration_s": 312.5,
            "eligible": 40,
            "saved": 6,
            "failures": 1,
            "paid_calls": {"llm": 7, "council_llm": 48},
            "report_delivered": True,
        }
        fields.update(overrides)
        return CycleRun(**fields)

    def test_inserts_one_row_with_the_run_identity_and_counts(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        repo.save_cycle_run(self._run())

        mock_client.table.assert_called_once_with("cycle_runs")
        insert = mock_client.table.return_value.insert
        insert.assert_called_once()
        assert insert.call_args.args[0] == {
            "loop": "fast",
            "run_id": "18324",
            "run_attempt": 2,
            "started_at": "2026-10-20T05:30:00+00:00",
            "duration_s": 312.5,
            "eligible": 40,
            "saved": 6,
            "failures": 1,
            "paid_calls": {"llm": 7, "council_llm": 48},
            "report_delivered": True,
        }
        insert.return_value.execute.assert_called_once()

    def test_local_run_without_attempt_or_email_keeps_nulls(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        repo.save_cycle_run(self._run(run_id="local", run_attempt=None, report_delivered=None))

        row = mock_client.table.return_value.insert.call_args.args[0]
        assert (row["run_id"], row["run_attempt"], row["report_delivered"]) == (
            "local",
            None,
            None,
        )


class TestGetCycleRunsSince:
    """NIEZ-33: odczyt historii przebiegów pod publiczną stronę statusu."""

    @staticmethod
    def _order_builder(mock_client: MagicMock) -> MagicMock:
        chain = mock_client.table.return_value.select.return_value.gte.return_value
        return cast(MagicMock, chain.order.return_value.order.return_value)

    def test_reads_the_runs_since_a_moment_into_cycle_runs(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.application.ports import CycleRun

        _set_paginated_response(
            self._order_builder(mock_client),
            [[
                {
                    "loop": "fast",
                    "run_id": "18324",
                    "run_attempt": 2,
                    "started_at": "2026-10-20T05:30:00+00:00",
                    "duration_s": 312.5,
                    "eligible": 40,
                    "saved": 6,
                    "failures": 1,
                    "paid_calls": {"llm": 7},
                    "report_delivered": None,
                }
            ]],
        )

        runs = repo.get_cycle_runs_since(datetime(2026, 7, 27, tzinfo=UTC))

        mock_client.table.assert_called_once_with("cycle_runs")
        gte = mock_client.table.return_value.select.return_value.gte
        assert gte.call_args.args == ("started_at", "2026-07-27T00:00:00+00:00")
        assert runs == [
            CycleRun(
                loop="fast",
                run_id="18324",
                run_attempt=2,
                started_at=datetime(2026, 10, 20, 5, 30, tzinfo=UTC),
                duration_s=312.5,
                eligible=40,
                saved=6,
                failures=1,
                paid_calls={"llm": 7},
                report_delivered=None,
            )
        ]

    def test_postgrest_error_is_a_port_error(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from postgrest.exceptions import APIError

        from src.application.errors import PortError

        self._order_builder(mock_client).range.side_effect = APIError(
            {"code": "PGRST205", "message": "relation cycle_runs does not exist"}
        )

        with pytest.raises(PortError):
            repo.get_cycle_runs_since(datetime(2026, 7, 27, tzinfo=UTC))


class TestGetLlmCostSince:
    """FIN-25: suma wycenionych wierszy ledgera `llm_calls` od początku okna limitu."""

    @staticmethod
    def _order_builder(mock_client: MagicMock) -> MagicMock:
        chain = mock_client.table.return_value.select.return_value.gte.return_value
        return cast(MagicMock, chain.order.return_value)

    def test_sums_priced_rows_since_the_window_start(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_paginated_response(
            self._order_builder(mock_client),
            [[{"cost_usd": 0.25}, {"cost_usd": None}, {"cost_usd": 1.5}]],
        )

        spent = repo.get_llm_cost_since(datetime(2026, 10, 25, tzinfo=UTC))

        mock_client.table.assert_called_once_with("llm_calls")
        mock_client.table.return_value.select.assert_called_once_with("cost_usd")
        gte = mock_client.table.return_value.select.return_value.gte
        assert gte.call_args.args == ("created_at", "2026-10-25T00:00:00+00:00")
        assert spent == pytest.approx(1.75)

    def test_postgrest_error_is_a_port_error(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from postgrest.exceptions import APIError

        from src.application.errors import PortError

        self._order_builder(mock_client).range.side_effect = APIError(
            {"code": "PGRST205", "message": "relation llm_calls does not exist"}
        )

        with pytest.raises(PortError):
            repo.get_llm_cost_since(datetime(2026, 10, 25, tzinfo=UTC))


class TestShockAlertTimes:
    """ALR-02, decyzja 2: chwila ostatniego alertu per symbol (`created_at` z 021)."""

    def test_returns_the_latest_alert_time_per_symbol(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "gte"],
            [
                {"symbol": "BTC", "created_at": "2026-10-16T01:00:00+00:00"},
                {"symbol": "BTC", "created_at": "2026-10-16T22:00:00+00:00"},
                {"symbol": "ETH", "created_at": "2026-10-16T05:00:00+00:00"},
            ],
        )

        times = repo.get_shock_alert_times(datetime(2026, 10, 16, 0, 0, tzinfo=UTC))

        mock_client.table.assert_called_with("shock_alerts")
        assert times == {
            "BTC": datetime(2026, 10, 16, 22, 0, tzinfo=UTC),
            "ETH": datetime(2026, 10, 16, 5, 0, tzinfo=UTC),
        }

    def test_filters_by_creation_time(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "select", "gte"], [])

        repo.get_shock_alert_times(datetime(2026, 10, 16, 0, 0, tzinfo=UTC))

        mock_client.table.return_value.select.return_value.gte.assert_called_once_with(
            "created_at", "2026-10-16T00:00:00+00:00"
        )

    def test_is_graceful_without_migration(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        mock_client.table.side_effect = Exception("no table")

        assert repo.get_shock_alert_times(datetime(2026, 10, 16, tzinfo=UTC)) == {}


class TestShockAlertsSince:
    """ALR-06: alerty szoku od ostatniego maila do sekcji „⚡ Między mailami”."""

    def test_returns_alerts_since_the_moment_oldest_first(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "gte", "order"],
            [
                {
                    "symbol": "NVDA",
                    "delta": -0.062,
                    "direction": "DOWN",
                    "created_at": "2026-10-14T22:07:00+00:00",
                },
                {"symbol": None, "delta": 0.1, "direction": "UP", "created_at": None},
            ],
        )

        alerts = repo.get_shock_alerts_since(datetime(2026, 10, 14, 5, 30, tzinfo=UTC))

        mock_client.table.assert_called_with("shock_alerts")
        select = mock_client.table.return_value.select.return_value
        select.gte.assert_called_once_with("created_at", "2026-10-14T05:30:00+00:00")
        select.gte.return_value.order.assert_called_once_with("created_at", desc=False)
        assert alerts == [
            StoredShockAlert(
                symbol="NVDA",
                delta=-0.062,
                direction="DOWN",
                created_at=datetime(2026, 10, 14, 22, 7, tzinfo=UTC),
            )
        ]


class TestLastReportAt:
    """ALR-06: start ostatniego Fast Loopa z dostarczonym mailem (`cycle_runs`, 041)."""

    def test_reads_the_latest_delivered_fast_loop_before_the_cycle(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        chain = ["table", "select", "eq", "eq", "lt", "order", "limit"]
        _set_chain_response(mock_client, chain, [{"started_at": "2026-10-14T05:31:00+00:00"}])

        last = repo.get_last_report_at(datetime(2026, 10, 15, 5, 30, tzinfo=UTC))

        mock_client.table.assert_called_with("cycle_runs")
        select = mock_client.table.return_value.select.return_value
        select.eq.assert_called_once_with("loop", "fast")
        select.eq.return_value.eq.assert_called_once_with("report_delivered", True)
        select.eq.return_value.eq.return_value.lt.assert_called_once_with(
            "started_at", "2026-10-15T05:30:00+00:00"
        )
        assert last == datetime(2026, 10, 14, 5, 31, tzinfo=UTC)

    def test_none_without_a_delivered_run(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        chain = ["table", "select", "eq", "eq", "lt", "order", "limit"]
        _set_chain_response(mock_client, chain, [])

        assert repo.get_last_report_at(datetime(2026, 10, 15, tzinfo=UTC)) is None


class TestReferencePredictions:
    """WIA-24 (migracja 042): odczyt, wywołanie i rozliczenie sygnału referencyjnego
    w jednym wierszu `reference_predictions` — nigdy w `prediction_logs`."""

    def test_save_writes_the_read_and_call_with_the_reference_label(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.prediction import TrendDirection
        from src.domain.reference_signal import ReferenceCall, ReferenceRead

        _set_chain_response(mock_client, ["table", "insert"], [{"id": 1}])
        at = datetime(2026, 10, 25, 3, 0, tzinfo=UTC)

        repo.save_reference_read(
            ReferenceRead(symbol="AAPL", price=Decimal("103.25"), read_at=at),
            ReferenceCall(
                symbol="AAPL",
                trend=TrendDirection.BULLISH,
                price_at_call=Decimal("103.25"),
                called_at=at,
            ),
        )

        mock_client.table.assert_called_with("reference_predictions")
        assert mock_client.table.return_value.insert.call_args.args[0] == {
            "symbol": "AAPL",
            "read_at": "2026-10-25T03:00:00+00:00",
            "price": "103.25",
            "predicted_trend": "BULLISH",
            "recipe_version": "reference-v1",
        }

    def test_save_of_an_abstention_has_no_trend(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.reference_signal import ReferenceRead

        _set_chain_response(mock_client, ["table", "insert"], [{"id": 1}])

        repo.save_reference_read(
            ReferenceRead(
                symbol="BTC", price=Decimal("61000"), read_at=datetime(2026, 10, 25, tzinfo=UTC)
            ),
            None,
        )

        payload = mock_client.table.return_value.insert.call_args.args[0]
        assert payload["predicted_trend"] is None

    def test_reads_since_parse_into_domain_reads(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.reference_signal import ReferenceRead

        _set_chain_response(
            mock_client,
            ["table", "select", "gte", "order", "order", "range"],
            [{"symbol": "AAPL", "price": "100.5", "read_at": "2026-10-18T03:00:00+00:00"}],
        )

        reads = repo.get_reference_reads(datetime(2026, 10, 17, 3, 0, tzinfo=UTC))

        assert reads == [
            ReferenceRead(
                symbol="AAPL",
                price=Decimal("100.5"),
                read_at=datetime(2026, 10, 18, 3, 0, tzinfo=UTC),
            )
        ]
        mock_client.table.assert_called_with("reference_predictions")
        mock_client.table.return_value.select.return_value.gte.assert_called_once_with(
            "read_at", "2026-10-17T03:00:00+00:00"
        )

    def test_open_calls_are_unsettled_rows_with_a_trend(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.application.ports import OpenReferenceCall
        from src.domain.prediction import TrendDirection
        from src.domain.reference_signal import ReferenceCall

        select = mock_client.table.return_value.select.return_value
        unsettled = select.not_.is_.return_value.is_.return_value
        _set_paginated_response(
            unsettled.order.return_value.order.return_value,
            [
                [
                    {
                        "id": 7,
                        "symbol": "AAPL",
                        "predicted_trend": "BEARISH",
                        "price": "100",
                        "read_at": "2026-10-18T03:00:00+00:00",
                    }
                ]
            ],
        )

        calls = repo.get_open_reference_calls()

        assert calls == [
            OpenReferenceCall(
                id="7",
                call=ReferenceCall(
                    symbol="AAPL",
                    trend=TrendDirection.BEARISH,
                    price_at_call=Decimal("100"),
                    called_at=datetime(2026, 10, 18, 3, 0, tzinfo=UTC),
                ),
            )
        ]
        select.not_.is_.assert_called_once_with("predicted_trend", "null")
        select.not_.is_.return_value.is_.assert_called_once_with("outcome", "null")

    def test_settle_writes_outcome_price_and_time(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.reference_signal import ReferenceOutcome

        _set_chain_response(mock_client, ["table", "update", "eq"], [{"id": 7}])

        repo.settle_reference_call(
            "7", ReferenceOutcome.HIT, Decimal("104"), datetime(2026, 10, 25, 3, 0, tzinfo=UTC)
        )

        mock_client.table.assert_called_with("reference_predictions")
        update = mock_client.table.return_value.update
        assert update.call_args.args[0] == {
            "outcome": "HIT",
            "settled_price": "104",
            "settled_at": "2026-10-25T03:00:00+00:00",
        }
        update.return_value.eq.assert_called_once_with("id", "7")

    def test_expiry_settles_without_a_price(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.reference_signal import ReferenceOutcome

        _set_chain_response(mock_client, ["table", "update", "eq"], [{"id": 7}])

        repo.settle_reference_call(
            "7", ReferenceOutcome.EXPIRED, None, datetime(2026, 10, 25, 3, 0, tzinfo=UTC)
        )

        payload = mock_client.table.return_value.update.call_args.args[0]
        assert payload["outcome"] == "EXPIRED" and payload["settled_price"] is None


# ---------------------------------------------------------------------------
# WIA-70: dosłowny plaintext commitmentu (migracja 043)
# ---------------------------------------------------------------------------


class TestCommitmentPayloadColumn:
    """Reveal publikuje zapisany plaintext, a baza bez migracji 043 nie może
    zatrzymać ani sweepu starych wierszy, ani zapisu predykcji."""

    @staticmethod
    def _missing_payload_column() -> Exception:
        from postgrest.exceptions import APIError

        return APIError(
            {"code": "42703", "message": "column prediction_logs.commitment_payload does not exist"}
        )

    def test_unrevealed_commitments_read_the_stored_payload(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        builder = _fluent_query(mock_client, [[{"id": "p-1", "commitment_payload": "{}"}]])

        rows = repo.get_unrevealed_commitments()

        assert "commitment_payload" in builder.select.call_args.args[0]
        assert rows[0]["commitment_payload"] == "{}"

    def test_unrevealed_commitments_fall_back_without_the_payload_column(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # Przed 043 stare wiersze ujawniamy z kolumn, jak przed WIA-70.
        builder = _fluent_query(mock_client, [])
        failing, legacy = MagicMock(), MagicMock()
        failing.execute.side_effect = self._missing_payload_column()
        legacy.execute.return_value = MagicMock(data=[{"id": "p-1"}])
        builder.range.side_effect = [failing, legacy]

        rows = repo.get_unrevealed_commitments()

        assert rows == [{"id": "p-1"}]
        assert "commitment_payload" not in builder.select.call_args.args[0]

    def test_unrevealed_commitments_have_no_time_window_by_default(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # WIA-03: nieujawniony commitment starszy niż okno track recordu wygląda
        # jak ukryte pudło — sweep czyta wszystkie, bez okna i bez filtra watchlisty.
        builder = _fluent_query(mock_client, [[]])

        repo.get_unrevealed_commitments()

        builder.gte.assert_not_called()
        builder.in_.assert_not_called()
        assert "symbol" not in [c.args[0] for c in builder.eq.call_args_list]

    def test_unrevealed_commitments_honour_an_explicit_window(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        builder = _fluent_query(mock_client, [[]])

        repo.get_unrevealed_commitments(30)

        assert builder.gte.call_args.args[0] == "timestamp"

    def test_attestation_columns_available_when_the_select_succeeds(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "select", "limit"], [])

        assert repo.attestation_columns_available() is True
        selected = mock_client.table.return_value.select.call_args.args[0]
        assert {"commitment_hash", "commitment_salt", "commitment_payload"} <= {
            column.strip() for column in selected.split(",")
        }

    def test_attestation_columns_unavailable_without_the_migration(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        chain = mock_client.table.return_value.select.return_value.limit.return_value
        chain.execute.side_effect = self._missing_payload_column()

        assert repo.attestation_columns_available() is False


class TestCurrentHourCommitmentSalt:
    """WIA-01: sól wiersza (symbol, bieżąca godzina UTC) — klucza upsertu
    `save_prediction`. Ponowny zapis w tej godzinie reużywa ją, więc ten sam
    payload daje ten sam, już opublikowany hash."""

    def test_reads_the_salt_of_this_hour_row(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        builder = _fluent_query(mock_client, [])
        builder.limit.return_value = builder
        builder.execute.return_value = MagicMock(data=[{"commitment_salt": "f" * 32}])
        before = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)

        assert repo.get_current_hour_commitment_salt("AAPL") == "f" * 32

        after = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        filters = {call.args[0]: call.args[1] for call in builder.eq.call_args_list}
        assert filters["symbol"] == "AAPL"
        # Klucz upsertu: date_trunc('hour', timestamp, 'UTC') z migracji 013.
        assert datetime.fromisoformat(filters["timestamp_hour"]) in {before, after}
        builder.not_.is_.assert_called_with("commitment_salt", "null")

    def test_no_row_this_hour_means_no_salt(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        builder = _fluent_query(mock_client, [])
        builder.limit.return_value = builder
        builder.execute.return_value = MagicMock(data=[])

        assert repo.get_current_hour_commitment_salt("AAPL") is None

    def test_read_failure_degrades_to_no_salt(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        builder = _fluent_query(mock_client, [])
        builder.limit.return_value = builder
        builder.execute.side_effect = Exception("column commitment_salt does not exist")

        assert repo.get_current_hour_commitment_salt("AAPL") is None


class TestCurrentHourCommitment:
    """WIA-01: commitment wiersza (symbol, bieżąca godzina UTC). Graf porównuje go
    z nowym payloadem — inny payload nadpisze wiersz, więc stary commitment trzeba
    ujawnić jako „superseded", póki sól i plaintext są jeszcze w bazie."""

    @staticmethod
    def _builder(mock_client: MagicMock, rows: list[dict[str, Any]]) -> MagicMock:
        builder = _fluent_query(mock_client, [])
        builder.limit.return_value = builder
        builder.execute.return_value = MagicMock(data=rows)
        return builder

    def test_reads_hash_salt_and_payload_of_this_hour_row(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.attestation import StoredCommitment

        builder = self._builder(
            mock_client,
            [
                {
                    "commitment_hash": "a" * 64,
                    "commitment_salt": "f" * 32,
                    "commitment_payload": '{"schema_version":2}',
                }
            ],
        )
        before = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)

        stored = repo.get_current_hour_commitment("AAPL")

        after = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        assert stored == StoredCommitment(
            commitment="a" * 64, salt="f" * 32, payload='{"schema_version":2}'
        )
        columns = builder.select.call_args.args[0]
        assert {c.strip() for c in columns.split(",")} == {
            "commitment_hash",
            "commitment_salt",
            "commitment_payload",
        }
        filters = {call.args[0]: call.args[1] for call in builder.eq.call_args_list}
        assert filters["symbol"] == "AAPL"
        assert datetime.fromisoformat(filters["timestamp_hour"]) in {before, after}

    def test_row_without_payload_keeps_the_hash_and_salt(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # Wiersz sprzed 043: porównanie idzie po hashu, reveal bez plaintextu.
        self._builder(
            mock_client,
            [
                {
                    "commitment_hash": "a" * 64,
                    "commitment_salt": "f" * 32,
                    "commitment_payload": None,
                }
            ],
        )

        stored = repo.get_current_hour_commitment("AAPL")

        assert stored is not None
        assert stored.payload is None

    @pytest.mark.parametrize(
        "rows",
        [
            [],
            [{"commitment_hash": None, "commitment_salt": "f" * 32, "commitment_payload": None}],
            [{"commitment_hash": "a" * 64, "commitment_salt": None, "commitment_payload": None}],
        ],
    )
    def test_no_complete_commitment_this_hour_means_none(
        self, repo: SupabaseRepository, mock_client: MagicMock, rows: list[dict[str, Any]]
    ) -> None:
        self._builder(mock_client, rows)

        assert repo.get_current_hour_commitment("AAPL") is None

    def test_read_failure_degrades_to_none(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        builder = self._builder(mock_client, [])
        builder.execute.side_effect = Exception("column commitment_payload does not exist")

        assert repo.get_current_hour_commitment("AAPL") is None


class TestSaveCalibrationScoreVersion:
    """WIA-37: kolumna `calibration_score_version` (migracja 044) idzie do zapisu
    tylko z wersją; bez niej payload nie ma nowego klucza (PGRST204 przed 044)."""

    def test_versioned_calibration_writes_the_version_column(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "update", "eq"], [{"id": "pred-1"}])

        repo.save_calibration("pred-1", 0.19, "Przepewna.", score_version=2)

        payload = mock_client.table.return_value.update.call_args.args[0]
        assert payload == {
            "confidence_calibration": 0.19,
            "calibration_insight": "Przepewna.",
            "calibration_score_version": 2,
        }

    def test_unversioned_calibration_payload_has_no_version_key(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "update", "eq"], [{"id": "pred-1"}])

        repo.save_calibration("pred-1", 0.1, "Przepewna.")

        payload = mock_client.table.return_value.update.call_args.args[0]
        assert "calibration_score_version" not in payload


class TestResolvedForCalibrationTrendAndCycle:
    """WIA-39 / WIA-38: kalibracja kierunkowa potrzebuje `predicted_trend`, a n_eff
    i bootstrap po cyklach — `timestamp`. Obie kolumny istnieją od 001."""

    def test_select_carries_trend_and_timestamp(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        builder = _fluent_query(mock_client, [[]])

        repo.get_resolved_for_calibration(30)

        selected = builder.select.call_args.args[0]
        assert "predicted_trend" in selected
        assert "timestamp" in selected

    def test_select_carries_symbol_for_the_asset_class(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # ML-54: klasa aktywa (krypto vs reszta) wynika z symbolu — kolumna od 001.
        builder = _fluent_query(mock_client, [[]])

        repo.get_resolved_for_calibration(30)

        selected = [column.strip() for column in builder.select.call_args.args[0].split(",")]
        assert "symbol" in selected

class TestGetFundamentalsFetchedAt:
    """DAT-19: wiek wpisów cache (bez filtra TTL) — podstawa rotacji refreshu."""

    def test_maps_symbols_with_an_entry_to_their_fetched_at(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "in_"],
            [
                {"symbol": "AAPL", "fetched_at": "2026-10-11T06:00:00+00:00"},
                {"symbol": "MSFT", "fetched_at": None},
            ],
        )

        result = repo.get_fundamentals_fetched_at(["AAPL", "MSFT", "GOOGL"])

        assert result == {"AAPL": datetime(2026, 10, 11, 6, 0, tzinfo=UTC)}
        mock_client.table.assert_called_with("fundamentals_cache")
        in_call = mock_client.table.return_value.select.return_value.in_
        assert in_call.call_args.args == ("symbol", ["AAPL", "MSFT", "GOOGL"])

    def test_no_symbols_means_no_query(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        assert repo.get_fundamentals_fetched_at([]) == {}
        mock_client.table.assert_not_called()

# NIEZ-16: diagnoza „zaaplikuj migrację” tylko dla brakującego obiektu schematu
# ---------------------------------------------------------------------------

def _persona_accuracy(repo: SupabaseRepository) -> Any:
    from src.domain.value_objects import AssetType

    return repo.get_persona_accuracy(AssetType.STOCK)


# Odczyty graceful: błąd → pusty wynik. Każdy dawniej logował każdy wyjątek jako
# brak migracji, także zerwane połączenie.
_GRACEFUL_READS: dict[str, Callable[[SupabaseRepository], Any]] = {
    "find_similar_predictions": lambda r: r.find_similar_predictions([0.1, 0.2]),
    "get_persona_accuracy": _persona_accuracy,
    "get_persona_track_record": lambda r: r.get_persona_track_record(),
    "get_unrevealed_commitments": lambda r: r.get_unrevealed_commitments(),
    "get_sent_shock_alerts": lambda r: r.get_sent_shock_alerts(date(2026, 10, 1)),
    "get_shock_alert_times": lambda r: r.get_shock_alert_times(
        datetime(2026, 10, 1, tzinfo=UTC)
    ),
    "get_recent_model_scorecards": lambda r: r.get_recent_model_scorecards(),
}


def _fail_every_request(client: MagicMock, error: Exception) -> None:
    client.rpc.side_effect = error
    client.table.side_effect = error


def _mentions_migration(caplog: pytest.LogCaptureFixture) -> bool:
    return any(
        "migrat" in record.getMessage().lower() or "migrac" in record.getMessage().lower()
        for record in caplog.records
    )


@pytest.mark.parametrize("read", sorted(_GRACEFUL_READS))
def test_network_error_is_not_reported_as_missing_migration(
    read: str,
    repo: SupabaseRepository,
    mock_client: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    _fail_every_request(mock_client, httpx.ConnectError("connection reset by peer"))

    with caplog.at_level(logging.WARNING):
        result = _GRACEFUL_READS[read](repo)

    assert not result
    assert not _mentions_migration(caplog)
    assert any("ConnectError" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("code", ["PGRST202", "PGRST205", "42P01", "42883"])
def test_missing_schema_object_still_points_at_the_migration(
    code: str,
    repo: SupabaseRepository,
    mock_client: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from postgrest.exceptions import APIError

    _fail_every_request(mock_client, APIError({"code": code, "message": "not found"}))

    with caplog.at_level(logging.WARNING):
        assert repo.get_persona_track_record() == {}

    assert _mentions_migration(caplog)


def test_other_postgrest_error_is_logged_with_its_type_not_as_migration(
    repo: SupabaseRepository,
    mock_client: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    from postgrest.exceptions import APIError

    _fail_every_request(
        mock_client, APIError({"code": "57014", "message": "canceling statement due to timeout"})
    )

    with caplog.at_level(logging.WARNING):
        assert repo.get_recent_model_scorecards() == []

    assert not _mentions_migration(caplog)
    assert any("APIError" in record.getMessage() for record in caplog.records)


class TestMissingPredictionColumns:
    """FIN-05: preflight kolumn `prediction_logs` — select `limit 0`, brak kolumny
    rozpoznawany wyłącznie po 42703. Blip sieci nie może wyglądać jak brak migracji,
    bo wyłączyłby cały cykl."""

    @staticmethod
    def _execute(client: MagicMock) -> MagicMock:
        return cast(
            MagicMock, client.table.return_value.select.return_value.limit.return_value.execute
        )

    def test_no_missing_columns_on_a_migrated_database(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        missing = repo.missing_prediction_columns(frozenset({"symbol", "gate_delta"}))

        assert missing == frozenset()
        mock_client.table.assert_called_with("prediction_logs")
        mock_client.table.return_value.select.assert_called_once_with("gate_delta,symbol")
        mock_client.table.return_value.select.return_value.limit.assert_called_once_with(0)

    def test_reports_every_undefined_column_one_query_each(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from postgrest.exceptions import APIError

        def undefined(column: str) -> APIError:
            return APIError(
                {"code": "42703", "message": f"column prediction_logs.{column} does not exist"}
            )

        self._execute(mock_client).side_effect = [
            undefined("gate_delta"),
            undefined("recipe_version"),
            MagicMock(data=[]),
        ]

        missing = repo.missing_prediction_columns(
            frozenset({"symbol", "gate_delta", "recipe_version"})
        )

        assert missing == frozenset({"gate_delta", "recipe_version"})
        selects = [c.args[0] for c in mock_client.table.return_value.select.call_args_list]
        assert selects == ["gate_delta,recipe_version,symbol", "recipe_version,symbol", "symbol"]

    def test_network_error_propagates_instead_of_counting_as_missing(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        self._execute(mock_client).side_effect = httpx.ConnectError("connection reset")

        with pytest.raises(httpx.ConnectError):
            repo.missing_prediction_columns(frozenset({"symbol"}))

    @pytest.mark.parametrize(
        "error",
        [
            {"code": "57014", "message": "canceling statement due to statement timeout"},
            {"code": "42703", "message": "column prediction_logs.other does not exist"},
        ],
        ids=["other-code", "unrequested-column"],
    )
    def test_other_api_errors_propagate(
        self, repo: SupabaseRepository, mock_client: MagicMock, error: dict[str, str]
    ) -> None:
        from postgrest.exceptions import APIError

        self._execute(mock_client).side_effect = APIError(error)

        with pytest.raises(APIError):
            repo.missing_prediction_columns(frozenset({"symbol"}))


class TestSchemaVersion:
    """NIEZ-30: RPC `schema_version` (migracja 071) mówi, do której migracji sięga baza.
    NULL to „nie wiem”, nie zero; brak funkcji i awaria bazy lecą taksonomią portów."""

    @staticmethod
    def _execute(client: MagicMock) -> MagicMock:
        return cast(MagicMock, client.rpc.return_value.execute)

    def test_returns_the_applied_migration_number(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        self._execute(mock_client).return_value = MagicMock(data=70)

        assert repo.schema_version() == 70
        mock_client.rpc.assert_called_once_with("schema_version")

    def test_null_is_unknown_not_zero(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        self._execute(mock_client).return_value = MagicMock(data=None)

        assert repo.schema_version() is None

    @pytest.mark.parametrize("payload", ["70", [], {"version": 70}, True, 70.5])
    def test_a_payload_that_is_not_an_integer_is_malformed(
        self, repo: SupabaseRepository, mock_client: MagicMock, payload: object
    ) -> None:
        from src.application.errors import MalformedPayload

        self._execute(mock_client).return_value = MagicMock(data=payload)

        with pytest.raises(MalformedPayload):
            repo.schema_version()

    @pytest.mark.parametrize(
        ("code", "kind"),
        [("PGRST202", NotFound), ("42883", NotFound), ("PGRST002", ProviderUnavailable)],
    )
    def test_api_errors_leave_as_the_port_taxonomy(
        self, repo: SupabaseRepository, mock_client: MagicMock, code: str, kind: type[Exception]
    ) -> None:
        from postgrest.exceptions import APIError

        self._execute(mock_client).side_effect = APIError({"code": code, "message": "boom"})

        with pytest.raises(kind):
            repo.schema_version()


class TestEarningsCache:
    """FIN-11 (migracja 053): data najbliższego raportu per symbol — Slow Loop
    zapisuje, Fast Loop czyta tylko wpis młodszy niż TTL."""

    def test_save_upserts_the_report_date_per_symbol(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.earnings import CachedEarningsDate

        _set_chain_response(mock_client, ["table", "upsert"], [])

        repo.save_earnings(
            CachedEarningsDate(
                "AAPL", date(2026, 10, 29), datetime(2026, 10, 25, 3, 0, tzinfo=UTC)
            )
        )

        mock_client.table.assert_called_with("earnings_cache")
        assert mock_client.table.return_value.upsert.call_args.args[0] == {
            "symbol": "AAPL",
            "report_date": "2026-10-29",
            "fetched_at": "2026-10-25T03:00:00+00:00",
        }

    def test_get_reads_a_fresh_row(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.earnings import CachedEarningsDate

        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "gte"],
            [
                {
                    "symbol": "AAPL",
                    "report_date": "2026-10-29",
                    "fetched_at": "2026-10-25T03:00:00+00:00",
                }
            ],
        )

        assert repo.get_cached_earnings("AAPL") == CachedEarningsDate(
            "AAPL", date(2026, 10, 29), datetime(2026, 10, 25, 3, 0, tzinfo=UTC)
        )
        mock_client.table.assert_called_with("earnings_cache")
        mock_client.table.return_value.select.return_value.eq.assert_called_once_with(
            "symbol", "AAPL"
        )

    def test_get_filters_by_the_cache_ttl(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.earnings import EARNINGS_CACHE_TTL_HOURS

        _set_chain_response(mock_client, ["table", "select", "eq", "gte"], [])

        before = datetime.now(UTC)
        assert repo.get_cached_earnings("AAPL") is None
        after = datetime.now(UTC)

        gte = mock_client.table.return_value.select.return_value.eq.return_value.gte
        column, cutoff = gte.call_args.args
        ttl = timedelta(hours=EARNINGS_CACHE_TTL_HOURS)
        assert column == "fetched_at"
        assert before - ttl - timedelta(minutes=1) <= datetime.fromisoformat(cutoff)
        assert datetime.fromisoformat(cutoff) <= after - ttl + timedelta(minutes=1)

# WIA-49: expire_unverified_predictions — jawne wygaszanie zamiast ciszy
# ---------------------------------------------------------------------------


def _expiry_select_builder(mock_client: MagicMock) -> MagicMock:
    """Builder, na którym `_paginate` woła `.range(...)` w odczycie kandydatów."""
    chain = mock_client.table.return_value
    builder = (
        chain.select.return_value.eq.return_value.is_.return_value.lt.return_value
        .order.return_value.order.return_value
    )
    return cast(MagicMock, builder)


def test_expire_unverified_predictions_updates_only_older_rows(
    repo: SupabaseRepository, mock_client: MagicMock
) -> None:
    """Predykcja starsza niż okno oceny nigdy nie zostanie rozliczona — dostaje
    flagę `expired_unverified` w `data_quality_flags`, a NIE `actual_price`
    (cena trafiłaby do ml_feature_store jako fałszywa etykieta)."""
    _set_paginated_response(
        _expiry_select_builder(mock_client),
        [[
            {"id": "a", "data_quality_flags": []},
            {"id": "b", "data_quality_flags": ["av_keys_exhausted"]},
            {"id": "c", "data_quality_flags": ["expired_unverified"]},
            {"id": "d", "data_quality_flags": None},
        ]],
    )
    _set_chain_response(mock_client, ["table", "update", "eq", "is_"], [{"id": "x"}])
    before = datetime.now(UTC)

    expired = repo.expire_unverified_predictions("NVDA", older_than_hours=96)

    assert expired == 3
    chain = mock_client.table.return_value
    chain.select.assert_called_once_with("id, data_quality_flags")
    chain.select.return_value.eq.assert_called_once_with("symbol", "NVDA")
    chain.select.return_value.eq.return_value.is_.assert_called_once_with(
        "actual_price_after_12h", "null"
    )
    lt = chain.select.return_value.eq.return_value.is_.return_value.lt
    column, cutoff = lt.call_args.args
    assert column == "timestamp"
    assert before - timedelta(hours=96, seconds=5) <= datetime.fromisoformat(cutoff)
    assert datetime.fromisoformat(cutoff) <= datetime.now(UTC) - timedelta(hours=96)
    payloads = [update.args[0] for update in chain.update.call_args_list]
    assert payloads == [
        {"data_quality_flags": ["expired_unverified"]},
        {"data_quality_flags": ["av_keys_exhausted", "expired_unverified"]},
        {"data_quality_flags": ["expired_unverified"]},
    ]
    assert [call.args for call in chain.update.return_value.eq.call_args_list] == [
        ("id", "a"),
        ("id", "b"),
        ("id", "d"),
    ]
    # Strażnik wyścigu: wiersz rozliczony w międzyczasie nie dostaje flagi.
    chain.update.return_value.eq.return_value.is_.assert_called_with(
        "actual_price_after_12h", "null"
    )


def test_expire_counts_only_rows_the_update_actually_changed(
    repo: SupabaseRepository, mock_client: MagicMock
) -> None:
    _set_paginated_response(
        _expiry_select_builder(mock_client), [[{"id": "a", "data_quality_flags": []}]]
    )
    # Wiersz rozliczony między odczytem a zapisem: update nie dotyka żadnego wiersza.
    _set_chain_response(mock_client, ["table", "update", "eq", "is_"], [])

    assert repo.expire_unverified_predictions("NVDA", older_than_hours=96) == 0


def test_expire_without_stale_rows_writes_nothing(
    repo: SupabaseRepository, mock_client: MagicMock
) -> None:
    _set_paginated_response(_expiry_select_builder(mock_client), [[]])

    assert repo.expire_unverified_predictions("NVDA", older_than_hours=96) == 0
    mock_client.table.return_value.update.assert_not_called()


def _expired_count_builder(mock_client: MagicMock) -> MagicMock:
    chain = mock_client.table.return_value
    builder = (
        chain.select.return_value.gte.return_value.is_.return_value.contains.return_value
        .order.return_value
    )
    return cast(MagicMock, builder)


class TestCountExpiredUnverified:
    """WIA-49: liczba wygasłych z całego okna track recordu — wiersze oznaczone we
    wcześniejszych przebiegach, nie tylko w bieżącym."""

    def test_counts_rows_flagged_in_earlier_runs_within_the_window(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_paginated_response(
            _expired_count_builder(mock_client), [[{"id": "a"}, {"id": "b"}, {"id": "c"}]]
        )
        before = datetime.now(UTC)

        count = repo.count_expired_unverified(days=30)

        assert count == 3
        chain = mock_client.table.return_value
        chain.select.assert_called_once_with("id")
        column, cutoff = chain.select.return_value.gte.call_args.args
        assert column == "timestamp"
        assert datetime.fromisoformat(cutoff) <= before - timedelta(days=30) + timedelta(seconds=5)
        chain.select.return_value.gte.return_value.is_.assert_called_once_with(
            "actual_price_after_12h", "null"
        )
        contains = chain.select.return_value.gte.return_value.is_.return_value.contains
        contains.assert_called_once_with("data_quality_flags", '["expired_unverified"]')

    def test_no_flagged_rows_count_zero(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_paginated_response(_expired_count_builder(mock_client), [[]])

        assert repo.count_expired_unverified(days=30) == 0

    def test_database_error_degrades_to_zero_with_a_warning(
        self, repo: SupabaseRepository, mock_client: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        from postgrest.exceptions import APIError

        builder = _expired_count_builder(mock_client)
        builder.range.return_value.execute.side_effect = APIError(
            {"code": "PGRST000", "message": "down"}
        )

        with caplog.at_level(logging.WARNING):
            assert repo.count_expired_unverified(days=30) == 0

        assert "expired" in caplog.text


# ---------------------------------------------------------------------------
# WIA-56: odczyt kwitów decyzyjnych (migracja 020)
# ---------------------------------------------------------------------------

_RECEIPT_ROW: dict[str, Any] = {
    "id": "pred-1",
    "symbol": "NVDA",
    "timestamp": "2026-08-14T05:31:00+00:00",
    "predicted_trend": "BULLISH",
    "is_trend_correct": False,
    "decision_receipts": {"schema_version": 1, "effective_threshold": "0.03"},
}


def test_get_receipt_maps_the_row_and_its_raw_receipt(
    repo: SupabaseRepository, mock_client: MagicMock
) -> None:
    _set_chain_response(mock_client, ["table", "select", "eq", "limit"], [_RECEIPT_ROW])

    stored = repo.get_receipt("pred-1")

    assert stored is not None
    assert (stored.prediction_id, stored.symbol, stored.predicted_trend) == (
        "pred-1",
        "NVDA",
        "BULLISH",
    )
    assert stored.predicted_at == datetime(2026, 8, 14, 5, 31, tzinfo=UTC)
    assert stored.is_trend_correct is False
    assert stored.payload == {"schema_version": 1, "effective_threshold": "0.03"}
    select = mock_client.table.return_value.select
    assert "decision_receipts" in select.call_args.args[0]
    select.return_value.eq.assert_called_once_with("id", "pred-1")


def test_get_receipt_of_an_unknown_prediction_is_none(
    repo: SupabaseRepository, mock_client: MagicMock
) -> None:
    _set_chain_response(mock_client, ["table", "select", "eq", "limit"], [])

    assert repo.get_receipt("pred-404") is None


def test_receipt_that_is_not_a_json_object_maps_to_none(
    repo: SupabaseRepository, mock_client: MagicMock
) -> None:
    _set_chain_response(
        mock_client,
        ["table", "select", "eq", "limit"],
        [{**_RECEIPT_ROW, "decision_receipts": None, "is_trend_correct": None}],
    )

    stored = repo.get_receipt("pred-1")

    assert stored is not None
    assert stored.payload is None
    assert stored.is_trend_correct is None


def test_find_receipts_reads_one_utc_day_of_the_symbol(
    repo: SupabaseRepository, mock_client: MagicMock
) -> None:
    _set_chain_response(
        mock_client, ["table", "select", "eq", "gte", "lt", "order"], [_RECEIPT_ROW]
    )

    found = repo.find_receipts("NVDA", date(2026, 8, 14))

    assert [stored.prediction_id for stored in found] == ["pred-1"]
    select = mock_client.table.return_value.select
    select.return_value.eq.assert_called_once_with("symbol", "NVDA")
    select.return_value.eq.return_value.gte.assert_called_once_with(
        "timestamp", "2026-08-14T00:00:00+00:00"
    )
    select.return_value.eq.return_value.gte.return_value.lt.assert_called_once_with(
        "timestamp", "2026-08-15T00:00:00+00:00"
    )


def test_recently_resolved_rows_carry_the_receipt_for_the_post_mortem(
    repo: SupabaseRepository, mock_client: MagicMock
) -> None:
    """WIA-56: linia kwitu pod chybieniem w mailu czyta `decision_receipts`
    z tego samego odczytu zamkniętych predykcji (migracja 020 jest na produkcji)."""
    chain = mock_client.table.return_value
    order_builder = (
        chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value
    )
    _set_paginated_response(order_builder, [[]])

    repo.get_recently_resolved_predictions(24)

    assert "decision_receipts" in chain.select.call_args.args[0]


def test_recently_resolved_select_has_no_receipt_when_receipts_are_off(
    mock_client: MagicMock,
) -> None:
    """WIA-56: wyłączenie kwitów (`receipts_enabled = false`) wyłącza też ich odczyt
    w 24-godzinnym odczycie rozliczeń — select nie niesie `decision_receipts`."""
    repo = SupabaseRepository(
        url="https://test.supabase.co", key="anon-key", receipts_enabled=False
    )
    chain = mock_client.table.return_value
    order_builder = (
        chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value
    )
    _set_paginated_response(order_builder, [[]])

    repo.get_recently_resolved_predictions(24)

    assert "decision_receipts" not in chain.select.call_args.args[0]


# ---------------------------------------------------------------------------
# DAT-14: `stale_quote` (migracja 055) — rozliczenie i odczyty trafności
# ---------------------------------------------------------------------------


class TestStaleQuote:
    """Rozliczenie z nieaktualnego notowania niesie `stale_quote` tylko, gdy graf
    je poda (flaga), a odczyty trafności wybierają kolumnę i odsiewają takie
    wiersze wyłącznie za flagą repozytorium — przed migracją SELECT dałby 400."""

    @staticmethod
    def _repo(mock_client: MagicMock) -> SupabaseRepository:
        return SupabaseRepository(
            url="https://test.supabase.co", key="anon-key", stale_quote_enabled=True
        )

    @staticmethod
    def _settle(repo: SupabaseRepository, **kwargs: Any) -> None:
        repo.update_prediction_accuracy(
            prediction_id="uuid-1",
            actual_price=Decimal("100.0"),
            accuracy_score=0.96,
            is_trend_correct=False,
            insight="x",
            **kwargs,
        )

    def test_settlement_carries_the_flag_only_when_given(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "update", "eq", "is_"], [{"id": "uuid-1"}])

        self._settle(repo)
        plain = mock_client.table.return_value.update.call_args.args[0]
        self._settle(repo, stale_quote=True)
        stale = mock_client.table.return_value.update.call_args.args[0]

        assert "stale_quote" not in plain
        assert stale["stale_quote"] is True

    def test_unverified_prediction_carries_its_creation_time(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "is_", "or_", "order", "limit"],
            [
                {
                    "id": "uuid-1",
                    "symbol": "AAPL",
                    "predicted_trend": "BULLISH",
                    "price_at_prediction": 100.0,
                    "predicted_target_price": 104.0,
                    "timestamp": "2026-10-19T05:30:00+00:00",
                }
            ],
        )

        prediction = repo.get_unverified_prediction("AAPL")

        assert prediction is not None
        assert prediction.created_at == datetime(2026, 10, 19, 5, 30, tzinfo=UTC)

    def test_accuracy_stats_drop_stale_settlements_with_the_flag(
        self, mock_client: MagicMock
    ) -> None:
        repo = self._repo(mock_client)
        chain = mock_client.table.return_value
        rows: list[dict[str, Any]] = [
            {"is_trend_correct": True, "stale_quote": None},
            {"is_trend_correct": False, "stale_quote": True},
            {"is_trend_correct": False, "stale_quote": False},
        ]
        _set_paginated_response(
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value,
            [rows],
        )

        stats = repo.get_accuracy_stats(days=30)

        assert stats["sample_count"] == 2
        assert stats["correct_count"] == 1
        assert "stale_quote" in chain.select.call_args.args[0]

    def test_without_the_flag_readers_do_not_select_the_column(
        self, repo: SupabaseRepository, mock_client: MagicMock, mocker: Any
    ) -> None:
        mocker.patch("src.infrastructure.adapters.supabase_repo._paginate", return_value=[])

        repo.get_accuracy_stats(days=30)
        repo.get_resolved_predictions(days=30)
        repo.get_resolved_for_calibration(days=30)

        selects = mock_client.table.return_value.select.call_args_list
        assert len(selects) == 3
        assert all("stale_quote" not in c.args[0] for c in selects)

    def test_every_accuracy_reader_drops_stale_settlements(
        self, mock_client: MagicMock, mocker: Any
    ) -> None:
        repo = self._repo(mock_client)
        rows = [
            {"symbol": "AAPL", "is_trend_correct": True, "stale_quote": True},
            {"symbol": "NVDA", "is_trend_correct": True},
        ]
        mocker.patch(
            "src.infrastructure.adapters.supabase_repo._paginate",
            side_effect=lambda _query: [dict(row) for row in rows],
        )

        readers = {
            "recent": repo.get_recently_resolved_predictions(24),
            "track_record": repo.get_resolved_predictions(days=30),
            "detailed": repo.get_resolved_predictions_detailed(days=7),
            "calibration": repo.get_resolved_for_calibration(days=30),
            "eval": repo.get_resolved_predictions_for_eval(days=30),
        }

        assert {name: [r["symbol"] for r in out] for name, out in readers.items()} == {
            name: ["NVDA"] for name in readers
        }
        selects = mock_client.table.return_value.select.call_args_list
        assert len(selects) == len(readers)
        assert all("stale_quote" in c.args[0] for c in selects)

    def test_settled_moves_drop_stale_settlements_with_the_flag(
        self, mock_client: MagicMock, mocker: Any
    ) -> None:
        # Rozliczenie ceną równą cenie wejścia nie jest ruchem — nie może zawężać
        # przedziału ruchu (ML-34) ani reszt ACI.
        repo = self._repo(mock_client)
        rows = [
            {"symbol": "AAPL", "actual_price_after_12h": 100.0, "stale_quote": True},
            {"symbol": "NVDA", "actual_price_after_12h": 101.0, "stale_quote": None},
        ]
        mocker.patch(
            "src.infrastructure.adapters.supabase_repo._paginate",
            side_effect=lambda _query: [dict(row) for row in rows],
        )

        moves = repo.get_settled_moves(90, with_intervals=True)

        assert [row["symbol"] for row in moves] == ["NVDA"]
        assert "stale_quote" in mock_client.table.return_value.select.call_args.args[0]


# ---------------------------------------------------------------------------
# WIA-61: rozliczenie także na oficjalnym zamknięciu sesji (migracja 059)
# ---------------------------------------------------------------------------


class TestSessionCloseSettlement:
    """Rozliczenie niesie zamknięcie sesji tylko, gdy graf je poda (flaga), a hit
    rate na nim liczy się obok trafności /quote wyłącznie za flagą repozytorium —
    przed migracją 059 SELECT kolumny dałby 400."""

    @staticmethod
    def _settlement() -> SessionCloseSettlement:
        from src.domain.session_close import (
            SessionClose,
            SessionCloseSource,
        )

        close = SessionClose(
            Money(Decimal("106.12345")), date(2026, 10, 19), SessionCloseSource.QUOTE_CLOSE
        )
        return SessionCloseSettlement(close, is_trend_correct=True)

    def test_settlement_carries_the_session_close_only_when_given(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "update", "eq", "is_"], [{"id": "uuid-1"}])
        settle: dict[str, Any] = {
            "prediction_id": "uuid-1",
            "actual_price": Decimal("105.0"),
            "accuracy_score": 0.96,
            "is_trend_correct": True,
            "insight": "x",
        }

        repo.update_prediction_accuracy(**settle)
        plain = mock_client.table.return_value.update.call_args.args[0]
        repo.update_prediction_accuracy(**settle, session_close=self._settlement())
        both = mock_client.table.return_value.update.call_args.args[0]

        assert not {"settlement_close_price", "settlement_session_date"} & set(plain)
        assert both["actual_price_after_12h"] == "105.0"
        # DECIMAL(14,4) jak pozostałe ceny (ARCH-38).
        assert both["settlement_close_price"] == "106.1235"
        assert both["settlement_session_date"] == "2026-10-19"
        assert both["settlement_source"] == "quote_close"
        assert both["is_trend_correct_session_close"] is True

    def test_accuracy_stats_add_the_session_close_hit_rate_with_the_flag(
        self, mock_client: MagicMock
    ) -> None:
        repo = SupabaseRepository(
            url="https://test.supabase.co", key="anon-key", session_close_settlement_enabled=True
        )
        chain = mock_client.table.return_value
        rows: list[dict[str, Any]] = [
            {"is_trend_correct": True, "is_trend_correct_session_close": True},
            {"is_trend_correct": False, "is_trend_correct_session_close": True},
            {"is_trend_correct": False, "is_trend_correct_session_close": False},
            {"is_trend_correct": True, "is_trend_correct_session_close": None},
        ]
        _set_paginated_response(
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value,
            [rows],
        )

        stats = repo.get_accuracy_stats(days=30)

        assert "is_trend_correct_session_close" in chain.select.call_args.args[0]
        assert stats["sample_count"] == 4
        # Te same trzy predykcje z zamknięciem sesji: 2 trafione na nim, 1 na /quote.
        assert stats["session_close_count"] == 3
        assert stats["session_close_correct_count"] == 2
        assert stats["session_close_quote_correct_count"] == 1

    def test_without_the_flag_accuracy_stats_stay_as_they_were(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        chain = mock_client.table.return_value
        _set_paginated_response(
            chain.select.return_value.gte.return_value.not_.is_.return_value.order.return_value,
            [[{"is_trend_correct": True}]],
        )

        stats = repo.get_accuracy_stats(days=30)

        assert "is_trend_correct_session_close" not in chain.select.call_args.args[0]
        assert "session_close_count" not in stats


# ---------------------------------------------------------------------------
# WIA-54: cena wykonalna od otwarcia sesji (migracja 072)
# ---------------------------------------------------------------------------


class TestExecutableEntryPrice:
    """Rozliczenie niesie `entry_price` tylko, gdy graf go poda (flaga), a odczyt
    track recordu wybiera kolumnę wyłącznie za flagą repozytorium — przed migracją
    072 SELECT kolumny dałby 400, a klucz w UPDATE — PGRST204."""

    _SETTLE: dict[str, Any] = {
        "prediction_id": "uuid-1",
        "actual_price": Decimal("106.0"),
        "accuracy_score": 0.96,
        "is_trend_correct": True,
        "insight": "x",
    }

    def test_settlement_carries_the_entry_price_only_when_given(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "update", "eq", "is_"], [{"id": "uuid-1"}])

        repo.update_prediction_accuracy(**self._SETTLE)
        plain = mock_client.table.return_value.update.call_args.args[0]
        repo.update_prediction_accuracy(**self._SETTLE, entry_price=Decimal("105.12345"))
        both = mock_client.table.return_value.update.call_args.args[0]

        assert "entry_price" not in plain
        # DECIMAL(14,4) jak pozostałe ceny (ARCH-38).
        assert both["entry_price"] == "105.1235"

    def test_track_record_selects_the_entry_price_only_with_the_flag(
        self, mock_client: MagicMock
    ) -> None:
        select = mock_client.table.return_value.select
        _set_chain_response(
            mock_client, ["table", "select", "gte", "not_", "is_", "order", "range"], []
        )
        plain = SupabaseRepository(url="https://test.supabase.co", key="anon-key")
        flagged = SupabaseRepository(
            url="https://test.supabase.co", key="anon-key", executable_entry_enabled=True
        )

        plain.get_resolved_predictions(30)
        assert "entry_price" not in select.call_args.args[0]
        flagged.get_resolved_predictions(30)
        assert "entry_price" in select.call_args.args[0]


# ---------------------------------------------------------------------------
# DAT-21: `fundamentals_cache.market_cap` (migracja 056)
# ---------------------------------------------------------------------------


class TestFundamentalsMarketCap:
    """Upsert niesie kapitalizację tylko za flagą (przed migracją PGRST204 zgubiłby
    cały wpis cache); odczyt `select("*")` bierze ją, gdy kolumna jest."""

    @staticmethod
    def _fundamentals() -> Fundamentals:
        return Fundamentals(
            trailing_pe=25.4,
            forward_pe=None,
            peg_ratio=None,
            eps_growth_yoy=None,
            fetched_at=datetime(2026, 10, 18, tzinfo=UTC),
            market_cap=3.5e12,
        )

    def test_flag_off_upsert_has_no_market_cap_key(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "upsert"], [])

        repo.save_fundamentals("AAPL", self._fundamentals())

        assert "market_cap" not in mock_client.table.return_value.upsert.call_args.args[0]

    def test_flag_on_upsert_carries_the_market_cap(self, mock_client: MagicMock) -> None:
        repo = SupabaseRepository(
            url="https://test.supabase.co", key="anon-key", fundamentals_market_cap_enabled=True
        )
        _set_chain_response(mock_client, ["table", "upsert"], [])

        repo.save_fundamentals("AAPL", self._fundamentals())

        payload = mock_client.table.return_value.upsert.call_args.args[0]
        assert payload["market_cap"] == pytest.approx(3.5e12)

    @pytest.mark.parametrize(("stored", "expected"), [("3500000000000", 3.5e12), (None, None)])
    def test_cached_fundamentals_read_the_market_cap_when_present(
        self,
        repo: SupabaseRepository,
        mock_client: MagicMock,
        stored: str | None,
        expected: float | None,
    ) -> None:
        row: dict[str, Any] = {
            "symbol": "AAPL",
            "trailing_pe": 25.4,
            "fetched_at": "2026-10-18T00:00:00+00:00",
        }
        if stored is not None:
            row["market_cap"] = stored
        _set_chain_response(mock_client, ["table", "select", "eq", "gte"], [row])

        result = repo.get_cached_fundamentals("AAPL")

        assert result is not None
        assert result.market_cap == expected


# ---------------------------------------------------------------------------
# ALR-11: `price_level_alerts` (migracja 057)
# ---------------------------------------------------------------------------


class TestPriceLevelAlerts:
    def test_active_levels_map_to_domain_alerts(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.price_levels import LevelDirection

        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "order"],
            [
                {
                    "id": 7,
                    "symbol": "NVDA",
                    "level": 150,
                    "direction": "ABOVE",
                    "note": "realizacja części zysku",
                    "last_seen_price": 148.5,
                },
                {
                    "id": 8,
                    "symbol": "BTC",
                    "level": "60000.5",
                    "direction": "BELOW",
                    "note": None,
                    "last_seen_price": None,
                },
            ],
        )

        nvda, btc = repo.get_active_price_levels()

        mock_client.table.assert_called_with("price_level_alerts")
        mock_client.table.return_value.select.return_value.eq.assert_called_with("active", True)
        assert nvda.id == "7"
        assert nvda.level.level == Decimal("150")
        assert nvda.level.direction is LevelDirection.ABOVE
        assert nvda.level.note == "realizacja części zysku"
        assert nvda.last_seen_price == Decimal("148.5")
        assert btc.level.level == Decimal("60000.5")
        assert btc.last_seen_price is None

    def test_adding_levels_never_overwrites_a_saved_one(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.price_levels import LevelDirection, PriceLevel

        repo.add_price_levels(
            [PriceLevel("NVDA", Decimal("150"), LevelDirection.ABOVE, "zysk")]
        )

        upsert = mock_client.table.return_value.upsert
        assert upsert.call_args.args[0] == [
            {"symbol": "NVDA", "level": "150", "direction": "ABOVE", "note": "zysk"}
        ]
        assert upsert.call_args.kwargs == {
            "on_conflict": "symbol,level,direction",
            "ignore_duplicates": True,
        }

    def test_no_levels_no_request(self, repo: SupabaseRepository, mock_client: MagicMock) -> None:
        repo.add_price_levels([])

        mock_client.table.assert_not_called()

    def test_reading_updates_only_an_active_alert(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        repo.record_price_level_reading("7", Decimal("151.25"))

        update = mock_client.table.return_value.update
        assert update.call_args.args[0] == {"last_seen_price": "151.25"}
        update.return_value.eq.assert_called_with("id", "7")
        update.return_value.eq.return_value.eq.assert_called_with("active", True)

    def test_trigger_disables_the_alert_with_its_time_and_price(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        moment = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)

        repo.mark_price_level_triggered("7", Decimal("151.2"), moment)

        update = mock_client.table.return_value.update
        assert update.call_args.args[0] == {
            "active": False,
            "triggered_at": moment.isoformat(),
            "last_seen_price": "151.2",
        }
        update.return_value.eq.assert_called_with("id", "7")


class TestPortErrorTaxonomy:
    """ARCH-33: zapisy i odczyty Fast Loopa bez własnej degradacji tłumaczą PostgREST
    `APIError` na taksonomię portów; wyjątek nadal jest `APIError` z kodem."""

    @staticmethod
    def _failing_upsert(mock_client: MagicMock, code: str) -> None:
        from postgrest.exceptions import APIError

        mock_client.table.return_value.upsert.return_value.execute.side_effect = APIError(
            {"code": code, "message": "boom"}
        )

    @pytest.mark.parametrize(
        ("code", "kind_name"),
        [
            ("PGRST003", "ProviderUnavailable"),
            ("57014", "ProviderUnavailable"),
            ("PGRST205", "NotFound"),
            ("42P01", "NotFound"),
            ("23505", "PortError"),
        ],
    )
    def test_snapshot_write_error_is_translated(
        self, repo: SupabaseRepository, mock_client: MagicMock, code: str, kind_name: str
    ) -> None:
        from postgrest.exceptions import APIError

        import src.application.errors as errors

        self._failing_upsert(mock_client, code)

        with pytest.raises(getattr(errors, kind_name)) as excinfo:
            repo.save_price_snapshot("AAPL", Money(Decimal("1")))

        assert isinstance(excinfo.value, APIError)
        assert excinfo.value.code == code

    def test_last_prediction_price_read_error_is_provider_unavailable(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from postgrest.exceptions import APIError

        from src.application.errors import ProviderUnavailable

        chain = mock_client.table.return_value.select.return_value.eq.return_value
        chain.order.return_value.limit.return_value.execute.side_effect = APIError(
            {"code": "PGRST000", "message": "could not connect"}
        )

        with pytest.raises(ProviderUnavailable):
            repo.get_last_prediction_price("AAPL")

    def test_transport_error_is_provider_unavailable(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.application.errors import ProviderUnavailable

        mock_client.table.return_value.insert.return_value.execute.side_effect = (
            httpx.ConnectError("connection reset")
        )

        with pytest.raises(ProviderUnavailable) as excinfo:
            repo.save_quota_alert(_quota_alert())

        assert isinstance(excinfo.value, httpx.ConnectError)

    def test_other_httpx_error_is_a_port_error(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.application.errors import PortError, ProviderUnavailable

        mock_client.table.return_value.insert.return_value.execute.side_effect = (
            httpx.DecodingError("bad gzip")
        )

        with pytest.raises(PortError) as excinfo:
            repo.save_quota_alert(_quota_alert())

        assert not isinstance(excinfo.value, ProviderUnavailable)

    def test_every_public_method_translates_provider_errors(self) -> None:
        # Nowa metoda repozytorium dostaje tłumaczenie bez pamiętania o dekoratorze.
        untranslated = sorted(
            name
            for name, member in vars(SupabaseRepository).items()
            if callable(member) and not name.startswith("_") and not hasattr(member, "__wrapped__")
        )

        assert untranslated == []


class TestTrainingDataReads:
    """ML-52: Slow Loop czyta zapisane wektory cech (`feature_snapshots`, 031) i szereg
    snapshotów cen do etykiety — wektor dokładnie w postaci zapisanej przez cykl."""

    def test_feature_snapshots_of_one_version_are_read_as_saved_vectors(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        vector = {"price": 100.0, "snapshot_delta": 0.0123, "regime": "RISK_ON", "x": None}
        builder = _fluent_query(
            mock_client,
            [[
                {
                    "symbol": "AAPL",
                    "as_of": "2026-10-01T07:30:00+00:00",
                    "gated": True,
                    "features": vector,
                    "feature_set_version": 4,
                }
            ]],
        )

        [snapshot] = repo.get_feature_snapshots(4)

        mock_client.table.assert_called_with("feature_snapshots")
        builder.eq.assert_called_with("feature_set_version", 4)
        assert [c.args[0] for c in builder.order.call_args_list] == ["as_of", "id"]
        assert snapshot.symbol == "AAPL"
        assert snapshot.as_of == datetime(2026, 10, 1, 7, 30, tzinfo=UTC)
        assert snapshot.gated is True
        assert snapshot.feature_set_version == 4
        assert dict(snapshot.features) == vector

    def test_feature_snapshot_rows_without_time_or_vector_are_skipped(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        good = {
            "symbol": "AAPL",
            "as_of": "2026-10-01T07:30:00+00:00",
            "gated": False,
            "features": {"price": 1.0},
            "feature_set_version": 4,
        }
        _fluent_query(
            mock_client,
            [[{**good, "as_of": None}, {**good, "features": "[]"}, {**good, "symbol": ""}, good]],
        )

        assert len(repo.get_feature_snapshots(4)) == 1

    def test_feature_snapshots_are_read_past_one_page(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        row = {
            "symbol": "AAPL",
            "as_of": "2026-10-01T07:30:00+00:00",
            "gated": False,
            "features": {"price": 1.0},
            "feature_set_version": 4,
        }
        builder = _fluent_query(mock_client, [[row] * 1000, [row] * 3])

        assert len(repo.get_feature_snapshots(4)) == 1003
        assert builder.range.call_count == 2

    def test_price_series_groups_snapshots_by_symbol_in_time_order(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        since = datetime(2026, 9, 1, tzinfo=UTC)
        builder = _fluent_query(
            mock_client,
            [[
                {"symbol": "AAPL", "timestamp": "2026-10-01T07:30:00+00:00", "price": 100.0},
                {"symbol": "BTC", "timestamp": "2026-10-01T07:31:00+00:00", "price": "60000.5"},
                {"symbol": "AAPL", "timestamp": "2026-10-02T07:30:00+00:00", "price": 101.0},
                {"symbol": "AAPL", "timestamp": None, "price": 1.0},
            ]],
        )

        series = repo.get_price_series(["AAPL", "BTC"], since=since)

        mock_client.table.assert_called_with("price_snapshots")
        builder.in_.assert_called_with("symbol", ["AAPL", "BTC"])
        builder.gte.assert_called_with("timestamp", since.isoformat())
        assert [c.args[0] for c in builder.order.call_args_list] == ["timestamp", "id"]
        assert series == {
            "AAPL": [
                (datetime(2026, 10, 1, 7, 30, tzinfo=UTC), 100.0),
                (datetime(2026, 10, 2, 7, 30, tzinfo=UTC), 101.0),
            ],
            "BTC": [(datetime(2026, 10, 1, 7, 31, tzinfo=UTC), 60000.5)],
        }

    def test_price_series_without_symbols_reads_nothing(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        assert repo.get_price_series([], since=datetime(2026, 9, 1, tzinfo=UTC)) == {}
        mock_client.table.assert_not_called()


class TestCorporateActions:
    """DAT-23: podejrzenie splitu w `corporate_actions` (migracja 062) — raz na symbol,
    rodzaj i dzień UTC, więc powtórka cyklu nie dubluje zdarzenia."""

    @staticmethod
    def _suspicion() -> Any:
        from src.domain.corporate_actions import SplitSuspicion

        return SplitSuspicion(
            symbol="NVDA",
            ratio=Decimal("0.1"),
            prev_close=Decimal("120.6"),
            reference_price=Decimal("1200"),
            detected_at=datetime(2026, 10, 27, 5, 30, tzinfo=UTC),
        )

    def test_split_suspicion_is_upserted_once_per_symbol_kind_and_day(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        repo.save_corporate_action(self._suspicion())

        mock_client.table.assert_called_with("corporate_actions")
        upsert = mock_client.table.return_value.upsert
        assert upsert.call_args.args[0] == {
            "symbol": "NVDA",
            "kind": "split_suspected",
            "ratio": "0.1",
            "prev_close": "120.6",
            "reference_price": "1200",
            "detected_at": "2026-10-27T05:30:00+00:00",
            "detected_on": "2026-10-27",
        }
        assert upsert.call_args.kwargs == {
            "on_conflict": "symbol,kind,detected_on",
            "ignore_duplicates": True,
        }

    def test_write_error_is_translated(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from postgrest.exceptions import APIError

        from src.application.errors import NotFound

        mock_client.table.return_value.upsert.return_value.execute.side_effect = APIError(
            {"code": "42P01", "message": 'relation "corporate_actions" does not exist'}
        )

        with pytest.raises(NotFound):
            repo.save_corporate_action(self._suspicion())



class TestPointInTimeRagSearch:
    """ML-41 (reszta): za `rag_point_in_time_enabled` RAG woła
    `match_news_embeddings_pit` (migracja 076) — tylko analogi rozliczone przed
    `before_ts`, bez tego samego symbolu w oknie horyzontu; bez flagi RPC jak dziś."""

    _BEFORE = datetime(2026, 10, 27, 5, 30, tzinfo=UTC)

    @staticmethod
    def _repo(enabled: bool) -> SupabaseRepository:
        return SupabaseRepository(
            url="https://test.supabase.co", key="anon-key", rag_point_in_time_enabled=enabled
        )

    def test_flag_calls_the_point_in_time_function_with_cutoff_and_symbol(
        self, mock_client: MagicMock
    ) -> None:
        mock_client.rpc.return_value.execute.return_value.data = [{"symbol": "MSFT"}]

        rows = self._repo(True).find_similar_predictions(
            [0.1, 0.2], limit=12, regime="RISK_OFF", symbol="AAPL", before=self._BEFORE
        )

        assert rows == [{"symbol": "MSFT"}]
        mock_client.rpc.assert_called_once_with(
            "match_news_embeddings_pit",
            {
                "query_embedding": [0.1, 0.2],
                "match_count": 12,
                "filter_regime": "RISK_OFF",
                "before_ts": "2026-10-27T05:30:00+00:00",
                "exclude_symbol": "AAPL",
                "exclude_window_hours": 72,
            },
        )

    def test_without_the_flag_the_cutoff_and_symbol_change_nothing(
        self, mock_client: MagicMock
    ) -> None:
        mock_client.rpc.return_value.execute.return_value.data = []

        self._repo(False).find_similar_predictions(
            [0.1], limit=3, symbol="AAPL", before=self._BEFORE
        )

        mock_client.rpc.assert_called_once_with(
            "match_news_embeddings", {"query_embedding": [0.1], "match_count": 3}
        )

    def test_missing_point_in_time_function_points_at_migration_076(
        self, mock_client: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        from postgrest.exceptions import APIError

        mock_client.rpc.return_value.execute.side_effect = APIError(
            {"code": "PGRST202", "message": "Could not find the function"}
        )

        rows = self._repo(True).find_similar_predictions([0.1], limit=3, symbol="AAPL")

        assert rows == []
        assert "076" in caplog.text


class TestExactRagSearch:
    """ML-23: za flagą RAG woła `match_news_embeddings_exact` (migracja 063, dokładne
    KNN) z tymi samymi parametrami; bez flagi zostaje przybliżone ivfflat."""

    @pytest.mark.parametrize(
        ("enabled", "function"),
        [(False, "match_news_embeddings"), (True, "match_news_embeddings_exact")],
    )
    def test_flag_picks_the_rag_function(
        self, mock_client: MagicMock, enabled: bool, function: str
    ) -> None:
        repo = SupabaseRepository(
            url="https://test.supabase.co", key="anon-key", rag_exact_search_enabled=enabled
        )
        mock_client.rpc.return_value.execute.return_value.data = [{"symbol": "AAPL"}]

        rows = repo.find_similar_predictions([0.1, 0.2], limit=4, regime="RISK_OFF")

        assert rows == [{"symbol": "AAPL"}]
        mock_client.rpc.assert_called_once_with(
            function,
            {"query_embedding": [0.1, 0.2], "match_count": 4, "filter_regime": "RISK_OFF"},
        )

    def test_missing_exact_function_points_at_migration_063(
        self, mock_client: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        from postgrest.exceptions import APIError

        repo = SupabaseRepository(
            url="https://test.supabase.co", key="anon-key", rag_exact_search_enabled=True
        )
        mock_client.rpc.return_value.execute.side_effect = APIError(
            {"code": "PGRST202", "message": "Could not find the function"}
        )

        assert repo.find_similar_predictions([0.1], limit=3) == []
        assert "063" in caplog.text


def test_calibration_read_failure_crosses_the_port_as_a_port_error(
    repo: SupabaseRepository, mock_client: MagicMock
) -> None:
    # ML-28 / ARCH-33: ranking łapie PortError przy odczycie 60 dni rozliczeń.
    from postgrest.exceptions import APIError

    from src.application.errors import PortError

    _fail_every_request(
        mock_client, APIError({"code": "57014", "message": "canceling statement due to timeout"})
    )

    with pytest.raises(PortError):
        repo.get_resolved_for_calibration(60)


def _quota_alert() -> Any:
    from src.domain.quota import QuotaAlert, QuotaSeverity

    return QuotaAlert(
        source="Finnhub",
        severity=QuotaSeverity.WARNING,
        message="limit",
        action="czekaj",
        occurred_at=datetime(2026, 10, 25, tzinfo=UTC),
    )


class TestShadowPredictions:
    """ML-33 (migracja 075): prognozy cienia championa i challengerów jednej predykcji
    oraz ich odczyt z rozliczonych predykcji do werdyktu promocji w Slow Loopie."""

    CHAMPION = "a" * 64
    DIRECTION = "c" * 64

    def _forecasts(self) -> list[Any]:
        from src.domain.promotion import ShadowForecast

        return [
            ShadowForecast("champion", self.CHAMPION, predicted_return=0.012),
            ShadowForecast("direction", self.DIRECTION, proba=0.64),
        ]

    def test_saves_one_row_per_model_upserted_by_prediction_and_model(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "upsert"], [{"id": 1}])

        repo.save_shadow_predictions("pred-1", "AAPL", self._forecasts())

        mock_client.table.assert_called_with("shadow_predictions")
        upsert = mock_client.table.return_value.upsert
        upsert.assert_called_once()
        assert upsert.call_args.args[0] == [
            {
                "prediction_id": "pred-1",
                "symbol": "AAPL",
                "model_name": "champion",
                "model_version": self.CHAMPION,
                "predicted_return": 0.012,
                "proba": None,
            },
            {
                "prediction_id": "pred-1",
                "symbol": "AAPL",
                "model_name": "direction",
                "model_version": self.DIRECTION,
                "predicted_return": None,
                "proba": 0.64,
            },
        ]
        assert upsert.call_args.kwargs["on_conflict"] == "prediction_id,model_name"

    def test_no_forecasts_send_no_request(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        repo.save_shadow_predictions("pred-1", "AAPL", [])

        mock_client.table.assert_not_called()

    def test_missing_table_is_a_not_found_port_error(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from postgrest.exceptions import APIError

        from src.application.errors import NotFound

        mock_client.table.return_value.upsert.return_value.execute.side_effect = APIError(
            {"code": "PGRST205", "message": "Could not find the table"}
        )

        with pytest.raises(NotFound):
            repo.save_shadow_predictions("pred-1", "AAPL", self._forecasts())

    def _read(
        self, repo: SupabaseRepository, mock_client: MagicMock, pages: list[list[dict[str, Any]]]
    ) -> tuple[list[Any], MagicMock]:
        chain = mock_client.table.return_value
        order_builder = (
            chain.select.return_value.gte.return_value.not_.is_.return_value
            .order.return_value.order.return_value
        )
        _set_paginated_response(order_builder, pages)
        rows = repo.get_settled_shadow_predictions(datetime(2026, 7, 27, tzinfo=UTC))
        return rows, chain

    def test_reads_settled_rows_with_the_realized_return(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.promotion import SettledShadowForecast, ShadowForecast

        settled = {"price_at_prediction": "200.0", "actual_price_after_12h": 202.0}
        rows, chain = self._read(
            repo,
            mock_client,
            [
                [
                    {
                        "prediction_id": "pred-1",
                        "model_name": "champion",
                        "model_version": self.CHAMPION,
                        "predicted_return": 0.012,
                        "proba": None,
                        "prediction_logs": settled,
                    },
                    {
                        "prediction_id": "pred-1",
                        "model_name": "direction",
                        "model_version": self.DIRECTION,
                        "predicted_return": None,
                        "proba": "0.64",
                        "prediction_logs": settled,
                    },
                ]
            ],
        )

        assert rows == [
            SettledShadowForecast(
                "pred-1",
                ShadowForecast("champion", self.CHAMPION, predicted_return=0.012),
                cast(float, pytest.approx(0.01)),
            ),
            SettledShadowForecast(
                "pred-1",
                ShadowForecast("direction", self.DIRECTION, proba=0.64),
                cast(float, pytest.approx(0.01)),
            ),
        ]
        mock_client.table.assert_called_with("shadow_predictions")
        select = chain.select.call_args.args[0]
        assert "prediction_logs!inner(price_at_prediction, actual_price_after_12h)" in select
        assert chain.select.return_value.gte.call_args.args == (
            "created_at",
            "2026-07-27T00:00:00+00:00",
        )
        assert chain.select.return_value.gte.return_value.not_.is_.call_args.args == (
            "prediction_logs.actual_price_after_12h",
            "null",
        )

    def test_rows_that_cannot_be_scored_are_skipped(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        def row(prediction: object, **overrides: object) -> dict[str, Any]:
            return {
                "prediction_id": "pred-x",
                "model_name": "direction",
                "model_version": self.DIRECTION,
                "predicted_return": None,
                "proba": 0.6,
                "prediction_logs": prediction,
            } | overrides

        settled = {"price_at_prediction": 100.0, "actual_price_after_12h": 101.0}
        rows, _ = self._read(
            repo,
            mock_client,
            [
                [
                    row({"price_at_prediction": 0, "actual_price_after_12h": 1.0}),
                    row({"price_at_prediction": 100.0, "actual_price_after_12h": None}),
                    row(None),
                    row(settled, proba=None),
                    row(settled, proba=2.0),
                    row(settled, model_name=None),
                ]
            ],
        )

        assert rows == []


# ---------------------------------------------------------------------------
# WIA-59: append-only paper ledger (migracja 073)
# ---------------------------------------------------------------------------


class TestPaperLedger:
    """Repozytorium dopisuje pozycję za bieżącą głową łańcucha (hash liczy domena),
    ponawia przy wyścigu o głowę (23505 / 40001), zamyka pozycję raz i oddaje
    ledger jako wiersze domeny — z hashami, które da się zweryfikować."""

    MONDAY = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)

    @classmethod
    def _entry(
        cls, prediction_id: str = "11111111-1111-1111-1111-111111111111"
    ) -> PaperTradeEntry:
        from src.domain.paper_ledger import TradeSide

        return PaperTradeEntry(
            prediction_id=prediction_id,
            symbol="NVDA",
            side=TradeSide.LONG,
            entry_price=Decimal("100"),
            entry_ts=cls.MONDAY,
            cost_bps=Decimal("10"),
        )

    @staticmethod
    def _head(chain: MagicMock, rows: list[dict[str, Any]]) -> None:
        response = MagicMock()
        response.data = rows
        chain.select.return_value.order.return_value.limit.return_value.execute.return_value = (
            response
        )

    def test_first_trade_chains_from_the_genesis_hash(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.paper_ledger import GENESIS_HASH, chain_entry

        chain = mock_client.table.return_value
        self._head(chain, [])
        chain.upsert.return_value.execute.return_value.data = [{"seq": 1}]

        row = repo.append_paper_trade(self._entry())

        assert row == chain_entry(self._entry(), GENESIS_HASH)
        record = chain.upsert.call_args.args[0]
        assert record["prev_hash"] == GENESIS_HASH
        assert record["row_hash"] == row.row_hash
        assert record["entry_price"] == "100.0000"
        assert record["side"] == "long"
        assert "ledger_version" not in record
        assert chain.upsert.call_args.kwargs == {
            "on_conflict": "prediction_id",
            "ignore_duplicates": True,
        }

    def test_trade_chains_after_the_current_head(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        chain = mock_client.table.return_value
        self._head(chain, [{"seq": 7, "row_hash": "a" * 64}])
        chain.upsert.return_value.execute.return_value.data = [{"seq": 8}]

        row = repo.append_paper_trade(self._entry())

        assert row is not None
        assert row.prev_hash == "a" * 64

    def test_lost_race_for_the_head_retries_from_the_new_head(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from postgrest.exceptions import APIError

        chain = mock_client.table.return_value
        heads = [MagicMock(data=[{"seq": 7, "row_hash": "a" * 64}]),
                 MagicMock(data=[{"seq": 8, "row_hash": "b" * 64}])]
        chain.select.return_value.order.return_value.limit.return_value.execute.side_effect = heads
        chain.upsert.return_value.execute.side_effect = [
            APIError({"code": "40001", "message": "not the chain head"}),
            MagicMock(data=[{"seq": 9}]),
        ]

        row = repo.append_paper_trade(self._entry())

        assert row is not None
        assert row.prev_hash == "b" * 64

    def test_trade_already_in_the_ledger_is_kept(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # Re-run w tej samej godzinie: pierwsza decyzja zostaje (append-only).
        chain = mock_client.table.return_value
        self._head(chain, [{"seq": 7, "row_hash": "a" * 64}])
        chain.upsert.return_value.execute.return_value.data = []

        assert repo.append_paper_trade(self._entry()) is None

    def test_contention_beyond_the_retries_is_a_port_error(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from postgrest.exceptions import APIError

        from src.application.errors import PortError

        chain = mock_client.table.return_value
        self._head(chain, [{"seq": 7, "row_hash": "a" * 64}])
        chain.upsert.return_value.execute.side_effect = APIError(
            {"code": "23505", "message": "duplicate key value violates unique constraint"}
        )

        with pytest.raises(PortError):
            repo.append_paper_trade(self._entry())
        assert chain.upsert.return_value.execute.call_count == 3

    def test_close_updates_only_an_open_trade(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.paper_ledger import ExitReason, PaperTradeClose

        _set_chain_response(mock_client, ["table", "update", "eq", "is_"], [])
        close = PaperTradeClose(
            exit_price=Decimal("106.0000"),
            exit_ts=datetime(2026, 10, 20, 5, 30, tzinfo=UTC),
            pnl=Decimal("0.06"),
            exit_reason=ExitReason.SETTLED,
        )

        repo.close_paper_trade("pred-1", close)

        update = mock_client.table.return_value.update
        assert update.call_args.args[0] == {
            "exit_price": "106.0000",
            "exit_ts": "2026-10-20T05:30:00.000000+00:00",
            "pnl": "0.06",
            "exit_reason": "settled",
        }
        update.return_value.eq.assert_called_once_with("prediction_id", "pred-1")
        update.return_value.eq.return_value.is_.assert_called_once_with("exit_price", "null")

    def test_ledger_rows_round_trip_into_a_verifiable_chain(
        self, repo: SupabaseRepository, mock_client: MagicMock, mocker: Any
    ) -> None:
        from src.domain.paper_ledger import (
            GENESIS_HASH,
            ExitReason,
            chain_entry,
            first_broken_row,
        )

        first = chain_entry(self._entry("p-1"), GENESIS_HASH)
        second = chain_entry(self._entry("p-2"), first.row_hash)
        rows = [
            {
                "seq": seq, "prediction_id": row.entry.prediction_id, "symbol": "NVDA",
                "side": "long", "entry_price": 100.0, "entry_ts": "2026-10-19T05:30:00+00:00",
                "weight": 1.0, "cost_bps": 10.0, "prev_hash": row.prev_hash,
                "row_hash": row.row_hash, **close,
            }
            for seq, row, close in (
                (1, first, {"exit_price": 106.0, "exit_ts": "2026-10-20T05:30:00+00:00",
                            "pnl": 0.06, "exit_reason": "settled"}),
                (2, second, {"exit_price": None, "exit_ts": None, "pnl": None,
                             "exit_reason": None}),
            )
        ]
        mocker.patch(
            "src.infrastructure.adapters.supabase_repo._paginate",
            side_effect=lambda _query: [dict(row) for row in rows],
        )

        trades = repo.get_paper_trades(30)

        assert [trade.seq for trade in trades] == [1, 2]
        assert first_broken_row([trade.row for trade in trades]) is None
        assert trades[0].close is not None
        assert trades[0].close.exit_reason is ExitReason.SETTLED
        assert trades[0].close.exit_price == Decimal("106.0")
        assert trades[1].close is None

    def test_head_is_the_last_row_of_the_chain(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.domain.paper_ledger import LedgerHead

        chain = mock_client.table.return_value
        self._head(chain, [{"seq": 7, "row_hash": "a" * 64}])

        assert repo.get_paper_ledger_head() == LedgerHead(seq=7, row_hash="a" * 64)
        chain.select.return_value.order.assert_called_with("seq", desc=True)

    def test_empty_ledger_has_no_head(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        self._head(mock_client.table.return_value, [])

        assert repo.get_paper_ledger_head() is None


# ---------------------------------------------------------------------------
# ML-27: poziom unieważnienia tezy — odczyt i stempel alarmu watcha (migracja 074)
# ---------------------------------------------------------------------------


class TestOpenInvalidations:
    def test_reads_open_predictions_with_a_level_and_no_alert(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        from src.application.ports import OpenInvalidation

        chain = mock_client.table.return_value.select.return_value
        filtered = chain.not_.is_.return_value.is_.return_value.is_.return_value
        response = MagicMock()
        response.data = [
            {
                "id": "pred-1",
                "symbol": "NVDA",
                "predicted_trend": "BULLISH",
                "price_at_prediction": 120.0,
                "invalidation_price": 118.0,
                "timestamp": "2026-10-22T05:30:00+00:00",
            }
        ]
        filtered.gte.return_value.order.return_value.execute.return_value = response
        since = datetime(2026, 10, 18, 15, 0, tzinfo=UTC)

        rows = repo.get_open_invalidations(since)

        assert rows == [
            OpenInvalidation(
                prediction_id="pred-1",
                symbol="NVDA",
                predicted_trend="BULLISH",
                price_at_prediction=Decimal("120.0"),
                invalidation_price=Decimal("118.0"),
                predicted_at=datetime(2026, 10, 22, 5, 30, tzinfo=UTC),
            )
        ]
        chain.not_.is_.assert_called_once_with("invalidation_price", "null")
        chain.not_.is_.return_value.is_.assert_called_once_with("actual_price_after_12h", "null")
        chain.not_.is_.return_value.is_.return_value.is_.assert_called_once_with(
            "invalidation_alerted_at", "null"
        )
        filtered.gte.assert_called_once_with("timestamp", since.isoformat())

    def test_alert_stamp_is_written_once(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "update", "eq", "is_"], [])
        at = datetime(2026, 10, 22, 15, 0, tzinfo=UTC)

        repo.mark_invalidation_alerted("pred-1", at)

        update = mock_client.table.return_value.update
        assert update.call_args.args[0] == {"invalidation_alerted_at": at.isoformat()}
        update.return_value.eq.assert_called_once_with("id", "pred-1")
        update.return_value.eq.return_value.is_.assert_called_once_with(
            "invalidation_alerted_at", "null"
        )
