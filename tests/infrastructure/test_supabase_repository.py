from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.application.ports import RepositoryPort
from src.domain.prediction import TrendDirection
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

        with pytest.raises(RuntimeError, match="AAPL"):
            repo.save_prediction(
                {
                    "symbol": "AAPL",
                    "price_at_prediction": Decimal("100.0"),
                    "predicted_trend": "BULLISH",
                    "predicted_target_price": Decimal("105.0"),
                }
            )

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


# ---------------------------------------------------------------------------
# get_unverified_prediction
# ---------------------------------------------------------------------------


class TestGetUnverifiedPrediction:
    def test_maps_row_to_prediction_domain_object(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "is_", "order", "limit"],
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

    def test_returns_none_when_no_unverified_prediction(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client, ["table", "select", "eq", "is_", "order", "limit"], []
        )
        assert repo.get_unverified_prediction("AAPL") is None

    def test_filters_by_null_actual_price(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client, ["table", "select", "eq", "is_", "order", "limit"], []
        )

        repo.get_unverified_prediction("AAPL")

        # is_("actual_price_after_12h", "null") — postgresT składnia dla IS NULL
        is_call = mock_client.table.return_value.select.return_value.eq.return_value.is_
        is_call.assert_called_with("actual_price_after_12h", "null")

    def test_orders_oldest_first_to_drain_backlog(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        """#19: self-reflection musi opróżniać backlog — bierzemy NAJSTARSZĄ
        nierozliczoną predykcję (timestamp ASC), nie najnowszą. Przy DESC
        starsze nierozliczone predykcje nigdy nie stałyby się znów "najnowsze"
        i utknęłyby na zawsze (brak actual_price_after_12h → nie wchodzą do
        ml_feature_store, nie liczą się do hit-rate)."""
        _set_chain_response(
            mock_client, ["table", "select", "eq", "is_", "order", "limit"], []
        )

        repo.get_unverified_prediction("AAPL")

        order_call = (
            mock_client.table.return_value.select.return_value.eq.return_value
            .is_.return_value.order
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
            ["table", "select", "eq", "is_", "lte", "order", "limit"],
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
            mock_client, ["table", "select", "eq", "is_", "order", "limit"], []
        )

        repo.get_unverified_prediction("AAPL")

        lte = (
            mock_client.table.return_value.select.return_value.eq.return_value
            .is_.return_value.lte
        )
        lte.assert_not_called()


# ---------------------------------------------------------------------------
# update_prediction_accuracy
# ---------------------------------------------------------------------------


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
            mock_client, ["table", "select", "eq", "order"], rows
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


class TestGetCouncilVoteHistory:
    def test_groups_rows_into_investor_histories(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "select", "in_", "gte", "order"],
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
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "gte", "order"],
            [
                {"timestamp": "2026-06-01T00:00:00+00:00", "price": 100.0},
                {"timestamp": "2026-06-02T00:00:00+00:00", "price": 110.0},
            ],
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
                key_factors=["moat"],
            ),
            InvestorOpinion(
                investor_name="Michael Burry",
                recommendation="SELL",
                confidence=0.7,
                reasoning="Zawyżone P/E.",
                key_factors=["bubble"],
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
