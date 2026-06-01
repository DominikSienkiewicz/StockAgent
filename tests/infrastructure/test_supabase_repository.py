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
    def test_inserts_symbol_and_price_to_snapshots_table(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(mock_client, ["table", "insert"], [{"id": "snap-1"}])

        repo.save_price_snapshot("AAPL", Money(Decimal("298.87")))

        mock_client.table.assert_called_with("price_snapshots")
        inserted = mock_client.table.return_value.insert.call_args.args[0]
        assert inserted["symbol"] == "AAPL"
        # Decimal serializowany do str (JSON-safe)
        assert str(inserted["price"]) == "298.87"


# ---------------------------------------------------------------------------
# save_prediction
# ---------------------------------------------------------------------------


class TestSavePrediction:
    def test_returns_inserted_uuid(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        _set_chain_response(
            mock_client,
            ["table", "insert"],
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

    def test_raises_when_insert_returns_empty_rows(
        self, repo: SupabaseRepository, mock_client: MagicMock
    ) -> None:
        # Supabase czasem zwraca puste response.data (np. przy konflikcie RLS).
        # Bez tego guardu prediction_id propagowałby się jako None / KeyError.
        _set_chain_response(mock_client, ["table", "insert"], data=[])

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
        _set_chain_response(mock_client, ["table", "insert"], [{"id": "uuid-1"}])

        repo.save_prediction(
            {
                "symbol": "AAPL",
                "price_at_prediction": Decimal("100.55"),
                "predicted_target_price": Decimal("105.0"),
            }
        )

        inserted = mock_client.table.return_value.insert.call_args.args[0]
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
        assert payload["accuracy_score"] == 0.87
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
        assert payload["trailing_pe"] == 25.4
        assert payload["forward_pe"] == 22.1
        assert payload["peg_ratio"] == 1.5
        assert payload["eps_growth_yoy"] == 0.12
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
        assert result.trailing_pe == 25.4
        assert result.forward_pe == 22.1
        assert result.peg_ratio == 1.5
        assert result.eps_growth_yoy == 0.12
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
