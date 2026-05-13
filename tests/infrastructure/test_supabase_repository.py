from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.application.ports import RepositoryPort
from src.domain.prediction import TrendDirection
from src.domain.value_objects import Money
from src.infrastructure.adapters.supabase_repo import SupabaseRepository

# ---------------------------------------------------------------------------
# Helpers — Supabase client uses a builder/chain API:
#   client.table("x").select(...).eq(...).order(...).limit(...).execute()
# ---------------------------------------------------------------------------


def _set_chain_response(client: MagicMock, methods: list[str], data: list[dict]) -> MagicMock:
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
def mock_client(mocker) -> MagicMock:
    create = mocker.patch("src.infrastructure.adapters.supabase_repo.create_client")
    client = MagicMock()
    create.return_value = client
    return client


@pytest.fixture
def repo(mock_client: MagicMock) -> SupabaseRepository:
    return SupabaseRepository(url="https://test.supabase.co", key="anon-key")


# ---------------------------------------------------------------------------
# get_last_price
# ---------------------------------------------------------------------------


class TestGetLastPrice:
    def test_returns_money_for_existing_symbol(self, repo, mock_client):
        _set_chain_response(
            mock_client,
            ["table", "select", "eq", "order", "limit"],
            [{"price_at_prediction": 192.5}],
        )

        price = repo.get_last_price("AAPL")

        assert isinstance(price, Money)
        assert price.amount == Decimal("192.5")

    def test_returns_none_when_no_history(self, repo, mock_client):
        _set_chain_response(
            mock_client, ["table", "select", "eq", "order", "limit"], data=[]
        )
        assert repo.get_last_price("UNKNOWN") is None

    def test_queries_prediction_logs_table_with_symbol_filter(self, repo, mock_client):
        _set_chain_response(
            mock_client, ["table", "select", "eq", "order", "limit"], data=[]
        )

        repo.get_last_price("VOO")

        mock_client.table.assert_called_with("prediction_logs")
        mock_client.table.return_value.select.return_value.eq.assert_called_with(
            "symbol", "VOO"
        )


# ---------------------------------------------------------------------------
# save_prediction
# ---------------------------------------------------------------------------


class TestSavePrediction:
    def test_returns_inserted_uuid(self, repo, mock_client):
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

    def test_serializes_decimal_values_for_json(self, repo, mock_client):
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
    def test_maps_row_to_prediction_domain_object(self, repo, mock_client):
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

    def test_returns_none_when_no_unverified_prediction(self, repo, mock_client):
        _set_chain_response(
            mock_client, ["table", "select", "eq", "is_", "order", "limit"], []
        )
        assert repo.get_unverified_prediction("AAPL") is None

    def test_filters_by_null_actual_price(self, repo, mock_client):
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
    def test_updates_record_with_actual_price_and_insight(self, repo, mock_client):
        _set_chain_response(mock_client, ["table", "update", "eq"], [{"id": "uuid-1"}])

        repo.update_prediction_accuracy(
            prediction_id="uuid-1",
            actual_price=Decimal("99.0"),
            insight="Zignorowałem makro.",
        )

        payload = mock_client.table.return_value.update.call_args.args[0]
        # Decimal musi być serializowany
        assert str(payload["actual_price_after_12h"]) == "99.0"
        assert payload["correction_insights"] == "Zignorowałem makro."
        mock_client.table.return_value.update.return_value.eq.assert_called_with(
            "id", "uuid-1"
        )


# ---------------------------------------------------------------------------
# get_feature_store_data
# ---------------------------------------------------------------------------


class TestGetFeatureStoreData:
    def test_returns_rows_from_materialized_view(self, repo, mock_client):
        rows = [
            {"price_current": 100.0, "sentiment_score": 75.0, "llm_trend_signal": 1},
            {"price_current": 102.0, "sentiment_score": 80.0, "llm_trend_signal": 1},
        ]
        _set_chain_response(
            mock_client, ["table", "select", "eq", "order"], rows
        )

        result = repo.get_feature_store_data("AAPL")

        assert result == rows
        mock_client.table.assert_any_call("ml_feature_store")

    def test_returns_empty_list_when_no_data(self, repo, mock_client):
        _set_chain_response(mock_client, ["table", "select", "eq", "order"], [])
        assert repo.get_feature_store_data("UNKNOWN") == []


# ---------------------------------------------------------------------------
# Port conformance
# ---------------------------------------------------------------------------


class TestAdapterImplementsPort:
    def test_is_repository_port(self):
        assert issubclass(SupabaseRepository, RepositoryPort)
