from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application.agent_graph import create_agent_graph
from src.application.ports import (
    EmbeddingPort,
    LLMPort,
    MarketDataPort,
    MLPredictionPort,
    NewsPort,
    RepositoryPort,
    SentimentPort,
)
from src.domain.prediction import Prediction, TrendDirection
from src.domain.value_objects import Money, Threshold

EXPECTED_ML_FEATURES = [
    "price_delta",
    "av_sentiment_score",
    "av_relevance_avg",
    "news_volume_24h",
    "high_relevance_count",
    "llm_trend_signal",
    "av_llm_agreement",
]


@pytest.fixture
def market_port() -> Mock:
    return Mock(spec=MarketDataPort)


@pytest.fixture
def sentiment_port() -> Mock:
    return Mock(spec=SentimentPort)


@pytest.fixture
def news_port() -> Mock:
    return Mock(spec=NewsPort)


@pytest.fixture
def repository_port() -> Mock:
    repo = Mock(spec=RepositoryPort)
    # Domyślnie brak historii — reflect_node (wykonywany w KAŻDYM cyklu)
    # nie ma czego oceniać. Testy full-analysis nadpisują to przez helper.
    repo.get_unverified_prediction.return_value = None
    return repo


@pytest.fixture
def ml_port() -> Mock:
    return Mock(spec=MLPredictionPort)


@pytest.fixture
def llm_port() -> Mock:
    return Mock(spec=LLMPort)


@pytest.fixture
def workflow(
    market_port: Mock,
    sentiment_port: Mock,
    news_port: Mock,
    repository_port: Mock,
    ml_port: Mock,
    llm_port: Mock,
):
    return create_agent_graph(
        market_port=market_port,
        sentiment_port=sentiment_port,
        news_port=news_port,
        repository_port=repository_port,
        ml_port=ml_port,
        llm_port=llm_port,
        threshold=Threshold(Decimal("0.02")),
    )


def _initial_state(previous_price: str) -> dict:
    return {
        "symbol": "AAPL",
        "previous_price": Decimal(previous_price),
    }


def _setup_full_analysis_mocks(
    sentiment_port: Mock,
    news_port: Mock,
    repository_port: Mock,
    ml_port: Mock,
    llm_port: Mock,
    has_prior_prediction: bool = False,
) -> None:
    """Boilerplate — komplet mocków dla pełnej ścieżki analizy."""
    sentiment_port.get_social_score.return_value = {
        "av_sentiment_score": -0.42,
        "av_relevance_avg": 0.72,
        "news_volume_24h": 4,
        "high_relevance_count": 2,
        "av_sentiment_label": "Bearish",
    }
    news_port.get_news_context.return_value = [
        {"title": "Reuters: Fed keeps rates", "source": "Reuters"}
    ]
    if has_prior_prediction:
        repository_port.get_unverified_prediction.return_value = Prediction(
            id="prev-uuid-456",
            symbol="AAPL",
            predicted_trend=TrendDirection.BULLISH,
            price_at_prediction=Decimal("100.0"),
            predicted_target_price=Decimal("105.0"),
        )
    else:
        repository_port.get_unverified_prediction.return_value = None
    repository_port.save_prediction.return_value = "pred-uuid-123"
    llm_port.analyze.return_value = {
        "trend_direction": "BEARISH",
        "confidence_score": 0.85,
        "av_agreement": 0.25,
        "target_price_12h": 88.0,
        "reasoning": "Strong negative sentiment + macro headwinds.",
    }
    llm_port.analyze_mistake.return_value = "Zignorowałem szerszy kontekst makro."
    ml_port.predict.return_value = Money(Decimal("89.5"))
    ml_port.is_trained = True   # domyślnie: model wytrenowany


# ---------------------------------------------------------------------------
# Price snapshot — zapisywany w KAŻDYM cyklu (cold-start deadlock fix)
# ---------------------------------------------------------------------------


class TestPriceSnapshot:
    def test_snapshot_saved_on_ignored_cycle(
        self, workflow, market_port, repository_port
    ):
        # Nawet gdy cykl kończy się "ignored", bieżąca cena MUSI zostać
        # zapisana — inaczej następny cykl nie ma punktu odniesienia.
        market_port.get_current_price.return_value = Money(Decimal("99.0"))

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert final["status"] == "ignored"
        repository_port.save_price_snapshot.assert_called_once()
        args = repository_port.save_price_snapshot.call_args.args
        assert args[0] == "AAPL"
        assert args[1].amount == Decimal("99.0")

    def test_snapshot_saved_on_full_analysis_cycle(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )

        workflow.compile().invoke(_initial_state("100.0"))

        repository_port.save_price_snapshot.assert_called_once_with(
            "AAPL", Money(Decimal("90.0"))
        )


# ---------------------------------------------------------------------------
# Path 1 — high volatility, no prior prediction (cold start)
# ---------------------------------------------------------------------------


class TestFullAnalysisColdStart:
    def test_runs_all_nodes_when_volatility_high_and_no_prior_prediction(
        self,
        workflow,
        market_port: Mock,
        sentiment_port: Mock,
        news_port: Mock,
        repository_port: Mock,
        ml_port: Mock,
        llm_port: Mock,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))  # -10%
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
            has_prior_prediction=False,
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert final["status"] == "saved"
        assert final["delta"] == Decimal("-0.10")
        assert final["prediction_id"] == "pred-uuid-123"
        assert final["llm_analysis"]["trend_direction"] == "BEARISH"
        assert final["ml_target_price"] == Decimal("89.5")

        # Kolejność wywołań
        market_port.get_current_price.assert_called_once_with("AAPL")
        repository_port.get_unverified_prediction.assert_called_once_with("AAPL")
        sentiment_port.get_social_score.assert_called_once_with("AAPL")
        news_port.get_news_context.assert_called_once_with("AAPL")
        llm_port.analyze.assert_called_once()
        ml_port.predict.assert_called_once()
        repository_port.save_prediction.assert_called_once()
        # Brak poprzedniej predykcji — nie diagnozujemy błędu
        llm_port.analyze_mistake.assert_not_called()
        repository_port.update_prediction_accuracy.assert_not_called()


# ---------------------------------------------------------------------------
# Path 2 — high volatility, prior prediction was WRONG → mistake diagnosis
# ---------------------------------------------------------------------------


class TestSelfReflectionOnWrongPrediction:
    def test_diagnoses_mistake_when_prior_prediction_was_wrong(
        self,
        workflow,
        market_port: Mock,
        sentiment_port: Mock,
        news_port: Mock,
        repository_port: Mock,
        ml_port: Mock,
        llm_port: Mock,
    ):
        # Poprzednio: BULLISH @ 100, target 105. Aktualnie: 90 (spadek) → zły kierunek.
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
            has_prior_prediction=True,
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        # LLM diagnozuje błąd
        llm_port.analyze_mistake.assert_called_once()
        # Repository dostaje correction insight
        repository_port.update_prediction_accuracy.assert_called_once()
        args = repository_port.update_prediction_accuracy.call_args
        assert args.kwargs.get("insight") == "Zignorowałem szerszy kontekst makro." or \
               args.args[2] == "Zignorowałem szerszy kontekst makro."

        # reflection_context trafia do stanu i zostaje użyty w predykcji
        assert "Zignorowałem szerszy kontekst makro." in final["reflection_context"]
        assert final["status"] == "saved"


# ---------------------------------------------------------------------------
# Path 3 — high volatility, prior prediction CORRECT → reinforcement
# ---------------------------------------------------------------------------


class TestSelfReflectionOnCorrectPrediction:
    def test_reinforces_when_prior_prediction_was_correct(
        self,
        workflow,
        market_port: Mock,
        sentiment_port: Mock,
        news_port: Mock,
        repository_port: Mock,
        ml_port: Mock,
        llm_port: Mock,
    ):
        # Poprzednio: BULLISH @ 100. Aktualnie: 110 (wzrost) → trafiona, kierunek OK.
        market_port.get_current_price.return_value = Money(Decimal("110.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
            has_prior_prediction=True,
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        # Nie wzywamy LLM-a do diagnozy błędu
        llm_port.analyze_mistake.assert_not_called()
        # Repository dostaje pozytywny feedback
        repository_port.update_prediction_accuracy.assert_called_once()
        assert "trafn" in final["reflection_context"].lower()
        assert final["status"] == "saved"


# ---------------------------------------------------------------------------
# Edge case — sentiment z explicit None values (AV brak danych dla tickera)
# ---------------------------------------------------------------------------


class TestSentimentWithNullValues:
    """AV NEWS_SENTIMENT czasem zwraca rekordy z null'ami (np. ticker bez
    sklasyfikowanego sentymentu). _float_or_default musi je przepuścić bez
    crashu, a save_node nie może wywalić się na None w polach JSON-safe."""

    def test_full_analysis_runs_when_sentiment_fields_are_null(
        self,
        workflow,
        market_port: Mock,
        sentiment_port: Mock,
        news_port: Mock,
        repository_port: Mock,
        ml_port: Mock,
        llm_port: Mock,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        # Wszystkie pola sentymentu jako None — najgorszy realny case z AV.
        sentiment_port.get_social_score.return_value = {
            "av_sentiment_score": None,
            "av_relevance_avg": None,
            "news_volume_24h": None,
            "high_relevance_count": None,
            "av_sentiment_label": None,
        }
        news_port.get_news_context.return_value = []
        repository_port.get_unverified_prediction.return_value = None
        repository_port.save_prediction.return_value = "pred-uuid"
        # LLM też może zwrócić av_agreement=None (gdy brak danych do oceny).
        llm_port.analyze.return_value = {
            "trend_direction": "SIDEWAYS",
            "confidence_score": 0.5,
            "av_agreement": None,
            "target_price_12h": 90.0,
            "reasoning": "Brak sygnału.",
        }
        ml_port.predict.return_value = Money(Decimal("90.0"))
        ml_port.is_trained = True

        final = workflow.compile().invoke(_initial_state("100.0"))

        # Cykl kończy się sukcesem — None'y nie wywalają grafu.
        assert final["status"] == "saved"
        # Predict node wywołał ML z domyślnymi 0.0 dla null'i (sanity check
        # _float_or_default — wartości muszą być float, nie None).
        ml_call_args = ml_port.predict.call_args.args[0]
        for value in ml_call_args.values():
            assert isinstance(value, float)


# ---------------------------------------------------------------------------
# Path 4 — low volatility → ignore early (FinOps: ZERO płatnych wywołań)
# ---------------------------------------------------------------------------


class TestVolatilityBelowThreshold:
    def test_ignores_minor_changes_and_skips_all_paid_ports(
        self,
        workflow,
        market_port: Mock,
        sentiment_port: Mock,
        news_port: Mock,
        repository_port: Mock,
        ml_port: Mock,
        llm_port: Mock,
    ):
        market_port.get_current_price.return_value = Money(Decimal("99.0"))  # -1%

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert final["status"] == "ignored"
        assert final["delta"] == Decimal("-0.01")
        # FINOPS: żadne PŁATNE API nie wolno wywołać przy zmianie poniżej progu.
        sentiment_port.get_social_score.assert_not_called()
        news_port.get_news_context.assert_not_called()
        llm_port.analyze.assert_not_called()
        ml_port.predict.assert_not_called()
        repository_port.save_prediction.assert_not_called()
        # reflect_node działa ZAWSZE — get_unverified_prediction (read-only, tanie)
        # JEST wołane, ale przy braku historii (None) nie generuje żadnych kosztów:
        # ani analyze_mistake (płatne LLM), ani update (zapis) się nie wykonuje.
        repository_port.get_unverified_prediction.assert_called_once_with("AAPL")
        llm_port.analyze_mistake.assert_not_called()
        repository_port.update_prediction_accuracy.assert_not_called()


    def test_reflect_runs_even_when_cycle_is_ignored(
        self, workflow, market_port, repository_port, llm_port,
    ):
        """Sedno naprawy: ocena poprzedniej predykcji odbywa się ZAWSZE,
        niezależnie od tego, czy bieżący cykl przekroczył próg zmienności.
        Bez tego predykcje z cykli przed 'ignored' nigdy nie dostawały
        accuracy_score (czekały >12h aż trafi się cykl z volatility)."""
        # Bieżący cykl: zmiana -1% → poniżej progu → "ignored".
        market_port.get_current_price.return_value = Money(Decimal("99.0"))
        # ALE jest zaległa predykcja sprzed 12h do oceny:
        repository_port.get_unverified_prediction.return_value = Prediction(
            id="stale-uuid",
            symbol="AAPL",
            predicted_trend=TrendDirection.BEARISH,
            price_at_prediction=Decimal("100.0"),
            predicted_target_price=Decimal("95.0"),
        )
        llm_port.analyze_mistake.return_value = "Trend był OK ale skala przeszacowana."

        final = workflow.compile().invoke(_initial_state("100.0"))

        # Cykl kończy się "ignored" (brak nowej prognozy)...
        assert final["status"] == "ignored"
        # ...ALE zaległa predykcja ZOSTAŁA oceniona — accuracy_score zapisany.
        repository_port.update_prediction_accuracy.assert_called_once()
        kwargs = repository_port.update_prediction_accuracy.call_args.kwargs
        assert kwargs["prediction_id"] == "stale-uuid"
        assert isinstance(kwargs["accuracy_score"], float)


# ---------------------------------------------------------------------------
# Threshold injection
# ---------------------------------------------------------------------------


class TestCustomThreshold:
    def test_respects_injected_threshold(
        self,
        market_port: Mock,
        sentiment_port: Mock,
        news_port: Mock,
        repository_port: Mock,
        ml_port: Mock,
        llm_port: Mock,
    ):
        workflow = create_agent_graph(
            market_port=market_port,
            sentiment_port=sentiment_port,
            news_port=news_port,
            repository_port=repository_port,
            ml_port=ml_port,
            llm_port=llm_port,
            threshold=Threshold(Decimal("0.10")),  # 10%
        )
        market_port.get_current_price.return_value = Money(Decimal("95.0"))  # -5%

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert final["status"] == "ignored"
        sentiment_port.get_social_score.assert_not_called()


# ---------------------------------------------------------------------------
# XGBoost cold-start — baseline zamiast crasha (Bug 2 fix)
# ---------------------------------------------------------------------------


class TestColdStartBaseline:
    def test_uses_current_price_as_baseline_when_model_untrained(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))  # -10%
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
            has_prior_prediction=False,
        )
        ml_port.is_trained = False   # cold start — brak wytrenowanego modelu

        final = workflow.compile().invoke(_initial_state("100.0"))

        # Agent NIE crashuje — kończy "saved" z baseline = bieżąca cena
        assert final["status"] == "saved"
        assert final["ml_target_price"] == Decimal("90.0")  # current_price
        ml_port.predict.assert_not_called()   # nie wołamy nietrenowanego modelu

    def test_uses_xgboost_prediction_when_model_trained(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        ml_port.is_trained = True

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert final["ml_target_price"] == Decimal("89.5")   # z ml_port.predict
        ml_port.predict.assert_called_once()


# ---------------------------------------------------------------------------
# Kontrakt cech ML — trening i predykcja muszą używać identycznych nazw
# ---------------------------------------------------------------------------


class TestMlFeatureContract:
    def test_prediction_uses_full_training_feature_contract(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        ml_port.is_trained = True

        workflow.compile().invoke(_initial_state("100.0"))

        features = ml_port.predict.call_args.args[0]
        assert list(features) == EXPECTED_ML_FEATURES
        assert features["price_delta"] == -0.1
        assert features["av_sentiment_score"] == -0.42
        assert features["av_relevance_avg"] == 0.72
        assert features["news_volume_24h"] == 4.0
        assert features["high_relevance_count"] == 2.0
        assert features["llm_trend_signal"] == -1.0
        assert features["av_llm_agreement"] == 0.25

    def test_save_persists_ml_feature_inputs_for_slow_loop(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )

        workflow.compile().invoke(_initial_state("100.0"))

        saved_record = repository_port.save_prediction.call_args.args[0]
        assert saved_record["sentiment_score"] == -0.42
        assert saved_record["av_relevance_avg"] == 0.72
        assert saved_record["news_volume_24h"] == 4
        assert saved_record["high_relevance_count"] == 2
        assert saved_record["av_llm_agreement"] == 0.25


# ---------------------------------------------------------------------------
# accuracy_score zapisywany przy reflekcji (Bug 1 fix)
# ---------------------------------------------------------------------------


class TestAccuracyScorePersisted:
    def test_reflect_node_passes_accuracy_score_to_repository(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # Poprzednia predykcja BULLISH @100 target 105; teraz cena 110 → trafiona.
        market_port.get_current_price.return_value = Money(Decimal("110.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
            has_prior_prediction=True,
        )

        workflow.compile().invoke(_initial_state("100.0"))

        repository_port.update_prediction_accuracy.assert_called_once()
        kwargs = repository_port.update_prediction_accuracy.call_args.kwargs
        # accuracy_score MUSI być przekazany (inaczej get_accuracy_stats zawsze pusty)
        assert "accuracy_score" in kwargs
        assert isinstance(kwargs["accuracy_score"], float)
        assert 0.0 <= kwargs["accuracy_score"] <= 1.0


# ---------------------------------------------------------------------------
# Embedding podsumowania newsów → pgvector (#4)
# ---------------------------------------------------------------------------


class TestEmbeddingPersistence:
    def test_embedding_added_to_record_when_port_provided(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        embedding_port = Mock(spec=EmbeddingPort)
        embedding_port.embed.return_value = [0.1, 0.2, 0.3]
        workflow = create_agent_graph(
            market_port=market_port, sentiment_port=sentiment_port,
            news_port=news_port, repository_port=repository_port,
            ml_port=ml_port, llm_port=llm_port,
            threshold=Threshold(Decimal("0.02")),
            embedding_port=embedding_port,
        )
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )

        workflow.compile().invoke(_initial_state("100.0"))

        saved_record = repository_port.save_prediction.call_args.args[0]
        assert saved_record["embedding"] == [0.1, 0.2, 0.3]
        embedding_port.embed.assert_called_once()

    def test_save_works_without_embedding_port(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # workflow fixture nie ma embedding_port — rekord po prostu bez 'embedding'
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert final["status"] == "saved"
        saved_record = repository_port.save_prediction.call_args.args[0]
        assert "embedding" not in saved_record

    def test_embedding_failure_does_not_break_save(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        embedding_port = Mock(spec=EmbeddingPort)
        embedding_port.embed.side_effect = RuntimeError("OpenAI down")
        workflow = create_agent_graph(
            market_port=market_port, sentiment_port=sentiment_port,
            news_port=news_port, repository_port=repository_port,
            ml_port=ml_port, llm_port=llm_port,
            threshold=Threshold(Decimal("0.02")),
            embedding_port=embedding_port,
        )
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        # Graceful — save mimo błędu embeddingu
        assert final["status"] == "saved"
        saved_record = repository_port.save_prediction.call_args.args[0]
        assert "embedding" not in saved_record
