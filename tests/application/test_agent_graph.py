import logging
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
    Tool,
    ToolUseLLMPort,
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
    # Brak wcześniejszej predykcji → cecha price_delta liczona jako 0.0 (z flagą).
    # Testy weryfikujące referencję cechy nadpisują to konkretną ceną.
    repo.get_last_prediction_price.return_value = None
    # Domyślnie brak zapisanych głosów rady (dissent-replay nie ma czego dociągać).
    repo.get_council_votes_for_prediction.return_value = []
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

    def test_snapshot_not_written_when_cycle_crashes_before_terminal(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # #6: idempotencja. Snapshot ceny musi być zapisany dopiero w węźle
        # TERMINALNYM (save / ignore), NIE na starcie check_price. Gdy cykl
        # pada w połowie (np. predict_node rzuca), żaden snapshot nie może
        # zostać zapisany — inaczej retry odczytałby świeży snapshot padłej
        # próby (delta≈0) i bramka volatility cicho zignorowałaby symbol.
        market_port.get_current_price.return_value = Money(Decimal("90.0"))  # -10%
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        # predict_node pada twardo (ML niezależny od guardu LLM z #12) —
        # symuluje crash PO check_price, PRZED save.
        ml_port.predict.side_effect = RuntimeError("XGBoost segfault")

        with pytest.raises(Exception):  # noqa: B017 - dowolny wyjątek z grafu
            workflow.compile().invoke(_initial_state("100.0"))

        # Sedno naprawy: brak terminala = brak snapshotu = retry reużyje
        # referencji z POPRZEDNIEGO ukończonego cyklu.
        repository_port.save_price_snapshot.assert_not_called()

    def test_snapshot_write_failure_does_not_abort_ignored_cycle(
        self, workflow, market_port, repository_port,
    ):
        # #15: przejściowy błąd Supabase na zapisie snapshotu (teraz w węźle
        # terminalnym) nie może wywalić grafu. Cykl kończy się normalnie.
        market_port.get_current_price.return_value = Money(Decimal("99.0"))  # -1%
        repository_port.save_price_snapshot.side_effect = RuntimeError("Supabase 503")

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert final["status"] == "ignored"
        repository_port.save_price_snapshot.assert_called_once()

    def test_snapshot_write_failure_does_not_abort_saved_cycle(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # #15: jak wyżej, ale dla cyklu z pełną analizą (save_node).
        market_port.get_current_price.return_value = Money(Decimal("90.0"))  # -10%
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        repository_port.save_price_snapshot.side_effect = RuntimeError("Supabase 503")

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert final["status"] == "saved"
        assert final["prediction_id"] == "pred-uuid-123"


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
        repository_port.get_unverified_prediction.assert_called_once_with(
            "AAPL", min_age_hours=0
        )
        sentiment_port.get_social_score.assert_called_once_with("AAPL")
        news_port.get_news_context.assert_called_once_with("AAPL")
        llm_port.analyze.assert_called_once()
        ml_port.predict.assert_called_once()
        repository_port.save_prediction.assert_called_once()
        # Brak poprzedniej predykcji — nie diagnozujemy błędu
        llm_port.analyze_mistake.assert_not_called()
        repository_port.update_prediction_accuracy.assert_not_called()


# ---------------------------------------------------------------------------
# P0: kontrakt cechy price_delta i rekonstrukcja ceny z ZWROTU
# ---------------------------------------------------------------------------


class TestPriceDeltaFeatureContract:
    """Cecha price_delta podawana do ML musi być liczona względem ceny
    POPRZEDNIEJ zalogowanej predykcji (jak LAG(price_at_prediction) w widoku
    ml_feature_store), a NIE względem snapshotu używanego przez bramkę
    volatility — inaczej cecha ma inny rozkład w treningu i w inference."""

    def test_ml_price_delta_uses_last_prediction_price_not_snapshot(
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
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        # Snapshot (referencja bramki, z initial_state) = 100 → gate delta = -10%.
        # Ostatnia ZALOGOWANA predykcja była przy 120 → cecha ML = (90-120)/120.
        repository_port.get_last_prediction_price.return_value = Money(Decimal("120.0"))

        workflow.compile().invoke(_initial_state("100.0"))

        features = ml_port.predict.call_args.args[0]
        assert features["price_delta"] == pytest.approx((90.0 - 120.0) / 120.0)

    def test_ml_predict_receives_current_price_for_reconstruction(
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
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )

        workflow.compile().invoke(_initial_state("100.0"))

        # Model przewiduje zwrot → predict potrzebuje bieżącej ceny do rekonstrukcji.
        assert ml_port.predict.call_args.kwargs.get("current_price") == Decimal("90.0")


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

    def test_dissent_replay_enriches_prompt_with_vindicated_dissenter(
        self,
        workflow,
        market_port: Mock,
        sentiment_port: Mock,
        news_port: Mock,
        repository_port: Mock,
        ml_port: Mock,
        llm_port: Mock,
    ):
        from src.domain.council import InvestorOpinion

        # Prior BULLISH(=BUY) @100; cena spadła do 90 (DOWN) → błędny kierunek.
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
            has_prior_prediction=True,
        )
        # Rada miała dysydenta SELL (Burry), który trafił spadek — agent szedł BUY.
        repository_port.get_council_votes_for_prediction.return_value = [
            InvestorOpinion("Burry", "SELL", 0.8, "Zadłużenie", ()),
            InvestorOpinion("Wood", "BUY", 0.7, "Wzrost", ()),
        ]

        workflow.compile().invoke(_initial_state("100.0"))

        repository_port.get_council_votes_for_prediction.assert_called_once_with(
            "prev-uuid-456"
        )
        prompt = llm_port.analyze_mistake.call_args.args[0]
        assert "Burry" in prompt


# ---------------------------------------------------------------------------
# #16 — paid analyze_mistake bounded by volatility gate; cheap bookkeeping
#        (accuracy/trend/update) runs ALWAYS, regardless of current volatility.
# ---------------------------------------------------------------------------


class TestReflectMistakeDiagnosisGatedByVolatility:
    def _wrong_prediction(self) -> Prediction:
        # BULLISH @100 target 105; spadek poniżej 100 → zły kierunek.
        return Prediction(
            id="stale-uuid",
            symbol="AAPL",
            predicted_trend=TrendDirection.BULLISH,
            price_at_prediction=Decimal("100.0"),
            predicted_target_price=Decimal("105.0"),
        )

    def test_skips_paid_diagnosis_when_below_threshold_but_still_records(
        self, workflow, market_port, repository_port, llm_port,
    ):
        # Cykl płaski: prev=100, current=99.5 → |Δ|=0.5% < 2% (bramka ignoruje).
        # Prior był BŁĘDNY (BULLISH, a cena spadła) — KIEDYŚ to płaciło za
        # analyze_mistake mimo niskiej zmienności. Teraz: tania księgowość
        # (accuracy/trend/update) leci, ale PŁATNE analyze_mistake jest pominięte.
        market_port.get_current_price.return_value = Money(Decimal("99.5"))
        repository_port.get_unverified_prediction.return_value = self._wrong_prediction()

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert final["status"] == "ignored"
        # PŁATNE LLM pominięte — sedno #16.
        llm_port.analyze_mistake.assert_not_called()
        # Tania księgowość WYKONANA mimo niskiej zmienności.
        repository_port.update_prediction_accuracy.assert_called_once()
        kwargs = repository_port.update_prediction_accuracy.call_args.kwargs
        assert kwargs["prediction_id"] == "stale-uuid"
        assert kwargs["is_trend_correct"] is False
        assert isinstance(kwargs["accuracy_score"], float)
        # Generyczny insight zamiast diagnozy LLM.
        assert "zmienno" in kwargs["insight"].lower()

    def test_diagnoses_paid_when_at_or_above_threshold_and_trend_wrong(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # Cykl zmienny: prev=100, current=90 → |Δ|=10% ≥ 2% → bramka przepuszcza.
        # Prior był błędny → płatna diagnoza LLM odpala się jak dotąd.
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
            has_prior_prediction=True,
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        llm_port.analyze_mistake.assert_called_once()
        kwargs = repository_port.update_prediction_accuracy.call_args.kwargs
        assert kwargs["insight"] == "Zignorowałem szerszy kontekst makro."
        assert "Zignorowałem szerszy kontekst makro." in final["reflection_context"]

    def test_crypto_uses_crypto_threshold_for_diagnosis_gate(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # #16 reużywa _pick_threshold: dla CRYPTO bramka diagnozy używa
        # crypto_threshold. prev=100, current=97 → |Δ|=3%. Akcje (2%) by
        # przepuściły, ale crypto_threshold=5% → diagnoza pominięta.
        from src.domain.asset import Asset
        from src.domain.value_objects import AssetType

        market_port.get_current_price.return_value = Money(Decimal("97.0"))
        repository_port.get_unverified_prediction.return_value = Prediction(
            id="crypto-uuid",
            symbol="BTC",
            predicted_trend=TrendDirection.BULLISH,
            price_at_prediction=Decimal("100.0"),
            predicted_target_price=Decimal("110.0"),
        )
        llm_port.analyze_mistake.return_value = "nie powinno się odpalić"
        workflow = create_agent_graph(
            market_port=market_port, sentiment_port=sentiment_port,
            news_port=news_port, repository_port=repository_port,
            ml_port=ml_port, llm_port=llm_port,
            threshold=Threshold(Decimal("0.02")),
            crypto_threshold=Threshold(Decimal("0.05")),
        )

        asset = Asset(symbol="BTC", asset_type=AssetType.CRYPTO)
        workflow.compile().invoke({
            "symbol": "BTC",
            "previous_price": Decimal("100.0"),
            "asset": asset,
        })

        # 3% < crypto 5% → płatna diagnoza pominięta, księgowość wykonana.
        llm_port.analyze_mistake.assert_not_called()
        repository_port.update_prediction_accuracy.assert_called_once()
        kwargs = repository_port.update_prediction_accuracy.call_args.kwargs
        assert kwargs["is_trend_correct"] is False


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
# #14 — instrumentacja: timing per-node + jawny log każdego płatnego wywołania
# ---------------------------------------------------------------------------


class TestPaidNodeInstrumentation:
    def test_paid_nodes_emit_timing_logs(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port, caplog,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))  # -10%
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        with caplog.at_level(logging.INFO, logger="src.application.agent_graph"):
            workflow.compile().invoke(_initial_state("100.0"))

        timing_logs = [
            rec.message for rec in caplog.records if "elapsed_ms" in rec.message
        ]
        joined = " ".join(timing_logs)
        # Każdy płatny węzeł raportuje swój czas wykonania z symbolem.
        for node_name in ("analyze_sentiment", "fetch_news", "predict"):
            assert node_name in joined, f"brak timing logu dla {node_name}"
        assert "AAPL" in joined

    def test_explicit_paid_call_log_emitted(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port, caplog,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        with caplog.at_level(logging.INFO, logger="src.application.agent_graph"):
            workflow.compile().invoke(_initial_state("100.0"))

        joined = " ".join(rec.message for rec in caplog.records)
        # Jawna linia, gdy odpala się płatne wywołanie (FinOps audit).
        assert "paid call" in joined.lower()

    def test_no_paid_timing_logs_when_cycle_ignored(
        self, workflow, market_port, repository_port, caplog,
    ):
        # Cykl poniżej progu → płatne węzły się nie odpalają → brak ich timingów.
        market_port.get_current_price.return_value = Money(Decimal("99.0"))  # -1%
        with caplog.at_level(logging.INFO, logger="src.application.agent_graph"):
            workflow.compile().invoke(_initial_state("100.0"))

        paid_timing = [
            rec.message for rec in caplog.records
            if "elapsed_ms" in rec.message
            and ("predict" in rec.message or "fetch_news" in rec.message
                 or "analyze_sentiment" in rec.message)
        ]
        assert not paid_timing, f"płatne węzły nie powinny się odpalić: {paid_timing}"


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
# #12-guard — predict_node degraduje gracefully gdy llm_port.analyze rzuca
# ---------------------------------------------------------------------------


class TestPredictNodeLlmGuard:
    """W przeciwieństwie do council_node, predict_node wołał llm_port.analyze
    bez guardu — czatliwa / zepsuta odpowiedź LLM (ValueError przy parsowaniu)
    wywalała cały symbol. Powinien degradować: neutralna analiza + flaga."""

    def test_llm_analyze_failure_degrades_to_neutral_analysis(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))  # -10%
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        # LLM zwraca śmieć → adapter rzuca ValueError przy parsowaniu JSON.
        llm_port.analyze.side_effect = ValueError("LLM returned non-JSON prose")

        final = workflow.compile().invoke(_initial_state("100.0"))

        # Symbol NIE crashuje — kończy "saved" z neutralną analizą.
        assert final["status"] == "saved"
        assert final["llm_analysis"]["trend_direction"] == "SIDEWAYS"
        assert final["llm_analysis"]["confidence_score"] == pytest.approx(0.5)
        # Ślad degradacji w data_quality_flags (trening odsieje te rekordy).
        assert "llm_analysis_failed" in final["data_quality_flags"]
        # Zapisany rekord też niesie flagę.
        saved_record = repository_port.save_prediction.call_args.args[0]
        assert "llm_analysis_failed" in saved_record["data_quality_flags"]

    def test_predict_continues_to_ml_when_llm_fails(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # Po fallbacku LLM, ML nadal liczy predykcję (llm_trend_signal=0 → SIDEWAYS).
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        llm_port.analyze.side_effect = ValueError("boom")

        workflow.compile().invoke(_initial_state("100.0"))

        ml_port.predict.assert_called_once()
        features = ml_port.predict.call_args.args[0]
        # Neutralny fallback → llm_trend_signal SIDEWAYS = 0.0.
        assert features["llm_trend_signal"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# #21 — walidacja trend_direction z LLM przed mapowaniem na sygnał ML
# ---------------------------------------------------------------------------


class TestTrendDirectionValidation:
    """trend_direction z LLM napędza cechę llm_trend_signal. Dotychczas `.get`
    z domyślnym 0 cicho traktował KAŻDY nieoczekiwany string (np. "UP",
    literówkę) jako neutralny SIDEWAYS bez żadnej flagi jakości — w przeciwień-
    stwie do pól liczbowych, które zostawiają ślad (_missing/_invalid). Po
    naprawie: present-but-invalid → sygnał 0 + flaga `trend_direction_invalid`;
    poprawna wartość (case-insensitive) → właściwy sygnał, bez flagi; brak
    wartości → SIDEWAYS, bez flagi (genuinely-absent ma prawo default'ować)."""

    def test_unexpected_trend_string_maps_to_neutral_and_flags(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))  # -10%
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        # "UP" nie należy do {BULLISH, BEARISH, SIDEWAYS} → neutralny sygnał + flaga.
        llm_port.analyze.return_value = {
            "trend_direction": "UP",
            "confidence_score": 0.7,
            "av_agreement": 0.5,
            "reasoning": "Up.",
        }

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert final["status"] == "saved"
        features = ml_port.predict.call_args.args[0]
        # Nieznany kierunek → neutralny sygnał 0.0 (jak SIDEWAYS).
        assert features["llm_trend_signal"] == pytest.approx(0.0)
        # ...ALE zostawia ślad jakości (inaczej śmieć udaje prawdziwy SIDEWAYS).
        assert "trend_direction_invalid" in final["data_quality_flags"]
        saved_record = repository_port.save_prediction.call_args.args[0]
        assert "trend_direction_invalid" in saved_record["data_quality_flags"]

    def test_lowercase_trend_is_normalized_without_flag(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        # "bullish" (lowercase) to wciąż poprawny kierunek — case-insensitive.
        llm_port.analyze.return_value = {
            "trend_direction": "bullish",
            "confidence_score": 0.8,
            "av_agreement": 0.9,
            "reasoning": "Bull.",
        }

        final = workflow.compile().invoke(_initial_state("100.0"))

        features = ml_port.predict.call_args.args[0]
        assert features["llm_trend_signal"] == pytest.approx(1.0)  # BULLISH
        assert "trend_direction_invalid" not in final["data_quality_flags"]

    def test_missing_trend_defaults_to_sideways_without_flag(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        # Brak klucza trend_direction — genuinely-absent ma prawo default'ować
        # do SIDEWAYS bez flagi (to nie jest skażony input, tylko brak pola).
        llm_port.analyze.return_value = {
            "confidence_score": 0.5,
            "av_agreement": 0.5,
            "reasoning": "Brak kierunku.",
        }

        final = workflow.compile().invoke(_initial_state("100.0"))

        features = ml_port.predict.call_args.args[0]
        assert features["llm_trend_signal"] == pytest.approx(0.0)  # SIDEWAYS
        assert "trend_direction_invalid" not in final["data_quality_flags"]


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
        repository_port.get_unverified_prediction.assert_called_once_with(
            "AAPL", min_age_hours=0
        )
        llm_port.analyze_mistake.assert_not_called()
        repository_port.update_prediction_accuracy.assert_not_called()


    def test_gate_decision_is_logged_when_ignored(
        self, workflow, market_port, repository_port, caplog,
    ):
        # #13: decyzja bramki volatility musi być widoczna w logach (wcześniej
        # cisza — niemożliwe było odróżnienie "ignore z premedytacją" od buga).
        market_port.get_current_price.return_value = Money(Decimal("99.0"))  # -1%
        with caplog.at_level(logging.INFO, logger="src.application.agent_graph"):
            workflow.compile().invoke(_initial_state("100.0"))

        joined = " ".join(rec.message for rec in caplog.records)
        assert "AAPL" in joined
        assert "ignore" in joined.lower()
        # Log zawiera |delta|, próg i typ aktywa do diagnostyki FinOps.
        assert "0.01" in joined  # |delta| = 1%
        assert "0.02" in joined  # próg

    def test_gate_decision_is_logged_when_analyzing(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port, caplog,
    ):
        # #13: decyzja "przepuść" też loguje się (symetria z "ignore").
        market_port.get_current_price.return_value = Money(Decimal("90.0"))  # -10%
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        with caplog.at_level(logging.INFO, logger="src.application.agent_graph"):
            workflow.compile().invoke(_initial_state("100.0"))

        gate_logs = [
            rec.message for rec in caplog.records
            if "volatility gate" in rec.message.lower()
        ]
        assert gate_logs, "spodziewany log decyzji bramki volatility"
        joined = " ".join(gate_logs)
        assert "AAPL" in joined
        assert "analyze" in joined.lower()

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

    def test_reflect_passes_configured_min_age_to_repository(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        """reflection_min_age_hours przeniesione z konfiguracji aż do repo —
        bez tego filtr przedwczesnej oceny nigdy by się nie aktywował."""
        market_port.get_current_price.return_value = Money(Decimal("99.0"))
        repository_port.get_unverified_prediction.return_value = None
        workflow = create_agent_graph(
            market_port=market_port,
            sentiment_port=sentiment_port,
            news_port=news_port,
            repository_port=repository_port,
            ml_port=ml_port,
            llm_port=llm_port,
            threshold=Threshold(Decimal("0.02")),
            reflection_min_age_hours=6,
        )

        workflow.compile().invoke(_initial_state("100.0"))

        repository_port.get_unverified_prediction.assert_called_once_with(
            "AAPL", min_age_hours=6
        )


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
        # price_delta liczone względem ceny poprzedniej predykcji (kontrakt jak w
        # widoku), nie względem snapshotu: (90-120)/120.
        repository_port.get_last_prediction_price.return_value = Money(Decimal("120.0"))

        workflow.compile().invoke(_initial_state("100.0"))

        features = ml_port.predict.call_args.args[0]
        assert list(features) == EXPECTED_ML_FEATURES
        assert features["price_delta"] == pytest.approx((90.0 - 120.0) / 120.0)
        assert features["av_sentiment_score"] == pytest.approx(-0.42)
        assert features["av_relevance_avg"] == pytest.approx(0.72)
        assert features["news_volume_24h"] == pytest.approx(4.0)
        assert features["high_relevance_count"] == pytest.approx(2.0)
        assert features["llm_trend_signal"] == pytest.approx(-1.0)
        assert features["av_llm_agreement"] == pytest.approx(0.25)

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
        assert saved_record["sentiment_score"] == pytest.approx(-0.42)
        assert saved_record["av_relevance_avg"] == pytest.approx(0.72)
        assert saved_record["news_volume_24h"] == 4
        assert saved_record["high_relevance_count"] == 2
        assert saved_record["av_llm_agreement"] == pytest.approx(0.25)


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
# is_trend_correct zapisywany przy reflekcji — accuracy_score (bliskość celu)
# nie rozróżnia kierunku, więc trafność raportu musi jechać po tym polu.
# ---------------------------------------------------------------------------


class TestTrendCorrectnessPersisted:
    def test_reflect_persists_trend_correct_false_when_direction_wrong(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # Poprzednio: BULLISH @100. Teraz cena 90 (spadek) → zły kierunek.
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
            has_prior_prediction=True,
        )

        workflow.compile().invoke(_initial_state("100.0"))

        kwargs = repository_port.update_prediction_accuracy.call_args.kwargs
        assert kwargs["is_trend_correct"] is False

    def test_reflect_persists_trend_correct_true_when_direction_right(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # Poprzednio: BULLISH @100. Teraz cena 110 (wzrost) → kierunek OK.
        market_port.get_current_price.return_value = Money(Decimal("110.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
            has_prior_prediction=True,
        )

        workflow.compile().invoke(_initial_state("100.0"))

        kwargs = repository_port.update_prediction_accuracy.call_args.kwargs
        assert kwargs["is_trend_correct"] is True


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


# ---------------------------------------------------------------------------
# RAG retrieval (#7) — embedding newsów wykorzystany do wstrzyknięcia
# podobnych historycznych sytuacji do promptu predykcji.
# ---------------------------------------------------------------------------


class TestRagRetrieval:
    def test_similar_past_situations_injected_into_prediction_prompt(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        embedding_port = Mock(spec=EmbeddingPort)
        embedding_port.embed.return_value = [0.1] * 8
        repository_port.find_similar_predictions.return_value = [
            {
                "news_summary": "Fed hawkish surprise spooks tech",
                "predicted_trend": "BEARISH",
                "is_trend_correct": True,
                "correction_insights": "reaguj na jastrzębi Fed",
            }
        ]
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

        # Pierwsze llm.analyze() = prompt predykcji (brak rady w tym grafie).
        prediction_prompt = llm_port.analyze.call_args_list[0].args[0]
        assert "Fed hawkish surprise spooks tech" in prediction_prompt
        repository_port.find_similar_predictions.assert_called_once()

    def test_retrieval_failure_does_not_break_prediction(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        embedding_port = Mock(spec=EmbeddingPort)
        embedding_port.embed.return_value = [0.1] * 8
        repository_port.find_similar_predictions.side_effect = RuntimeError(
            "pgvector RPC missing"
        )
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

        assert final["status"] == "saved"

    def test_no_retrieval_without_embedding_port(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )

        workflow.compile().invoke(_initial_state("100.0"))

        repository_port.find_similar_predictions.assert_not_called()


# ---------------------------------------------------------------------------
# Q5 — RAG precedent receipts: analogi RAG (już pobrane w tym cyklu) wystawione
# jako auditowalny "dlaczego ta decyzja". Zero nowych płatnych wywołań —
# reużywamy embedding + RPC, które predict_node i tak wykonał.
# ---------------------------------------------------------------------------


class TestRagPrecedentReceipts:
    def _make_workflow(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port, embedding_port,
    ):
        return create_agent_graph(
            market_port=market_port, sentiment_port=sentiment_port,
            news_port=news_port, repository_port=repository_port,
            ml_port=ml_port, llm_port=llm_port,
            threshold=Threshold(Decimal("0.02")),
            embedding_port=embedding_port,
        )

    def test_precedents_captured_into_node_output(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        embedding_port = Mock(spec=EmbeddingPort)
        embedding_port.embed.return_value = [0.1] * 8
        repository_port.find_similar_predictions.return_value = [
            {
                "news_summary": "Fed hawkish surprise spooks tech",
                "predicted_trend": "BEARISH",
                "is_trend_correct": True,
                "correction_insights": "reaguj na jastrzębi Fed",
            },
            {
                "news_summary": "Strong earnings beat",
                "predicted_trend": "BULLISH",
                "is_trend_correct": False,
            },
        ]
        workflow = self._make_workflow(
            market_port, sentiment_port, news_port,
            repository_port, ml_port, llm_port, embedding_port,
        )
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        precedents = final["similar_precedents"]
        assert len(precedents) == 2
        assert precedents[0]["summary"] == "Fed hawkish surprise spooks tech"
        assert precedents[0]["predicted_trend"] == "BEARISH"
        assert precedents[0]["is_trend_correct"] is True
        assert precedents[1]["is_trend_correct"] is False

    def test_precedents_capped_at_top_3(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        embedding_port = Mock(spec=EmbeddingPort)
        embedding_port.embed.return_value = [0.1] * 8
        repository_port.find_similar_predictions.return_value = [
            {"news_summary": f"sytuacja {i}", "predicted_trend": "BULLISH",
             "is_trend_correct": True}
            for i in range(5)
        ]
        workflow = self._make_workflow(
            market_port, sentiment_port, news_port,
            repository_port, ml_port, llm_port, embedding_port,
        )
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert len(final["similar_precedents"]) == 3

    def test_records_without_summary_skipped(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        embedding_port = Mock(spec=EmbeddingPort)
        embedding_port.embed.return_value = [0.1] * 8
        repository_port.find_similar_predictions.return_value = [
            {"news_summary": "", "predicted_trend": "BULLISH"},
            {"news_summary": "realny analog", "predicted_trend": "BEARISH",
             "is_trend_correct": True},
        ]
        workflow = self._make_workflow(
            market_port, sentiment_port, news_port,
            repository_port, ml_port, llm_port, embedding_port,
        )
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        precedents = final["similar_precedents"]
        assert len(precedents) == 1
        assert precedents[0]["summary"] == "realny analog"

    def test_empty_rag_yields_empty_precedents(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        embedding_port = Mock(spec=EmbeddingPort)
        embedding_port.embed.return_value = [0.1] * 8
        repository_port.find_similar_predictions.return_value = []
        workflow = self._make_workflow(
            market_port, sentiment_port, news_port,
            repository_port, ml_port, llm_port, embedding_port,
        )
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert final["similar_precedents"] == []

    def test_retrieval_failure_yields_empty_precedents(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        embedding_port = Mock(spec=EmbeddingPort)
        embedding_port.embed.return_value = [0.1] * 8
        repository_port.find_similar_predictions.side_effect = RuntimeError("boom")
        workflow = self._make_workflow(
            market_port, sentiment_port, news_port,
            repository_port, ml_port, llm_port, embedding_port,
        )
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        # Graceful — brak analogów, ale cykl się kończy zapisem.
        assert final["status"] == "saved"
        assert final["similar_precedents"] == []

    def test_find_similar_called_once_no_extra_paid_call(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # FinOps: receipts NIE mogą dodać drugiego wywołania RPC/embeddingu.
        embedding_port = Mock(spec=EmbeddingPort)
        embedding_port.embed.return_value = [0.1] * 8
        repository_port.find_similar_predictions.return_value = [
            {"news_summary": "analog", "predicted_trend": "BULLISH",
             "is_trend_correct": True},
        ]
        workflow = self._make_workflow(
            market_port, sentiment_port, news_port,
            repository_port, ml_port, llm_port, embedding_port,
        )
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )

        workflow.compile().invoke(_initial_state("100.0"))

        repository_port.find_similar_predictions.assert_called_once()
        # Embedding: raz w predict (RAG) — reużyty w save_node (brak 2. wywołania).
        embedding_port.embed.assert_called_once()


# ---------------------------------------------------------------------------
# Q6 — Data-provenance: sygnały świeżości/jakości (degraded_reason, wiek
# fundamentów, data_quality_flags) wystawione w wyjściu predict_node, by
# to_symbol_result mógł zbudować odznaki proweniencji. Zero płatnych wywołań.
# ---------------------------------------------------------------------------


class TestProvenanceSignals:
    def test_degraded_reason_surfaced_in_data_quality_flags(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        sentiment_port.get_social_score.return_value = {
            "av_sentiment_score": 0.0,
            "av_relevance_avg": 0.0,
            "news_volume_24h": 0,
            "high_relevance_count": 0,
            "av_sentiment_label": "Neutral",
            "degraded_reason": "av_keys_exhausted",
        }

        final = workflow.compile().invoke(_initial_state("100.0"))

        # degraded_reason musi trafić do flag, by to_symbol_result zbudował
        # odznakę DEGRADED. (Provenance buduje się w report z raw outputu.)
        assert "av_keys_exhausted" in final["data_quality_flags"]

    def test_clean_cycle_has_no_quality_flags(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        workflow = create_agent_graph(
            market_port=market_port, sentiment_port=sentiment_port,
            news_port=news_port, repository_port=repository_port,
            ml_port=ml_port, llm_port=llm_port,
            threshold=Threshold(Decimal("0.02")),
        )
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
            has_prior_prediction=True,
        )
        repository_port.get_last_prediction_price.return_value = Money(
            Decimal("100.0")
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        # Czysty cykl → brak flag (FRESH w raporcie).
        assert final["data_quality_flags"] == []


# ---------------------------------------------------------------------------
# #6 — Tool-use research agent: na NAJWIĘKSZYCH ruchach (osobny, wyższy próg)
# predict_node przełącza główną analizę na pętlę tool-use (ToolUseLLMPort),
# która pozwala modelowi dociągnąć read-only toole (fundamenty/makro) przed
# werdyktem. Poniżej progu tool-use — zwykły llm_port.analyze (FinOps).
# ---------------------------------------------------------------------------


class TestToolUseResearchAgent:
    def _tools(self) -> list[Tool]:
        # Atrapa read-only toola — w teście nie wywołujemy func (adapter to mock).
        return [
            Tool(
                name="get_macro",
                description="Polski makro snapshot.",
                parameters={"type": "object", "properties": {}, "required": []},
                func=lambda args: {"available": False},
            )
        ]

    def test_uses_tool_loop_when_move_exceeds_tool_threshold(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # -10% ≥ 5% próg tool-use → analiza idzie pętlą tool-use, nie zwykłym analyze.
        tool_use_port = Mock(spec=ToolUseLLMPort)
        tool_use_port.analyze_with_tools.return_value = {
            "trend_direction": "BEARISH",
            "confidence_score": 0.9,
            "av_agreement": 0.3,
            "reasoning": "Po sprawdzeniu fundamentów: pogorszenie.",
        }
        workflow = create_agent_graph(
            market_port=market_port, sentiment_port=sentiment_port,
            news_port=news_port, repository_port=repository_port,
            ml_port=ml_port, llm_port=llm_port,
            threshold=Threshold(Decimal("0.02")),
            tool_use_port=tool_use_port,
            research_tools=self._tools(),
            tool_use_threshold=Threshold(Decimal("0.05")),
        )
        market_port.get_current_price.return_value = Money(Decimal("90.0"))  # -10%
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert final["status"] == "saved"
        tool_use_port.analyze_with_tools.assert_called_once()
        # Pętla tool-use przejmuje główną analizę — zwykły analyze NIE leci.
        llm_port.analyze.assert_not_called()
        # Werdykt z tool-use trafia do analizy i napędza dalsze węzły.
        assert final["llm_analysis"]["trend_direction"] == "BEARISH"
        # Tool i prompt przekazane do adaptera.
        args = tool_use_port.analyze_with_tools.call_args.args
        assert isinstance(args[0], str)  # prompt
        assert args[1][0].name == "get_macro"

    def test_falls_back_to_plain_analyze_below_tool_threshold(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # -3%: główna bramka 2% przepuszcza (analiza leci), ale < 5% próg tool-use
        # → zwykły llm_port.analyze, BEZ kosztownej pętli tool-use.
        tool_use_port = Mock(spec=ToolUseLLMPort)
        workflow = create_agent_graph(
            market_port=market_port, sentiment_port=sentiment_port,
            news_port=news_port, repository_port=repository_port,
            ml_port=ml_port, llm_port=llm_port,
            threshold=Threshold(Decimal("0.02")),
            tool_use_port=tool_use_port,
            research_tools=self._tools(),
            tool_use_threshold=Threshold(Decimal("0.05")),
        )
        market_port.get_current_price.return_value = Money(Decimal("97.0"))  # -3%
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert final["status"] == "saved"
        tool_use_port.analyze_with_tools.assert_not_called()
        llm_port.analyze.assert_called_once()

    def test_tool_loop_failure_degrades_gracefully(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # Pętla tool-use rzuca (np. cap rund + zła odpowiedź) → predict_node
        # degraduje do neutralnej analizy + flagi, jak guard #12. Brak crashu.
        tool_use_port = Mock(spec=ToolUseLLMPort)
        tool_use_port.analyze_with_tools.side_effect = RuntimeError("tool loop boom")
        workflow = create_agent_graph(
            market_port=market_port, sentiment_port=sentiment_port,
            news_port=news_port, repository_port=repository_port,
            ml_port=ml_port, llm_port=llm_port,
            threshold=Threshold(Decimal("0.02")),
            tool_use_port=tool_use_port,
            research_tools=self._tools(),
            tool_use_threshold=Threshold(Decimal("0.05")),
        )
        market_port.get_current_price.return_value = Money(Decimal("90.0"))  # -10%
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert final["status"] == "saved"
        assert final["llm_analysis"]["trend_direction"] == "SIDEWAYS"
        assert "llm_analysis_failed" in final["data_quality_flags"]

    def test_no_tool_port_always_uses_plain_analyze(
        self, workflow, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # Backward-compat: brak tool_use_port → zwykły analyze nawet przy dużym ruchu.
        market_port.get_current_price.return_value = Money(Decimal("90.0"))  # -10%
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )

        final = workflow.compile().invoke(_initial_state("100.0"))

        assert final["status"] == "saved"
        llm_port.analyze.assert_called_once()


# ---------------------------------------------------------------------------
# Cross-Asset Vector Memory — tag reżimu przy embeddingu + filtr reżimu w RAG.
# Bez flagi: zachowanie wsteczne (regime=None, brak kolumny regime w rekordzie).
# ---------------------------------------------------------------------------


class TestVectorMemory:
    def _make(self, ports: dict, *, enabled: bool, embedding_port: Mock):
        return create_agent_graph(
            **ports,
            threshold=Threshold(Decimal("0.02")),
            embedding_port=embedding_port,
            vector_memory_enabled=enabled,
        )

    def _state(self) -> dict:
        return {
            "symbol": "AAPL",
            "previous_price": Decimal("100.0"),
            "regime_context": "RISK-OFF",
        }

    def test_regime_passed_to_retrieval_and_stored_when_enabled(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        embedding_port = Mock(spec=EmbeddingPort)
        embedding_port.embed.return_value = [0.1] * 8
        repository_port.find_similar_predictions.return_value = []
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        ports = {
            "market_port": market_port, "sentiment_port": sentiment_port,
            "news_port": news_port, "repository_port": repository_port,
            "ml_port": ml_port, "llm_port": llm_port,
        }

        self._make(ports, enabled=True, embedding_port=embedding_port).compile().invoke(
            self._state()
        )

        # RAG filtrowany po reżimie (surowa etykieta — repo kanonizuje).
        _, kwargs = repository_port.find_similar_predictions.call_args
        assert kwargs.get("regime") == "RISK-OFF"
        # Zapisany rekord NIESIE kanoniczny tag reżimu (do tagowania pamięci).
        saved = repository_port.save_prediction.call_args.args[0]
        assert saved["regime"] == "RISK-OFF"

    def test_no_regime_filter_or_tag_when_disabled(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        embedding_port = Mock(spec=EmbeddingPort)
        embedding_port.embed.return_value = [0.1] * 8
        repository_port.find_similar_predictions.return_value = []
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        ports = {
            "market_port": market_port, "sentiment_port": sentiment_port,
            "news_port": news_port, "repository_port": repository_port,
            "ml_port": ml_port, "llm_port": llm_port,
        }

        self._make(ports, enabled=False, embedding_port=embedding_port).compile().invoke(
            self._state()
        )

        _, kwargs = repository_port.find_similar_predictions.call_args
        assert kwargs.get("regime") is None
        saved = repository_port.save_prediction.call_args.args[0]
        assert "regime" not in saved


class TestEarningsThresholdGate:
    """#6 — mnożnik earnings zacieśnia bramkę volatility i mnoży się
    z mnożnikiem reżimu. Oba są >= 1.0, więc bramka może się tylko zacieśnić."""

    @staticmethod
    def _state(**overrides: object) -> dict[str, object]:
        state: dict[str, object] = {"asset": None}
        state.update(overrides)
        return state

    def test_earnings_multiplier_tightens_threshold(self) -> None:
        from src.application.agent_graph import _pick_threshold

        base = Threshold(Decimal("0.02"))
        chosen = _pick_threshold(
            self._state(earnings_multiplier=1.5), base, None  # type: ignore[arg-type]
        )

        assert chosen.value == Decimal("0.03")

    def test_earnings_and_regime_multipliers_compound(self) -> None:
        # IMMINENT (1.5) w RISK_OFF (2.0) → próg 2% * 3.0 = 6%.
        from src.application.agent_graph import _pick_threshold

        base = Threshold(Decimal("0.02"))
        chosen = _pick_threshold(
            self._state(earnings_multiplier=1.5, regime_multiplier=2.0),  # type: ignore[arg-type]
            base,
            None,
        )

        assert chosen.value == Decimal("0.06")

    def test_absent_earnings_multiplier_leaves_threshold_untouched(self) -> None:
        from src.application.agent_graph import _pick_threshold

        base = Threshold(Decimal("0.02"))
        chosen = _pick_threshold(self._state(), base, None)  # type: ignore[arg-type]

        assert chosen.value == Decimal("0.02")

    def test_earnings_multiplier_never_loosens_the_gate(self) -> None:
        # Kontrakt domenowy: mnożnik < 1.0 nie ma prawa powstać. Gdyby jednak
        # przeciekł, bramka nie może się rozluźnić — to reguła FinOps.
        from src.application.agent_graph import _pick_threshold

        base = Threshold(Decimal("0.02"))
        chosen = _pick_threshold(
            self._state(earnings_multiplier=0.5), base, None  # type: ignore[arg-type]
        )

        assert chosen.value >= Decimal("0.02")


class TestDecisionReceipts:
    """#13 — persystowany audit trail predykcji (migracja 020).

    Flaga `receipts_enabled` NIE jest ozdobnikiem: nowy klucz w rekordzie przed
    zaaplikowaniem migracji powoduje PGRST204 i kładzie zapis CAŁEJ predykcji.
    Off = klucza nie ma w ogóle.
    """

    def _make(self, ports: dict, *, enabled: bool):
        return create_agent_graph(
            **ports,
            threshold=Threshold(Decimal("0.02")),
            receipts_enabled=enabled,
        )

    def _record(self, repository_port: Mock) -> dict:
        return repository_port.save_prediction.call_args.args[0]

    def test_flag_off_omits_the_column_entirely(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        ports = dict(
            market_port=market_port, sentiment_port=sentiment_port,
            news_port=news_port, repository_port=repository_port,
            ml_port=ml_port, llm_port=llm_port,
        )

        self._make(ports, enabled=False).compile().invoke(_initial_state("100.0"))

        assert "decision_receipts" not in self._record(repository_port)

    def test_flag_on_writes_versioned_receipts(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        ports = dict(
            market_port=market_port, sentiment_port=sentiment_port,
            news_port=news_port, repository_port=repository_port,
            ml_port=ml_port, llm_port=llm_port,
        )

        self._make(ports, enabled=True).compile().invoke(_initial_state("100.0"))

        receipts = self._record(repository_port)["decision_receipts"]
        assert receipts["schema_version"] == 1
        # Próg EFEKTYWNY (po mnożnikach), nie surowy z konfiguracji.
        assert "effective_threshold" in receipts

    def test_receipts_are_json_serialisable(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # Regresja: `_serialize` konwertuje Decimal tylko na najwyższym poziomie,
        # więc zagnieżdżony Decimal wywala zapis do Supabase.
        import json

        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        ports = dict(
            market_port=market_port, sentiment_port=sentiment_port,
            news_port=news_port, repository_port=repository_port,
            ml_port=ml_port, llm_port=llm_port,
        )

        self._make(ports, enabled=True).compile().invoke(_initial_state("100.0"))

        json.dumps(self._record(repository_port)["decision_receipts"])


class TestFormatAnalogCarriesIdentity:
    """#13 — RPC zwraca `id` i `similarity`, a `_format_analog` je gubiło.
    Bez nich kwit nie pozwala zaudytować, KTÓRY analog stał za predykcją."""

    def test_precedent_keeps_id_and_similarity(self):
        from src.application.agent_graph import _format_analog

        formatted = _format_analog({
            "id": "pred-123",
            "similarity": 0.87,
            "news_summary": "Fed hawkish",
            "predicted_trend": "BEARISH",
            "is_trend_correct": True,
        })

        assert formatted is not None
        _, precedent = formatted
        assert precedent["id"] == "pred-123"
        assert precedent["similarity"] == 0.87


class TestImpliedEdge:
    """#18 — edge vs rynek opcji. Fetch opcji siedzi w `_predict_node`, czyli
    NATURALNIE za bramką volatility: przy niskiej zmienności płatny port nie rusza."""

    @staticmethod
    def _options(iv: float | None) -> Mock:
        from src.application.ports import OptionsFlowPort
        from src.domain.options_flow import OptionsFlowSnapshot

        port = Mock(spec=OptionsFlowPort)
        port.get_options_flow.return_value = (
            None if iv is None
            else OptionsFlowSnapshot(symbol="AAPL", put_call_ratio=0.6, implied_vol=iv)
        )
        return port

    def _make(self, ports: dict, options_port):
        return create_agent_graph(
            **ports,
            threshold=Threshold(Decimal("0.02")),
            options_port=options_port,
        )

    def _ports(self, market_port, sentiment_port, news_port, repository_port, ml_port, llm_port):
        return dict(
            market_port=market_port, sentiment_port=sentiment_port,
            news_port=news_port, repository_port=repository_port,
            ml_port=ml_port, llm_port=llm_port,
        )

    def test_low_volatility_never_touches_the_options_port(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # Δ = 0.5% < 2% → bramka odcina cykl PRZED predict_node.
        market_port.get_current_price.return_value = Money(Decimal("100.5"))
        options = self._options(0.45)
        ports = self._ports(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
        )

        final = self._make(ports, options).compile().invoke(_initial_state("100.0"))

        assert final["status"] == "ignored"
        options.get_options_flow.assert_not_called()

    def test_edge_sigma_reaches_the_saved_record(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        options = self._options(0.45)
        ports = self._ports(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
        )

        self._make(ports, options).compile().invoke(_initial_state("100.0"))

        record = repository_port.save_prediction.call_args.args[0]
        assert "edge_sigma" in record

    def test_no_options_port_leaves_the_cycle_identical(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # options_flow_enabled=false → Null/None → sygnał nie powstaje.
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        ports = self._ports(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
        )

        self._make(ports, None).compile().invoke(_initial_state("100.0"))

        record = repository_port.save_prediction.call_args.args[0]
        assert "edge_sigma" not in record

    def test_options_failure_does_not_break_the_prediction(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        options = self._options(0.45)
        options.get_options_flow.side_effect = RuntimeError("Finnhub 403")
        ports = self._ports(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
        )

        final = self._make(ports, options).compile().invoke(_initial_state("100.0"))

        assert final["status"] == "saved"


class TestAlphaFusionInGraph:
    """#14 — score fuzji sygnałów alfa dociera do promptu LLM i do rekordu.
    Bez niego 5 pobieranych źródeł alfa było render-only."""

    def _make(self, ports: dict):
        return create_agent_graph(**ports, threshold=Threshold(Decimal("0.02")))

    def _ports(self, market_port, sentiment_port, news_port, repository_port, ml_port, llm_port):
        return dict(
            market_port=market_port, sentiment_port=sentiment_port,
            news_port=news_port, repository_port=repository_port,
            ml_port=ml_port, llm_port=llm_port,
        )

    def _state(self, score: float | None) -> dict:
        from src.domain.alpha_fusion import AlphaFusionScore

        state: dict = {"symbol": "AAPL", "previous_price": Decimal("100.0")}
        if score is not None:
            state["alpha_fusion"] = AlphaFusionScore(
                score=score,
                contributions={"insider": 0.30, "options": 0.12},
                available_sources=("insider", "options"),
                earnings_confidence=1.0,
            )
        return state

    def test_score_reaches_the_saved_record(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        ports = self._ports(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
        )

        self._make(ports).compile().invoke(self._state(0.42))

        record = repository_port.save_prediction.call_args.args[0]
        assert record["alpha_fusion_score"] == 0.42

    def test_score_reaches_the_llm_prompt(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        ports = self._ports(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
        )

        self._make(ports).compile().invoke(self._state(0.42))

        prompt = llm_port.analyze.call_args.args[0]
        assert "Alpha Fusion" in prompt
        assert "0.42" in prompt

    def test_neutral_score_is_omitted_from_the_prompt(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # Wszystkie flagi alfa off → score 0.0 → prompt identyczny jak dziś.
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        ports = self._ports(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
        )

        self._make(ports).compile().invoke(self._state(None))

        prompt = llm_port.analyze.call_args.args[0]
        assert "Alpha Fusion" not in prompt
        record = repository_port.save_prediction.call_args.args[0]
        assert record["alpha_fusion_score"] == 0.0


class TestAttestationCommitment:
    """#16 — commit-reveal. Przy zapisie predykcji publikujemy SHA-256
    commitment (hash treści + sól); sól ujawniamy dopiero przy reveal."""

    @staticmethod
    def _publisher() -> Mock:
        from src.application.ports import AttestationPublisherPort

        pub = Mock(spec=AttestationPublisherPort)
        pub.publish_commitment.return_value = "public/attestation/commitments.jsonl"
        return pub

    def _ports(self, market_port, sentiment_port, news_port, repository_port, ml_port, llm_port):
        return dict(
            market_port=market_port, sentiment_port=sentiment_port,
            news_port=news_port, repository_port=repository_port,
            ml_port=ml_port, llm_port=llm_port,
        )

    def _run(self, ports, publisher):
        return create_agent_graph(
            **ports,
            threshold=Threshold(Decimal("0.02")),
            attestation_publisher=publisher,
        ).compile().invoke(_initial_state("100.0"))

    def test_without_publisher_no_commitment_columns_are_written(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # attestation_enabled=false → nowe klucze nie mogą trafić do rekordu
        # (PGRST204 przed migracją 024 zabiłby zapis całej predykcji).
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        ports = self._ports(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
        )

        self._run(ports, None)

        record = repository_port.save_prediction.call_args.args[0]
        assert "commitment_hash" not in record
        assert "commitment_salt" not in record

    def test_commitment_hash_and_salt_are_persisted(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        ports = self._ports(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
        )
        publisher = self._publisher()

        self._run(ports, publisher)

        record = repository_port.save_prediction.call_args.args[0]
        assert len(record["commitment_hash"]) == 64
        assert record["commitment_salt"]
        publisher.publish_commitment.assert_called_once()

    def test_published_commitment_never_leaks_the_salt(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # Sedno commit-reveal: przed ujawnieniem hash bez soli jest nieodwracalny.
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        ports = self._ports(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
        )
        publisher = self._publisher()

        self._run(ports, publisher)

        published = publisher.publish_commitment.call_args.args[0]
        assert "salt" not in published
        assert "commitment" in published

    def test_publisher_failure_never_breaks_the_cycle(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        ports = self._ports(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
        )
        publisher = self._publisher()
        publisher.publish_commitment.side_effect = OSError("disk full")

        final = self._run(ports, publisher)

        assert final["status"] == "saved"


class TestPaidCallMeter:
    """FinOps transparency — agent raportuje własny rachunek. Licznik zbiera
    dokładnie te wywołania, które już dziś loguje `_log_paid_call`."""

    def _ports(self, market_port, sentiment_port, news_port, repository_port, ml_port, llm_port):
        return dict(
            market_port=market_port, sentiment_port=sentiment_port,
            news_port=news_port, repository_port=repository_port,
            ml_port=ml_port, llm_port=llm_port,
        )

    def test_gated_cycle_is_free(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # Sedno FinOps repo: cykl odcięty bramką volatility nie kosztuje nic.
        from collections import Counter

        market_port.get_current_price.return_value = Money(Decimal("100.5"))
        meter: Counter[str] = Counter()
        ports = self._ports(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
        )

        create_agent_graph(
            **ports, threshold=Threshold(Decimal("0.02")), paid_call_meter=meter
        ).compile().invoke(_initial_state("100.0"))

        assert sum(meter.values()) == 0

    def test_full_cycle_counts_llm_sentiment_and_news(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        from collections import Counter

        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        meter: Counter[str] = Counter()
        ports = self._ports(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
        )

        create_agent_graph(
            **ports, threshold=Threshold(Decimal("0.02")), paid_call_meter=meter
        ).compile().invoke(_initial_state("100.0"))

        assert meter["llm"] >= 1
        assert meter["sentiment"] == 1
        assert meter["news"] == 1

    def test_meter_is_optional(
        self, market_port, sentiment_port, news_port,
        repository_port, ml_port, llm_port,
    ):
        # Bez licznika graf działa dokładnie jak dotąd.
        market_port.get_current_price.return_value = Money(Decimal("90.0"))
        _setup_full_analysis_mocks(
            sentiment_port, news_port, repository_port, ml_port, llm_port,
        )
        ports = self._ports(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
        )

        final = create_agent_graph(
            **ports, threshold=Threshold(Decimal("0.02"))
        ).compile().invoke(_initial_state("100.0"))

        assert final["status"] == "saved"
