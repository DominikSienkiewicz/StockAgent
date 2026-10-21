from collections import Counter
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application.agent_graph import AgentGraphDeps
from src.application.ports import (
    EmbeddingPort,
    LLMPort,
    MarketDataPort,
    MLPredictionPort,
    NewsPort,
    RepositoryPort,
    SentimentPort,
)
from src.application.use_cases.analyze_market import AnalyzeMarketUseCase
from src.domain.prediction import Prediction, TrendDirection
from src.domain.value_objects import Money, Threshold


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
    # reflect_node działa w każdym cyklu — domyślnie brak historii do oceny.
    repo.get_unverified_prediction.return_value = None
    repo.get_last_prediction_price.return_value = None
    return repo


@pytest.fixture
def ml_port() -> Mock:
    return Mock(spec=MLPredictionPort)


@pytest.fixture
def llm_port() -> Mock:
    return Mock(spec=LLMPort)


@pytest.fixture
def use_case(
    market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
) -> AnalyzeMarketUseCase:
    return AnalyzeMarketUseCase(AgentGraphDeps(
        market_port=market_port,
        sentiment_port=sentiment_port,
        news_port=news_port,
        repository_port=repository_port,
        ml_port=ml_port,
        llm_port=llm_port,
        threshold=Threshold(Decimal("0.02")),
    ))


class TestRun:
    def test_fetches_previous_price_from_repository(
        self, use_case, market_port, repository_port
    ):
        repository_port.get_last_price.return_value = Money(Decimal("100.0"))
        market_port.get_current_price.return_value = Money(Decimal("99.5"))  # -0.5%

        use_case.run("AAPL")

        repository_port.get_last_price.assert_called_once_with("AAPL")

    def test_skips_analysis_on_cold_start_no_previous_price(
        self, use_case, market_port, repository_port, sentiment_port
    ):
        # Brak historii — repo zwraca None
        repository_port.get_last_price.return_value = None
        market_port.get_current_price.return_value = Money(Decimal("100.0"))

        result = use_case.run("AAPL")

        # Cold start: nie ma do czego porównać → ignorujemy (delta=0)
        assert result["status"] == "ignored"
        sentiment_port.get_social_score.assert_not_called()

    def test_runs_full_graph_when_volatility_above_threshold(
        self,
        use_case,
        market_port,
        sentiment_port,
        news_port,
        repository_port,
        ml_port,
        llm_port,
    ):
        repository_port.get_last_price.return_value = Money(Decimal("100.0"))
        market_port.get_current_price.return_value = Money(Decimal("90.0"))  # -10%
        repository_port.get_unverified_prediction.return_value = None
        sentiment_port.get_social_score.return_value = {"galaxy_score": 85}
        news_port.get_news_context.return_value = [{"title": "Reuters story"}]
        llm_port.analyze.return_value = {
            "trend_direction": "BEARISH",
            "confidence_score": 0.8,
            "target_price_12h": 88.0,
            "reasoning": "macro headwinds",
        }
        ml_port.predict.return_value = Money(Decimal("89.0"))
        repository_port.save_prediction.return_value = "new-uuid"

        result = use_case.run("AAPL")

        assert result["status"] == "saved"
        assert result["prediction_id"] == "new-uuid"
        assert result["delta"] == Decimal("-0.10")
        sentiment_port.get_social_score.assert_called_once_with("AAPL")
        news_port.get_news_context.assert_called_once_with("AAPL")

    def test_ignores_minor_change_below_threshold(
        self, use_case, market_port, repository_port, sentiment_port, ml_port
    ):
        repository_port.get_last_price.return_value = Money(Decimal("100.0"))
        market_port.get_current_price.return_value = Money(Decimal("100.5"))  # +0.5%

        result = use_case.run("AAPL")

        assert result["status"] == "ignored"
        sentiment_port.get_social_score.assert_not_called()
        ml_port.predict.assert_not_called()


def test_analyze_market_passes_council_port_to_graph():
    """AnalyzeMarketUseCase forwards council_port — council_node gets invoked."""
    from decimal import Decimal
    from unittest.mock import Mock

    from src.application.ports import (
        AdvisoryCouncilPort,
        LLMPort,
        MarketDataPort,
        MLPredictionPort,
        NewsPort,
        RepositoryPort,
        SentimentPort,
    )
    from src.application.use_cases.analyze_market import AnalyzeMarketUseCase
    from src.domain.council import CouncilVerdict, InvestorOpinion
    from src.domain.value_objects import Money, Threshold

    market = Mock(spec=MarketDataPort)
    market.get_current_price.return_value = Money(Decimal("180.00"))
    sentiment = Mock(spec=SentimentPort)
    sentiment.get_social_score.return_value = {
        "av_sentiment_score": 0.5, "av_sentiment_label": "Bullish",
        "av_relevance_avg": 0.7, "news_volume_24h": 10, "high_relevance_count": 3,
    }
    news = Mock(spec=NewsPort)
    news.get_news_context.return_value = [{"title": "Apple gains"}]
    repo = Mock(spec=RepositoryPort)
    repo.get_unverified_prediction.return_value = None
    repo.get_last_prediction_price.return_value = Money(Decimal("100.0"))
    repo.save_prediction.return_value = "pred-123"
    ml = Mock(spec=MLPredictionPort)
    ml.is_trained = True
    ml.predict.return_value = Money(Decimal("185.00"))
    llm = Mock(spec=LLMPort)
    llm.analyze.return_value = {
        "trend_direction": "BULLISH", "confidence_score": 0.8,
        "av_agreement": 0.9, "target_price_12h": 185.0, "reasoning": "ok",
    }
    council_port = Mock(spec=AdvisoryCouncilPort)
    verdict = CouncilVerdict(
        final_recommendation="BUY",
        consensus_strength=0.7,
        summary="Rada kupuje.",
        dissenting_views=[],
        investor_opinions=[
            InvestorOpinion(
                investor_name="Warren Buffett",
                recommendation="BUY",
                confidence=0.9,
                reasoning="moat",
                key_factors=["moat"],
            )
        ],
    )
    council_port.analyze.return_value = verdict

    use_case = AnalyzeMarketUseCase(AgentGraphDeps(
        market_port=market,
        sentiment_port=sentiment,
        news_port=news,
        repository_port=repo,
        ml_port=ml,
        llm_port=llm,
        threshold=Threshold(Decimal("0.02")),
        council_port=council_port,
    ))
    # previous_price much lower to trigger volatility gate
    repo.get_last_price.return_value = Money(Decimal("170.00"))
    use_case.run("AAPL")
    council_port.analyze.assert_called_once()


class TestGraphCompiledOnce:
    """Graf kompilujemy RAZ przy konstrukcji, nie per `run()`.

    Przy 43 symbolach na cykl rekompilacja w każdym `run()` to 43× zbędna
    praca. Kompilacja jest deterministyczna (zależy tylko od topologii i
    wstrzykniętych portów), więc skompilowaną aplikację można reużyć.
    """

    def test_compile_called_once_across_multiple_runs(
        self, monkeypatch, repository_port, market_port
    ):
        from unittest.mock import Mock

        from src.application.use_cases import analyze_market as module
        from src.domain.value_objects import Money, Threshold

        repository_port.get_last_price.return_value = Money(Decimal("100.0"))
        market_port.get_current_price.return_value = Money(Decimal("100.0"))

        compiled_app = Mock()
        compiled_app.invoke.return_value = {"status": "ignored"}
        fake_workflow = Mock()
        fake_workflow.compile.return_value = compiled_app
        monkeypatch.setattr(
            module, "create_agent_graph", lambda _deps: fake_workflow
        )

        use_case = module.AnalyzeMarketUseCase(AgentGraphDeps(
            market_port=market_port,
            sentiment_port=Mock(),
            news_port=Mock(),
            repository_port=repository_port,
            ml_port=Mock(),
            llm_port=Mock(),
            threshold=Threshold(Decimal("0.02")),
        ))
        use_case.run("AAPL")
        use_case.run("MSFT")
        use_case.run("NVDA")

        fake_workflow.compile.assert_called_once()
        assert compiled_app.invoke.call_count == 3


class TestPaidCallMeter:
    """`use_case.paid_calls` musi być TYM SAMYM egzemplarzem licznika, który
    inkrementują węzły grafu.

    To niezmiennik okablowania, nie detal implementacyjny: `main_agent` liczy
    z niego koszt cyklu (`estimate_cycle_cost(dict(use_case.paid_calls))`).
    Gdyby graf dostał inny egzemplarz niż ten wystawiony na use case'ie,
    raport FinOps pokazywałby 0 zł mimo realnych płatnych wywołań — cicho,
    bez żadnego błędu.
    """

    def _arrange_paid_cycle(
        self, market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
    ) -> None:
        repository_port.get_last_price.return_value = Money(Decimal("100.0"))
        market_port.get_current_price.return_value = Money(Decimal("90.0"))  # -10%
        sentiment_port.get_social_score.return_value = {"galaxy_score": 85}
        news_port.get_news_context.return_value = [{"title": "Reuters story"}]
        llm_port.analyze.return_value = {
            "trend_direction": "BEARISH",
            "confidence_score": 0.8,
            "target_price_12h": 88.0,
            "reasoning": "macro headwinds",
        }
        ml_port.predict.return_value = Money(Decimal("89.0"))
        repository_port.save_prediction.return_value = "new-uuid"

    def test_counts_paid_calls_when_caller_supplied_no_meter(
        self,
        use_case,
        market_port,
        sentiment_port,
        news_port,
        repository_port,
        ml_port,
        llm_port,
    ):
        self._arrange_paid_cycle(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
        )

        use_case.run("AAPL")

        assert use_case.paid_calls["sentiment"] == 1
        assert use_case.paid_calls["news"] == 1
        assert use_case.paid_calls["llm"] == 1

    def test_counts_into_the_meter_the_caller_supplied(
        self,
        market_port,
        sentiment_port,
        news_port,
        repository_port,
        ml_port,
        llm_port,
    ):
        meter: Counter[str] = Counter()
        use_case = AnalyzeMarketUseCase(AgentGraphDeps(
            market_port=market_port,
            sentiment_port=sentiment_port,
            news_port=news_port,
            repository_port=repository_port,
            ml_port=ml_port,
            llm_port=llm_port,
            threshold=Threshold(Decimal("0.02")),
            paid_call_meter=meter,
        ))
        self._arrange_paid_cycle(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port
        )

        use_case.run("AAPL")

        # Ten sam obiekt, nie kopia — wołający trzyma referencję i na niej raportuje.
        assert use_case.paid_calls is meter
        assert meter["llm"] == 1


class TestMlModelLoaded:
    """MAIL-01 (G03): raport pyta use case, czy XGBoost ma wagi — inaczej karta
    kondycji twierdziłaby, że jedziemy na sprawdzonym modelu, którego nie ma."""

    def test_reports_cold_start_when_model_untrained(self, use_case, ml_port):
        ml_port.is_trained = False
        assert use_case.ml_model_loaded is False

    def test_reports_loaded_model_when_trained(self, use_case, ml_port):
        ml_port.is_trained = True
        assert use_case.ml_model_loaded is True


class TestRequiredPredictionColumns:
    """FIN-05: use case podaje DI listę kolumn zapisu liczoną z tych samych deps."""

    def test_matches_the_graph_rule_for_its_deps(self, use_case) -> None:
        from src.application.agent_graph import required_prediction_columns

        assert use_case.required_prediction_columns() == required_prediction_columns(
            use_case._deps
        )
        assert "symbol" in use_case.required_prediction_columns()


class TestThresholdInjection:
    def test_respects_custom_threshold(
        self,
        market_port,
        sentiment_port,
        news_port,
        repository_port,
        ml_port,
        llm_port,
    ):
        use_case = AnalyzeMarketUseCase(AgentGraphDeps(
            market_port=market_port,
            sentiment_port=sentiment_port,
            news_port=news_port,
            repository_port=repository_port,
            ml_port=ml_port,
            llm_port=llm_port,
            threshold=Threshold(Decimal("0.20")),  # 20% próg
        ))
        repository_port.get_last_price.return_value = Money(Decimal("100.0"))
        market_port.get_current_price.return_value = Money(Decimal("90.0"))  # -10%

        result = use_case.run("AAPL")

        assert result["status"] == "ignored"
        sentiment_port.get_social_score.assert_not_called()


class TestStaleReferenceRebaseline:
    """FIN-14 (decyzja 2026-10-19: re-baseline): referencja ceny starsza niż
    `REFERENCE_MAX_AGE_HOURS` to nie poprzedni cykl, tylko stan sprzed pauzy.
    Ruch od niej otworzyłby bramkę wszystkim symbolom naraz — cykl zapisuje
    więc nowy snapshot BEZ analizy: zero LLM, Alpha Vantage i embeddingów."""

    _NOW = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)

    def _use_case(
        self, market_port, sentiment_port, news_port, repository_port, ml_port, llm_port,
        embedding_port,
    ) -> AnalyzeMarketUseCase:
        return AnalyzeMarketUseCase(AgentGraphDeps(
            market_port=market_port,
            sentiment_port=sentiment_port,
            news_port=news_port,
            repository_port=repository_port,
            ml_port=ml_port,
            llm_port=llm_port,
            threshold=Threshold(Decimal("0.02")),
            embedding_port=embedding_port,
        ))

    def _arrange_big_move(self, market_port, repository_port, *, reference_age: timedelta) -> None:
        reference = Money(Decimal("100.0"))
        repository_port.get_last_price.return_value = reference
        repository_port.get_last_price_snapshot.return_value = (
            reference, self._NOW - reference_age,
        )
        market_port.get_current_price.return_value = Money(Decimal("90.0"))  # -10% ≥ próg

    def test_month_old_reference_saves_snapshot_without_paid_calls(
        self, market_port, sentiment_port, news_port, repository_port, ml_port, llm_port,
    ):
        embedding_port = Mock(spec=EmbeddingPort)
        use_case = self._use_case(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port,
            embedding_port,
        )
        self._arrange_big_move(market_port, repository_port, reference_age=timedelta(days=30))

        result = use_case.run("AAPL", now=self._NOW)

        assert result["status"] == "ignored"
        assert result["stale_reference"] is True
        assert result["reference_at"] == self._NOW - timedelta(days=30)
        sentiment_port.get_social_score.assert_not_called()
        news_port.get_news_context.assert_not_called()
        llm_port.analyze.assert_not_called()
        llm_port.analyze_mistake.assert_not_called()
        embedding_port.embed.assert_not_called()
        ml_port.predict.assert_not_called()
        repository_port.save_prediction.assert_not_called()
        assert sum(use_case.paid_calls.values()) == 0
        # Nowy punkt odniesienia: następny cykl porówna się do dzisiejszej ceny.
        repository_port.save_price_snapshot.assert_called_once_with(
            "AAPL", Money(Decimal("90.0"))
        )

    def test_month_old_reference_skips_paid_diagnosis_but_still_scores_prior(
        self, market_port, sentiment_port, news_port, repository_port, ml_port, llm_port,
    ):
        # Błędna wcześniejsza predykcja + ruch ≥ progu od stęchłej referencji:
        # tania ocena leci, płatna diagnoza LLM — nie (ta sama bramka).
        repository_port.get_unverified_prediction.return_value = Prediction(
            id="prior-uuid",
            symbol="AAPL",
            predicted_trend=TrendDirection.BULLISH,
            price_at_prediction=Decimal("100.0"),
            predicted_target_price=Decimal("105.0"),
        )
        use_case = self._use_case(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port,
            None,
        )
        self._arrange_big_move(market_port, repository_port, reference_age=timedelta(days=30))

        use_case.run("AAPL", now=self._NOW)

        llm_port.analyze_mistake.assert_not_called()
        repository_port.update_prediction_accuracy.assert_called_once()

    def test_reference_from_previous_cycle_still_opens_the_gate(
        self, market_port, sentiment_port, news_port, repository_port, ml_port, llm_port,
    ):
        sentiment_port.get_social_score.return_value = {"galaxy_score": 85}
        news_port.get_news_context.return_value = [{"title": "Reuters story"}]
        llm_port.analyze.return_value = {
            "trend_direction": "BEARISH",
            "confidence_score": 0.8,
            "target_price_12h": 88.0,
            "reasoning": "macro headwinds",
        }
        ml_port.predict.return_value = Money(Decimal("89.0"))
        repository_port.save_prediction.return_value = "new-uuid"
        use_case = self._use_case(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port,
            None,
        )
        self._arrange_big_move(market_port, repository_port, reference_age=timedelta(hours=24))

        result = use_case.run("AAPL", now=self._NOW)

        assert result["status"] == "saved"
        assert result["stale_reference"] is False
        sentiment_port.get_social_score.assert_called_once_with("AAPL")

    def test_cold_start_reads_no_reference_age(
        self, use_case, market_port, repository_port
    ):
        # Cold-start bez zmian: brak ceny → brak referencji → brak odczytu wieku.
        repository_port.get_last_price.return_value = None
        market_port.get_current_price.return_value = Money(Decimal("100.0"))

        result = use_case.run("AAPL", now=self._NOW)

        assert result["status"] == "ignored"
        assert result["stale_reference"] is False
        assert result["reference_at"] is None
        repository_port.get_last_price_snapshot.assert_not_called()

    def test_unreadable_reference_age_keeps_the_gate_as_before(
        self, market_port, sentiment_port, news_port, repository_port, ml_port, llm_port,
    ):
        # Odczyt wieku jest best-effort: błąd Supabase nie wywala symbolu,
        # a bramka działa jak przed FIN-14 (sam próg volatility).
        repository_port.get_last_price.return_value = Money(Decimal("100.0"))
        repository_port.get_last_price_snapshot.side_effect = RuntimeError("Supabase 503")
        market_port.get_current_price.return_value = Money(Decimal("100.5"))  # +0.5%
        use_case = self._use_case(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port,
            None,
        )

        result = use_case.run("AAPL", now=self._NOW)

        assert result["status"] == "ignored"
        assert result["stale_reference"] is False
        assert result["reference_at"] is None


class TestExperimentArms:
    """EXP-05: use case przydziela ramiona AKTYWNYCH eksperymentów per
    symbol-dzień UTC cyklu i podaje je grafowi w stanie. Bez aktywnego
    eksperymentu stan jest taki jak przed EXP-05."""

    _NOW = datetime(2026, 11, 2, 21, 30, tzinfo=UTC)

    @staticmethod
    def _experiment(*, enabled: bool):
        from datetime import date

        from src.domain.experiment import ARM_NO_MEMORY, Experiment, ExperimentMetric

        return Experiment(
            id="EXP-03",
            hypothesis="Lekcje i analogi poprawiają trafność.",
            arms=("memory", ARM_NO_MEMORY),
            start=date(2026, 11, 1),
            end=date(2026, 12, 31),
            primary_metric=ExperimentMetric.HIT_RATE,
            enabled=enabled,
        )

    def _result(
        self, market_port, sentiment_port, news_port, repository_port, ml_port, llm_port,
        *, enabled: bool,
    ) -> dict:
        use_case = AnalyzeMarketUseCase(AgentGraphDeps(
            market_port=market_port,
            sentiment_port=sentiment_port,
            news_port=news_port,
            repository_port=repository_port,
            ml_port=ml_port,
            llm_port=llm_port,
            threshold=Threshold(Decimal("0.02")),
            experiments=(self._experiment(enabled=enabled),),
        ))
        repository_port.get_last_price.return_value = Money(Decimal("100.0"))
        market_port.get_current_price.return_value = Money(Decimal("100.5"))  # +0.5%
        return use_case.run("AAPL", now=self._NOW)

    def test_active_experiment_puts_the_assigned_arm_into_the_state(
        self, market_port, sentiment_port, news_port, repository_port, ml_port, llm_port,
    ):
        from datetime import date

        from src.domain.experiment import assign_arm

        result = self._result(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port,
            enabled=True,
        )

        expected = assign_arm("EXP-03", "AAPL", date(2026, 11, 2), ("memory", "no_memory"))
        assert result["experiment_arms"] == {"EXP-03": expected}

    def test_inactive_experiment_leaves_the_state_untouched(
        self, market_port, sentiment_port, news_port, repository_port, ml_port, llm_port,
    ):
        result = self._result(
            market_port, sentiment_port, news_port, repository_port, ml_port, llm_port,
            enabled=False,
        )

        assert "experiment_arms" not in result


class TestCycleTelemetry:
    """ARCH-11: use case wystawia TEN SAM `CycleTelemetry`, który dostają węzły,
    a `paid_calls` zostaje żywym Counterem z tego obiektu."""

    def test_exposes_the_graph_telemetry_and_its_live_counter(
        self, market_port, sentiment_port, news_port, repository_port, ml_port, llm_port,
    ):
        from src.application.cycle_telemetry import CycleTelemetry

        telemetry = CycleTelemetry()
        use_case = AnalyzeMarketUseCase(AgentGraphDeps(
            market_port=market_port,
            sentiment_port=sentiment_port,
            news_port=news_port,
            repository_port=repository_port,
            ml_port=ml_port,
            llm_port=llm_port,
            threshold=Threshold(Decimal("0.02")),
            telemetry=telemetry,
        ))
        repository_port.get_last_price.return_value = Money(Decimal("100.0"))
        market_port.get_current_price.return_value = Money(Decimal("99.5"))  # -0.5%

        use_case.run("AAPL")

        assert use_case.telemetry is telemetry
        assert use_case.paid_calls is telemetry.paid_calls
        assert telemetry.snapshot().gated == 1


class TestTwoPhaseCycle:
    """EXP-27: use case wystawia fazę bramki (`run_gate`) i fazę płatną
    (`paid_phase`) na porty feedu podane dopiero po fazie 1 — DI buduje je
    z klientem AV tylko dla symboli za bramką."""

    def _big_move(self, market_port, repository_port, llm_port, ml_port) -> None:
        repository_port.get_last_price.return_value = Money(Decimal("100.0"))
        repository_port.get_last_price_snapshot.return_value = None
        market_port.get_current_price.return_value = Money(Decimal("95.0"))  # -5%
        llm_port.analyze.return_value = {
            "trend_direction": "BEARISH",
            "confidence_score": 0.7,
            "target_price_12h": 94.0,
            "reasoning": "r",
        }
        ml_port.predict.return_value = Money(Decimal("94.5"))
        repository_port.save_prediction.return_value = "uuid-2p"

    def test_flat_symbol_ends_in_the_gate_phase(
        self, use_case, market_port, repository_port, sentiment_port
    ):
        repository_port.get_last_price.return_value = Money(Decimal("100.0"))
        repository_port.get_last_price_snapshot.return_value = None
        market_port.get_current_price.return_value = Money(Decimal("100.5"))

        outcome = use_case.run_gate("AAPL")

        assert outcome.passed is False
        assert outcome.state["status"] == "ignored"
        sentiment_port.get_social_score.assert_not_called()

    def test_paid_phase_uses_the_feeds_bound_after_the_gate(
        self, use_case, market_port, repository_port, sentiment_port, news_port,
        llm_port, ml_port,
    ):
        self._big_move(market_port, repository_port, llm_port, ml_port)
        phase_two_sentiment = Mock(spec=SentimentPort)
        phase_two_sentiment.get_social_score.return_value = {"av_sentiment_score": 0.1}
        phase_two_news = Mock(spec=NewsPort)
        phase_two_news.get_news_context.return_value = [{"title": "Headline"}]

        outcome = use_case.run_gate("AAPL")
        paid = use_case.paid_phase(sentiment_port=phase_two_sentiment, news_port=phase_two_news)
        result = paid.run(outcome.state, peer_context=(("MSFT", "BULLISH"),))

        assert outcome.passed is True
        assert result["status"] == "saved"
        phase_two_sentiment.get_social_score.assert_called_once_with("AAPL")
        phase_two_news.get_news_context.assert_called_once_with("AAPL")
        sentiment_port.get_social_score.assert_not_called()
        news_port.get_news_context.assert_not_called()
        # Contagion fazy 2: peery z werdyktami z tego cyklu trafiają do stanu.
        assert result["peer_context"] == (("MSFT", "BULLISH"),)
        # Jedna telemetria cyklu dla obu faz.
        assert use_case.telemetry.snapshot().passed == 1
        assert use_case.paid_calls["llm"] == 1

    def test_gate_phase_compiles_once_for_the_whole_cycle(
        self, use_case, market_port, repository_port, monkeypatch
    ):
        from src.application.use_cases import analyze_market as module

        repository_port.get_last_price.return_value = Money(Decimal("100.0"))
        repository_port.get_last_price_snapshot.return_value = None
        market_port.get_current_price.return_value = Money(Decimal("100.0"))
        compiled = []
        real_factory = module.create_gate_graph

        def counting_factory(deps):
            compiled.append(deps)
            return real_factory(deps)

        monkeypatch.setattr(module, "create_gate_graph", counting_factory)

        for symbol in ("AAPL", "MSFT", "NVDA"):
            use_case.run_gate(symbol)

        assert len(compiled) == 1
