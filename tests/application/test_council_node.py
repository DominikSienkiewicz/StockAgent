# tests/application/test_council_node.py
from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

from src.application.agent_graph import create_agent_graph
from src.application.ports import (
    AdvisoryCouncilPort,
    LLMPort,
    MarketDataPort,
    MLPredictionPort,
    NewsPort,
    RepositoryPort,
    SentimentPort,
)
from src.domain.asset import Asset
from src.domain.council import CouncilVerdict, InvestorOpinion
from src.domain.value_objects import AssetType, Money, Threshold


def _make_verdict() -> CouncilVerdict:
    return CouncilVerdict(
        final_recommendation="BUY",
        consensus_strength=0.73,
        summary="Rada radzi kupować.",
        dissenting_views=["Soros: ryzykowne"],
        investor_opinions=[
            InvestorOpinion(
                investor_name="Warren Buffett",
                recommendation="BUY",
                confidence=0.9,
                reasoning="Silna fosa.",
                key_factors=["moat"],
            )
        ],
    )


def _make_graph_with_repo(
    council_port: AdvisoryCouncilPort | None = None,
    council_threshold: Threshold | None = None,
):
    market = Mock(spec=MarketDataPort)
    market.get_current_price.return_value = Money(Decimal("180.00"))
    sentiment = Mock(spec=SentimentPort)
    sentiment.get_social_score.return_value = {
        "av_sentiment_score": 0.5,
        "av_sentiment_label": "Bullish",
        "av_relevance_avg": 0.7,
        "news_volume_24h": 10,
        "high_relevance_count": 3,
    }
    news = Mock(spec=NewsPort)
    news.get_news_context.return_value = [{"title": "Apple gains", "source": "Reuters"}]
    repo = Mock(spec=RepositoryPort)
    repo.get_unverified_prediction.return_value = None
    repo.get_last_prediction_price.return_value = Money(Decimal("100.0"))
    repo.save_prediction.return_value = "pred-123"
    ml = Mock(spec=MLPredictionPort)
    ml.is_trained = True
    ml.predict.return_value = Money(Decimal("185.00"))
    llm = Mock(spec=LLMPort)
    llm.analyze.return_value = {
        "trend_direction": "BULLISH",
        "confidence_score": 0.8,
        "av_agreement": 0.9,
        "target_price_12h": 185.0,
        "reasoning": "Strong momentum.",
    }
    graph = create_agent_graph(
        market_port=market,
        sentiment_port=sentiment,
        news_port=news,
        repository_port=repo,
        ml_port=ml,
        llm_port=llm,
        threshold=Threshold(Decimal("0.02")),
        council_port=council_port,
        council_threshold=council_threshold,
    )
    return graph, repo


def _make_graph(
    council_port: AdvisoryCouncilPort | None = None,
    council_threshold: Threshold | None = None,
):
    return _make_graph_with_repo(council_port, council_threshold)[0]


def _make_crypto_graph(
    council_port: AdvisoryCouncilPort,
    *,
    council_threshold: Threshold,
    crypto_council_threshold: Threshold,
):
    """Wariant grafu dla aktywa CRYPTO z ROZRÓŻNIALNYMI progami rady:
    stock `council_threshold` vs `crypto_council_threshold`. Próg główny
    volatility ustawiony nisko (0.5%), żeby zawsze przepuścił i izolować
    zachowanie bramki RADY (council_node), a nie głównej bramki."""
    market = Mock(spec=MarketDataPort)
    market.get_current_price.return_value = Money(Decimal("180.00"))
    sentiment = Mock(spec=SentimentPort)
    sentiment.get_social_score.return_value = {
        "av_sentiment_score": 0.5,
        "av_sentiment_label": "Bullish",
        "av_relevance_avg": 0.7,
        "news_volume_24h": 10,
        "high_relevance_count": 3,
    }
    news = Mock(spec=NewsPort)
    news.get_news_context.return_value = [{"title": "BTC gains", "source": "Reuters"}]
    repo = Mock(spec=RepositoryPort)
    repo.get_unverified_prediction.return_value = None
    repo.get_last_prediction_price.return_value = Money(Decimal("100.0"))
    repo.save_prediction.return_value = "pred-123"
    ml = Mock(spec=MLPredictionPort)
    ml.is_trained = True
    ml.predict.return_value = Money(Decimal("185.00"))
    llm = Mock(spec=LLMPort)
    llm.analyze.return_value = {
        "trend_direction": "BULLISH",
        "confidence_score": 0.8,
        "av_agreement": 0.9,
        "target_price_12h": 185.0,
        "reasoning": "Strong momentum.",
    }
    graph = create_agent_graph(
        market_port=market,
        sentiment_port=sentiment,
        news_port=news,
        repository_port=repo,
        ml_port=ml,
        llm_port=llm,
        # Główna bramka nisko (0.5%) — żeby krypto zawsze ją przeszło.
        threshold=Threshold(Decimal("0.005")),
        crypto_threshold=Threshold(Decimal("0.005")),
        council_port=council_port,
        council_threshold=council_threshold,
        crypto_council_threshold=crypto_council_threshold,
    )
    return graph, repo


class TestCouncilNodeIntegration:
    def test_council_verdict_in_state_when_port_provided(self):
        council_port = Mock(spec=AdvisoryCouncilPort)
        council_port.analyze.return_value = _make_verdict()
        graph = _make_graph(council_port=council_port)
        app = graph.compile()
        result = app.invoke({"symbol": "AAPL", "previous_price": Decimal("173.00")})
        assert result.get("council_verdict") is not None
        assert result["council_verdict"].final_recommendation == "BUY"

    def test_council_called_with_symbol(self):
        council_port = Mock(spec=AdvisoryCouncilPort)
        council_port.analyze.return_value = _make_verdict()
        graph = _make_graph(council_port=council_port)
        app = graph.compile()
        app.invoke({"symbol": "AAPL", "previous_price": Decimal("173.00")})
        council_port.analyze.assert_called_once()
        args = council_port.analyze.call_args
        assert args[0][0] == "AAPL"

    def test_council_verdict_none_when_no_port(self):
        graph = _make_graph(council_port=None)
        app = graph.compile()
        result = app.invoke({"symbol": "AAPL", "previous_price": Decimal("173.00")})
        assert result.get("council_verdict") is None

    def test_council_not_called_when_volatility_gate_blocks(self):
        council_port = Mock(spec=AdvisoryCouncilPort)
        council_port.analyze.return_value = _make_verdict()
        graph = _make_graph(council_port=council_port)
        app = graph.compile()
        # previous_price == current_price → delta=0 → gate blocks → council_node not reached
        result = app.invoke({"symbol": "AAPL", "previous_price": Decimal("180.00")})
        council_port.analyze.assert_not_called()
        assert result.get("council_verdict") is None


class TestCouncilVolatilityThreshold:
    """Osobny próg dla rady — pozwala odpalić główną analizę (Δ ≥ 2%) bez
    odpalania kosztownej rady (15 wywołań LLM) dla średnich zmienności.
    Rada odpala się dopiero przy Δ ≥ council_threshold (np. 5%)."""

    def test_council_skipped_when_delta_below_council_threshold(self):
        council_port = Mock(spec=AdvisoryCouncilPort)
        council_port.analyze.return_value = _make_verdict()
        # Główny próg 2%, próg rady 5%. Current=180, prev=176 → Δ≈+2.27%.
        # Główna analiza powinna lecieć, rada powinna być pominięta.
        graph = _make_graph(
            council_port=council_port,
            council_threshold=Threshold(Decimal("0.05")),
        )
        app = graph.compile()
        result = app.invoke({"symbol": "AAPL", "previous_price": Decimal("176.00")})
        council_port.analyze.assert_not_called()
        assert result.get("council_verdict") is None
        # Główna analiza zakończona zapisem (rada to opcjonalny dodatek).
        assert result.get("status") == "saved"

    def test_council_called_when_delta_above_council_threshold(self):
        council_port = Mock(spec=AdvisoryCouncilPort)
        council_port.analyze.return_value = _make_verdict()
        # Current=180, prev=160 → Δ=+12.5%, ponad 5%.
        graph = _make_graph(
            council_port=council_port,
            council_threshold=Threshold(Decimal("0.05")),
        )
        app = graph.compile()
        result = app.invoke({"symbol": "AAPL", "previous_price": Decimal("160.00")})
        council_port.analyze.assert_called_once()
        assert result.get("council_verdict") is not None

    def test_council_votes_persisted_after_save_prediction(self):
        """save_node po zapisie predykcji powinien zapisać strukturalne głosy
        rady (po jednej linii per inwestor) do osobnej tabeli, z FK do
        prediction_id. Audit trail niezależny od JSONB blobu."""
        council_port = Mock(spec=AdvisoryCouncilPort)
        council_port.analyze.return_value = _make_verdict()
        graph, repo = _make_graph_with_repo(council_port=council_port)
        app = graph.compile()
        app.invoke({"symbol": "AAPL", "previous_price": Decimal("160.00")})

        repo.save_council_votes.assert_called_once()
        call = repo.save_council_votes.call_args
        # Akceptujemy zarówno kwargs jak i pozycyjne
        all_args = dict(call.kwargs)
        if call.args:
            # pozycyjne: (prediction_id, symbol, votes)
            for key, val in zip(
                ["prediction_id", "symbol", "votes"], call.args, strict=False
            ):
                all_args.setdefault(key, val)
        assert all_args["prediction_id"] == "pred-123"
        assert all_args["symbol"] == "AAPL"
        assert len(all_args["votes"]) == 1
        assert all_args["votes"][0].investor_name == "Warren Buffett"

    def test_no_council_votes_persisted_when_council_skipped(self):
        # Gdy rada nie odpaliła (próg blokuje), nie ma głosów do zapisu.
        council_port = Mock(spec=AdvisoryCouncilPort)
        council_port.analyze.return_value = _make_verdict()
        graph, repo = _make_graph_with_repo(
            council_port=council_port,
            council_threshold=Threshold(Decimal("0.5")),  # bardzo wysoki próg
        )
        app = graph.compile()
        app.invoke({"symbol": "AAPL", "previous_price": Decimal("176.00")})
        repo.save_council_votes.assert_not_called()

    def test_council_called_when_threshold_is_none(self):
        # Brak osobnego progu → rada zawsze leci gdy główna bramka przepuści.
        council_port = Mock(spec=AdvisoryCouncilPort)
        council_port.analyze.return_value = _make_verdict()
        graph = _make_graph(council_port=council_port, council_threshold=None)
        app = graph.compile()
        # Δ≈+2.27%, ponad główny próg 2%
        app.invoke({"symbol": "AAPL", "previous_price": Decimal("176.00")})
        council_port.analyze.assert_called_once()


class TestCryptoCouncilVolatilityThreshold:
    """#38: bramka rady (council_node) wybiera próg per asset_type przez
    _pick_threshold. Dla CRYPTO MUSI użyć `crypto_council_threshold`, nie
    `council_threshold` akcji — main_agent wiąże osobny próg dla krypto, bo ma
    ono inną natywną zmienność. Bez tego pokrycia regresja (krypto spada do
    progu akcji) przeszłaby niezauważona. Progi celowo ROZRÓŻNIALNE."""

    def test_crypto_council_skipped_using_crypto_threshold_not_stock(self):
        # stock council_threshold=2%, crypto_council_threshold=8%.
        # current=180, prev=170 → Δ≈+5.88%. To JEST ponad próg akcji (2%), więc
        # gdyby council_node błędnie użył progu akcji → rada by ODPALIŁA.
        # Skoro używa progu KRYPTO (8%), Δ 5.88% < 8% → rada POMINIĘTA.
        council_port = Mock(spec=AdvisoryCouncilPort)
        council_port.analyze.return_value = _make_verdict()
        graph, _ = _make_crypto_graph(
            council_port,
            council_threshold=Threshold(Decimal("0.02")),
            crypto_council_threshold=Threshold(Decimal("0.08")),
        )
        app = graph.compile()
        asset = Asset(symbol="BTC", asset_type=AssetType.CRYPTO)
        result = app.invoke({
            "symbol": "BTC",
            "previous_price": Decimal("170.00"),
            "asset": asset,
        })
        # Sedno: rada pominięta DZIĘKI progowi krypto (5.88% < 8%), mimo że
        # próg akcji (2%) by ją przepuścił. To dowodzi, że crypto override działa.
        council_port.analyze.assert_not_called()
        assert result.get("council_verdict") is None
        # Główna analiza i tak się zakończyła (rada to opcjonalny dodatek).
        assert result.get("status") == "saved"

    def test_crypto_council_called_using_crypto_threshold_not_stock(self):
        # stock council_threshold=20%, crypto_council_threshold=3%.
        # current=180, prev=170 → Δ≈+5.88%. PONIŻEJ progu akcji (20%), więc
        # gdyby council_node błędnie użył progu akcji → rada by była POMINIĘTA.
        # Skoro używa progu KRYPTO (3%), Δ 5.88% ≥ 3% → rada ODPALA.
        council_port = Mock(spec=AdvisoryCouncilPort)
        council_port.analyze.return_value = _make_verdict()
        graph, _ = _make_crypto_graph(
            council_port,
            council_threshold=Threshold(Decimal("0.20")),
            crypto_council_threshold=Threshold(Decimal("0.03")),
        )
        app = graph.compile()
        asset = Asset(symbol="BTC", asset_type=AssetType.CRYPTO)
        result = app.invoke({
            "symbol": "BTC",
            "previous_price": Decimal("170.00"),
            "asset": asset,
        })
        # Sedno: rada ODPALIŁA DZIĘKI progowi krypto (5.88% ≥ 3%), mimo że
        # próg akcji (20%) by ją zablokował.
        council_port.analyze.assert_called_once()
        assert result.get("council_verdict") is not None
