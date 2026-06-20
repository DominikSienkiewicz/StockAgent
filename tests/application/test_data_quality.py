# tests/application/test_data_quality.py
"""Bug-regression: dane wejściowe z None/NaN/"" cicho zamieniane na 0.0
maskują problemy w pipeline — model uczy się na śmieciach bez śladu.

Po naprawie:
- save_node zapisuje `data_quality_flags` (lista stringów) w prediction_logs
- logger.warning() emitowane gdy są jakieś flagi
- brak flag = lista pusta (NIE None — łatwiejsze SQL: `array_length(...) = 0`)
"""
from __future__ import annotations

import logging
from decimal import Decimal
from unittest.mock import Mock

from src.application.agent_graph import create_agent_graph
from src.application.ports import (
    LLMPort,
    MarketDataPort,
    MLPredictionPort,
    NewsPort,
    RepositoryPort,
    SentimentPort,
)
from src.domain.value_objects import Money, Threshold


def _make_graph(sentiment_payload: dict, llm_payload: dict | None = None):
    market = Mock(spec=MarketDataPort)
    market.get_current_price.return_value = Money(Decimal("180.00"))
    sentiment = Mock(spec=SentimentPort)
    sentiment.get_social_score.return_value = sentiment_payload
    news = Mock(spec=NewsPort)
    news.get_news_context.return_value = [{"title": "Apple gains", "source": "Reuters"}]
    repo = Mock(spec=RepositoryPort)
    repo.get_unverified_prediction.return_value = None
    # Cena poprzedniej predykcji (referencja price_delta) — ustawiona, by te
    # testy skupiały się na flagach sentymentu/numerycznych, bez flagi "no_prior".
    repo.get_last_prediction_price.return_value = Money(Decimal("100.0"))
    repo.save_prediction.return_value = "pred-1"
    ml = Mock(spec=MLPredictionPort)
    ml.is_trained = True
    ml.predict.return_value = Money(Decimal("185.00"))
    llm = Mock(spec=LLMPort)
    llm.analyze.return_value = llm_payload or {
        "trend_direction": "BULLISH",
        "confidence_score": 0.8,
        "av_agreement": 0.9,
        "reasoning": "OK",
    }
    graph = create_agent_graph(
        market_port=market,
        sentiment_port=sentiment,
        news_port=news,
        repository_port=repo,
        ml_port=ml,
        llm_port=llm,
        threshold=Threshold(Decimal("0.02")),
    )
    return graph, repo


class TestDataQualityFlags:
    def test_clean_inputs_produce_empty_flag_list(self):
        clean = {
            "av_sentiment_score": 0.5,
            "av_sentiment_label": "Bullish",
            "av_relevance_avg": 0.7,
            "news_volume_24h": 10,
            "high_relevance_count": 3,
        }
        graph, repo = _make_graph(sentiment_payload=clean)
        graph.compile().invoke(
            {"symbol": "AAPL", "previous_price": Decimal("160.00")}
        )
        record = repo.save_prediction.call_args.args[0]
        assert "data_quality_flags" in record
        assert record["data_quality_flags"] == []

    def test_missing_sentiment_score_produces_flag(self):
        # None — typowo gdy AlphaVantage zwraca pusty NEWS_SENTIMENT response
        bad = {
            "av_sentiment_score": None,
            "av_sentiment_label": "Neutral",
            "av_relevance_avg": 0.7,
            "news_volume_24h": 10,
            "high_relevance_count": 3,
        }
        graph, repo = _make_graph(sentiment_payload=bad)
        graph.compile().invoke(
            {"symbol": "AAPL", "previous_price": Decimal("160.00")}
        )
        record = repo.save_prediction.call_args.args[0]
        flags = record["data_quality_flags"]
        assert "av_sentiment_score_missing" in flags

    def test_unparseable_news_volume_produces_flag(self):
        # AlphaVantage czasem zwraca "N/A" lub "" w polach liczbowych
        bad = {
            "av_sentiment_score": 0.5,
            "av_sentiment_label": "Bullish",
            "av_relevance_avg": 0.7,
            "news_volume_24h": "N/A",
            "high_relevance_count": 3,
        }
        graph, repo = _make_graph(sentiment_payload=bad)
        graph.compile().invoke(
            {"symbol": "AAPL", "previous_price": Decimal("160.00")}
        )
        record = repo.save_prediction.call_args.args[0]
        flags = record["data_quality_flags"]
        assert "news_volume_24h_invalid" in flags

    def test_nan_sentiment_score_produces_flag(self):
        bad = {
            "av_sentiment_score": float("nan"),
            "av_sentiment_label": "Neutral",
            "av_relevance_avg": 0.7,
            "news_volume_24h": 10,
            "high_relevance_count": 3,
        }
        graph, repo = _make_graph(sentiment_payload=bad)
        graph.compile().invoke(
            {"symbol": "AAPL", "previous_price": Decimal("160.00")}
        )
        record = repo.save_prediction.call_args.args[0]
        flags = record["data_quality_flags"]
        assert "av_sentiment_score_invalid" in flags

    def test_multiple_bad_fields_accumulate_flags(self):
        bad = {
            "av_sentiment_score": None,
            "av_sentiment_label": "Neutral",
            "av_relevance_avg": "",
            "news_volume_24h": None,
            "high_relevance_count": "junk",
        }
        graph, repo = _make_graph(sentiment_payload=bad)
        graph.compile().invoke(
            {"symbol": "AAPL", "previous_price": Decimal("160.00")}
        )
        record = repo.save_prediction.call_args.args[0]
        flags = record["data_quality_flags"]
        assert len(flags) >= 3

    def test_warning_logged_when_flags_present(self, caplog):
        bad = {
            "av_sentiment_score": None,
            "av_sentiment_label": "Neutral",
            "av_relevance_avg": 0.7,
            "news_volume_24h": 10,
            "high_relevance_count": 3,
        }
        graph, _ = _make_graph(sentiment_payload=bad)
        with caplog.at_level(logging.WARNING, logger="src.application.agent_graph"):
            graph.compile().invoke(
                {"symbol": "AAPL", "previous_price": Decimal("160.00")}
            )
        joined = " ".join(rec.message for rec in caplog.records)
        assert "data_quality" in joined.lower() or "av_sentiment_score_missing" in joined

    def test_av_degraded_reason_propagated_as_quality_flag(self):
        # Adapter sentiment dorzucił sentiel `degraded_reason` (np. wszystkie
        # klucze AV wyczerpane). Predict_node przepuszcza to do
        # data_quality_flags, żeby trening filtrował zdegradowane cykle.
        payload_with_degradation = {
            "av_sentiment_score": 0.0,
            "av_sentiment_label": "Neutral",
            "av_relevance_avg": 0.0,
            "news_volume_24h": 0,
            "high_relevance_count": 0,
            "degraded_reason": "av_keys_exhausted",
        }
        graph, repo = _make_graph(sentiment_payload=payload_with_degradation)
        graph.compile().invoke(
            {"symbol": "AAPL", "previous_price": Decimal("160.00")}
        )
        record = repo.save_prediction.call_args.args[0]
        assert "av_keys_exhausted" in record["data_quality_flags"]

    def test_no_warning_when_inputs_clean(self, caplog):
        clean = {
            "av_sentiment_score": 0.5,
            "av_sentiment_label": "Bullish",
            "av_relevance_avg": 0.7,
            "news_volume_24h": 10,
            "high_relevance_count": 3,
        }
        graph, _ = _make_graph(sentiment_payload=clean)
        with caplog.at_level(logging.WARNING, logger="src.application.agent_graph"):
            graph.compile().invoke(
                {"symbol": "AAPL", "previous_price": Decimal("160.00")}
            )
        joined = " ".join(rec.message for rec in caplog.records)
        assert "data_quality" not in joined.lower()
