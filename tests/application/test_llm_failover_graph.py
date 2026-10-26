"""NIEZ-44: predykcja od zapasowego dostawcy LLM jest odróżnialna w track recordzie.

Adapter failover znakuje wynik `analyze` kluczami `LLM_FAILOVER_PROVIDER_KEY`
i `LLM_FAILOVER_MODEL_KEY`. Graf zdejmuje je przed normalizacją (nie trafiają do
analizy ani rekordu), dopisuje flagę jakości `llm_failover` i zapisuje FAKTYCZNY
przepis tam, gdzie predykcja dziś zapisuje model: w kwicie v2 (`llm_provider`,
`llm_model`, `recipe_version`) i w kolumnie `recipe_version`. Przepis z innym
modelem ma inną wersję, więc segmentacja po wersji sama odsiewa takie predykcje.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from typing import Any
from unittest.mock import Mock

import pytest

from src.application.agent_graph import AgentGraphDeps, create_agent_graph
from src.application.ports import (
    LLM_FAILOVER_MODEL_KEY,
    LLM_FAILOVER_PROVIDER_KEY,
    LLMPort,
    MarketDataPort,
    MLPredictionPort,
    NewsPort,
    RepositoryPort,
    SentimentPort,
)
from src.domain.recipe import PredictionRecipe
from src.domain.value_objects import Money, Threshold

_ANALYSIS: dict[str, Any] = {
    "trend_direction": "BEARISH",
    "confidence_score": 0.85,
    "av_agreement": 0.25,
    "target_price_12h": 88.0,
    "reasoning": "Strong negative sentiment + macro headwinds.",
}


def _recipe() -> PredictionRecipe:
    return PredictionRecipe(
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-6",
        council_provider="openai",
        council_model="gpt-5-mini",
        prompt_sha256="a" * 64,
        personas_sha256="b" * 64,
        config_hash="c" * 64,
        model_version="cold_start",
        universe_version="d" * 64,
        git_sha="local",
        run_id="local",
    )


def _record(analysis: dict[str, Any], *, recipe: PredictionRecipe | None) -> dict[str, Any]:
    market = Mock(spec=MarketDataPort)
    market.get_current_price.return_value = Money(Decimal("90.0"))
    sentiment = Mock(spec=SentimentPort)
    sentiment.get_social_score.return_value = {
        "av_sentiment_score": -0.42,
        "av_relevance_avg": 0.72,
        "news_volume_24h": 4,
        "high_relevance_count": 2,
        "av_sentiment_label": "Bearish",
    }
    news = Mock(spec=NewsPort)
    news.get_news_context.return_value = [{"title": "Fed keeps rates", "source": "Reuters"}]
    repository = Mock(spec=RepositoryPort)
    repository.get_unverified_prediction.return_value = None
    repository.get_last_prediction_price.return_value = None
    repository.get_council_votes_for_prediction.return_value = []
    repository.save_prediction.return_value = "pred-1"
    ml = Mock(spec=MLPredictionPort)
    ml.predict.return_value = Money(Decimal("89.5"))
    ml.is_trained = True
    llm = Mock(spec=LLMPort)
    llm.analyze.return_value = analysis
    create_agent_graph(
        AgentGraphDeps(
            market_port=market,
            sentiment_port=sentiment,
            news_port=news,
            repository_port=repository,
            ml_port=ml,
            llm_port=llm,
            threshold=Threshold(Decimal("0.02")),
            receipts_enabled=True,
            recipe=recipe,
            recipe_version_enabled=True,
        )
    ).compile().invoke({"symbol": "AAPL", "previous_price": Decimal("100.0")})
    record: dict[str, Any] = repository.save_prediction.call_args.args[0]
    return record


def _failover_analysis() -> dict[str, Any]:
    return {**_ANALYSIS, LLM_FAILOVER_PROVIDER_KEY: "openai", LLM_FAILOVER_MODEL_KEY: "gpt-4o"}


def test_failover_prediction_carries_the_quality_flag() -> None:
    record = _record(_failover_analysis(), recipe=_recipe())

    assert "llm_failover" in record["data_quality_flags"]
    assert record["predicted_trend"] == "BEARISH"


def test_failover_prediction_records_the_actual_provider_and_model() -> None:
    recipe = _recipe()
    actual = dataclasses.replace(recipe, llm_provider="openai", llm_model="gpt-4o")

    record = _record(_failover_analysis(), recipe=recipe)

    receipt = record["decision_receipts"]
    assert (receipt["llm_provider"], receipt["llm_model"]) == ("openai", "gpt-4o")
    assert receipt["recipe_version"] == actual.version
    assert record["recipe_version"] == actual.version
    assert actual.version != recipe.version


def test_failover_markers_never_reach_the_record() -> None:
    record = _record(_failover_analysis(), recipe=_recipe())

    assert LLM_FAILOVER_PROVIDER_KEY not in str(record)
    assert LLM_FAILOVER_MODEL_KEY not in str(record)


def test_primary_prediction_keeps_the_configured_recipe() -> None:
    recipe = _recipe()

    record = _record(dict(_ANALYSIS), recipe=recipe)

    assert "llm_failover" not in record["data_quality_flags"]
    assert record["decision_receipts"]["llm_model"] == "claude-sonnet-4-6"
    assert record["recipe_version"] == recipe.version


@pytest.mark.parametrize("model", [None, ""])
def test_marker_without_a_model_still_flags_the_prediction(model: str | None) -> None:
    analysis = {**_ANALYSIS, LLM_FAILOVER_PROVIDER_KEY: "openai", LLM_FAILOVER_MODEL_KEY: model}

    record = _record(analysis, recipe=_recipe())

    assert "llm_failover" in record["data_quality_flags"]
    assert record["decision_receipts"]["llm_provider"] == "openai"


def test_failover_without_a_recipe_only_flags() -> None:
    record = _record(_failover_analysis(), recipe=None)

    assert "llm_failover" in record["data_quality_flags"]
    assert "recipe_version" not in record
