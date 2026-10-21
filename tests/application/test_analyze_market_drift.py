"""ALR-10 — use case podaje grafowi referencję dryfu (`drift_reference`).

Referencja to cena ostatniej predykcji z okna `DRIFT_WINDOW_DAYS` albo pierwsza
cena snapshotu w tym oknie — same darmowe odczyty Supabase, tylko za flagą
`AgentGraphDeps.drift_gate_enabled`. Każdy błąd albo odczyt spoza kontraktu
zamyka dryf (fail-closed): bramka działa wtedy dokładnie jak przed ALR-10."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application.agent_graph import AgentGraphDeps
from src.application.ports import (
    LatestPrediction,
    LLMPort,
    MarketDataPort,
    MLPredictionPort,
    NewsPort,
    RepositoryPort,
    SentimentPort,
)
from src.application.use_cases.analyze_market import AnalyzeMarketUseCase
from src.domain.asset import DRIFT_WINDOW_DAYS
from src.domain.value_objects import Money, Threshold

NOW = datetime(2026, 10, 21, 13, 30, tzinfo=UTC)


@pytest.fixture
def repo() -> Mock:
    repo = Mock(spec=RepositoryPort)
    repo.get_unverified_prediction.return_value = None
    repo.get_last_prediction_price.return_value = None
    repo.get_last_price.return_value = Money(Decimal("100.0"))
    repo.get_last_price_snapshot.return_value = (Money(Decimal("100.0")), NOW - timedelta(hours=20))
    return repo


@pytest.fixture
def market() -> Mock:
    market = Mock(spec=MarketDataPort)
    market.get_current_price.return_value = Money(Decimal("98.1"))  # -1,9%: pod progiem
    return market


def _use_case(repo: Mock, market: Mock, *, drift_gate_enabled: bool) -> AnalyzeMarketUseCase:
    llm = Mock(spec=LLMPort)
    llm.analyze.return_value = {
        "trend_direction": "BEARISH",
        "confidence_score": 0.7,
        "target_price_12h": 97.0,
        "reasoning": "dryf",
    }
    ml = Mock(spec=MLPredictionPort)
    ml.predict.return_value = Money(Decimal("97.5"))
    sentiment = Mock(spec=SentimentPort)
    sentiment.get_social_score.return_value = {"av_sentiment_score": -0.2}
    news = Mock(spec=NewsPort)
    news.get_news_context.return_value = [{"title": "Headline"}]
    repo.save_prediction.return_value = "uuid-drift"
    return AnalyzeMarketUseCase(AgentGraphDeps(
        market_port=market,
        sentiment_port=sentiment,
        news_port=news,
        repository_port=repo,
        ml_port=ml,
        llm_port=llm,
        threshold=Threshold(Decimal("0.02")),
        drift_gate_enabled=drift_gate_enabled,
    ))


def _latest(price: str, days_ago: float) -> LatestPrediction:
    return LatestPrediction(
        price=Money(Decimal(price)), timestamp=NOW - timedelta(days=days_ago), reasoning=None
    )


def test_flag_off_reads_no_drift_reference(repo: Mock, market: Mock) -> None:
    result = _use_case(repo, market, drift_gate_enabled=False).run("AAPL", now=NOW)

    assert result["status"] == "ignored"
    repo.get_latest_prediction.assert_not_called()
    repo.get_price_history.assert_not_called()


def test_drift_since_the_last_prediction_opens_the_gate(repo: Mock, market: Mock) -> None:
    repo.get_latest_prediction.return_value = _latest("118.9", days_ago=4)
    repo.get_price_history.return_value = []

    result = _use_case(repo, market, drift_gate_enabled=True).run("AAPL", now=NOW)

    assert result["status"] == "saved"
    repo.get_price_history.assert_called_once_with("AAPL", DRIFT_WINDOW_DAYS)


def test_window_start_is_the_reference_without_a_recent_prediction(
    repo: Mock, market: Mock
) -> None:
    repo.get_latest_prediction.return_value = _latest("60", days_ago=40)
    repo.get_price_history.return_value = [
        (NOW - timedelta(days=9), Money(Decimal("99.0"))),
        (NOW - timedelta(days=1), Money(Decimal("100.0"))),
    ]

    result = _use_case(repo, market, drift_gate_enabled=True).run("AAPL", now=NOW)

    # Od 99,00 to tylko -0,9% — stara predykcja (60) nie otwiera bramki.
    assert result["status"] == "ignored"


def test_failing_reads_close_the_drift(repo: Mock, market: Mock) -> None:
    repo.get_latest_prediction.side_effect = RuntimeError("Supabase 503")
    repo.get_price_history.side_effect = RuntimeError("Supabase 503")

    result = _use_case(repo, market, drift_gate_enabled=True).run("AAPL", now=NOW)

    assert result["status"] == "ignored"


def test_reads_outside_the_contract_close_the_drift(repo: Mock, market: Mock) -> None:
    # Mock bez skonfigurowanych odczytów (fixture'y innych testów) nie może otworzyć bramki.
    result = _use_case(repo, market, drift_gate_enabled=True).run("AAPL", now=NOW)

    assert result["status"] == "ignored"
