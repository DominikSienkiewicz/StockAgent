"""FIN-35: deterministyczne atrapy płatnych portów dla lokalnego runu.

Przy `PAID_CALLS_ENABLED=false` DI podstawia je zamiast LLM i Alpha Vantage.
Atrapa nie robi I/O, zawsze daje ten sam wynik i jawnie mówi, że jest atrapą —
sentyment niesie `degraded_reason`, więc predykcja dostaje flagę jakości,
a mail odznakę DEGRADED zamiast udawać spokojny rynek."""

from __future__ import annotations

from src.application.agent_graph import SentimentPayload
from src.application.ports import LLMPort, NewsPort, SentimentPort
from src.infrastructure.adapters.local_stubs import (
    PAID_CALLS_DISABLED,
    LocalStubNewsAdapter,
    LocalStubSentimentAdapter,
)
from src.infrastructure.llm.local_stub import LocalStubLLMAdapter


def test_stubs_implement_the_ports() -> None:
    assert isinstance(LocalStubLLMAdapter(), LLMPort)
    assert isinstance(LocalStubSentimentAdapter(), SentimentPort)
    assert isinstance(LocalStubNewsAdapter(), NewsPort)


def test_llm_stub_gives_the_same_neutral_analysis_for_any_prompt() -> None:
    stub = LocalStubLLMAdapter()

    first = stub.analyze("AAPL -5%")

    assert first == stub.analyze("zupełnie inny prompt")
    assert first["trend_direction"] == "SIDEWAYS"
    assert first["confidence_score"] == 0.5
    assert "PAID_CALLS_ENABLED" in first["reasoning"]


def test_llm_stub_has_no_narrative_so_the_recap_falls_back() -> None:
    # WeeklyRecapUseCase bierze `narrative` z wyniku LLM — atrapa nie udaje opowieści.
    assert "narrative" not in LocalStubLLMAdapter().analyze("recap")


def test_llm_stub_diagnosis_names_itself() -> None:
    insight = LocalStubLLMAdapter().analyze_mistake("diagnoza")

    assert insight == LocalStubLLMAdapter().analyze_mistake("inna")
    assert "PAID_CALLS_ENABLED" in insight


def test_sentiment_stub_is_neutral_and_flagged_degraded() -> None:
    payload = LocalStubSentimentAdapter().get_social_score("AAPL")

    assert payload == {
        "av_sentiment_score": 0.0,
        "av_relevance_avg": 0.0,
        "news_volume_24h": 0,
        "high_relevance_count": 0,
        "av_sentiment_label": "Neutral",
        "degraded_reason": PAID_CALLS_DISABLED,
    }


def test_sentiment_stub_payload_fits_the_graph_state_contract() -> None:
    # ARCH-10: graf rzutuje wynik portu na `SentimentPayload` bez kopiowania, więc
    # każdy klucz, który daje adapter (atrapa ma komplet pól AV), musi tam być.
    payload = LocalStubSentimentAdapter().get_social_score("AAPL")

    assert set(payload) <= set(SentimentPayload.__annotations__)


def test_news_stub_returns_no_articles() -> None:
    assert LocalStubNewsAdapter().get_news_context("AAPL") == []
