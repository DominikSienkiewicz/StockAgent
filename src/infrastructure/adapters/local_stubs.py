# src/infrastructure/adapters/local_stubs.py
"""FIN-35: atrapy portów Alpha Vantage dla lokalnego runu bez płatnych wywołań.

Lokalny run zużywał limit kluczy AV (25 zapytań na dobę na klucz) wspólny z
produkcją. Atrapy nie robią I/O i zawsze dają ten sam wynik."""

from __future__ import annotations

from typing import Any

from src.application.ports import NewsPort, SentimentPort

# Powód degradacji feedu: graf dopisuje go do `data_quality_flags`, a mail
# pokazuje odznakę DEGRADED — neutralne zera nie udają spokojnego rynku.
PAID_CALLS_DISABLED = "paid_calls_disabled"


class LocalStubSentimentAdapter(SentimentPort):
    """Neutralny sentyment (jak pusty feed AV) oznaczony `degraded_reason`."""

    def get_social_score(self, symbol: str) -> dict[str, Any]:
        return {
            "av_sentiment_score": 0.0,
            "av_relevance_avg": 0.0,
            "news_volume_24h": 0,
            "high_relevance_count": 0,
            "av_sentiment_label": "Neutral",
            "degraded_reason": PAID_CALLS_DISABLED,
        }


class LocalStubNewsAdapter(NewsPort):
    """Brak artykułów — bez newsów graf nie liczy też embeddingu ani RAG."""

    def get_news_context(self, symbol: str) -> list[dict[str, Any]]:
        return []
