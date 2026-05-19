# src/domain/council.py
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from src.domain.value_objects import Fundamentals, ValuationVerdict


@dataclass(frozen=True)
class CouncilInput:
    symbol: str
    current_price: Decimal
    price_delta_pct: Decimal
    sentiment_score: float
    news_articles: list[str]
    llm_trend: str
    llm_confidence: float
    ml_price_target: Decimal
    # Pola opcjonalne — domyślne wartości zapewniają wsteczną kompatybilność
    fundamentals: Fundamentals | None = field(default=None)
    valuation_verdict: ValuationVerdict = field(default=ValuationVerdict.UNKNOWN)


# Próg "wyraźnego konsensusu" w radzie. 0.7 = chairman uznał, że co najmniej
# 70% siły opinii prze w jedną stronę. Wartość trzymana w domenie, nie w
# raporcie — żeby logika decyzyjna nie była rozsmarowana po warstwie prezentacji.
STRONG_CONSENSUS_THRESHOLD = 0.7

# Progi etykiet pewności pojedynczego inwestora. Wartości dobrane empirycznie
# z obserwacji rozkładu confidence w radzie (LLM rzadko schodzi <0.4, rzadko
# >0.9 — bardziej rozróżniający split na 0.5/0.75 niż 0.33/0.66).
CONFIDENCE_HIGH_THRESHOLD = 0.75
CONFIDENCE_LOW_THRESHOLD = 0.5

_RECOMMENDATIONS: tuple[Literal["BUY", "SELL", "HOLD"], ...] = ("BUY", "SELL", "HOLD")


@dataclass(frozen=True)
class InvestorOpinion:
    investor_name: str
    recommendation: Literal["BUY", "SELL", "HOLD"]
    confidence: float
    reasoning: str
    key_factors: list[str]

    def confidence_label(self) -> Literal["HIGH", "MEDIUM", "LOW"]:
        """Kategoryzacja pewności inwestora — używana w mailu i przy filtracji
        opinii (np. "pokaż tylko HIGH-confidence dissenters").
        """
        if self.confidence >= CONFIDENCE_HIGH_THRESHOLD:
            return "HIGH"
        if self.confidence >= CONFIDENCE_LOW_THRESHOLD:
            return "MEDIUM"
        return "LOW"


@dataclass(frozen=True)
class CouncilVerdict:
    final_recommendation: Literal["BUY", "SELL", "HOLD"]
    consensus_strength: float
    summary: str
    dissenting_views: list[str]
    investor_opinions: list[InvestorOpinion]

    def has_strong_consensus(
        self, threshold: float = STRONG_CONSENSUS_THRESHOLD
    ) -> bool:
        """Czy rada wyraźnie się zgadza (consensus_strength ≥ threshold).

        Używane przez raport (czerwona/zielona banda) i potencjalnie przez
        portfolio sizing (mocniejsza pozycja gdy rada jednogłośna).
        """
        return self.consensus_strength >= threshold

    def is_split_decision(self) -> bool:
        """Czy w radzie są jednocześnie głosy BUY i SELL (pomijając HOLD).

        Sygnalizuje fundamentalny brak zgody co do kierunku — ostrzeżenie
        że final_recommendation jest tylko statystyczną większością, nie
        wnioskiem płynącym z analizy.
        """
        recs = {op.recommendation for op in self.investor_opinions}
        return "BUY" in recs and "SELL" in recs

    def vote_distribution(self) -> dict[str, int]:
        """Liczba głosów na każdą rekomendację (zawsze 3 klucze).

        Zwraca dict z deterministycznym setem kluczy `BUY/SELL/HOLD`, nawet
        gdy któraś rekomendacja nie padła ani razu — dzięki temu konsumenci
        (raport HTML, dashboard) nie muszą obsługiwać brakujących kluczy.
        """
        dist: dict[str, int] = {rec: 0 for rec in _RECOMMENDATIONS}
        for op in self.investor_opinions:
            if op.recommendation in dist:
                dist[op.recommendation] += 1
        return dist

    def dissent_ratio(self) -> float:
        """Frakcja inwestorów niezgodnych z finalną rekomendacją (0.0-1.0).

        Wysoki dissent_ratio przy `has_strong_consensus()`==False to klasyczny
        sygnał: chairman wymusił werdykt, ale rada się sypie pod nim.
        """
        if not self.investor_opinions:
            return 0.0
        dissenting = sum(
            1 for op in self.investor_opinions
            if op.recommendation != self.final_recommendation
        )
        return dissenting / len(self.investor_opinions)


@dataclass(frozen=True)
class InvestorPersona:
    """Tożsamość i filozofia inwestycyjna członka rady doradczej.

    Persony są częścią domeny — to one definiują skład rady, a nie szczegół
    techniczny adaptera LLM. Warstwa application składa z nich prompty,
    warstwa infrastructure wykonuje wywołania.
    """

    name: str
    style: str
