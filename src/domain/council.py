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
    news_articles: tuple[str, ...]
    llm_trend: str
    llm_confidence: float
    ml_price_target: Decimal
    # Pola opcjonalne — domyślne wartości zapewniają wsteczną kompatybilność
    fundamentals: Fundamentals | None = field(default=None)
    valuation_verdict: ValuationVerdict = field(default=ValuationVerdict.UNKNOWN)

    def __post_init__(self) -> None:
        # Kolekcja jako tuple — frozen=True blokuje tylko rebinding atrybutu, NIE
        # mutację listy w miejscu. Koercja gwarantuje realną niezmienność value
        # objectu (i przyjmuje listę od callera dla wstecznej kompatybilności).
        object.__setattr__(self, "news_articles", tuple(self.news_articles))


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
    key_factors: tuple[str, ...]

    def __post_init__(self) -> None:
        # Tuple zamiast listy — realna niezmienność frozen value objectu.
        object.__setattr__(self, "key_factors", tuple(self.key_factors))

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
    dissenting_views: tuple[str, ...]
    investor_opinions: tuple[InvestorOpinion, ...]

    def __post_init__(self) -> None:
        # Tuple zamiast list — frozen=True nie chroni przed mutacją kolekcji
        # w miejscu; koercja daje realną niezmienność (i przyjmuje listę callera).
        object.__setattr__(self, "dissenting_views", tuple(self.dissenting_views))
        object.__setattr__(self, "investor_opinions", tuple(self.investor_opinions))

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


def derive_consensus(
    opinions: list[InvestorOpinion],
) -> tuple[Literal["BUY", "SELL", "HOLD"], float]:
    """Wylicza autorytatywną rekomendację rady i siłę konsensusu z REALNYCH głosów.

    Decyzja należy do domeny, nie do chairman-LLM: bierzemy faktyczne opinie
    inwestorów (po rozwiązaniu/timeoutach) i liczymy z nich deterministyczny
    wynik. Dzięki temu jednogłośne BUY zostaje BUY nawet gdy wywołanie chairmana
    padnie — żadna "wymyślona" liczba 0.5/HOLD nie nadpisze prawdziwych głosów.

    Algorytm:
    - Każda rekomendacja zbiera "masę" = sumę confidence swoich głosów.
    - Zwycięski koszyk = ten z największą masą (confidence-weighted, nie headcount).
      Remis rozstrzygany preferencją BUY > SELL > HOLD (stabilna kolejność).
    - `consensus_strength` = masa zwycięzcy / masa wszystkich głosów ∈ [0, 1].

    Przypadki brzegowe:
    - brak opinii → ("HOLD", 0.0) — bezpieczny default, nic nie zgaduje.
    - wszystkie confidence == 0.0 → waga bezużyteczna (dzielenie przez zero),
      więc cofamy się do liczenia głosów (headcount) zamiast masy pewności.
    """
    if not opinions:
        return "HOLD", 0.0

    total_confidence = sum(op.confidence for op in opinions)
    # Gdy wszyscy mają zerową pewność, ważenie traci sens — używamy headcountu.
    use_headcount = total_confidence <= 0.0

    buckets: dict[str, float] = {rec: 0.0 for rec in _RECOMMENDATIONS}
    for op in opinions:
        if op.recommendation in buckets:
            buckets[op.recommendation] += 1.0 if use_headcount else op.confidence

    total_weight = sum(buckets.values())
    if total_weight <= 0.0:
        # Teoretycznie nieosiągalne (opinions niepuste), ale chroni przed dzieleniem.
        return "HOLD", 0.0

    # max() z kluczem stabilnym względem preferencji BUY > SELL > HOLD przy remisie.
    winner = max(
        _RECOMMENDATIONS,
        key=lambda rec: (buckets[rec], -_RECOMMENDATIONS.index(rec)),
    )
    strength = buckets[winner] / total_weight
    return winner, strength


@dataclass(frozen=True)
class InvestorPersona:
    """Tożsamość i filozofia inwestycyjna członka rady doradczej.

    Persony są częścią domeny — to one definiują skład rady, a nie szczegół
    techniczny adaptera LLM. Warstwa application składa z nich prompty,
    warstwa infrastructure wykonuje wywołania.
    """

    name: str
    style: str
