# src/domain/council.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from src.domain.value_objects import AssetType, Fundamentals, ValuationVerdict


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
    # Klasa aktywa — pozwala adaptacyjnemu ważeniu person (feature #3) dobrać
    # tabelę wag zależnie od typu (np. inne trafności person dla krypto niż akcji).
    asset_type: AssetType = field(default=AssetType.STOCK)

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
        dist: dict[str, int] = dict.fromkeys(_RECOMMENDATIONS, 0)
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

    def opinion_shift(self, before: Sequence[InvestorOpinion]) -> int:
        """Ilu inwestorów zmieniło rekomendację względem stanu `before`.

        Używane przez debate mode (feature #1): porównuje rekomendacje z rundy 1
        (`before`) z opiniami w tym werdykcie (po rewizji), dopasowując po
        `investor_name`. Inwestor obecny tylko w `before` (brak w werdykcie) jest
        ignorowany — liczymy realne zmiany zdania, nie różnice w składzie.
        """
        prior: dict[str, str] = {op.investor_name: op.recommendation for op in before}
        return sum(
            1
            for op in self.investor_opinions
            if op.investor_name in prior
            and prior[op.investor_name] != op.recommendation
        )


def derive_consensus(
    opinions: list[InvestorOpinion],
    weights: Mapping[str, float] | None = None,
) -> tuple[Literal["BUY", "SELL", "HOLD"], float]:
    """Wylicza autorytatywną rekomendację rady i siłę konsensusu z REALNYCH głosów.

    Decyzja należy do domeny, nie do chairman-LLM: bierzemy faktyczne opinie
    inwestorów (po rozwiązaniu/timeoutach) i liczymy z nich deterministyczny
    wynik. Dzięki temu jednogłośne BUY zostaje BUY nawet gdy wywołanie chairmana
    padnie — żadna "wymyślona" liczba 0.5/HOLD nie nadpisze prawdziwych głosów.

    Algorytm:
    - Każda rekomendacja zbiera "masę" = sumę `confidence * waga_persony` swoich
      głosów. Waga (feature #3) pochodzi z historycznej trafności persony;
      `weights.get(name, 1.0)` — brak wpisu = waga neutralna 1.0.
    - Zwycięski koszyk = ten z największą masą (confidence-weighted, nie headcount).
      Remis rozstrzygany preferencją BUY > SELL > HOLD (stabilna kolejność).
    - `consensus_strength` = masa zwycięzcy / masa wszystkich głosów ∈ [0, 1].

    `weights=None` (lub pusta mapa) → zachowanie identyczne jak bez ważenia:
    każda persona dostaje wagę 1.0, więc masa = sama confidence.

    Przypadki brzegowe:
    - brak opinii → ("HOLD", 0.0) — bezpieczny default, nic nie zgaduje.
    - wszystkie confidence == 0.0 → waga bezużyteczna (dzielenie przez zero),
      więc cofamy się do liczenia głosów (headcount) zamiast masy pewności.
    """
    if not opinions:
        return "HOLD", 0.0

    def _weight(op: InvestorOpinion) -> float:
        # Brak mapy wag albo brak wpisu dla persony → waga neutralna 1.0.
        return weights.get(op.investor_name, 1.0) if weights is not None else 1.0

    total_confidence = sum(op.confidence for op in opinions)
    # Gdy wszyscy mają zerową pewność, ważenie traci sens — używamy headcountu.
    use_headcount = total_confidence <= 0.0

    buckets: dict[str, float] = dict.fromkeys(_RECOMMENDATIONS, 0.0)
    for op in opinions:
        if op.recommendation in buckets:
            mass = 1.0 if use_headcount else op.confidence * _weight(op)
            buckets[op.recommendation] += mass

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


_DIRECTION_TO_REC: dict[str, Literal["BUY", "SELL", "HOLD"]] = {
    "UP": "BUY",
    "DOWN": "SELL",
    "FLAT": "HOLD",
}


def vindicated_dissenters(
    opinions: Sequence[InvestorOpinion],
    actual_direction: Literal["UP", "DOWN", "FLAT"],
    final_recommendation: Literal["BUY", "SELL", "HOLD"],
) -> tuple[InvestorOpinion, ...]:
    """Inwestorzy 'wybronieni' przez rynek: głosowali zgodnie z RZECZYWISTYM
    ruchem (BUY↔UP, SELL↔DOWN, HOLD↔FLAT), ale niezgodnie z finalną
    rekomendacją rady. Napędza counterfactual dissent-replay w self-reflection:
    'bearish dysydent miał rację — ważenie jego głosu odwróciłoby decyzję'.

    Gdy trafna rekomendacja == finalna (rada miała rację), zwraca pustą krotkę —
    nie ma kogo 'bronić'.
    """
    right_rec = _DIRECTION_TO_REC[actual_direction]
    if right_rec == final_recommendation:
        return ()
    return tuple(op for op in opinions if op.recommendation == right_rec)


@dataclass(frozen=True)
class InvestorPersona:
    """Tożsamość i filozofia inwestycyjna członka rady doradczej.

    Persony są częścią domeny — to one definiują skład rady, a nie szczegół
    techniczny adaptera LLM. Warstwa application składa z nich prompty,
    warstwa infrastructure wykonuje wywołania.
    """

    name: str
    style: str
