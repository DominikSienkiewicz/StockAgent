"""Domena: rozkład prawdopodobieństwa trendu (ML-29).

Jedna pewność dla wybranej klasy nie mówi, czy model „prawie” wybrał inną, a
właściwe reguły punktacji (RPS, 3-klasowy Brier) wymagają pełnego rozkładu.
`ProbabilityForecast` normalizuje rozkład P(BULLISH/BEARISH/SIDEWAYS) z odpowiedzi
LLM i wyprowadza z niego trend (argmax) oraz pewność (max p). Gdy rozkładu brak
albo jest bezużyteczny, zostają stare pola odpowiedzi: `trend_direction`
i `confidence_score`. Czysta logika, stdlib only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from src.domain.prediction import TrendDirection

# Kolejność rozstrzygania remisu argmaxu: najpierw SIDEWAYS (brak przewagi kierunku
# to brak sygnału), potem BULLISH i BEARISH — deterministycznie, bez losowania.
_TIE_BREAK_ORDER: tuple[TrendDirection, ...] = (
    TrendDirection.SIDEWAYS,
    TrendDirection.BULLISH,
    TrendDirection.BEARISH,
)


def _probability(value: object) -> float | None:
    """Liczba skończona albo None (brak, tekst, bool, NaN, nieskończoność)."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def normalize_distribution(raw: object) -> dict[TrendDirection, float] | None:
    """Rozkład z odpowiedzi LLM znormalizowany do sumy 1 albo None.

    Klucze bez rozróżniania wielkości liter i spacji; brakująca klasa = 0.
    Ujemne p przycinamy do 0; skala nie ma znaczenia (procenty 60/25/15 dają
    0.6/0.25/0.15), bo i tak dzielimy przez sumę. None, gdy to nie mapa, gdy
    którakolwiek podana klasa nie jest liczbą, gdy pojawia się klucz spoza trzech
    klas albo gdy suma po przycięciu wynosi 0 — wtedy obowiązują stare pola."""
    if not isinstance(raw, Mapping) or not raw:
        return None
    weights = dict.fromkeys(TrendDirection, 0.0)
    for key, value in raw.items():
        name = str(key).strip().upper()
        if name not in TrendDirection.__members__:
            return None
        probability = _probability(value)
        if probability is None:
            return None
        weights[TrendDirection(name)] = max(probability, 0.0)
    total = sum(weights.values())
    if total <= 0.0:
        return None
    return {trend: weight / total for trend, weight in weights.items()}


@dataclass(frozen=True)
class ProbabilityForecast:
    """Trend, pewność i — gdy LLM go podał — pełny rozkład trzech klas.

    `probabilities` None znaczy „rozkład niedostępny”: trend i pewność pochodzą
    wtedy wprost ze starych pól odpowiedzi (fallback)."""

    trend: TrendDirection
    confidence: float
    probabilities: tuple[tuple[TrendDirection, float], ...] | None = None

    @classmethod
    def from_llm(
        cls,
        raw_probabilities: object,
        *,
        fallback_trend: TrendDirection,
        fallback_confidence: float,
    ) -> ProbabilityForecast:
        distribution = normalize_distribution(raw_probabilities)
        if distribution is None:
            return cls(trend=fallback_trend, confidence=fallback_confidence)
        trend = max(_TIE_BREAK_ORDER, key=lambda candidate: distribution[candidate])
        return cls(
            trend=trend,
            confidence=distribution[trend],
            probabilities=tuple((t, distribution[t]) for t in TrendDirection),
        )

    def as_dict(self) -> dict[str, float] | None:
        """Rozkład jako JSON `{"BULLISH": p, "BEARISH": p, "SIDEWAYS": p}` albo None."""
        if self.probabilities is None:
            return None
        return {trend.value: probability for trend, probability in self.probabilities}
