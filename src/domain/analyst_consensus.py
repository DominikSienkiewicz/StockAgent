"""Domena: konsensus analityków i cel cenowy — tło sell-side.

Czysta logika: rozkład rekomendacji (strong buy … strong sell) daje ocenę
konsensusu, a cel cenowy vs cena bieżąca — implikowany potencjał wzrostu.
Tło, nie wyrocznia: konsensus bywa opóźniony, ale wyłapuje skrajne rozjazdy."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


class AnalystRating(Enum):
    """Zagregowana ocena konsensusu sell-side."""

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


# Wagi rekomendacji do uśrednionego score'u (5 = strong buy … 1 = strong sell).
_RATING_WEIGHTS = (
    ("strong_buy", 5.0),
    ("buy", 4.0),
    ("hold", 3.0),
    ("sell", 2.0),
    ("strong_sell", 1.0),
)


@dataclass(frozen=True)
class AnalystConsensus:
    """Rozkład rekomendacji + cel cenowy dla symbolu.

    Liczniki rekomendacji + opcjonalny `price_target` (mediana/średnia celu).
    `rating()` mapuje ważoną średnią na koszyk; `upside()` liczy potencjał."""

    symbol: str
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0
    price_target: float | None = None

    def total(self) -> int:
        """Łączna liczba rekomendacji w rozkładzie."""
        return self.strong_buy + self.buy + self.hold + self.sell + self.strong_sell

    def mean_score(self) -> float | None:
        """Ważona średnia ocen (5..1). None gdy brak rekomendacji."""
        counts = {
            "strong_buy": self.strong_buy,
            "buy": self.buy,
            "hold": self.hold,
            "sell": self.sell,
            "strong_sell": self.strong_sell,
        }
        total = self.total()
        if total == 0:
            return None
        weighted = sum(counts[name] * weight for name, weight in _RATING_WEIGHTS)
        return weighted / total

    def rating(self) -> AnalystRating:
        """Mapuje ważoną średnią na koszyk. Brak rekomendacji → HOLD."""
        score = self.mean_score()
        if score is None:
            return AnalystRating.HOLD
        if score >= 4.5:
            return AnalystRating.STRONG_BUY
        if score >= 3.5:
            return AnalystRating.BUY
        if score >= 2.5:
            return AnalystRating.HOLD
        if score >= 1.5:
            return AnalystRating.SELL
        return AnalystRating.STRONG_SELL

    def upside(self, current_price: Decimal) -> float | None:
        """Implikowany potencjał: (cel - cena) / cena. None gdy brak celu lub
        cena ≤ 0 (brak odniesienia)."""
        if self.price_target is None or current_price <= 0:
            return None
        return (self.price_target - float(current_price)) / float(current_price)
