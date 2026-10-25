"""Domena: konsensus analityków i cel cenowy — tło sell-side.

Czysta logika: rozkład rekomendacji (strong buy … strong sell) daje ocenę
konsensusu, a cel cenowy vs cena bieżąca — implikowany potencjał wzrostu.
Tło, nie wyrocznia: konsensus bywa opóźniony, ale wyłapuje skrajne rozjazdy.

DAT-13: poziom konsensusu dużych spółek jest trwale „Buy”, więc więcej mówi jego
zmiana miesiąc do miesiąca — `revision()` porównuje najświeższy okres z poprzednim
z tej samej odpowiedzi Finnhub (UPGRADE / DOWNGRADE / STABLE). Okresy są
miesięczne i bywają opóźnione o miesiąc: to sygnał kontekstowy, nie dzienny."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from enum import Enum

# DAT-13: rewizja to zmiana średniej oceny o co najmniej tyle punktów skali 5..1…
REVISION_MEAN_SCORE_STEP = 0.3
# …albo przesunięcie netto co najmniej tylu analityków między stroną kupna
# (strong buy + buy) a sprzedaży (sell + strong sell).
REVISION_ANALYST_SHIFT = 2
# Średnia to iloraz floatów: 3.8 − 3.5 wychodzi 0.29999…, a to ma być równo próg.
_MEAN_TOLERANCE = 1e-9


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


class RevisionDirection(Enum):
    """Kierunek rewizji konsensusu względem poprzedniego okresu (DAT-13)."""

    UPGRADE = "UPGRADE"
    DOWNGRADE = "DOWNGRADE"
    STABLE = "STABLE"


@dataclass(frozen=True)
class AnalystRevision:
    """Rewizja konsensusu m/m: kierunek i zmiana liczby analityków w każdym koszyku
    (dodatnia = więcej rekomendacji w koszyku niż miesiąc wcześniej)."""

    direction: RevisionDirection
    strong_buy_change: int = 0
    buy_change: int = 0
    hold_change: int = 0
    sell_change: int = 0
    strong_sell_change: int = 0

    @property
    def bullish_change(self) -> int:
        """Zmiana liczby rekomendacji kupna (strong buy + buy)."""
        return self.strong_buy_change + self.buy_change

    @property
    def bearish_change(self) -> int:
        """Zmiana liczby rekomendacji sprzedaży (sell + strong sell)."""
        return self.sell_change + self.strong_sell_change


_STABLE = AnalystRevision(direction=RevisionDirection.STABLE)


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
    # DAT-13: rozkład z poprzedniego okresu tej samej odpowiedzi; None = jeden okres.
    previous: AnalystConsensus | None = None

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

    def revision(self) -> AnalystRevision:
        """DAT-13: rewizja względem poprzedniego okresu; jeden okres → STABLE."""
        if self.previous is None:
            return _STABLE
        return self.revision_vs(self.previous)

    def revision_vs(self, previous: AnalystConsensus) -> AnalystRevision:
        """DAT-13: rewizja tego rozkładu względem `previous`.

        W górę: średnia ocena wzrosła o ≥ `REVISION_MEAN_SCORE_STEP` albo strona
        kupna zyskała netto ≥ `REVISION_ANALYST_SHIFT` analityków względem strony
        sprzedaży; w dół symetrycznie. Sygnały sprzeczne (średnia w górę, analitycy
        w dół — np. skurczone pokrycie) albo poniżej progów → STABLE."""
        changes = AnalystRevision(
            direction=RevisionDirection.STABLE,
            strong_buy_change=self.strong_buy - previous.strong_buy,
            buy_change=self.buy - previous.buy,
            hold_change=self.hold - previous.hold,
            sell_change=self.sell - previous.sell,
            strong_sell_change=self.strong_sell - previous.strong_sell,
        )
        shift = changes.bullish_change - changes.bearish_change
        mean_change = _mean_change(previous.mean_score(), self.mean_score())
        step = REVISION_MEAN_SCORE_STEP - _MEAN_TOLERANCE
        up = mean_change >= step or shift >= REVISION_ANALYST_SHIFT
        down = mean_change <= -step or shift <= -REVISION_ANALYST_SHIFT
        if up == down:
            return changes
        direction = RevisionDirection.UPGRADE if up else RevisionDirection.DOWNGRADE
        return replace(changes, direction=direction)


def _mean_change(before: float | None, after: float | None) -> float:
    """Zmiana średniej oceny; okres bez rekomendacji nie daje zmiany średniej."""
    if before is None or after is None:
        return 0.0
    return after - before
