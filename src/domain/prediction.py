from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


def calibration_error(samples: Sequence[tuple[float, bool]]) -> float:
    """#4 — luka kalibracji: |śr. deklarowana pewność − realna trafność| ∈ [0,1].

    0 = idealnie skalibrowane; wysokie = rada przepewna (lub niedopewna).
    Każdy sample to (confidence∈[0,1], czy_trafiona_kierunkowo). Pusta → 0.0."""
    if not samples:
        return 0.0
    mean_conf = sum(c for c, _ in samples) / len(samples)
    accuracy = sum(1 for _, ok in samples if ok) / len(samples)
    return abs(mean_conf - accuracy)


def calibration_score(confidence: float, correct: bool) -> float:
    """#4 — kalibracja pojedynczej predykcji: `1 − |confidence − wynik(0/1)|`
    ∈ [0,1]. 1 = pewność idealnie dopasowana do wyniku."""
    outcome = 1.0 if correct else 0.0
    return 1.0 - abs(confidence - outcome)


class TrendDirection(StrEnum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"


SIDEWAYS_TOLERANCE = Decimal("0.005")  # ±0.5%


@dataclass(frozen=True)
class Prediction:
    symbol: str
    predicted_trend: TrendDirection
    price_at_prediction: Decimal
    predicted_target_price: Decimal
    id: str | None = None  # ustawiane przy odczycie z repozytorium

    def is_trend_correct(self, actual_price: Decimal) -> bool:
        actual_delta = actual_price - self.price_at_prediction
        if self.predicted_trend == TrendDirection.BULLISH:
            return actual_delta > 0
        if self.predicted_trend == TrendDirection.BEARISH:
            return actual_delta < 0
        # SIDEWAYS — w granicach ±SIDEWAYS_TOLERANCE. Guard: cena bazowa 0
        # (skażony rekord) → brak punktu odniesienia. Traktujemy brak ruchu
        # jako SIDEWAYS-correct zamiast dzielić przez zero.
        if self.price_at_prediction == 0:
            return actual_delta == 0
        return abs(actual_delta / self.price_at_prediction) <= SIDEWAYS_TOLERANCE

    def accuracy_score(self, actual_price: Decimal) -> Decimal:
        # Guard: bez ceny bazowej (0) nie da się policzyć błędu względnego —
        # zwracamy 0.0 (brak wiarygodnego sygnału), nie ZeroDivisionError.
        if self.price_at_prediction == 0:
            return Decimal("0.0")
        error = abs(actual_price - self.predicted_target_price) / self.price_at_prediction
        return max(Decimal("0.0"), Decimal("1.0") - error)
