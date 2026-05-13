from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


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
        # SIDEWAYS — w granicach ±SIDEWAYS_TOLERANCE
        return abs(actual_delta / self.price_at_prediction) <= SIDEWAYS_TOLERANCE

    def accuracy_score(self, actual_price: Decimal) -> Decimal:
        error = abs(actual_price - self.predicted_target_price) / self.price_at_prediction
        return max(Decimal("0.0"), Decimal("1.0") - error)
