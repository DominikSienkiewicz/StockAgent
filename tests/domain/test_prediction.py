from decimal import Decimal

from src.domain.prediction import Prediction, TrendDirection


def _make_prediction(
    trend: TrendDirection,
    price_at: Decimal = Decimal("100.0"),
    target: Decimal = Decimal("105.0"),
) -> Prediction:
    return Prediction(
        symbol="AAPL",
        predicted_trend=trend,
        price_at_prediction=price_at,
        predicted_target_price=target,
    )


class TestIsTrendCorrect:
    def test_bullish_trend_is_correct_when_price_rises(self):
        prediction = _make_prediction(TrendDirection.BULLISH)
        assert prediction.is_trend_correct(Decimal("110.0")) is True

    def test_bullish_trend_is_wrong_when_price_falls(self):
        prediction = _make_prediction(TrendDirection.BULLISH)
        assert prediction.is_trend_correct(Decimal("90.0")) is False

    def test_bearish_trend_is_correct_when_price_falls(self):
        prediction = _make_prediction(TrendDirection.BEARISH)
        assert prediction.is_trend_correct(Decimal("90.0")) is True

    def test_bearish_trend_is_wrong_when_price_rises(self):
        prediction = _make_prediction(TrendDirection.BEARISH)
        assert prediction.is_trend_correct(Decimal("110.0")) is False

    def test_sideways_is_correct_within_half_percent(self):
        prediction = _make_prediction(TrendDirection.SIDEWAYS)
        assert prediction.is_trend_correct(Decimal("100.4")) is True
        assert prediction.is_trend_correct(Decimal("99.6")) is True

    def test_sideways_is_wrong_outside_half_percent(self):
        prediction = _make_prediction(TrendDirection.SIDEWAYS)
        assert prediction.is_trend_correct(Decimal("101.0")) is False
        assert prediction.is_trend_correct(Decimal("99.0")) is False


class TestAccuracyScore:
    def test_perfect_prediction_scores_one(self):
        prediction = _make_prediction(
            TrendDirection.BULLISH,
            price_at=Decimal("100.0"),
            target=Decimal("105.0"),
        )
        assert prediction.accuracy_score(Decimal("105.0")) == Decimal("1.0")

    def test_score_decreases_with_error(self):
        # error 5/100 = 0.05 → score = 1 - 0.05 = 0.95
        prediction = _make_prediction(
            TrendDirection.BULLISH,
            price_at=Decimal("100.0"),
            target=Decimal("105.0"),
        )
        assert prediction.accuracy_score(Decimal("110.0")) == Decimal("0.95")

    def test_score_floored_at_zero_for_huge_error(self):
        prediction = _make_prediction(
            TrendDirection.BULLISH,
            price_at=Decimal("100.0"),
            target=Decimal("105.0"),
        )
        # error = |10 - 105| / 100 = 0.95 → 1 - 0.95 = 0.05 (jeszcze pozytywny)
        # ale dla dużych odchyleń wynik zostaje sklipowany do 0.0
        assert prediction.accuracy_score(Decimal("1000.0")) == Decimal("0.0")


class TestZeroPriceAtPredictionGuard:
    """Skażona cena 0 w prediction_logs nie może wywalać reflect_node.

    Obie metody dzielą przez `price_at_prediction` — bez guardu pojedynczy
    rekord z ceną 0 (np. błędny snapshot, wadliwy feed) rzuca
    ZeroDivisionError i ubija cały symbol w cyklu.
    """

    def test_is_trend_correct_does_not_raise_on_zero_price(self):
        prediction = _make_prediction(
            TrendDirection.SIDEWAYS, price_at=Decimal("0"), target=Decimal("0")
        )
        # SIDEWAYS dzieli actual_delta / price_at_prediction → 0/0.
        # Cena nie zmieniła się względem (nieznanego) 0 → traktujemy jak SIDEWAYS-correct.
        assert prediction.is_trend_correct(Decimal("0")) is True

    def test_accuracy_score_does_not_raise_on_zero_price(self):
        prediction = _make_prediction(
            TrendDirection.BULLISH, price_at=Decimal("0"), target=Decimal("105.0")
        )
        # Bez punktu odniesienia (cena 0) nie da się policzyć błędu względnego →
        # zwracamy 0.0 (brak wiarygodnego sygnału), nie crash.
        assert prediction.accuracy_score(Decimal("110.0")) == Decimal("0.0")
