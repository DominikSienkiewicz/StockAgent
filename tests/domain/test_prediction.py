from decimal import Decimal

import pytest

from src.domain.prediction import (
    CRYPTO_SIDEWAYS_TOLERANCE,
    SIDEWAYS_TOLERANCE,
    Prediction,
    TrendDirection,
    calibration_error,
    calibration_score,
)


class TestCalibration:
    def test_calibration_error_zero_when_confidence_matches_accuracy(self):
        samples = [(0.5, True), (0.5, False)]
        assert calibration_error(samples) == pytest.approx(0.0)

    def test_calibration_error_high_when_overconfident(self):
        samples = [(0.9, True), (0.9, False)]
        assert calibration_error(samples) == pytest.approx(0.4)

    def test_calibration_error_empty(self):
        assert calibration_error([]) == pytest.approx(0.0)

    def test_calibration_score_per_prediction(self):
        assert calibration_score(0.9, correct=False) == pytest.approx(0.1)
        assert calibration_score(0.8, correct=True) == pytest.approx(0.8)
        assert calibration_score(1.0, correct=True) == pytest.approx(1.0)


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


class TestCryptoSidewaysTolerance:
    """Bug pomiarowy: ±0.5% to sensowne pasmo FLAT dla akcji, ale przy dziennej
    zmienności BTC 3–5% niemal każdy dzień jest „ruchem", więc predykcja
    SIDEWAYS na krypto jest praktycznie zawsze uznawana za chybioną."""

    @staticmethod
    def _sideways(price: str) -> Prediction:
        return Prediction(
            symbol="BTC",
            predicted_trend=TrendDirection.SIDEWAYS,
            price_at_prediction=Decimal(price),
            predicted_target_price=Decimal(price),
        )

    def test_crypto_tolerance_is_wider_than_equity(self) -> None:
        assert CRYPTO_SIDEWAYS_TOLERANCE > SIDEWAYS_TOLERANCE

    def test_two_percent_crypto_move_is_still_sideways(self) -> None:
        # 2% na BTC to szum w paśmie dziennej zmienności 3–5%.
        pred = self._sideways("100")

        assert pred.is_trend_correct(
            Decimal("102"), tolerance=CRYPTO_SIDEWAYS_TOLERANCE
        )

    def test_same_move_is_not_sideways_for_equity(self) -> None:
        # Kontrola: ta sama delta na akcji to realny ruch.
        pred = self._sideways("100")

        assert not pred.is_trend_correct(Decimal("102"))

    def test_default_tolerance_is_unchanged(self) -> None:
        # Wsteczna kompatybilność: bez parametru zachowanie jak dotąd.
        pred = self._sideways("100")

        assert pred.is_trend_correct(Decimal("100.4"))
        assert not pred.is_trend_correct(Decimal("101"))
