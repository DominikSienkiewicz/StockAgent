from collections.abc import Callable
from decimal import Decimal

import pytest

from src.domain.prediction import (
    BRIER_CALIBRATION_SCORE_VERSION,
    CRYPTO_SIDEWAYS_TOLERANCE,
    LINEAR_CALIBRATION_SCORE_VERSION,
    SIDEWAYS_TOLERANCE,
    Prediction,
    TrendDirection,
    brier_calibration_score,
    calibration_error,
    calibration_score,
    normalize_trend,
    sideways_tolerance_for,
)
from src.domain.value_objects import AssetType


class TestCalibration:
    def test_calibration_error_zero_when_confidence_matches_accuracy(self) -> None:
        samples = [(0.5, True), (0.5, False)]
        assert calibration_error(samples) == pytest.approx(0.0)

    def test_calibration_error_high_when_overconfident(self) -> None:
        samples = [(0.9, True), (0.9, False)]
        assert calibration_error(samples) == pytest.approx(0.4)

    def test_calibration_error_empty(self) -> None:
        assert calibration_error([]) == pytest.approx(0.0)

    def test_calibration_score_per_prediction(self) -> None:
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
    def test_bullish_trend_is_correct_when_price_rises(self) -> None:
        prediction = _make_prediction(TrendDirection.BULLISH)
        assert prediction.is_trend_correct(Decimal("110.0")) is True

    def test_bullish_trend_is_wrong_when_price_falls(self) -> None:
        prediction = _make_prediction(TrendDirection.BULLISH)
        assert prediction.is_trend_correct(Decimal("90.0")) is False

    def test_bearish_trend_is_correct_when_price_falls(self) -> None:
        prediction = _make_prediction(TrendDirection.BEARISH)
        assert prediction.is_trend_correct(Decimal("90.0")) is True

    def test_bearish_trend_is_wrong_when_price_rises(self) -> None:
        prediction = _make_prediction(TrendDirection.BEARISH)
        assert prediction.is_trend_correct(Decimal("110.0")) is False

    def test_sideways_is_correct_within_half_percent(self) -> None:
        prediction = _make_prediction(TrendDirection.SIDEWAYS)
        assert prediction.is_trend_correct(Decimal("100.4")) is True
        assert prediction.is_trend_correct(Decimal("99.6")) is True

    def test_sideways_is_wrong_outside_half_percent(self) -> None:
        prediction = _make_prediction(TrendDirection.SIDEWAYS)
        assert prediction.is_trend_correct(Decimal("101.0")) is False
        assert prediction.is_trend_correct(Decimal("99.0")) is False


class TestDeadBand:
    """EXP-06: martwa strefa kierunku — trafienie BULLISH przy +0,01% to szum.
    Domyślnie (`dead_band=None`) reguła bez zmian; szerokość pasma żyje w protokole
    ewaluacji (`directional_dead_zone` w evaluation_protocol.toml)."""

    def test_bullish_inside_dead_band_is_not_a_hit(self) -> None:
        prediction = _make_prediction(TrendDirection.BULLISH)

        assert prediction.is_trend_correct(Decimal("100.01")) is True
        assert prediction.is_trend_correct(Decimal("100.01"), dead_band=Decimal("0.001")) is False

    def test_bearish_inside_dead_band_is_not_a_hit(self) -> None:
        prediction = _make_prediction(TrendDirection.BEARISH)

        assert prediction.is_trend_correct(Decimal("99.99"), dead_band=Decimal("0.001")) is False

    def test_moves_beyond_the_dead_band_are_hits(self) -> None:
        band = Decimal("0.001")

        assert _make_prediction(TrendDirection.BULLISH).is_trend_correct(
            Decimal("100.2"), dead_band=band
        )
        assert _make_prediction(TrendDirection.BEARISH).is_trend_correct(
            Decimal("99.8"), dead_band=band
        )

    def test_a_move_exactly_on_the_edge_stays_inside(self) -> None:
        band = Decimal("0.001")

        assert not _make_prediction(TrendDirection.BULLISH).is_trend_correct(
            Decimal("100.1"), dead_band=band
        )
        assert not _make_prediction(TrendDirection.BEARISH).is_trend_correct(
            Decimal("99.9"), dead_band=band
        )

    def test_zero_band_is_todays_rule(self) -> None:
        for trend in (TrendDirection.BULLISH, TrendDirection.BEARISH):
            prediction = _make_prediction(trend)
            for actual in ("99.99", "100.0", "100.01"):
                price = Decimal(actual)
                assert prediction.is_trend_correct(
                    price, dead_band=Decimal("0")
                ) is prediction.is_trend_correct(price)

    def test_sideways_keeps_its_own_band(self) -> None:
        prediction = _make_prediction(TrendDirection.SIDEWAYS)

        assert prediction.is_trend_correct(Decimal("100.4"), dead_band=Decimal("0.01")) is True
        assert prediction.is_trend_correct(Decimal("101.0"), dead_band=Decimal("0.01")) is False

    def test_zero_base_price_falls_back_to_the_raw_move(self) -> None:
        prediction = _make_prediction(TrendDirection.BULLISH, price_at=Decimal("0"))

        assert prediction.is_trend_correct(Decimal("0.01"), dead_band=Decimal("0.001")) is True

    def test_negative_band_is_rejected(self) -> None:
        prediction = _make_prediction(TrendDirection.BULLISH)

        with pytest.raises(ValueError, match="dead_band"):
            prediction.is_trend_correct(Decimal("101"), dead_band=Decimal("-0.001"))


class TestAccuracyScore:
    def test_perfect_prediction_scores_one(self) -> None:
        prediction = _make_prediction(
            TrendDirection.BULLISH,
            price_at=Decimal("100.0"),
            target=Decimal("105.0"),
        )
        assert prediction.accuracy_score(Decimal("105.0")) == Decimal("1.0")

    def test_score_decreases_with_error(self) -> None:
        # error 5/100 = 0.05 → score = 1 - 0.05 = 0.95
        prediction = _make_prediction(
            TrendDirection.BULLISH,
            price_at=Decimal("100.0"),
            target=Decimal("105.0"),
        )
        assert prediction.accuracy_score(Decimal("110.0")) == Decimal("0.95")

    def test_score_floored_at_zero_for_huge_error(self) -> None:
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

    def test_is_trend_correct_does_not_raise_on_zero_price(self) -> None:
        prediction = _make_prediction(
            TrendDirection.SIDEWAYS, price_at=Decimal("0"), target=Decimal("0")
        )
        # SIDEWAYS dzieli actual_delta / price_at_prediction → 0/0.
        # Cena nie zmieniła się względem (nieznanego) 0 → traktujemy jak SIDEWAYS-correct.
        assert prediction.is_trend_correct(Decimal("0")) is True

    def test_accuracy_score_does_not_raise_on_zero_price(self) -> None:
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


class TestSidewaysToleranceFor:
    """Jedna reguła pasma „bez ruchu” dla reflect i recapu tygodniowego."""

    def test_crypto_uses_the_wide_band(self) -> None:
        assert sideways_tolerance_for(AssetType.CRYPTO) == CRYPTO_SIDEWAYS_TOLERANCE

    @pytest.mark.parametrize("asset_type", [AssetType.STOCK, AssetType.ETF, AssetType.OTHER, None])
    def test_everything_else_uses_the_equity_band(self, asset_type: AssetType | None) -> None:
        assert sideways_tolerance_for(asset_type) == SIDEWAYS_TOLERANCE


class TestNormalizeTrend:
    """Jedna reguła dla nowych odpowiedzi LLM (ML-07) i starych wierszy bazy:
    lowercase/spacje → kanon, brak → SIDEWAYS bez flagi, śmieć → SIDEWAYS z flagą."""

    @pytest.mark.parametrize(
        ("raw", "expected", "invalid"),
        [
            ("BULLISH", TrendDirection.BULLISH, False),
            ("bullish", TrendDirection.BULLISH, False),
            (" Bearish ", TrendDirection.BEARISH, False),
            (None, TrendDirection.SIDEWAYS, False),
            ("", TrendDirection.SIDEWAYS, False),
            ("UP", TrendDirection.SIDEWAYS, True),
        ],
    )
    def test_maps_raw_value_to_canonical_trend(
        self, raw: object, expected: TrendDirection, invalid: bool
    ) -> None:
        assert normalize_trend(raw) == (expected, invalid)


class TestSubstantiveInsight:
    """WIA-10: reflect_node zapisuje w `correction_insights` placeholdery, gdy nie
    ma prawdziwej diagnozy. Predykat odsiewa je (także historyczny tekst
    „odłożona”), żeby nie trafiały do lessons, analogów RAG, „dlaczego” i recapu."""

    def test_placeholders_are_not_substantive(self) -> None:
        from src.domain.prediction import (
            CORRECT_PREDICTION_INSIGHT,
            DIAGNOSIS_SKIPPED_INSIGHT,
            LEGACY_DIAGNOSIS_DEFERRED_INSIGHT,
            is_substantive_insight,
        )

        for placeholder in (
            CORRECT_PREDICTION_INSIGHT,
            DIAGNOSIS_SKIPPED_INSIGHT,
            LEGACY_DIAGNOSIS_DEFERRED_INSIGHT,
        ):
            assert not is_substantive_insight(placeholder)
            assert not is_substantive_insight(f"  {placeholder}\n")

    def test_historical_placeholder_texts_are_recognised(self) -> None:
        # Wiersze sprzed WIA-10 niosą dokładnie te literały — zostają w bazie.
        from src.domain.prediction import is_substantive_insight

        assert not is_substantive_insight("Niska zmienność — diagnoza LLM odłożona.")
        assert not is_substantive_insight("Trafiona predykcja.")

    def test_empty_and_missing_are_not_substantive(self) -> None:
        from src.domain.prediction import is_substantive_insight

        assert not is_substantive_insight(None)
        assert not is_substantive_insight("")
        assert not is_substantive_insight("   ")

    def test_real_diagnosis_is_substantive(self) -> None:
        from src.domain.prediction import is_substantive_insight

        assert is_substantive_insight("Zignorowałem jastrzębi ton Fed.")
        # Diagnoza wspominająca „nietrafioną” prognozę to wciąż diagnoza.
        assert is_substantive_insight("Prognoza nietrafiona, bo rynek wycenił Fed.")

    def test_skipped_placeholder_says_not_retried(self) -> None:
        # Diagnoza pominięta przy niskiej zmienności NIE jest ponawiana —
        # tekst nie może obiecywać „odłożenia”.
        from src.domain.prediction import DIAGNOSIS_SKIPPED_INSIGHT

        assert "odłożona" not in DIAGNOSIS_SKIPPED_INSIGHT
        assert "zmienno" in DIAGNOSIS_SKIPPED_INSIGHT.lower()


class TestPredictionDiagnosisInputs:
    """WIA-18: diagnoza błędu potrzebuje oryginalnych newsów i uzasadnienia.
    Pola są opcjonalne — stare wywołania konstruktora działają bez zmian."""

    def test_news_and_reasoning_default_to_none(self) -> None:
        pred = Prediction(
            symbol="AAPL",
            predicted_trend=TrendDirection.BULLISH,
            price_at_prediction=Decimal("100"),
            predicted_target_price=Decimal("105"),
        )

        assert pred.news_summary is None
        assert pred.reasoning is None

    def test_news_and_reasoning_are_carried(self) -> None:
        pred = Prediction(
            symbol="AAPL",
            predicted_trend=TrendDirection.BULLISH,
            price_at_prediction=Decimal("100"),
            predicted_target_price=Decimal("105"),
            id="p-1",
            news_summary="Fed holds | Chips rally",
            reasoning="Popyt na GPU rośnie.",
        )

        assert pred.news_summary == "Fed holds | Chips rally"
        assert pred.reasoning == "Popyt na GPU rośnie."


class TestBrierCalibrationScore:
    """WIA-37: `1 − |c − o|` jest regułą niewłaściwą — przy prawdziwej trafności
    0.6 zgłoszenie c = 1.0 daje więcej niż uczciwe c = 0.6. `1 − (c − o)²` jest
    właściwa: oczekiwany wynik jest największy przy c równym trafności."""

    @staticmethod
    def _expected(
        score: Callable[[float, bool], float], confidence: float, true_rate: float
    ) -> float:
        return true_rate * score(confidence, True) + (1 - true_rate) * score(confidence, False)

    def test_brier_calibration_score_is_proper(self) -> None:
        honest = self._expected(brier_calibration_score, 0.6, 0.6)
        overconfident = self._expected(brier_calibration_score, 1.0, 0.6)

        assert honest > overconfident

    def test_linear_score_rewards_overconfidence(self) -> None:
        # Dlatego v1 zostaje tylko do odczytu starych wierszy.
        honest = self._expected(calibration_score, 0.6, 0.6)
        overconfident = self._expected(calibration_score, 1.0, 0.6)

        assert overconfident > honest

    @pytest.mark.parametrize(
        ("confidence", "correct", "expected"),
        [(0.9, False, 0.19), (0.8, True, 0.96), (1.0, True, 1.0), (0.0, True, 0.0)],
    )
    def test_brier_calibration_score_values(
        self, confidence: float, correct: bool, expected: float
    ) -> None:
        assert brier_calibration_score(confidence, correct) == pytest.approx(expected)

    def test_score_versions_are_distinct(self) -> None:
        assert LINEAR_CALIBRATION_SCORE_VERSION == 1
        assert BRIER_CALIBRATION_SCORE_VERSION == 2


def test_stale_quote_note_is_a_placeholder_not_a_lesson() -> None:
    # DAT-14: rozliczenie na nieaktualnym notowaniu nie ma diagnozy — to nie lekcja.
    from src.domain.prediction import STALE_QUOTE_INSIGHT, is_substantive_insight

    assert not is_substantive_insight(STALE_QUOTE_INSIGHT)
