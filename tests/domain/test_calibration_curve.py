"""Testy domeny krzywej kalibracji zaufania (reliability diagram).

Czysta statystyka — partycjonujemy próbki (confidence, was_correct) na
kubełki o równej szerokości, liczymy hit_rate vs. mean_confidence per kubełek
oraz ECE (Expected Calibration Error). Zero zależności zewnętrznych.

To NIE jest sędzia LLM z `domain/prediction.py` — to czysto liczbowy
diagram niezawodności predykcji.
"""

from __future__ import annotations

import math

import pytest

from src.domain.calibration_curve import (
    CalibrationBucket,
    brier_score,
    calibrated_confidence,
    expected_calibration_error,
    log_loss,
    murphy_decomposition,
    probability_scores,
    reliability_curve,
)


class TestReliabilityCurve:
    def test_empty_input_yields_empty_curve(self) -> None:
        # Brak próbek = brak kubełków.
        assert reliability_curve([]) == []

    def test_omits_empty_buckets(self) -> None:
        # Wszystkie próbki w jednym kubełku (0.1) → tylko on jest zwracany,
        # pozostałe 9 kubełków (puste) są pomijane.
        samples = [(0.05, True), (0.05, False), (0.08, True)]
        curve = reliability_curve(samples, n_buckets=10)
        assert len(curve) == 1
        assert curve[0].lower == pytest.approx(0.0)
        assert curve[0].upper == pytest.approx(0.1)
        assert curve[0].count == 3

    def test_confidence_one_lands_in_last_bucket(self) -> None:
        # confidence == 1.0 nie wypada poza zakres — trafia do ostatniego
        # kubełka [0.9, 1.0].
        curve = reliability_curve([(1.0, True)], n_buckets=10)
        assert len(curve) == 1
        bucket = curve[0]
        assert isinstance(bucket, CalibrationBucket)
        assert bucket.lower == pytest.approx(0.9)
        assert bucket.upper == pytest.approx(1.0)
        assert bucket.count == 1
        assert bucket.mean_confidence == pytest.approx(1.0)
        assert bucket.hit_rate == pytest.approx(1.0)

    def test_buckets_returned_in_ascending_order(self) -> None:
        samples = [(0.95, True), (0.05, False), (0.55, True)]
        curve = reliability_curve(samples, n_buckets=10)
        lowers = [b.lower for b in curve]
        assert lowers == sorted(lowers)

    def test_per_bucket_count_mean_and_hit_rate(self) -> None:
        # Kubełek [0.9, 1.0]: 10 próbek, 9 poprawnych → hit_rate 0.9.
        samples: list[tuple[float, bool]] = [(0.9, True)] * 9 + [(0.9, False)]
        curve = reliability_curve(samples, n_buckets=10)
        assert len(curve) == 1
        bucket = curve[0]
        assert bucket.count == 10
        # Średnia z dziesięciu 0.9 to ~0.9 z dokładnością zmiennoprzecinkową.
        assert abs(bucket.mean_confidence - 0.9) < 1e-9
        assert bucket.hit_rate == pytest.approx(0.9)

    def test_assigns_samples_to_correct_buckets(self) -> None:
        # 0.05 → [0.0,0.1); 0.15 → [0.1,0.2); 0.95 → [0.9,1.0].
        samples = [(0.05, True), (0.15, False), (0.95, True)]
        curve = reliability_curve(samples, n_buckets=10)
        lowers = [round(b.lower, 1) for b in curve]
        assert lowers == [0.0, 0.1, 0.9]


class TestExpectedCalibrationError:
    def test_no_buckets_is_zero(self) -> None:
        assert expected_calibration_error([]) == pytest.approx(0.0)

    def test_perfectly_calibrated_set_has_near_zero_ece(self) -> None:
        # Idealna kalibracja: w kubełku 0.9 dokładnie 9/10 trafień,
        # w kubełku 0.1 dokładnie 1/10 trafień → ECE ≈ 0.
        high = [(0.9, True)] * 9 + [(0.9, False)]
        low = [(0.1, True)] + [(0.1, False)] * 9
        curve = reliability_curve([*high, *low], n_buckets=10)
        ece = expected_calibration_error(curve)
        assert ece < 1e-9

    def test_overconfident_set_has_high_ece(self) -> None:
        # Systematyczna nadmierna pewność: model deklaruje 0.95, a trafia
        # tylko w połowie przypadków → |0.5 - 0.95| = 0.45 na cały zbiór.
        samples = [(0.95, i % 2 == 0) for i in range(10)]
        curve = reliability_curve(samples, n_buckets=10)
        ece = expected_calibration_error(curve)
        assert ece > 0.4

    def test_ece_is_count_weighted(self) -> None:
        # Kubełek A: 9 próbek, 1 trafienie → hit_rate = 1/9,
        #   mean_conf 0.1 → |diff| = |1/9 − 0.1| = 1/90.
        # Kubełek B: 1 próbka, 0 trafień → hit_rate 0.0,
        #   mean_conf 0.9 → |diff| = 0.9.
        # ECE count-weighted = (9 * 1/90 + 1 * 0.9) / 10 = (0.1 + 0.9)/10 = 0.1,
        # a nie zwykła średnia po kubełkach (≈ 0.4556).
        bucket_a = [(0.1, True)] + [(0.1, False)] * 8
        bucket_b = [(0.9, False)]
        curve = reliability_curve([*bucket_a, *bucket_b], n_buckets=10)
        ece = expected_calibration_error(curve)
        assert abs(ece - 0.1) < 1e-9


def _bucket(
    lower: float, upper: float, count: int, hit_rate: float
) -> CalibrationBucket:
    # Pomocnik: mean_confidence nieistotny dla kalibracji, ustawiamy w środku.
    return CalibrationBucket(
        lower=lower,
        upper=upper,
        count=count,
        mean_confidence=(lower + upper) / 2,
        hit_rate=hit_rate,
    )


class TestCalibratedConfidence:
    # ---- Kontrakt graceful: brak historii → raw bez zmian ----

    def test_empty_curve_returns_raw_unchanged(self) -> None:
        # Zamrożenie ryzyka (c): pusta krzywa → zachowanie identyczne jak dziś.
        assert calibrated_confidence(0.85, []) == pytest.approx(0.85)

    def test_no_matching_bucket_returns_raw(self) -> None:
        # raw=0.85 nie wpada do żadnego dostarczonego kubełka [0.0,0.1).
        buckets = [_bucket(0.0, 0.1, 50, hit_rate=0.05)]
        assert calibrated_confidence(0.85, buckets) == pytest.approx(0.85)

    # ---- Zamrożenie shrinkage (największe ryzyko: niestacjonarność) ----

    def test_tiny_sample_barely_moves_raw(self) -> None:
        # (a) count=1 przy min_count=10 → waga = 1/11 ≈ 0.0909.
        # Kubełek [0.8,0.9] ma hit_rate 0.2, raw=0.85.
        # Oczekiwane: 0.85 + (1/11)*(0.2 - 0.85) = 0.85 - 0.0590909... ≈ 0.79090909.
        buckets = [_bucket(0.8, 0.9, 1, hit_rate=0.2)]
        result = calibrated_confidence(0.85, buckets, min_count=10)
        assert result == pytest.approx(0.85 + (1 / 11) * (0.2 - 0.85))
        # Prawie nie odbiega od raw — mała próbka nie przeciąga.
        assert abs(result - 0.85) < 0.06

    def test_large_sample_pulls_raw_close_to_hit_rate(self) -> None:
        # (b) count=100 przy min_count=10 → waga = 100/110 ≈ 0.909.
        # Kubełek [0.8,0.9] ma hit_rate 0.55, raw=0.85.
        # Oczekiwane: 0.85 + (100/110)*(0.55 - 0.85) ≈ 0.5772727.
        buckets = [_bucket(0.8, 0.9, 100, hit_rate=0.55)]
        result = calibrated_confidence(0.85, buckets, min_count=10)
        assert result == pytest.approx(0.85 + (100 / 110) * (0.55 - 0.85))
        # Wynik blisko hit_rate kubełka, wyraźnie poniżej raw.
        assert abs(result - 0.55) < 0.03

    def test_shrinkage_weight_is_count_over_count_plus_min_count(self) -> None:
        # Jawna formuła: waga = count / (count + min_count).
        # count=30, min_count=10 → waga = 30/40 = 0.75.
        buckets = [_bucket(0.8, 0.9, 30, hit_rate=0.4)]
        result = calibrated_confidence(0.85, buckets, min_count=10)
        expected = 0.85 + 0.75 * (0.4 - 0.85)
        assert result == pytest.approx(expected)

    def test_default_min_count_is_ten(self) -> None:
        # Domyślny min_count=10 → waga = count/(count+10).
        buckets = [_bucket(0.8, 0.9, 10, hit_rate=0.5)]
        result = calibrated_confidence(0.85, buckets)
        expected = 0.85 + (10 / 20) * (0.5 - 0.85)
        assert result == pytest.approx(expected)

    # ---- Dopasowanie kubełka wg semantyki reliability_curve ----

    def test_raw_on_lower_edge_matches_that_bucket(self) -> None:
        # raw=0.8 wpada do [0.8,0.9), nie do [0.7,0.8).
        buckets = [
            _bucket(0.7, 0.8, 100, hit_rate=0.1),
            _bucket(0.8, 0.9, 100, hit_rate=0.9),
        ]
        result = calibrated_confidence(0.8, buckets, min_count=10)
        # Ciągnie w stronę 0.9 (górny kubełek), nie 0.1.
        assert result > 0.8

    def test_raw_one_matches_top_bucket(self) -> None:
        # raw=1.0 wpada do ostatniego kubełka [0.9,1.0] (górna krawędź).
        buckets = [_bucket(0.9, 1.0, 100, hit_rate=0.7)]
        result = calibrated_confidence(1.0, buckets, min_count=10)
        assert result == pytest.approx(1.0 + (100 / 110) * (0.7 - 1.0))

    # ---- Wynik zawsze w [0, 1] ----

    def test_result_clamped_to_unit_interval(self) -> None:
        # Nawet gdyby dane były zdegenerowane, wynik nie wychodzi poza [0,1].
        buckets = [_bucket(0.9, 1.0, 100, hit_rate=1.0)]
        result = calibrated_confidence(1.0, buckets, min_count=10)
        assert 0.0 <= result <= 1.0


class TestProbabilityScores:
    """WIA-36: Brier, log loss i rozkład Murphy'ego. ECE jest 0 dla prognozy
    „zawsze 55%” przy 55% trafień — czyli przy ZEROWEJ umiejętności — więc nie
    może być główną metryką pewności."""

    def test_brier_of_perfect_forecast_is_zero(self) -> None:
        assert brier_score([(1.0, True), (0.0, False), (1.0, True)]) == pytest.approx(0.0)

    def test_brier_is_mean_squared_error_of_confidence(self) -> None:
        # (0.8 - 1)² = 0.04, (0.6 - 0)² = 0.36 → średnia 0.2.
        assert brier_score([(0.8, True), (0.6, False)]) == pytest.approx(0.2)

    def test_log_loss_clips_certain_misses(self) -> None:
        # Pewność 1.0 przy pudle dałaby nieskończoność — eps ją ogranicza.
        loss = log_loss([(1.0, False)], eps=1e-6)

        assert loss == pytest.approx(-math.log(1e-6))

    def test_log_loss_of_coin_flip_is_ln_two(self) -> None:
        assert log_loss([(0.5, True), (0.5, False)]) == pytest.approx(math.log(2))

    def test_murphy_components_reconstruct_brier_when_forecasts_constant_within_bucket(
        self,
    ) -> None:
        samples = (
            [(0.85, True)] * 3 + [(0.85, False)]
            + [(0.35, True)] + [(0.35, False)] * 3
        )
        reliability, resolution, uncertainty = murphy_decomposition(
            reliability_curve(samples)
        )

        assert reliability - resolution + uncertainty == pytest.approx(brier_score(samples))
        assert uncertainty == pytest.approx(0.25)

    def test_constant_base_rate_forecast_has_zero_skill(self) -> None:
        # „Zawsze 60%” przy 60% trafień: ECE = 0, a umiejętności brak (BSS = 0).
        samples = [(0.6, True)] * 6 + [(0.6, False)] * 4

        scores = probability_scores(samples)

        assert scores is not None
        assert scores.skill == pytest.approx(0.0)
        assert scores.brier == pytest.approx(scores.climatology_brier)
        assert expected_calibration_error(reliability_curve(samples)) < 1e-9

    def test_skill_is_undefined_when_outcomes_never_vary(self) -> None:
        scores = probability_scores([(0.7, True), (0.9, True)])

        assert scores is not None
        assert scores.climatology_brier == pytest.approx(0.0)
        assert scores.skill is None

    def test_no_samples_no_scores(self) -> None:
        assert probability_scores([]) is None
        assert brier_score([]) == pytest.approx(0.0)
        assert murphy_decomposition([]) == (0.0, 0.0, 0.0)
