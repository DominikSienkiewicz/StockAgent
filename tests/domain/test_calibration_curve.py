"""Testy domeny krzywej kalibracji zaufania (reliability diagram).

Czysta statystyka — partycjonujemy próbki (confidence, was_correct) na
kubełki o równej szerokości, liczymy hit_rate vs. mean_confidence per kubełek
oraz ECE (Expected Calibration Error). Zero zależności zewnętrznych.

To NIE jest sędzia LLM z `domain/prediction.py` — to czysto liczbowy
diagram niezawodności predykcji.
"""

from __future__ import annotations

from src.domain.calibration_curve import (
    CalibrationBucket,
    expected_calibration_error,
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
        assert curve[0].lower == 0.0
        assert curve[0].upper == 0.1
        assert curve[0].count == 3

    def test_confidence_one_lands_in_last_bucket(self) -> None:
        # confidence == 1.0 nie wypada poza zakres — trafia do ostatniego
        # kubełka [0.9, 1.0].
        curve = reliability_curve([(1.0, True)], n_buckets=10)
        assert len(curve) == 1
        bucket = curve[0]
        assert isinstance(bucket, CalibrationBucket)
        assert bucket.lower == 0.9
        assert bucket.upper == 1.0
        assert bucket.count == 1
        assert bucket.mean_confidence == 1.0
        assert bucket.hit_rate == 1.0

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
        assert bucket.hit_rate == 0.9

    def test_assigns_samples_to_correct_buckets(self) -> None:
        # 0.05 → [0.0,0.1); 0.15 → [0.1,0.2); 0.95 → [0.9,1.0].
        samples = [(0.05, True), (0.15, False), (0.95, True)]
        curve = reliability_curve(samples, n_buckets=10)
        lowers = [round(b.lower, 1) for b in curve]
        assert lowers == [0.0, 0.1, 0.9]


class TestExpectedCalibrationError:
    def test_no_buckets_is_zero(self) -> None:
        assert expected_calibration_error([]) == 0.0

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
