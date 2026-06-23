"""Testy domeny hedge — Hedge-Effectiveness Score (R3).

Czysta domena, stdlib only. Sprawdzamy: realized_correlation (Pearson z
truncacją, bramką na <2 punkty i zerową wariancję), hedge_effectiveness
(odwrócenie znaku dla hedge'a inverse) oraz assess_hedge (etykiety po polsku
na granicach progu).
"""

from __future__ import annotations

import math

import pytest

from src.domain.hedge import (
    HedgeAssessment,
    assess_hedge,
    hedge_effectiveness,
    realized_correlation,
)


def test_perfectly_inverse_series_has_correlation_minus_one() -> None:
    """Książka w górę, hedge w idealne lustro w dół → korelacja -1."""
    book = [1.0, 2.0, 3.0, 4.0]
    hedge = [4.0, 3.0, 2.0, 1.0]
    assert math.isclose(realized_correlation(book, hedge), -1.0, abs_tol=1e-9)


def test_perfectly_inverse_hedge_is_effective() -> None:
    """Inverse hedge poruszający się przeciwnie → effectiveness +1, "skuteczny"."""
    book = [1.0, 2.0, 3.0, 4.0]
    hedge = [4.0, 3.0, 2.0, 1.0]
    eff = hedge_effectiveness(book, hedge, expect_inverse=True)
    assert math.isclose(eff, 1.0, abs_tol=1e-9)
    assessment = assess_hedge("SH", book, hedge, expect_inverse=True)
    assert assessment.label == "skuteczny"


def test_positively_correlated_with_expect_inverse_is_inverted() -> None:
    """Para dodatnio skorelowana, ale oczekiwany inverse → effectiveness -1, "odwrócony"."""
    book = [1.0, 2.0, 3.0, 4.0]
    hedge = [1.0, 2.0, 3.0, 4.0]
    eff = hedge_effectiveness(book, hedge, expect_inverse=True)
    assert math.isclose(eff, -1.0, abs_tol=1e-9)
    assessment = assess_hedge("QQQ", book, hedge, expect_inverse=True)
    assert assessment.label == "odwrócony"


def test_expect_inverse_false_keeps_correlation_sign() -> None:
    """Bez oczekiwanego odwrócenia effectiveness = +korelacja."""
    book = [1.0, 2.0, 3.0, 4.0]
    hedge = [1.0, 2.0, 3.0, 4.0]
    eff = hedge_effectiveness(book, hedge, expect_inverse=False)
    assert math.isclose(eff, 1.0, abs_tol=1e-9)


def test_zero_variance_returns_zero() -> None:
    """Stała seria (zerowa wariancja) → korelacja 0.0, brak mierzalnej zależności."""
    book = [5.0, 5.0, 5.0, 5.0]
    hedge = [1.0, 2.0, 3.0, 4.0]
    assert realized_correlation(book, hedge) == pytest.approx(0.0)
    assert hedge_effectiveness(book, hedge, expect_inverse=True) == pytest.approx(0.0)


def test_too_short_series_returns_zero() -> None:
    """Mniej niż 2 punkty → 0.0 (nie da się policzyć korelacji)."""
    assert realized_correlation([1.0], [2.0]) == pytest.approx(0.0)
    assert realized_correlation([], []) == pytest.approx(0.0)


def test_unequal_length_truncates_to_shorter() -> None:
    """Różne długości → truncacja do krótszego, ogon hedge'a ignorowany."""
    book = [1.0, 2.0, 3.0]
    hedge = [3.0, 2.0, 1.0, 999.0, -999.0]
    # Po truncacji to idealne lustro [1,2,3] vs [3,2,1] → -1.
    assert math.isclose(realized_correlation(book, hedge), -1.0, abs_tol=1e-9)


def test_correlation_in_range() -> None:
    """Korelacja zawsze w [-1, 1] mimo błędów zmiennoprzecinkowych."""
    book = [0.1, -0.2, 0.3, -0.15, 0.25]
    hedge = [-0.1, 0.18, -0.31, 0.16, -0.24]
    corr = realized_correlation(book, hedge)
    assert -1.0 <= corr <= 1.0


def test_assess_hedge_returns_frozen_dataclass() -> None:
    """assess_hedge zwraca zamrożony HedgeAssessment z poprawnymi polami."""
    book = [1.0, 2.0, 3.0, 4.0]
    hedge = [4.0, 3.0, 2.0, 1.0]
    assessment = assess_hedge("SH", book, hedge, expect_inverse=True)
    assert isinstance(assessment, HedgeAssessment)
    assert assessment.hedge_symbol == "SH"
    assert math.isclose(assessment.correlation, -1.0, abs_tol=1e-9)
    assert math.isclose(assessment.effectiveness, 1.0, abs_tol=1e-9)


def test_label_at_effective_threshold_boundary() -> None:
    """Effectiveness dokładnie == próg → "skuteczny" (granica włącznie)."""
    # Konstruujemy parę dającą korelację -0.3, więc effectiveness == +0.3.
    book = [1.0, 2.0, 3.0]
    # Dobieramy hedge tak, by korelacja wyszła -0.3 (sprawdzane w teście).
    # Wystarczy dowolny szereg — przeliczamy realny próg z faktycznej korelacji.
    hedge = [3.0, 2.5, 1.4]
    corr = realized_correlation(book, hedge)
    threshold = -corr  # tak by effectiveness == próg
    assessment = assess_hedge(
        "SH", book, hedge, expect_inverse=True, effective_threshold=threshold
    )
    assert assessment.label == "skuteczny"


def test_label_just_below_threshold_is_ineffective() -> None:
    """Effectiveness w (-próg, próg) → "nieskuteczny"."""
    book = [1.0, 2.0, 3.0, 4.0]
    # Korelacja bliska zera → effectiveness bliskie zera, w środku pasma.
    hedge = [2.0, 1.0, 2.5, 1.5]
    assessment = assess_hedge(
        "DUMMY", book, hedge, expect_inverse=True, effective_threshold=0.9
    )
    assert assessment.label == "nieskuteczny"


def test_label_at_negative_threshold_boundary_is_inverted() -> None:
    """Effectiveness == -próg → "odwrócony" (granica włącznie)."""
    book = [1.0, 2.0, 3.0]
    hedge = [1.0, 1.5, 2.6]
    corr = realized_correlation(book, hedge)  # dodatnia
    # effectiveness = -corr; chcemy effectiveness <= -próg, więc próg = corr.
    threshold = corr
    assessment = assess_hedge(
        "QQQ", book, hedge, expect_inverse=True, effective_threshold=threshold
    )
    assert assessment.label == "odwrócony"
