"""Testy minimalnych prób w matematyce ryzyka (PORT-22).

VaR 95% z kilku obserwacji to najgorszy dzień próbki przebrany za kwantyl,
a korelacja z dwóch zwrotów wynosi ±1. Progi żyją w domenie; `historical_var`,
`pearson` i `beta_to_portfolio` liczą dalej to, co dostaną.
"""

from __future__ import annotations

from src.domain.risk_sample import (
    MIN_PAIRED_RETURNS,
    MIN_VAR_OBSERVATIONS,
    Sufficiency,
    sample_sufficiency,
)


def test_var_requires_60_observations() -> None:
    assert sample_sufficiency(59, "var").ok is False
    assert sample_sufficiency(60, "var").ok is True
    assert sample_sufficiency(19, "beta").ok is False


def test_correlation_and_beta_require_20_paired_returns() -> None:
    assert sample_sufficiency(19, "correlation").ok is False
    assert sample_sufficiency(20, "correlation").ok is True
    assert sample_sufficiency(20, "beta").ok is True


def test_sufficiency_carries_the_sample_and_the_requirement() -> None:
    assert sample_sufficiency(9, "var") == Sufficiency(ok=False, n=9, required=60)


def test_thresholds_are_the_owner_decision_of_2026_10_19() -> None:
    assert MIN_VAR_OBSERVATIONS == 60
    assert MIN_PAIRED_RETURNS == 20
