"""Testy domeny earnings — klasyfikacja bliskości raportu wyników."""

from __future__ import annotations

from src.domain.earnings import EarningsEvent, EarningsProximity


def test_proximity_imminent_within_two_days() -> None:
    # Domyślny próg IMMINENT to <= 2 dni.
    assert (
        EarningsEvent("AAPL", days_until=0).proximity()
        is EarningsProximity.IMMINENT
    )
    assert (
        EarningsEvent("AAPL", days_until=2).proximity()
        is EarningsProximity.IMMINENT
    )


def test_proximity_near_within_a_week() -> None:
    # Powyżej IMMINENT, ale w obrębie tygodnia → NEAR.
    assert (
        EarningsEvent("AAPL", days_until=3).proximity() is EarningsProximity.NEAR
    )
    assert (
        EarningsEvent("AAPL", days_until=7).proximity() is EarningsProximity.NEAR
    )


def test_proximity_far_beyond_week() -> None:
    assert (
        EarningsEvent("AAPL", days_until=8).proximity() is EarningsProximity.FAR
    )
    assert (
        EarningsEvent("AAPL", days_until=90).proximity() is EarningsProximity.FAR
    )


def test_proximity_none_is_far() -> None:
    # Brak danych o nadchodzącym raporcie → FAR.
    assert (
        EarningsEvent("AAPL", days_until=None).proximity()
        is EarningsProximity.FAR
    )


def test_proximity_negative_is_far() -> None:
    # Raport w przeszłości (ujemne dni) → FAR.
    assert (
        EarningsEvent("AAPL", days_until=-1).proximity() is EarningsProximity.FAR
    )


def test_proximity_respects_custom_thresholds() -> None:
    event = EarningsEvent("AAPL", days_until=5)
    assert (
        event.proximity(imminent_days=5, near_days=10)
        is EarningsProximity.IMMINENT
    )
    assert (
        event.proximity(imminent_days=1, near_days=3) is EarningsProximity.FAR
    )
