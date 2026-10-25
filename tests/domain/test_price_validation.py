"""DAT-15: odporny detektor anomalii ceny (MAD) przed bramką, snapshotem i rozliczeniem.

Zwrot odczytu od ostatniej ceny historii mierzymy w jednostkach MAD zwrotów historii
symbolu. Zły tick (0,01, cena innego instrumentu) leży setki MAD od mediany, zwykły
ruch kilka. `pc` notowania potwierdza prawdziwy skok: oficjalne zamknięcie poprzedniej
sesji już go niesie, a bieżąca cena leży blisko niego.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.domain.price_validation import (
    MIN_HISTORY_RETURNS,
    is_price_outlier,
    is_unconfirmed_outlier,
)


def _history(count: int = 21) -> list[Decimal]:
    """Ceny z dziennymi ruchami ±1% na przemian z ±0,5% (MAD zwrotów 0,75 pp)."""
    prices = [Decimal("100")]
    steps = (Decimal("1.01"), Decimal("0.99"), Decimal("1.005"), Decimal("0.995"))
    for index in range(count - 1):
        prices.append(prices[-1] * steps[index % len(steps)])
    return prices


def test_hundredfold_jump_is_outlier() -> None:
    history = _history()

    assert is_price_outlier(history[-1] * 100, history)


def test_tick_to_one_cent_is_outlier() -> None:
    assert is_price_outlier(Decimal("0.01"), _history())


def test_three_sigma_move_is_not_outlier() -> None:
    # σ zwrotów historii ≈ 0,8%, więc 3σ ≈ 2,4% — zwykły, choć duży dzień.
    history = _history()

    assert not is_price_outlier(history[-1] * Decimal("0.976"), history)


def test_short_history_never_flags() -> None:
    history = _history(MIN_HISTORY_RETURNS)

    assert not is_price_outlier(history[-1] * 100, history)


def test_flat_history_never_flags() -> None:
    # MAD = 0 (cena stała) nie daje skali — bez skali nie tłumimy odczytu.
    history = [Decimal("50")] * 30

    assert not is_price_outlier(Decimal("5000"), history)


def test_non_positive_price_is_always_an_outlier() -> None:
    assert is_price_outlier(Decimal("0"), [Decimal("100")])


def test_non_positive_history_price_is_ignored() -> None:
    history = [*_history(), Decimal("0")]

    assert is_price_outlier(history[-2] * 100, history)


class TestUnconfirmedOutlier:
    def test_jump_without_previous_close_stays_unconfirmed(self) -> None:
        history = _history()

        assert is_unconfirmed_outlier(history[-1] * Decimal("0.7"), history, prev_close=None)

    def test_previous_close_carrying_the_jump_confirms_it(self) -> None:
        # Spadek −30% po wynikach: zamknięcie poprzedniej sesji już go niesie,
        # bieżąca cena leży blisko niego — to prawdziwy ruch, nie zły tick.
        history = _history()
        shocked = history[-1] * Decimal("0.7")

        assert not is_unconfirmed_outlier(shocked * Decimal("1.004"), history, prev_close=shocked)

    def test_bad_tick_against_an_ordinary_previous_close_stays_unconfirmed(self) -> None:
        history = _history()

        assert is_unconfirmed_outlier(Decimal("0.01"), history, prev_close=history[-1])

    def test_ordinary_reading_is_never_an_outlier(self) -> None:
        history = _history()

        assert not is_unconfirmed_outlier(history[-1], history, prev_close=None)


@pytest.mark.parametrize("k", [Decimal("0"), Decimal("-1")])
def test_non_positive_k_is_rejected(k: Decimal) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        is_price_outlier(Decimal("100"), _history(), k=k)
