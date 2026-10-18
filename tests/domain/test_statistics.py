"""Testy przedziału ufności Wilsona (WIA-32) — czysty stdlib.

Goły procent trafności bez przedziału i liczności jest nieweryfikowalny:
52% z 10 predykcji i 52% z 1000 to zupełnie inne dowody.
"""

from __future__ import annotations

import pytest

from src.domain.statistics import wilson_interval


def test_wilson_interval_known_value() -> None:
    low, high = wilson_interval(50, 100)

    assert low == pytest.approx(0.404, abs=1e-3)
    assert high == pytest.approx(0.596, abs=1e-3)


def test_interval_stays_inside_unit_range_at_the_edges() -> None:
    # Wilson (w odróżnieniu od przybliżenia Walda) nie wychodzi poza [0, 1]
    # i nie zapada się do zera szerokości przy 0 albo n trafieniach.
    low, high = wilson_interval(10, 10)

    assert 0.0 < low < 1.0
    assert high == pytest.approx(1.0)
    assert wilson_interval(0, 10)[0] == pytest.approx(0.0)


def test_small_sample_gives_a_wide_interval() -> None:
    low, high = wilson_interval(2, 3)

    assert high - low > 0.5


def test_accepts_fractional_effective_counts() -> None:
    # n_eff (WIA-23) bywa ułamkowe — przedział liczony na efektywnej liczności.
    low, high = wilson_interval(30.5, 61.0)

    assert low < 0.5 < high


def test_empty_sample_is_maximally_uncertain() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
