"""DAT-23: detekcja splitów z `pc` (/quote) względem ostatniego snapshotu ceny.

Przy cyklu sprzed otwarcia `pc` to zamknięcie sesji sprzed ostatniej, czyli dokładnie
cena poprzedniego snapshotu. Stosunek `pc / snapshot` daleko od 1 i bliski stosunkowi
splitu znaczy, że dostawca skorygował historię — zwykły ruch ceny widać w `c / pc`.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from src.domain.corporate_actions import (
    SPLIT_RATIOS,
    CorporateActionKind,
    SplitSuspicion,
    detect_split,
    to_pre_split_price,
)

_AT = datetime(2026, 10, 27, 5, 30, tzinfo=UTC)


def test_ratio_near_one_tenth_is_split_10_for_1() -> None:
    # NVDA 10:1 — snapshot 1200 sprzed splitu, skorygowane `pc` 120,6 po nim.
    suspicion = detect_split("NVDA", Decimal("120.6"), Decimal("1200"), at=_AT)

    assert suspicion == SplitSuspicion(
        symbol="NVDA",
        ratio=Decimal("0.1"),
        prev_close=Decimal("120.6"),
        reference_price=Decimal("1200"),
        detected_at=_AT,
    )
    assert suspicion.kind is CorporateActionKind.SPLIT_SUSPECTED
    assert suspicion.detected_on == date(2026, 10, 27)


def test_ordinary_move_is_not_split() -> None:
    # Zwykły dzień: `pc` to cena poprzedniego snapshotu z dokładnością do szumu.
    assert detect_split("AAPL", Decimal("101.5"), Decimal("100"), at=_AT) is None


def test_reverse_split_doubles_the_reference() -> None:
    suspicion = detect_split("XYZ", Decimal("39.4"), Decimal("20"), at=_AT)

    assert suspicion is not None
    assert suspicion.ratio == Decimal("2")


def test_ratio_outside_the_tolerance_is_not_split() -> None:
    # 0,47 jest 6% od 1/2 — poza tolerancją ±2%.
    assert detect_split("XYZ", Decimal("47"), Decimal("100"), at=_AT) is None


def test_ratio_within_the_tolerance_of_one_third() -> None:
    suspicion = detect_split("XYZ", Decimal("33.0"), Decimal("100"), at=_AT)

    assert suspicion is not None
    assert suspicion.ratio == Decimal(1) / Decimal(3)


@pytest.mark.parametrize("reference", [Decimal("0"), Decimal("-5")])
def test_no_reference_price_never_flags(reference: Decimal) -> None:
    assert detect_split("XYZ", Decimal("10"), reference, at=_AT) is None


def test_non_positive_prev_close_never_flags() -> None:
    assert detect_split("XYZ", Decimal("0"), Decimal("100"), at=_AT) is None


def test_split_ratios_cover_the_catalogue() -> None:
    expected = {Decimal(1) / Decimal(n) for n in (2, 3, 4, 5, 10)} | {Decimal(2), Decimal(3)}
    assert set(SPLIT_RATIOS) == expected


def test_settlement_price_returns_to_the_units_of_the_prediction() -> None:
    # Predykcja z ceną 1000 przed splitem 10:1, cena po splicie 95 → 950 sprzed splitu.
    assert to_pre_split_price(Decimal("95"), Decimal("0.1")) == Decimal("950")


def test_naive_detection_moment_is_utc() -> None:
    suspicion = detect_split(
        "NVDA", Decimal("120"), Decimal("1200"), at=datetime(2026, 10, 27, 23, 30)
    )

    assert suspicion is not None
    assert suspicion.detected_on == date(2026, 10, 27)
