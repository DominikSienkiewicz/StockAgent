"""ALR-05 (krok 2): zmiana reżimu warta leadu maila — wejście w RISK_OFF albo
wyjście z niego, liczona wyłącznie z reżimu efektywnego po histerezie."""

from __future__ import annotations

import pytest

from src.domain.regime import MarketRegime, RegimeChange, risk_off_change


def test_entering_risk_off_is_a_change() -> None:
    change = risk_off_change(MarketRegime.NEUTRAL, MarketRegime.RISK_OFF)

    assert change == RegimeChange(previous=MarketRegime.NEUTRAL, current=MarketRegime.RISK_OFF)
    assert change.enters_risk_off


def test_leaving_risk_off_is_a_change() -> None:
    change = risk_off_change(MarketRegime.RISK_OFF, MarketRegime.RISK_ON)

    assert change is not None
    assert not change.enters_risk_off


@pytest.mark.parametrize(
    ("previous", "current"),
    [
        (MarketRegime.RISK_OFF, MarketRegime.RISK_OFF),
        (MarketRegime.NEUTRAL, MarketRegime.RISK_ON),
        (MarketRegime.RISK_ON, MarketRegime.NEUTRAL),
        (None, MarketRegime.RISK_OFF),
    ],
)
def test_no_change_without_crossing_risk_off_or_without_history(
    previous: MarketRegime | None, current: MarketRegime
) -> None:
    assert risk_off_change(previous, current) is None
