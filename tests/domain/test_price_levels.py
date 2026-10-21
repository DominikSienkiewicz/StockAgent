"""ALR-11 (część domenowa) — alert poziomu cenowego odpala przy PRZEKROCZENIU
poziomu między dwoma odczytami, a nie przy samym położeniu ceny po jego stronie.
Inaczej cena stojąca nad poziomem alarmowałaby w każdym przebiegu watcha."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.domain.price_levels import LevelDirection, PriceLevel, PriceLevelAlert, crossed


def _d(value: str) -> Decimal:
    return Decimal(value)


def test_crossing_requires_previous_on_other_side() -> None:
    assert crossed(_d("148"), _d("151"), _d("150"), LevelDirection.ABOVE) is True
    assert crossed(_d("152"), _d("155"), _d("150"), LevelDirection.ABOVE) is False


def test_falling_through_a_level_triggers_below() -> None:
    assert crossed(_d("152"), _d("149"), _d("150"), LevelDirection.BELOW) is True
    assert crossed(_d("148"), _d("145"), _d("150"), LevelDirection.BELOW) is False


def test_direction_matters() -> None:
    # Spadek przez 150 to nie „przebiła 150 w górę”.
    assert crossed(_d("152"), _d("149"), _d("150"), LevelDirection.ABOVE) is False
    assert crossed(_d("148"), _d("151"), _d("150"), LevelDirection.BELOW) is False


@pytest.mark.parametrize(
    ("previous", "current", "direction", "expected"),
    [
        # Dotknięcie poziomu od właściwej strony to już trafienie...
        ("149", "150", LevelDirection.ABOVE, True),
        ("151", "150", LevelDirection.BELOW, True),
        # ...a start z samego poziomu nie: trafienie zaliczył poprzedni odczyt.
        ("150", "151", LevelDirection.ABOVE, False),
        ("150", "149", LevelDirection.BELOW, False),
    ],
)
def test_touching_the_level_counts_once(
    previous: str, current: str, direction: LevelDirection, expected: bool
) -> None:
    assert crossed(_d(previous), _d(current), _d("150"), direction) is expected


class TestPriceLevelAlert:
    """ALR-11: alert z poprzednim odczytem watcha (`last_seen_price`) — pierwszy
    odczyt po dodaniu to punkt odniesienia, nie trafienie."""

    @staticmethod
    def _alert(last_seen: str | None) -> PriceLevelAlert:
        level = PriceLevel(symbol="NVDA", level=_d("150"), direction=LevelDirection.ABOVE)
        return PriceLevelAlert(
            id="1", level=level, last_seen_price=_d(last_seen) if last_seen else None
        )

    def test_hit_needs_the_previous_reading_on_the_other_side(self) -> None:
        assert self._alert("148").is_hit(_d("151"))
        assert not self._alert("152").is_hit(_d("155"))

    def test_first_reading_after_adding_the_level_is_not_a_hit(self) -> None:
        assert not self._alert(None).is_hit(_d("151"))

    @pytest.mark.parametrize("level", ["0", "-150", "NaN", "Infinity"])
    def test_level_must_be_a_positive_number(self, level: str) -> None:
        with pytest.raises(ValueError):
            PriceLevel(symbol="NVDA", level=_d(level), direction=LevelDirection.BELOW)
