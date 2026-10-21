"""ALR-11 (część domenowa) — przekroczenie poziomu cenowego ustawionego przez
użytkownika („daj znać, gdy NVDA przebije 150”).

Alert odpala przy PRZEKROCZENIU poziomu między poprzednim a bieżącym odczytem,
nie przy samym położeniu ceny po jego stronie — inaczej cena stojąca nad poziomem
alarmowałaby w każdym przebiegu watcha. Poprzedni odczyt nie może pochodzić ze
snapshotów Fast Loopa (watch ich nie zapisuje), więc docelowo żyje przy alercie
(`last_seen_price`). Czysta funkcja, tylko stdlib.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum


class LevelDirection(Enum):
    """ABOVE — cena przebija poziom w górę, BELOW — spada przez poziom."""

    ABOVE = "ABOVE"
    BELOW = "BELOW"


def crossed(
    previous: Decimal, current: Decimal, level: Decimal, direction: LevelDirection
) -> bool:
    """Czy cena przekroczyła `level` w kierunku `direction` między dwoma odczytami.

    Poprzedni odczyt musi leżeć po drugiej stronie poziomu, bieżący na nim albo za
    nim. Start z samego poziomu nie liczy się: trafienie zaliczył już odczyt, który
    na poziom wszedł — jedno przejście daje jeden alert."""
    if direction is LevelDirection.ABOVE:
        return previous < level <= current
    return previous > level >= current
