"""ALR-11 (część domenowa) — przekroczenie poziomu cenowego ustawionego przez
użytkownika („daj znać, gdy NVDA przebije 150”).

Alert odpala przy PRZEKROCZENIU poziomu między poprzednim a bieżącym odczytem,
nie przy samym położeniu ceny po jego stronie — inaczej cena stojąca nad poziomem
alarmowałaby w każdym przebiegu watcha. Poprzedni odczyt nie może pochodzić ze
snapshotów Fast Loopa (watch ich nie zapisuje), więc żyje przy alercie
(`last_seen_price`, tabela `price_level_alerts`, migracja 057). Tylko stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class PriceLevel:
    """Poziom ustawiony przez użytkownika: symbol, cena, kierunek i opcjonalna
    notatka („realizacja części zysku”). Źródło startowe: `price_alerts` w config.toml."""

    symbol: str
    level: Decimal
    direction: LevelDirection
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.level.is_finite() or self.level <= 0:
            raise ValueError(f"price level must be a positive number, got {self.level}")


@dataclass(frozen=True)
class PriceLevelAlert:
    """Aktywny alert poziomu z ostatnim odczytem ceny, który widział watch.

    Bez poprzedniego odczytu (`last_seen_price` None, pierwszy przebieg po dodaniu)
    alert nie odpala: cena po drugiej stronie poziomu w chwili jego dodania to
    stan wyjściowy, nie przekroczenie."""

    id: str
    level: PriceLevel
    last_seen_price: Decimal | None = None

    def is_hit(self, current: Decimal) -> bool:
        """Czy odczyt `current` przekroczył poziom od poprzedniego odczytu."""
        if self.last_seen_price is None:
            return False
        return crossed(self.last_seen_price, current, self.level.level, self.level.direction)
