"""Poziom unieważnienia tezy predykcji — „co by zmieniło zdanie” (ML-27).

Mail mówi, co model myśli, ale nie kiedy przestać mu wierzyć. Za flagą
`invalidation_level_enabled` LLM podaje obok kierunku poziom ceny, którego
przekroczenie obala tezę (np. „BULLISH upada poniżej 118”), a watch alarmuje,
gdy cena go przekroczy. To falsyfikowalna część predykcji, więc jej reguły żyją
tutaj, a nie w prompcie:

  * BULLISH — poziom POD ceną z chwili predykcji, BEARISH — NAD nią, SIDEWAYS —
    po dowolnej stronie (wyjście z pasma w którąkolwiek stronę obala „bez ruchu”);
  * najdalej `INVALIDATION_BAND` (15%) od ceny i nie na samej cenie — dalszy
    poziom to zmyślone „wsparcie”, którego horyzont 12 h i tak nie sprawdzi.

Zero I/O, tylko stdlib i domena.
"""

from __future__ import annotations

from decimal import Decimal

from src.domain.prediction import TrendDirection

# Najdalsza dopuszczalna odległość poziomu od ceny predykcji (ułamek).
INVALIDATION_BAND = Decimal("0.15")


def is_valid_invalidation(
    trend: TrendDirection, current_price: Decimal, level: Decimal
) -> bool:
    """Czy `level` jest poziomem unieważnienia tezy `trend` przy cenie `current_price`:
    po właściwej stronie ceny i najdalej o `INVALIDATION_BAND` od niej."""
    if current_price <= 0 or level <= 0 or level == current_price:
        return False
    distance = (level - current_price) / current_price
    if abs(distance) > INVALIDATION_BAND:
        return False
    if trend is TrendDirection.BULLISH:
        return distance < 0
    if trend is TrendDirection.BEARISH:
        return distance > 0
    return True


def is_thesis_invalidated(entry_price: Decimal, level: Decimal, price: Decimal) -> bool:
    """Czy cena `price` przekroczyła poziom `level` od strony ceny wejścia: poziom pod
    wejściem pęka przy spadku do niego, poziom nad wejściem — przy wzroście."""
    if level < entry_price:
        return price <= level
    return price >= level
