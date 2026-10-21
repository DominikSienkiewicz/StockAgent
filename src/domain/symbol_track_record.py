"""Domena: trafność kierunkowa per symbol (WIA-35).

Globalny hit-rate nie mówi, czy model rozumie konkretną spółkę, a decyzja dotyczy
konkretnej spółki. Ten moduł dzieli rozliczone predykcje po symbolu i na każdej
grupie liczy trafność JEDYNĄ definicją repo (`directional_hit_rate`, G24/WIA-05):
SIDEWAYS nie stawia zakładu, więc nie wchodzi do mianownika.

Próg `MIN_SYMBOL_SAMPLE` to reguła decyzyjna, nie kosmetyka: poniżej niego wynik
jest oznaczony jako niewystarczający, bo „3/4” wygląda jak umiejętność, a jest
szumem. Rekord zawsze niesie n i przedział Wilsona, żeby raport mógł je pokazać.

Czysta logika, tylko stdlib i inne moduły domeny — wejściem są prymitywy
`(symbol, predicted_trend, is_correct)`, nie DTO warstwy aplikacji.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from src.domain.equity_curve import DirectionalHitRate, directional_hit_rate
from src.domain.statistics import wilson_interval

# Minimalna liczba rozliczonych predykcji KIERUNKOWYCH symbolu, od której
# pokazujemy jego trafność zamiast „za mało historii”.
MIN_SYMBOL_SAMPLE = 5


@dataclass(frozen=True)
class SymbolTrackRecord:
    """Trafność kierunkowa jednego symbolu w oknie track recordu."""

    symbol: str
    hit_rate: DirectionalHitRate
    min_sample: int = MIN_SYMBOL_SAMPLE

    @property
    def hits(self) -> int:
        return self.hit_rate.hits

    @property
    def directional(self) -> int:
        """Mianownik: predykcje BULLISH/BEARISH (bez SIDEWAYS)."""
        return self.hit_rate.directional

    @property
    def sideways(self) -> int:
        return self.hit_rate.sideways

    @property
    def rate(self) -> float | None:
        return self.hit_rate.rate

    @property
    def insufficient(self) -> bool:
        """Za mała próba, by trafność na symbolu cokolwiek znaczyła."""
        return self.directional < self.min_sample

    @property
    def interval(self) -> tuple[float, float]:
        """95% przedział Wilsona; pusta próba daje pełną niepewność (0, 1).

        Symbol ma zwykle jedną predykcję na cykl, więc jego próby nie są
        skorelowane jak predykcje jednego dnia na całej watchliście — liczymy na
        surowym n, bez n_eff."""
        return wilson_interval(self.hits, self.directional)


def symbol_track_record(
    calls: Iterable[tuple[str, str, bool]], *, min_sample: int = MIN_SYMBOL_SAMPLE
) -> dict[str, SymbolTrackRecord]:
    """Rekord trafności per symbol z trójek `(symbol, predicted_trend, is_correct)`.

    Symbol bez żadnej rozliczonej predykcji nie dostaje rekordu — raport traktuje
    go jak „za mało historii”."""
    grouped: dict[str, list[tuple[str, bool]]] = {}
    for symbol, trend, is_correct in calls:
        grouped.setdefault(symbol, []).append((trend, is_correct))
    return {
        symbol: SymbolTrackRecord(
            symbol=symbol,
            hit_rate=directional_hit_rate(symbol_calls),
            min_sample=min_sample,
        )
        for symbol, symbol_calls in grouped.items()
    }
