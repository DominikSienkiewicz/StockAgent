"""Pokrycie danych track recordu (DAT-16) — czysta logika, tylko stdlib.

Track record z dziurami (pauza Fast Loopa, symbole bez ceny, padnięte cykle)
bez informacji o pokryciu sugeruje ciągłość, której nie było. Moduł liczy,
jaki odsetek oczekiwanych sesji ma snapshot ceny, gdzie jest najdłuższa luka
i które symbole nie mają w oknie ani jednego rozliczenia.

Oczekiwane sesje zależą od klasy aktywa: akcje i ETF-y handlują w dni handlowe
NYSE (`market_calendar.is_trading_day` — weekendy i znane święta wypadają),
krypto codziennie. Poza latami znanymi kalendarzowi święta są nieznane i mogą
się pokazać jako fałszywa luka jednej sesji.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta

from src.domain.market_calendar import is_trading_day
from src.domain.value_objects import AssetType


@dataclass(frozen=True)
class CoverageGap:
    """Ciąg kolejnych oczekiwanych sesji bez snapshotu: pierwsza i ostatnia
    brakująca sesja oraz ich liczba."""

    first: date
    last: date
    sessions: int


@dataclass(frozen=True)
class CoverageReport:
    """Pokrycie okna: `covered` z `expected` sesji ma snapshot, `longest_gap` to
    najdłuższa luka (None = brak luk). `unsettled_symbols` — symbole bez żadnego
    rozliczenia w oknie (uzupełnia wołający, `unsettled_symbols()`)."""

    expected: int
    covered: int
    longest_gap: CoverageGap | None
    unsettled_symbols: tuple[str, ...] = ()

    @property
    def ratio(self) -> float | None:
        """Odsetek sesji ze snapshotem; None, gdy okno nie ma żadnej sesji."""
        if self.expected == 0:
            return None
        return self.covered / self.expected


def expected_sessions(start: date, end: date, asset_type: AssetType) -> list[date]:
    """Sesje w `[start, end]`, w których klasa aktywa powinna mieć notowanie."""
    days = [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
    if asset_type is AssetType.CRYPTO:
        return days
    return [day for day in days if is_trading_day(day)]


def coverage(expected_sessions: Iterable[date], observed_dates: Iterable[date]) -> CoverageReport:
    """Pokrycie oczekiwanych sesji obserwowanymi datami snapshotów.

    Daty spoza oczekiwanych (np. snapshot krypto w sobotę przy kalendarzu akcji)
    nie liczą się. Luka to ciąg KOLEJNYCH oczekiwanych sesji bez snapshotu;
    przy remisie wygrywa wcześniejsza."""
    expected = sorted(set(expected_sessions))
    observed = set(observed_dates)
    longest: CoverageGap | None = None
    run: list[date] = []
    for session in [*expected, None]:
        if session is not None and session not in observed:
            run.append(session)
            continue
        if run and (longest is None or len(run) > longest.sessions):
            longest = CoverageGap(first=run[0], last=run[-1], sessions=len(run))
        run = []
    covered = sum(1 for session in expected if session in observed)
    return CoverageReport(expected=len(expected), covered=covered, longest_gap=longest)


def unsettled_symbols(symbols: Iterable[str], settled: Iterable[str]) -> tuple[str, ...]:
    """Symbole bez żadnego rozliczenia w oknie, posortowane, bez powtórzeń."""
    settled_set = set(settled)
    return tuple(sorted({symbol for symbol in symbols if symbol not in settled_set}))
