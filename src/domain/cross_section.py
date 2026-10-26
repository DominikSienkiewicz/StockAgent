"""Domena: cechy cross-sectional cyklu (ML-32) — ruch symbolu na tle uniwersum.

Ruch −3% w dniu, gdy cały koszyk tech spada −3%, to beta, a nie wiadomość o spółce.
Model wspólny potrzebuje kontekstu „na tle rynku”: rangi ruchu w uniwersum cyklu,
ruchu ponad benchmark (IVV, ten sam co w dowodach raportu), ruchu ponad medianę
klasy aktywa i szerokości rynku (odsetka symboli z ruchem co najmniej progu bramki).

Wejście to jednodniowe zwroty symboli, których kurs w tym cyklu się udał — symbol
bez kursu nie wchodzi do uniwersum, a jego cechy są None. Stdlib only.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from dataclasses import asdict, dataclass

# Benchmark rynku USA (S&P 500) — ten sam ETF, z którym raport porównuje agenta.
BENCHMARK_SYMBOL = "IVV"

CROSS_SECTION_FEATURE_NAMES: tuple[str, ...] = (
    "ret_1_rank",
    "ret_1_excess_vs_benchmark",
    "ret_1_excess_vs_class_median",
    "breadth",
)


@dataclass(frozen=True)
class SymbolMove:
    """Jednodniowy zwrot symbolu w cyklu, jego klasa aktywa i bazowy próg bramki."""

    ret_1: float
    asset_class: str
    gate_threshold: float


@dataclass(frozen=True)
class CrossSectionFeatures:
    """Cechy symbolu na tle uniwersum cyklu; None = brak danych.

    Pola:
        ret_1_rank: percentylowa ranga ret_1 w uniwersum, 0 = najsłabszy, 1 = najsilniejszy,
        ret_1_excess_vs_benchmark: ret_1 − ret_1(IVV),
        ret_1_excess_vs_class_median: ret_1 − mediana ret_1 klasy aktywa,
        breadth: odsetek uniwersum z |ret_1| ≥ bazowy próg bramki swojej klasy.
    """

    ret_1_rank: float | None = None
    ret_1_excess_vs_benchmark: float | None = None
    ret_1_excess_vs_class_median: float | None = None
    breadth: float | None = None

    def as_features(self) -> dict[str, float | None]:
        values = asdict(self)
        return {name: values[name] for name in CROSS_SECTION_FEATURE_NAMES}


def percentile_ranks(values: Mapping[str, float]) -> dict[str, float]:
    """Ranga percentylowa w [0, 1]: (średnia ranga − 1) / (n − 1).

    Remisy dostają średnią rangę; pojedynczy symbol to środek skali (0.5)."""
    count = len(values)
    if count == 0:
        return {}
    if count == 1:
        return dict.fromkeys(values, 0.5)
    ordered = sorted(values.items(), key=lambda item: item[1])
    ranks: dict[str, float] = {}
    start = 0
    while start < count:
        end = start
        while end + 1 < count and ordered[end + 1][1] == ordered[start][1]:
            end += 1
        average_rank = (start + end) / 2.0
        for symbol, _ in ordered[start : end + 1]:
            ranks[symbol] = average_rank / (count - 1)
        start = end + 1
    return ranks


def cross_section_features(
    moves: Mapping[str, SymbolMove], *, benchmark: str = BENCHMARK_SYMBOL
) -> dict[str, CrossSectionFeatures]:
    """Cechy cross-sectional każdego symbolu z `moves` (uniwersum cyklu z kursem)."""
    if not moves:
        return {}
    ranks = percentile_ranks({symbol: move.ret_1 for symbol, move in moves.items()})
    benchmark_move = moves.get(benchmark)
    medians = {
        asset_class: statistics.median(
            move.ret_1 for move in moves.values() if move.asset_class == asset_class
        )
        for asset_class in {move.asset_class for move in moves.values()}
    }
    breadth = sum(1 for move in moves.values() if abs(move.ret_1) >= move.gate_threshold) / len(
        moves
    )
    return {
        symbol: CrossSectionFeatures(
            ret_1_rank=ranks[symbol],
            ret_1_excess_vs_benchmark=(
                move.ret_1 - benchmark_move.ret_1 if benchmark_move is not None else None
            ),
            ret_1_excess_vs_class_median=move.ret_1 - medians[move.asset_class],
            breadth=breadth,
        )
        for symbol, move in moves.items()
    }
