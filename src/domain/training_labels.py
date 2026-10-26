"""Domena: etykieta point-in-time dla wektorów cech z `feature_snapshots` (ML-52).

Fast Loop zapisuje wektor cech w KAŻDYM ukończonym cyklu (ML-14, także ignored),
bez etykiety — w chwili zapisu przyszłości nie ma. Ten moduł dokleja ją później ze
snapshotów ceny tego samego symbolu (`price_snapshots`):

    target_return = (cena_etykiety − cena_wektora) / cena_wektora,

gdzie cena wektora to `features["price"]` (kurs, który widział cykl), a cena
etykiety to PIERWSZY snapshot symbolu w chwili ≥ `as_of + horizon` (12 h, ten sam
`LABEL_HORIZON`, którym walk-forward purguje foldy). Wektor cech zostaje dokładnie
taki, jaki zapisał cykl — trening nie odtwarza cech, więc nie ma skew „SQL
w treningu vs Python w servingu”.

Point-in-time: etykieta bierze jedną cenę — pierwszą dojrzałą — i nic późniejszego.
Wektor bez dojrzałej ceny (cykl sprzed < 12 h) nie ma etykiety. Pierwsza dojrzała
cena dalej niż `MAX_LABEL_HORIZON` (pauza harmonogramu) też nie daje etykiety:
„zwrot do następnego cyklu” przez tydzień przerwy to nie obserwacja cyklu.
Stdlib only.
"""

from __future__ import annotations

import math
import statistics
from bisect import bisect_left
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.domain.backtest import LABEL_HORIZON

# Najdalsza dopuszczalna cena etykiety: następny cykl przez weekend (~72 h, ML-25)
# z zapasem na święto. Dalej to już przerwa w harmonogramie, nie horyzont cyklu.
MAX_LABEL_HORIZON = timedelta(hours=96)


@dataclass(frozen=True)
class FeatureSnapshot:
    """Wiersz `feature_snapshots` (migracja 031): wektor cech cyklu bez etykiety.

    `features` to JSONB dokładnie w postaci zapisanej przez Fast Loop — cechy
    darmowe zawsze, wektor modelu i flagi jakości tylko przy `gated`."""

    symbol: str
    as_of: datetime
    gated: bool
    features: Mapping[str, object]
    feature_set_version: int

    @property
    def price(self) -> float | None:
        """Kurs cyklu z wektora albo None (brak, zero, ujemny, nie-liczba)."""
        return _positive_price(self.features.get("price"))


@dataclass(frozen=True)
class LabeledSnapshot:
    """Wektor cyklu z etykietą: zwrot do pierwszej dojrzałej ceny i jej chwila."""

    snapshot: FeatureSnapshot
    target_return: float
    settled_at: datetime


def _positive_price(raw: object) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    price = float(raw)
    return price if math.isfinite(price) and price > 0 else None


def _clean_series(points: Sequence[tuple[datetime, float]]) -> list[tuple[datetime, float]]:
    """Punkty z dodatnią, skończoną ceną, rosnąco po czasie."""
    cleaned = [
        (moment, price) for moment, raw in points if (price := _positive_price(raw)) is not None
    ]
    return sorted(cleaned, key=lambda point: point[0])


def label_snapshots(
    snapshots: Iterable[FeatureSnapshot],
    price_series: Mapping[str, Sequence[tuple[datetime, float]]],
    *,
    horizon: timedelta = LABEL_HORIZON,
    max_horizon: timedelta = MAX_LABEL_HORIZON,
) -> list[LabeledSnapshot]:
    """Wektory z etykietą `horizon`, rosnąco po (as_of, symbol).

    `price_series` to snapshoty cen per symbol `(chwila, cena)` w dowolnej
    kolejności. Wektor bez kursu, bez dojrzałej ceny albo z ceną dalej niż
    `max_horizon` jest pomijany."""
    series = {symbol: _clean_series(points) for symbol, points in price_series.items()}
    moments = {symbol: [moment for moment, _ in points] for symbol, points in series.items()}
    labeled: list[LabeledSnapshot] = []
    for snapshot in snapshots:
        base = snapshot.price
        points = series.get(snapshot.symbol)
        if base is None or not points:
            continue
        index = bisect_left(moments[snapshot.symbol], snapshot.as_of + horizon)
        if index == len(points):
            continue
        settled_at, label_price = points[index]
        if settled_at - snapshot.as_of > max_horizon:
            continue
        labeled.append(
            LabeledSnapshot(
                snapshot=snapshot,
                target_return=(label_price - base) / base,
                settled_at=settled_at,
            )
        )
    return sorted(labeled, key=lambda item: (item.snapshot.as_of, item.snapshot.symbol))


@dataclass(frozen=True)
class ReturnDistribution:
    """Rozkład zwrotów etykiety jednej grupy cykli; statystyki None przy n = 0
    (odchylenie także przy n = 1). `p10`/`p90` — kwantyle z interpolacją liniową."""

    n: int
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    mean_abs: float | None = None
    p10: float | None = None
    p90: float | None = None


@dataclass(frozen=True)
class SelectionReport:
    """ML-52 (3): raport selekcji bramki — zwroty cykli za bramką i pod nią.

    `ml_feature_store` powstaje tylko z cykli za bramką, więc model uczył się na
    dniach z ruchem. `abs_return_ratio` = średni |zwrot| gated / non-gated: ile
    razy większe ruchy widzi model uczony tylko na gated (None bez obu grup)."""

    gated: ReturnDistribution
    non_gated: ReturnDistribution

    @property
    def abs_return_ratio(self) -> float | None:
        gated, non_gated = self.gated.mean_abs, self.non_gated.mean_abs
        if gated is None or non_gated is None or non_gated == 0:
            return None
        return gated / non_gated


def _quantile(ordered: Sequence[float], q: float) -> float:
    """Kwantyl posortowanej, niepustej próbki z interpolacją liniową."""
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def return_distribution(returns: Sequence[float]) -> ReturnDistribution:
    """Statystyki rozkładu zwrotów (patrz `ReturnDistribution`)."""
    if not returns:
        return ReturnDistribution(n=0)
    ordered = sorted(returns)
    return ReturnDistribution(
        n=len(ordered),
        mean=statistics.fmean(ordered),
        median=statistics.median(ordered),
        std=statistics.stdev(ordered) if len(ordered) > 1 else None,
        mean_abs=statistics.fmean(abs(value) for value in ordered),
        p10=_quantile(ordered, 0.1),
        p90=_quantile(ordered, 0.9),
    )


def selection_report(labeled: Iterable[LabeledSnapshot]) -> SelectionReport:
    """Rozkład zwrotów etykiety osobno dla cykli gated i non-gated."""
    gated: list[float] = []
    non_gated: list[float] = []
    for item in labeled:
        (gated if item.snapshot.gated else non_gated).append(item.target_return)
    return SelectionReport(
        gated=return_distribution(gated), non_gated=return_distribution(non_gated)
    )
