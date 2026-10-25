"""Domena korelacji portfela — radar koncentracji ("43 symbole = 3 zakłady").

Czysta logika, ZERO importów zewnętrznych (tylko stdlib). Liczymy zwroty
z szeregu cen, korelację Pearsona z bramką na zerową wariancję, macierz
korelacji parami oraz klastry symboli, które poruszają się razem (pairwise
korelacja ≥ próg, sklejane przez union-find — także tranzytywnie).

Wszystko karmione DARMOWYMI snapshotami cen — żadnych płatnych portów.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime

from src.domain.value_objects import require_finite

# Minimalna liczba par zwrotów potrzebna, by korelacja miała sens.
_MIN_RETURNS = 2


def _utc_day(moment: datetime) -> date:
    """Dzień UTC znacznika; naiwny znacznik traktujemy jak UTC."""
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC)
    return moment.date()


def align_by_utc_day(
    series: Mapping[str, Sequence[tuple[datetime, float]]],
) -> dict[str, list[float]]:
    """Wyrównuje szeregi cen po WSPÓLNYCH dniach UTC (PORT-02, G19).

    Snapshoty różnych symboli z jednego cyklu dzielą sekundy, więc przecięcie
    dokładnych timestampów bywało puste. Kluczem jest dzień UTC: przy kilku
    snapshotach w dniu wygrywa najpóźniejszy (stan na koniec dnia), a zostają
    tylko dni obecne we WSZYSTKICH szeregach, w rosnącej kolejności. Kubełek
    godzinowy rozciąłby jeden cykl między dwie godziny. Brak wspólnego dnia →
    puste listy; puste wejście → `{}`.
    """
    by_day: dict[str, dict[date, float]] = {}
    for symbol, points in series.items():
        days: dict[date, float] = {}
        for moment, price in sorted(points, key=lambda point: point[0]):
            days[_utc_day(moment)] = price
        by_day[symbol] = days
    if not by_day:
        return {}
    common = set.intersection(*(set(days) for days in by_day.values()))
    ordered = sorted(common)
    return {symbol: [days[day] for day in ordered] for symbol, days in by_day.items()}


def returns_from_prices(prices: list[float]) -> list[float]:
    """Zwroty procentowe (proste) między kolejnymi cenami.

    `[100, 110, 121]` → `[0.1, 0.1]`. Krok z ceną bazową 0 jest pomijany
    (dzielenie przez zero nie produkuje sensownego zwrotu) — dzięki temu
    wynik jest zawsze skończony.
    """
    out: list[float] = []
    for prev, curr in zip(prices, prices[1:], strict=False):
        if prev == 0:
            continue
        out.append((curr - prev) / prev)
    return out


def pearson(a: list[float], b: list[float]) -> float:
    """Współczynnik korelacji Pearsona dla dwóch szeregów.

    Wyrównuje długości do wspólnego prefiksu (misaligned history). Gdy
    którakolwiek seria ma zerową wariancję (stała), zwraca 0.0 zamiast
    NaN — brak zmienności = brak mierzalnej współzależności. Za krótkie
    serie (<2 punktów) też dają 0.0. NaN/inf w którejkolwiek serii →
    `ValueError` (PORT-09): domknięcie `min`/`max` zamieniało NaN w 1.0.
    """
    require_finite(a, "pearson series")
    require_finite(b, "pearson series")
    n = min(len(a), len(b))
    if n < _MIN_RETURNS:
        return 0.0
    xs = a[:n]
    ys = b[:n]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    # Pierwiastki osobno: iloczyn bardzo małych wariancji schodzi do 0.0
    # (np. 1e-90 × 1e-281), choć każda z nich jest dodatnia.
    denominator = math.sqrt(var_x) * math.sqrt(var_y)
    if denominator == 0:
        return 0.0
    corr = cov / denominator
    # Domknięcie do [-1, 1] — błędy zmiennoprzecinkowe potrafią dać 1.0000002.
    return max(-1.0, min(1.0, corr))


def correlation_matrix(
    returns_by_symbol: dict[str, list[float]],
) -> dict[tuple[str, str], float]:
    """Macierz korelacji parami (górny trójkąt, klucze posortowane).

    Symbole z za krótką historią zwrotów (<2) są pomijane — nie da się dla
    nich policzyć sensownej korelacji. Klucz to zawsze `(min, max)` symboli,
    więc para występuje raz: `(A,B)`, nigdy `(B,A)`.
    """
    usable = sorted(sym for sym, rets in returns_by_symbol.items() if len(rets) >= _MIN_RETURNS)
    matrix: dict[tuple[str, str], float] = {}
    for i, left in enumerate(usable):
        for right in usable[i + 1 :]:
            matrix[(left, right)] = pearson(returns_by_symbol[left], returns_by_symbol[right])
    return matrix


def detect_clusters(
    matrix: dict[tuple[str, str], float],
    symbols: list[str],
    threshold: float,
) -> list[tuple[str, ...]]:
    """Klastry symboli poruszających się razem (pairwise korelacja ≥ próg).

    Union-find: każda para powyżej progu skleja swoje zbiory, więc klastry
    łączą się także tranzytywnie (A~B i B~C → {A,B,C}, nawet gdy A~C poniżej
    progu). Singletony (symbole bez żadnego silnego powiązania) są pomijane
    — radar pokazuje tylko realne skupiska ryzyka.
    """
    parent: dict[str, str] = {s: s for s in symbols}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        # Kompresja ścieżki — płaska struktura przy kolejnych find.
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for (left, right), corr in matrix.items():
        if left in parent and right in parent and corr >= threshold:
            union(left, right)

    groups: dict[str, list[str]] = {}
    for sym in symbols:
        groups.setdefault(find(sym), []).append(sym)

    clusters = [tuple(sorted(members)) for members in groups.values() if len(members) >= 2]
    # Stabilny porządek wyniku — sort po pierwszym członku klastra.
    clusters.sort()
    return clusters
