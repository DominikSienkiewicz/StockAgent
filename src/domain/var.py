"""Domena ryzyka ogona — historyczny Value-at-Risk i Conditional VaR (CVaR).

Czysta logika, ZERO importów zewnętrznych (tylko stdlib, BEZ numpy). VaR liczymy
metodą empiryczną na próbce zwrotów: sortujemy rosnąco i bierzemy kwantyl dolnego
ogona z LINIOWĄ interpolacją między statystykami pozycyjnymi (order statistics).
CVaR to średnia magnituda strat w tym ogonie. Oba zwracane jako DODATNIE
magnitudy straty (loss magnitude), więc 0.09 = "9% straty w ogonie".

Karmione DARMOWYMI snapshotami cen / zwrotów — żadnych płatnych portów.
"""

from __future__ import annotations

from collections.abc import Sequence

# Minimalna liczba obserwacji, by kwantyl ogona miał sens (potrzebne ≥2 punkty
# do interpolacji między statystykami pozycyjnymi).
_MIN_OBSERVATIONS = 2


def _clamp_confidence(confidence: float) -> float:
    """Domyka poziom ufności do (0,1).

    Wartości spoza zakresu (np. 0.0, 1.0, 1.5, -0.2) są clampowane do wąskiego
    otwartego przedziału — dzięki temu interpolacja kwantyla nie wybucha ani nie
    wychodzi poza tablicę. Świadomy wybór: clamp zamiast wyjątku, bo wejście
    pochodzi z konfiguracji i lepiej zwrócić sensowny skraj niż przerwać cykl.
    """
    return max(1e-9, min(1.0 - 1e-9, confidence))


def _lower_tail_quantile(returns: Sequence[float], confidence: float) -> float:
    """Empiryczny kwantyl dolnego ogona z liniową interpolacją order statistics.

    Próbka jest sortowana rosnąco. Pozycja interpolacji to
    `h = (1 - confidence) * (n - 1)` (konwencja typu "linear"/numpy default).
    Wynik leży między statystykami pozycyjnymi `sorted[floor(h)]` i
    `sorted[ceil(h)]`, interpolowany ułamkiem `h - floor(h)`.
    """
    ordered = sorted(returns)
    n = len(ordered)
    pos = (1.0 - confidence) * (n - 1)
    lower = int(pos)  # floor — pos jest nieujemne
    upper = min(lower + 1, n - 1)
    frac = pos - lower
    return ordered[lower] + frac * (ordered[upper] - ordered[lower])


def historical_var(returns: Sequence[float], confidence: float = 0.95) -> float:
    """Empiryczny Value-at-Risk jako DODATNIA magnituda straty.

    Bierzemy kwantyl `(1 - confidence)` dolnego ogona rozkładu zwrotów (liniowa
    interpolacja między statystykami pozycyjnymi) i zwracamy `max(0, -kwantyl)`.
    Dodatni kwantyl (sam zysk w ogonie) → VaR 0.0. Pusta lub jednoelementowa
    próbka → 0.0 (za mało punktów do interpolacji). `confidence` clampowane do
    (0,1).
    """
    if len(returns) < _MIN_OBSERVATIONS:
        return 0.0
    quantile = _lower_tail_quantile(returns, _clamp_confidence(confidence))
    return max(0.0, -quantile)


def conditional_var(returns: Sequence[float], confidence: float = 0.95) -> float:
    """Conditional VaR (expected shortfall) jako DODATNIA średnia strata w ogonie.

    Liczymy próg kwantyla dolnego ogona (jak w `historical_var`), bierzemy
    wszystkie zwroty NA POZIOMIE progu lub PONIŻEJ niego i zwracamy magnitudę
    ich średniej: `max(0, -mean(tail))`. Gdy żaden zwrot nie wpada do ogona
    (numerycznie pusty ogon), cofamy się do najgorszej obserwacji, by spełnić
    CVaR >= VaR. Pusta/jednoelementowa próbka → 0.0.
    """
    if len(returns) < _MIN_OBSERVATIONS:
        return 0.0
    threshold = _lower_tail_quantile(returns, _clamp_confidence(confidence))
    tail = [r for r in returns if r <= threshold]
    if not tail:
        # Zabezpieczenie numeryczne — bierzemy najgorszy zwrot, by CVaR >= VaR.
        tail = [min(returns)]
    mean_tail = sum(tail) / len(tail)
    return max(0.0, -mean_tail)
