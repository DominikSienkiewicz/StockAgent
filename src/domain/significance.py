"""Istotność trafności: test dwumianowy, BH-FDR, n_eff i bootstrap blokowy.

Czysty stdlib (WIA-22, WIA-23). Każda trafność była dotąd gołym procentem:
  - `binomial_test_greater` mówi, czy wynik bije NAIWNY BASELINE (p0 to
    trafność najlepszej naiwnej strategii w tym samym oknie, nie 0.5),
  - `benjamini_hochberg` koryguje wielokrotne porównania — przy 7 personach
    i kilkunastu segmentach „najlepszy” wynik jest z konstrukcji selekcją szumu,
  - `effective_n` i `cluster_bootstrap_ci` uwzględniają, że ~40 spółek w jednym
    cyklu rusza się razem: dzień spadku NASDAQ daje 30 „trafień BEARISH” naraz,
    czyli jeden zakład, nie trzydzieści.

Test dwumianowy zakłada niezależne próby, więc bez n_eff p-wartości są zbyt
optymistyczne — `verdict_pl` mówi to wprost, gdy n_eff nie jest znane.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

# Powyżej tej liczności dokładny ogon przez `math.comb` traci sens numeryczny
# (iloczyny rzędu 1e±300) — przechodzimy na przybliżenie normalne.
EXACT_BINOMIAL_MAX_N = 1000

# Poniżej tej liczby cykli bootstrap po cyklach jest niestabilny — zamiast
# przedziału zostaje n_eff i ostrzeżenie.
MIN_BOOTSTRAP_CYCLES = 20

DEFAULT_ALPHA = 0.05
DEFAULT_FDR_Q = 0.10


def binomial_test_greater(k: int, n: int, p0: float) -> float:
    """Jednostronna p-wartość P(X >= k) dla X ~ Binomial(n, p0).

    Dla `n <= 1000` dokładny ogon przez `math.comb`; powyżej przybliżenie
    normalne z korektą ciągłości. Brzegi: `k <= 0` → 1.0, `k > n` → 0.0,
    `p0 <= 0` → 0.0 (każde trafienie jest wtedy niemożliwe pod H0),
    `p0 >= 1` → 1.0.
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p0 <= 0.0:
        return 0.0
    if p0 >= 1.0:
        return 1.0
    if n > EXACT_BINOMIAL_MAX_N:
        mean = n * p0
        sd = math.sqrt(n * p0 * (1.0 - p0))
        z = (k - 0.5 - mean) / sd
        return 0.5 * math.erfc(z / math.sqrt(2.0))
    tail = sum(
        math.comb(n, i) * p0**i * (1.0 - p0) ** (n - i) for i in range(k, n + 1)
    )
    return min(1.0, tail)


def benjamini_hochberg(pvalues: Sequence[float], q: float = DEFAULT_FDR_Q) -> list[bool]:
    """Procedura step-up Benjaminiego–Hochberga kontrolująca FDR na poziomie `q`.

    Zwraca listę flag „odrzucamy H0” w kolejności wejścia. Odrzucane są wszystkie
    hipotezy o randze <= największej randze k, dla której p_(k) <= q·k/m.
    Przy kilku hipotezach procedura jest konserwatywna — to cena uczciwości.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    largest_passing_rank = 0
    for rank, index in enumerate(order, start=1):
        if pvalues[index] <= q * rank / m:
            largest_passing_rank = rank
    rejected = [False] * m
    for index in order[:largest_passing_rank]:
        rejected[index] = True
    return rejected


def _format_p(p: float) -> str:
    """p-wartość do copy: dwa miejsca, a bardzo małe jako „p<0.001”."""
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return f"p={p:.3f}"
    return f"p={p:.2f}"


def verdict_pl(
    p: float, n: int, n_eff: float | None = None, alpha: float = DEFAULT_ALPHA
) -> str:
    """Polski werdykt istotności, np. „nieodróżnialne od rzutu monetą (p=0.41, n=37)”.

    `p` to p-wartość względem najlepszego naiwnego baseline'u. Bez `n_eff`
    werdykt dopisuje, że zakłada niezależne próby (p-wartość jest wtedy zbyt
    optymistyczna przy skorelowanych predykcjach jednego cyklu).
    """
    details = f"{_format_p(p)}, n={n}"
    if n_eff is not None:
        details += f", n_eff≈{n_eff:.0f}"
    else:
        details += "; zakłada niezależne próby"
    if p < alpha:
        return f"istotnie lepsze od naiwnego baseline'u ({details})"
    return f"nieodróżnialne od rzutu monetą ({details})"


def effective_n(outcomes_by_cycle: Sequence[Sequence[bool]]) -> float:
    """Efektywna liczność n_eff = n / (1 + (m̄ − 1)·ICC).

    `outcomes_by_cycle` to trafienia pogrupowane po cyklu (dniu UTC). ICC to
    korelacja wewnątrzklasowa trafień w obrębie cyklu, estymowana jednoczynnikową
    ANOVA i przycięta do [0, 1]; m̄ to średnia liczność cyklu.

    Brzegi rozstrzygamy na niekorzyść dowodu: jeden cykl albo zerowa wariancja
    (np. same trafienia) nie pozwalają oszacować korelacji, więc przyjmujemy
    najgorszy przypadek — jeden zakład na cykl. Cykle jednoelementowe są z
    definicji niezależne (n_eff = n).
    """
    cycles = [cycle for cycle in outcomes_by_cycle if cycle]
    k = len(cycles)
    n = sum(len(cycle) for cycle in cycles)
    if n == 0:
        return 0.0
    if n == k:
        return float(n)
    if k < 2:
        return float(k)
    icc = _intraclass_correlation(cycles, n)
    mean_size = n / k
    return n / (1.0 + (mean_size - 1.0) * icc)


def _intraclass_correlation(cycles: Sequence[Sequence[bool]], n: int) -> float:
    """ICC(1) z jednoczynnikowej ANOVA dla wyników binarnych, przycięte do [0, 1].

    Zerowa wariancja całkowita → 1.0 (najgorszy przypadek, patrz `effective_n`).
    """
    k = len(cycles)
    rates = [sum(cycle) / len(cycle) for cycle in cycles]
    pooled = sum(sum(cycle) for cycle in cycles) / n
    between = sum(len(c) * (r - pooled) ** 2 for c, r in zip(cycles, rates, strict=True))
    within = sum(len(c) * r * (1.0 - r) for c, r in zip(cycles, rates, strict=True))
    ms_between = between / (k - 1)
    ms_within = within / (n - k)
    # m0 — skorygowana średnia liczność dla nierównych cykli.
    m0 = (n - sum(len(c) ** 2 for c in cycles) / n) / (k - 1)
    denominator = ms_between + (m0 - 1.0) * ms_within
    if denominator <= 0.0:
        return 1.0
    return min(1.0, max(0.0, (ms_between - ms_within) / denominator))


def cluster_bootstrap_ci(
    outcomes_by_cycle: Sequence[Sequence[bool]],
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float] | None:
    """Percentylowy przedział hit-rate z bootstrapu losującego CAŁE cykle.

    Losowanie cykli ze zwracaniem zachowuje korelację wewnątrz cyklu, której
    zwykły bootstrap po predykcjach by nie widział. Deterministyczny dla
    `seed`. Przy < `MIN_BOOTSTRAP_CYCLES` niepustych cyklach zwraca None —
    wtedy raport pokazuje tylko n_eff i ostrzeżenie.
    """
    cycles = [(sum(cycle), len(cycle)) for cycle in outcomes_by_cycle if cycle]
    if len(cycles) < MIN_BOOTSTRAP_CYCLES or n_boot <= 0:
        return None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_boot):
        sample = rng.choices(cycles, k=len(cycles))
        hits = sum(h for h, _ in sample)
        total = sum(t for _, t in sample)
        estimates.append(hits / total)
    estimates.sort()
    tail = (1.0 - level) / 2.0
    low_index = int(math.floor(tail * (n_boot - 1)))
    high_index = int(math.ceil((1.0 - tail) * (n_boot - 1)))
    return estimates[low_index], estimates[high_index]
