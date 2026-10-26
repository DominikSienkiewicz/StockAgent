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
from collections.abc import Callable, Sequence
from datetime import date, timedelta
from decimal import Decimal
from statistics import NormalDist

from src.domain.number_format import DecimalMark, format_number_pl

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
    tail = sum(math.comb(n, i) * p0**i * (1.0 - p0) ** (n - i) for i in range(k, n + 1))
    return min(1.0, tail)


# DOC-16: moc testu, którą zakłada kalkulator próby — konwencja 80%.
DEFAULT_POWER = 0.80


def required_n(
    p0: float, p1: float, *, alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER
) -> int:
    """DOC-16: ile niezależnych predykcji kierunkowych (w n_eff) potrzeba, żeby
    jednostronny test proporcji na poziomie `alpha` odróżnił trafność `p1` od `p0`
    z mocą `power` (przybliżenie normalne).

    55% wobec 50% przy α = 0,05 i mocy 80% to 617: wynik z 40 predykcji to
    anegdota, nie dowód. Pytanie bez odpowiedzi (`p1 <= p0`, wartości spoza
    (0, 1)) → ValueError."""
    if not 0.0 < p0 < p1 < 1.0:
        raise ValueError(f"required_n needs 0 < p0 < p1 < 1 (got p0={p0}, p1={p1})")
    if not (0.0 < alpha < 1.0 and 0.0 < power < 1.0):
        raise ValueError(f"alpha and power must be in (0, 1) (got {alpha}, {power})")
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha)
    z_beta = normal.inv_cdf(power)
    spread = z_alpha * math.sqrt(p0 * (1.0 - p0)) + z_beta * math.sqrt(p1 * (1.0 - p1))
    return math.ceil((spread / (p1 - p0)) ** 2)


def sample_reached_on(have: float, required: int, *, window_days: int, today: date) -> date | None:
    """DOC-16: dzień, w którym przy tempie okna (`have` w `window_days` dni) próba
    dojdzie do `required`. Zero napływu (pauza Fast Loopa) → None, bo „nigdy”
    to też informacja; próba już wystarczająca → `today`."""
    if have >= required:
        return today
    if have <= 0 or window_days <= 0:
        return None
    return today + timedelta(days=math.ceil((required - have) * window_days / have))


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


def _format_decimal(value: float, decimals: int, mark: DecimalMark) -> str:
    if mark is DecimalMark.COMMA:
        return format_number_pl(Decimal(str(value)), False, decimals)
    return f"{value:.{decimals}f}"


def _format_p(p: float, mark: DecimalMark) -> str:
    """p-wartość do copy: dwa miejsca, a bardzo małe jako „p<0.001”."""
    if p < 0.001:
        return f"p<{_format_decimal(0.001, 3, mark)}"
    if p < 0.01:
        return f"p={_format_decimal(p, 3, mark)}"
    return f"p={_format_decimal(p, 2, mark)}"


def verdict_pl(
    p: float,
    n: int,
    n_eff: float | None = None,
    alpha: float = DEFAULT_ALPHA,
    *,
    mark: DecimalMark = DecimalMark.POINT,
    significant: bool | None = None,
) -> str:
    """Polski werdykt istotności, np. „nieodróżnialne od rzutu monetą (p=0.41, n=37)”.

    `p` to p-wartość względem najlepszego naiwnego baseline'u. Bez `n_eff`
    werdykt dopisuje, że zakłada niezależne próby (p-wartość jest wtedy zbyt
    optymistyczna przy skorelowanych predykcjach jednego cyklu). `mark` to znak
    dziesiętny p-wartości: kropka dla strony publicznej, przecinek dla maila
    (MAIL-58, zaokrąglenie ROUND_HALF_UP). `significant` (WIA-22) podaje
    decyzję wprost — np. po korekcie BH-FDR — zamiast porównania `p < alpha`;
    p-wartość w nawiasie zostaje surowa.
    """
    details = f"{_format_p(p, mark)}, n={n}"
    if n_eff is not None:
        details += f", n_eff≈{n_eff:.0f}"
    else:
        details += "; zakłada niezależne próby"
    if (p < alpha) if significant is None else significant:
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
    rates = [sum(cycle) / len(cycle) for cycle in cycles]
    pooled = sum(sum(cycle) for cycle in cycles) / n
    between = sum(len(c) * (r - pooled) ** 2 for c, r in zip(cycles, rates, strict=True))
    within = sum(len(c) * r * (1.0 - r) for c, r in zip(cycles, rates, strict=True))
    return _icc_from_sums([len(c) for c in cycles], between, within, n)


def _icc_from_sums(sizes: Sequence[int], between: float, within: float, n: int) -> float:
    """ICC(1) z sum kwadratów między cyklami i w cyklach, przycięte do [0, 1]."""
    k = len(sizes)
    ms_between = between / (k - 1)
    ms_within = within / (n - k)
    # m0 — skorygowana średnia liczność dla nierównych cykli.
    m0 = (n - sum(size**2 for size in sizes) / n) / (k - 1)
    denominator = ms_between + (m0 - 1.0) * ms_within
    if denominator <= 0.0:
        return 1.0
    return min(1.0, max(0.0, (ms_between - ms_within) / denominator))


def effective_n_values(values_by_cycle: Sequence[Sequence[float]]) -> float:
    """EXP-05: `effective_n` dla wartości ciągłych (np. kwadratów błędu Briera).

    Ta sama reguła n / (1 + (m̄ − 1)·ICC) i te same brzegi rozstrzygane na
    niekorzyść dowodu; ICC z jednoczynnikowej ANOVA na surowych wartościach.
    Dla wyników 0/1 daje to samo co `effective_n`.
    """
    cycles = [cycle for cycle in values_by_cycle if cycle]
    k = len(cycles)
    n = sum(len(cycle) for cycle in cycles)
    if n == 0:
        return 0.0
    if n == k:
        return float(n)
    if k < 2:
        return float(k)
    means = [sum(cycle) / len(cycle) for cycle in cycles]
    pooled = sum(sum(cycle) for cycle in cycles) / n
    between = sum(len(c) * (m - pooled) ** 2 for c, m in zip(cycles, means, strict=True))
    within = sum((x - m) ** 2 for c, m in zip(cycles, means, strict=True) for x in c)
    icc = _icc_from_sums([len(c) for c in cycles], between, within, n)
    return n / (1.0 + (n / k - 1.0) * icc)


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
    return _percentile_interval(estimates, level)


def cluster_bootstrap_statistic_ci[T](
    samples_by_cycle: Sequence[Sequence[T]],
    statistic: Callable[[list[T]], float | None],
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float] | None:
    """WIA-58 / WIA-38: percentylowy przedział dowolnej statystyki próbki
    (ECE, AUC) z bootstrapu losującego CAŁE cykle.

    Ten sam schemat losowania co `cluster_bootstrap_ci`: cykle ze zwracaniem,
    statystyka liczona na spłaszczonej próbie. Losowanie, dla którego statystyka
    jest nieokreślona (None, np. AUC bez pudeł), odpada. Przy <
    `MIN_BOOTSTRAP_CYCLES` niepustych cyklach albo bez żadnej estymaty → None.
    """
    cycles = [list(cycle) for cycle in samples_by_cycle if cycle]
    if len(cycles) < MIN_BOOTSTRAP_CYCLES or n_boot <= 0:
        return None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_boot):
        sample = rng.choices(cycles, k=len(cycles))
        value = statistic([item for cycle in sample for item in cycle])
        if value is not None:
            estimates.append(value)
    if not estimates:
        return None
    return _percentile_interval(estimates, level)


def cluster_bootstrap_difference_ci(
    arms_by_cycle: Sequence[tuple[Sequence[float], Sequence[float]]],
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float] | None:
    """EXP-05: percentylowy przedział różnicy średnich (drugie ramię − pierwsze)
    z bootstrapu losującego CAŁE cykle.

    `arms_by_cycle` to pary `(wartości ramienia A, wartości ramienia B)` jednego
    cyklu. Oba ramiona dnia losują się razem, więc wspólny ruch rynku tego dnia
    nie udaje różnicy między ramionami. Próba bez obserwacji któregoś ramienia
    odpada. Deterministyczny dla `seed`; przy < `MIN_BOOTSTRAP_CYCLES`
    niepustych cyklach albo pustym ramieniu zwraca None.
    """
    cycles = [
        (sum(first), len(first), sum(second), len(second))
        for first, second in arms_by_cycle
        if first or second
    ]
    if len(cycles) < MIN_BOOTSTRAP_CYCLES or n_boot <= 0:
        return None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_boot):
        sample = rng.choices(cycles, k=len(cycles))
        first_n = sum(c[1] for c in sample)
        second_n = sum(c[3] for c in sample)
        if first_n == 0 or second_n == 0:
            continue
        first_mean = sum(c[0] for c in sample) / first_n
        estimates.append(sum(c[2] for c in sample) / second_n - first_mean)
    if not estimates:
        return None
    return _percentile_interval(estimates, level)


def _percentile_interval(estimates: list[float], level: float) -> tuple[float, float]:
    """Percentylowy przedział `level` z niepustej listy estymat bootstrapu."""
    estimates.sort()
    last = len(estimates) - 1
    tail = (1.0 - level) / 2.0
    low_index = int(math.floor(tail * last))
    high_index = int(math.ceil((1.0 - tail) * last))
    return estimates[low_index], estimates[high_index]
