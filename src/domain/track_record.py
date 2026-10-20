"""Uczciwy dowód trafności kierunkowej: baseline'y, przedziały i istotność.

Czysta domena (stdlib + inne moduły domeny). Wejściem są prymitywy
(`ScoredCall`), nie DTO raportu — mapowanie z `ResolvedPrediction` robi
warstwa aplikacji.

Hit-rate bez punktu odniesienia nic nie dowodzi (WIA-31): w hossie strategia
„zawsze BULLISH” trafia w ponad połowie dni bez żadnej wiedzy. Dlatego
`assess_directional_hit_rate` zbiera w jednym obiekcie:
  - hit-rate kierunkowy (jedna definicja: `directional_hit_rate`, WIA-05),
  - naiwne baseline'y liczone na TYM SAMYM zbiorze predykcji (po bramce
    volatility), a nie na wszystkich dniach rynku,
  - efektywną liczność n_eff i bootstrap po cyklach (WIA-23),
  - przedział Wilsona na n_eff (WIA-32),
  - p-wartość testu dwumianowego względem najlepszego kierunkowego baseline'u
    i polski werdykt (WIA-22).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.domain.equity_curve import DirectionalHitRate, directional_hit_rate, signal_of
from src.domain.number_format import DecimalMark
from src.domain.prediction import SIDEWAYS_TOLERANCE
from src.domain.significance import (
    binomial_test_greater,
    cluster_bootstrap_ci,
    effective_n,
    verdict_pl,
)
from src.domain.statistics import wilson_interval

DEFAULT_TOLERANCE = float(SIDEWAYS_TOLERANCE)


@dataclass(frozen=True)
class NaiveBaselines:
    """Trafność strategii bez wiedzy na tym samym zbiorze ruchów cen.

    Reguły trafienia jak w `Prediction.is_trend_correct`: BULLISH wymaga
    zmiany > 0, BEARISH < 0, SIDEWAYS |zmiana| <= tolerancja. `None`, gdy
    zbiór jest pusty.
    """

    always_bullish: float | None
    always_bearish: float | None
    always_sideways: float | None
    n: int

    @property
    def best_directional(self) -> tuple[str, float] | None:
        """Mocniejszy z baseline'ów kierunkowych — p0 dla hit-rate kierunkowego.

        „Zawsze SIDEWAYS” nie stawia zakładu na kierunek, więc nie jest
        porównywalny z trafnością liczoną bez SIDEWAYS.
        """
        if self.always_bullish is None or self.always_bearish is None:
            return None
        if self.always_bearish > self.always_bullish:
            return "BEARISH", self.always_bearish
        return "BULLISH", self.always_bullish


def naive_baselines(
    changes: Sequence[float], tolerance: float = DEFAULT_TOLERANCE
) -> NaiveBaselines:
    """Liczy baseline'y „zawsze BULLISH / BEARISH / SIDEWAYS” z ruchów cen.

    `changes` to zrealizowane zmiany ceny (ułamki) predykcji, z którymi
    porównujemy hit-rate — wołający odpowiada za to, żeby był to ten sam
    zbiór i horyzont.
    """
    n = len(changes)
    if n == 0:
        return NaiveBaselines(
            always_bullish=None, always_bearish=None, always_sideways=None, n=0
        )
    return NaiveBaselines(
        always_bullish=sum(1 for c in changes if c > 0) / n,
        always_bearish=sum(1 for c in changes if c < 0) / n,
        always_sideways=sum(1 for c in changes if abs(c) <= tolerance) / n,
        n=n,
    )


def group_by_cycle[T](keyed: Iterable[tuple[str, T]]) -> list[list[T]]:
    """Grupuje elementy po kluczu cyklu, w rosnącej (chronologicznej) kolejności.

    Klucz to data UTC w formacie ISO, więc porządek leksykograficzny jest
    chronologiczny. Kolejność elementów w obrębie cyklu zostaje zachowana.
    """
    groups: dict[str, list[T]] = {}
    for key, item in keyed:
        groups.setdefault(key, []).append(item)
    return [groups[key] for key in sorted(groups)]


@dataclass(frozen=True)
class ScoredCall:
    """Jedna rozliczona predykcja jako prymitywy domeny.

    - `cycle` — klucz cyklu (data UTC predykcji, ISO); predykcje jednego
      cyklu są skorelowane,
    - `realized_change` — zrealizowana zmiana ceny (ułamek) albo None, gdy
      brak cen; bez niej predykcja nie wchodzi do porównania z baseline'em.
    """

    cycle: str
    predicted_trend: str
    is_correct: bool
    realized_change: float | None


@dataclass(frozen=True)
class HitRateEvidence:
    """Hit-rate kierunkowy razem z tym, co pozwala go ocenić.

    - `interval` — Wilson 95% na n_eff (None bez predykcji kierunkowych),
    - `bootstrap` — percentylowy przedział z bootstrapu po cyklach; None przy
      < `significance.MIN_BOOTSTRAP_CYCLES` cyklach (niestabilny),
    - `p_value` — test dwumianowy na n_eff względem najlepszego kierunkowego
      baseline'u; None, gdy baseline'u nie da się policzyć.
    """

    hit_rate: DirectionalHitRate
    cycles: int
    n_eff: float
    interval: tuple[float, float] | None
    bootstrap: tuple[float, float] | None
    baselines: NaiveBaselines
    p_value: float | None

    @property
    def verdict(self) -> str | None:
        """Polski werdykt istotności (z n i n_eff) z kropką albo None bez p-wartości."""
        return self.verdict_in(DecimalMark.POINT)

    def verdict_in(self, mark: DecimalMark) -> str | None:
        """Werdykt z p-wartością zapisaną znakiem `mark` (mail: przecinek, MAIL-58)."""
        if self.p_value is None:
            return None
        return verdict_pl(self.p_value, self.hit_rate.directional, self.n_eff, mark=mark)


def assess_directional_hit_rate(
    calls: Sequence[ScoredCall],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    n_boot: int = 2000,
    seed: int = 0,
) -> HitRateEvidence:
    """Zbiera dowód trafności kierunkowej z rozliczonych predykcji.

    Hit-rate liczy `directional_hit_rate` (SIDEWAYS poza mianownikiem).
    Baseline'y, n_eff, przedziały i p-wartość liczone są wyłącznie na
    predykcjach kierunkowych. Test dwumianowy i Wilson idą na n_eff, bo
    predykcje jednego cyklu nie są niezależne.
    """
    hit_rate = directional_hit_rate((c.predicted_trend, c.is_correct) for c in calls)
    directional = [c for c in calls if signal_of(c.predicted_trend) != 0]
    by_cycle = group_by_cycle((c.cycle, c.is_correct) for c in directional)
    n_eff = effective_n(by_cycle)

    interval = None
    if hit_rate.rate is not None:
        interval = wilson_interval(hit_rate.rate * n_eff, n_eff)

    comparable = [c for c in directional if c.realized_change is not None]
    baselines = naive_baselines(
        [c.realized_change for c in comparable if c.realized_change is not None],
        tolerance,
    )
    return HitRateEvidence(
        hit_rate=hit_rate,
        cycles=len(by_cycle),
        n_eff=n_eff,
        interval=interval,
        bootstrap=cluster_bootstrap_ci(by_cycle, n_boot=n_boot, seed=seed),
        baselines=baselines,
        p_value=_p_value_vs_best_baseline(comparable, baselines, n_eff),
    )


def _p_value_vs_best_baseline(
    comparable: Sequence[ScoredCall], baselines: NaiveBaselines, n_eff: float
) -> float | None:
    """p-wartość trafności `comparable` vs najlepszy kierunkowy baseline, na n_eff.

    Liczności skalujemy do n_eff (efekt projektu): k_eff = rate · n_eff,
    zaokrąglone do całości, bo test dwumianowy działa na liczbach całkowitych.
    """
    best = baselines.best_directional
    if best is None or not comparable:
        return None
    rate = sum(1 for c in comparable if c.is_correct) / len(comparable)
    scaled_n = max(1, round(min(n_eff, len(comparable))))
    return binomial_test_greater(round(rate * scaled_n), scaled_n, best[1])
