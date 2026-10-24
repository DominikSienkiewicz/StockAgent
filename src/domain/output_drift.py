"""NIEZ-19: monitor rozkładu wyjść LLM między dwoma oknami (tydzień do tygodnia).

Cicha podmiana modelu u dostawcy albo edycja promptu zmienia zachowanie, a bez
monitora widać to dopiero w trafności po tygodniach. Porównujemy trzy rzeczy:

- udziały klas BULLISH / BEARISH / SIDEWAYS — test chi² niezależności 2×K,
- średnią pewność (`confidence_score`),
- bias kierunkowy: udział BULLISH minus udział realnych wzrostów ceny w tych
  samych (rozliczonych) predykcjach.

Rynek sam zmienia udział wzrostów: w hossie model, który częściej mówi BULLISH,
nie zmienił zachowania. Dlatego skok udziałów flaguje dopiero razem ze skokiem
biasu; stały bias BULLISH w hossie udawałby umiejętność, a tu wychodzi na jaw.

Progi są pre-rejestrowane (stałe poniżej) — nie dobiera się ich po fakcie.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

TRENDS = ("BULLISH", "BEARISH", "SIDEWAYS")
BULLISH = "BULLISH"

# Pre-rejestrowane progi monitora (NIEZ-19).
DRIFT_ALPHA = 0.01
MIN_BIAS_SHIFT = 0.15
MIN_CONFIDENCE_SHIFT = 0.10
# Poniżej tej liczby predykcji w którymkolwiek oknie monitor nie flaguje.
MIN_WINDOW_SIZE = 20


@dataclass(frozen=True)
class OutputSample:
    """Jedno wyjście LLM: trend, pewność, czy cena realnie wzrosła (None =
    nierozliczona albo bez cen) i wersja przepisu (None = nieznana)."""

    trend: str
    confidence: float | None = None
    realized_up: bool | None = None
    recipe_version: str | None = None


@dataclass(frozen=True)
class WindowSummary:
    """Okno predykcji: liczby klas, średnia pewność, bias i przepisy."""

    n: int
    counts: Mapping[str, int]
    mean_confidence: float | None
    bullish_bias: float | None
    recipes: frozenset[str] = field(default_factory=frozenset)

    @property
    def bullish_share(self) -> float | None:
        return self.counts[BULLISH] / self.n if self.n else None


def summarize_window(samples: Sequence[OutputSample]) -> WindowSummary:
    """Podsumowanie okna. Etykiety spoza `TRENDS` (bez rozróżniania wielkości
    liter) są pomijane. Bias liczony tylko na predykcjach z `realized_up`."""
    known = [
        (sample, sample.trend.upper()) for sample in samples if sample.trend.upper() in TRENDS
    ]
    counts = {trend: 0 for trend in TRENDS}
    for _, trend in known:
        counts[trend] += 1
    confidences = [s.confidence for s, _ in known if s.confidence is not None]
    settled = [(s, trend) for s, trend in known if s.realized_up is not None]
    bias = None
    if settled:
        bullish = sum(1 for _, trend in settled if trend == BULLISH)
        ups = sum(1 for s, _ in settled if s.realized_up)
        bias = (bullish - ups) / len(settled)
    return WindowSummary(
        n=len(known),
        counts=counts,
        mean_confidence=sum(confidences) / len(confidences) if confidences else None,
        bullish_bias=bias,
        recipes=frozenset(s.recipe_version for s, _ in known if s.recipe_version),
    )


def chi_square_p_value(statistic: float, df: int) -> float:
    """P(χ²(df) ≥ statistic) dla df ∈ {0, 1, 2} — tyle daje tabela 2×3.

    df = 1: erfc(√(x/2)); df = 2: exp(−x/2) (rozkład wykładniczy); df = 0: brak
    swobody, więc brak dowodu zmiany (1.0)."""
    if df <= 0:
        return 1.0
    if df == 1:
        return math.erfc(math.sqrt(statistic / 2.0))
    if df == 2:
        return math.exp(-statistic / 2.0)
    raise ValueError(f"unsupported degrees of freedom: {df}")


def _chi_square(previous: Mapping[str, int], current: Mapping[str, int]) -> tuple[float, int]:
    """Statystyka chi² niezależności 2×K i stopnie swobody (klasy niepuste − 1)."""
    columns = [t for t in TRENDS if previous.get(t, 0) + current.get(t, 0) > 0]
    rows = (previous, current)
    row_totals = [sum(row.get(t, 0) for t in columns) for row in rows]
    total = sum(row_totals)
    if total == 0 or 0 in row_totals:
        return 0.0, 0
    statistic = 0.0
    for t in columns:
        column_total = previous.get(t, 0) + current.get(t, 0)
        for row, row_total in zip(rows, row_totals, strict=True):
            expected = row_total * column_total / total
            statistic += (row.get(t, 0) - expected) ** 2 / expected
    return statistic, len(columns) - 1


@dataclass(frozen=True)
class OutputDriftReport:
    """Wynik porównania okien; `flagged` = WARNING w Slow Loopie."""

    previous: WindowSummary
    current: WindowSummary
    chi_square: float
    p_value: float
    bias_shift: float | None
    confidence_shift: float | None
    flagged: bool
    new_recipes: tuple[str, ...] = ()

    @property
    def recipe_changed(self) -> bool:
        return bool(self.new_recipes)

    @property
    def enough_data(self) -> bool:
        return min(self.previous.n, self.current.n) >= MIN_WINDOW_SIZE

    def describe_pl(self) -> str:
        """Jedno zdanie dla alertu i logu (polski)."""
        if not self.enough_data:
            return (
                f"Monitor wyjść LLM: za mało predykcji do porównania "
                f"(n = {self.previous.n} i {self.current.n}, minimum {MIN_WINDOW_SIZE})."
            )
        details = self._details_pl()
        if not self.flagged:
            return f"Rozkład wyjść LLM stabilny: {details}."
        if self.recipe_changed:
            recipes = ", ".join(self.new_recipes)
            return f"Rozkład wyjść LLM: zmiana zachowania po zmianie przepisu {recipes}: {details}."
        return f"Rozkład wyjść LLM zmienił się bez zmiany przepisu: {details}."

    def _details_pl(self) -> str:
        parts = [
            f"udział BULLISH {_pct(self.previous.bullish_share)} → "
            f"{_pct(self.current.bullish_share)} (chi² p = {self.p_value:.3g})"
        ]
        if self.previous.bullish_bias is not None and self.current.bullish_bias is not None:
            parts.append(
                f"bias kierunkowy {_pp(self.previous.bullish_bias)} → "
                f"{_pp(self.current.bullish_bias)}"
            )
        if self.previous.mean_confidence is not None and self.current.mean_confidence is not None:
            parts.append(
                f"średnia pewność {self.previous.mean_confidence:.2f} → "
                f"{self.current.mean_confidence:.2f}"
            )
        return ", ".join(parts)


def _pct(share: float | None) -> str:
    return "—" if share is None else f"{round(share * 100)}%"


def _pp(value: float) -> str:
    return f"{round(value * 100):+d} pp"


def _shift(previous: float | None, current: float | None) -> float | None:
    if previous is None or current is None:
        return None
    return current - previous


def output_distribution_shift(
    previous_window: Sequence[OutputSample],
    current_window: Sequence[OutputSample],
    *,
    alpha: float = DRIFT_ALPHA,
) -> OutputDriftReport:
    """Porównuje okno bieżące z poprzednim.

    Flaga: (udziały klas różne na poziomie `alpha` ORAZ |zmiana biasu| ≥
    `MIN_BIAS_SHIFT`) ALBO |zmiana średniej pewności| ≥ `MIN_CONFIDENCE_SHIFT`.
    Bez rozliczonych predykcji bias jest nieznany, więc sam skok udziałów nie
    flaguje — nie da się go odróżnić od rynku. Poniżej `MIN_WINDOW_SIZE`
    predykcji w którymkolwiek oknie flagi nie ma."""
    previous = summarize_window(previous_window)
    current = summarize_window(current_window)
    statistic, df = _chi_square(previous.counts, current.counts)
    p_value = chi_square_p_value(statistic, df)
    bias_shift = _shift(previous.bullish_bias, current.bullish_bias)
    confidence_shift = _shift(previous.mean_confidence, current.mean_confidence)
    share_drift = (
        p_value < alpha and bias_shift is not None and abs(bias_shift) >= MIN_BIAS_SHIFT
    )
    confidence_drift = confidence_shift is not None and (
        abs(confidence_shift) >= MIN_CONFIDENCE_SHIFT
    )
    enough = min(previous.n, current.n) >= MIN_WINDOW_SIZE
    return OutputDriftReport(
        previous=previous,
        current=current,
        chi_square=statistic,
        p_value=p_value,
        bias_shift=bias_shift,
        confidence_shift=confidence_shift,
        flagged=enough and (share_drift or confidence_drift),
        # Bez przepisów w oknie poprzednim nie wiadomo, czy przepis się zmienił.
        new_recipes=tuple(sorted(current.recipes - previous.recipes)) if previous.recipes else (),
    )
