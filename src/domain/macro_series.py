"""Domena: zmienne stanu rynku z FRED (DAT-22) — VIX, VXN, HY OAS, T10Y2Y, NFCI.

Reżim (`regime.py`) opierał się na dziennej zmianie cen ETF-ów proxy z progami 1%
i 3% (`macro_risk.py`) — dla VIXY (contango) to szum. Poziomy VIX i spreadów
kredytowych to standardowe, stabilniejsze zmienne stanu. Ten moduł liczy z nich:

- percentyl VIX z ostatniego roku (VIXCLS) — głos dla `RegimeDetector`,
- 5-obserwacyjną zmianę spreadu high yield (BAMLH0A0HYM2, punkty procentowe) — głos,
- cechy dla modelu wspólnego (poziomy VIX, VXN, HY OAS, T10Y2Y, NFCI i powyższe).

Daty obserwacji, nie pobrania: FRED publikuje z opóźnieniem (VIXCLS z poprzedniej
sesji), a NFCI jest tygodniowe — `observed_on` mówi, z którego dnia jest wartość.
Stdlib only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from src.domain.macro_risk import MacroAlertLevel

# Minimalna liczba obserwacji VIX w oknie roku, od której percentyl coś znaczy
# (~kwartał sesji); mniej → brak percentyla i brak głosu.
MIN_VIX_OBSERVATIONS = 60
# Percentyl VIX z roku: od 75 (górny kwartyl) ELEVATED, powyżej 90 CRITICAL.
VIX_ELEVATED_PERCENTILE = 75.0
VIX_CRITICAL_PERCENTILE = 90.0
# Zmiana HY OAS w 5 obserwacjach (pp): +0.25 (25 pb) ELEVATED, +0.50 CRITICAL.
HY_OAS_CHANGE_OBSERVATIONS = 5
HY_OAS_ELEVATED_CHANGE = 0.25
HY_OAS_CRITICAL_CHANGE = 0.50

MACRO_FEATURE_NAMES: tuple[str, ...] = (
    "vix_level",
    "vix_percentile_1y",
    "vxn_level",
    "hy_oas",
    "hy_oas_change_5d",
    "term_spread_10y2y",
    "nfci",
    "macro_observed_on",
)


@dataclass(frozen=True)
class SeriesObservation:
    """Jedna obserwacja serii FRED: dzień, którego dotyczy wartość, i wartość."""

    observed_on: date
    value: float


@dataclass(frozen=True)
class MacroSeriesSnapshot:
    """Serie makro jednego cyklu, rosnąco po dniu obserwacji.

    `vix` to okno roku (percentyl), `hy_oas` okno kilku tygodni (zmiana 5 obs.);
    pozostałe serie to ostatnia obserwacja. Brak serii = pusta krotka / None."""

    vix: tuple[SeriesObservation, ...] = ()
    hy_oas: tuple[SeriesObservation, ...] = ()
    vxn: SeriesObservation | None = None
    term_spread: SeriesObservation | None = None
    nfci: SeriesObservation | None = None

    @property
    def is_empty(self) -> bool:
        return not self.vix and not self.hy_oas and all(
            series is None for series in (self.vxn, self.term_spread, self.nfci)
        )


def vix_percentile(window: Sequence[SeriesObservation]) -> float | None:
    """Percentyl (0–100) ostatniej obserwacji VIX w oknie: odsetek obserwacji ≤ niej.

    None przy mniej niż `MIN_VIX_OBSERVATIONS` obserwacjach."""
    if len(window) < MIN_VIX_OBSERVATIONS:
        return None
    latest = window[-1].value
    return 100.0 * sum(1 for obs in window if obs.value <= latest) / len(window)


def hy_oas_change(
    window: Sequence[SeriesObservation], observations: int = HY_OAS_CHANGE_OBSERVATIONS
) -> float | None:
    """Zmiana HY OAS (pp) względem obserwacji sprzed `observations` sesji, albo None."""
    if len(window) <= observations:
        return None
    return window[-1].value - window[-1 - observations].value


def vix_percentile_alert_level(percentile: float) -> MacroAlertLevel:
    """Głos VIX dla `RegimeDetector`: > 90 CRITICAL, ≥ 75 ELEVATED, inaczej NORMAL."""
    if percentile > VIX_CRITICAL_PERCENTILE:
        return MacroAlertLevel.CRITICAL
    if percentile >= VIX_ELEVATED_PERCENTILE:
        return MacroAlertLevel.ELEVATED
    return MacroAlertLevel.NORMAL


def credit_spread_alert_level(change: float) -> MacroAlertLevel:
    """Głos spreadu HY: wzrost ≥ 0.50 pp CRITICAL, ≥ 0.25 pp ELEVATED (5 sesji)."""
    if change >= HY_OAS_CRITICAL_CHANGE:
        return MacroAlertLevel.CRITICAL
    if change >= HY_OAS_ELEVATED_CHANGE:
        return MacroAlertLevel.ELEVATED
    return MacroAlertLevel.NORMAL


def macro_series_votes(snapshot: MacroSeriesSnapshot) -> tuple[MacroAlertLevel, ...]:
    """Głosy VIX i HY OAS do licznika `RegimeDetector` (brak danych = brak głosu).

    Głos, nie weto, jak krzywa rentowności: CRITICAL i tak daje RISK_OFF, ale
    pojedynczy ELEVATED tylko NEUTRAL."""
    votes: list[MacroAlertLevel] = []
    percentile = vix_percentile(snapshot.vix)
    if percentile is not None:
        votes.append(vix_percentile_alert_level(percentile))
    change = hy_oas_change(snapshot.hy_oas)
    if change is not None:
        votes.append(credit_spread_alert_level(change))
    return tuple(votes)


def _latest(window: Sequence[SeriesObservation]) -> SeriesObservation | None:
    return window[-1] if window else None


def macro_features(snapshot: MacroSeriesSnapshot) -> dict[str, Any]:
    """Cechy makro cyklu (`MACRO_FEATURE_NAMES`); brak serii = None.

    `macro_observed_on` to mapa seria → dzień obserwacji (ISO), żeby trening nie
    pomylił wartości z dnia publikacji z wartością z dnia cyklu."""
    latest = {
        "VIXCLS": _latest(snapshot.vix),
        "VXNCLS": snapshot.vxn,
        "BAMLH0A0HYM2": _latest(snapshot.hy_oas),
        "T10Y2Y": snapshot.term_spread,
        "NFCI": snapshot.nfci,
    }
    return {
        "vix_level": latest["VIXCLS"].value if latest["VIXCLS"] else None,
        "vix_percentile_1y": vix_percentile(snapshot.vix),
        "vxn_level": snapshot.vxn.value if snapshot.vxn else None,
        "hy_oas": latest["BAMLH0A0HYM2"].value if latest["BAMLH0A0HYM2"] else None,
        "hy_oas_change_5d": hy_oas_change(snapshot.hy_oas),
        "term_spread_10y2y": snapshot.term_spread.value if snapshot.term_spread else None,
        "nfci": snapshot.nfci.value if snapshot.nfci else None,
        "macro_observed_on": {
            series: obs.observed_on.isoformat() for series, obs in latest.items() if obs
        },
    }
