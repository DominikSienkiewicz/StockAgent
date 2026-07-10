"""Domena: stopy i krzywa rentowności (FRED) — tło makro reżimu.

Czysta logika: spread 10Y-2Y i poziom stóp klasyfikują stan krzywej
rentowności. Inwersja (10Y < 2Y) to klasyczny sygnał recesyjny — istotne tło
makro, które może zacieśniać reżim (komplementarne do Risk Watch / AI #7)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.domain.macro_risk import MacroAlertLevel


class YieldCurveState(Enum):
    """Stan krzywej rentowności wyprowadzony ze spreadu 10Y-2Y."""

    INVERTED = "INVERTED"   # 10Y < 2Y — sygnał recesyjny
    FLAT = "FLAT"           # spread dodatni, ale wąski
    NORMAL = "NORMAL"       # zdrowy dodatni spread


@dataclass(frozen=True)
class YieldCurveSnapshot:
    """Snapshot kluczowych stóp (w procentach, np. 4.25 = 4.25%).

    Wszystkie pola opcjonalne (None gdy FRED nie zwrócił serii) — klasyfikacja
    działa gracefully na brakach."""

    ten_year: float | None
    two_year: float | None
    fed_funds: float | None = None

    @property
    def spread_10y_2y(self) -> float | None:
        """Spread 10Y-2Y w punktach procentowych. None gdy brak którejś serii."""
        if self.ten_year is None or self.two_year is None:
            return None
        return self.ten_year - self.two_year

    def state(
        self, invert_below: float = 0.0, flat_below: float = 0.5
    ) -> YieldCurveState:
        """Klasyfikuje krzywą. Spread < `invert_below` → INVERTED; <
        `flat_below` → FLAT; w pozostałych przypadkach (też brak danych,
        traktowany neutralnie) → NORMAL."""
        spread = self.spread_10y_2y
        if spread is None:
            return YieldCurveState.NORMAL
        if spread < invert_below:
            return YieldCurveState.INVERTED
        if spread < flat_below:
            return YieldCurveState.FLAT
        return YieldCurveState.NORMAL


def yield_curve_alert_level(state: YieldCurveState) -> MacroAlertLevel:
    """Mapuje stan krzywej rentowności na poziom alertu makro.

    Inwersja (10Y < 2Y) to klasyczny sygnał recesyjny → ELEVATED; pozostałe
    stany (FLAT/NORMAL) nie są sygnałem ryzyka → NORMAL.

    Zwraca POJEDYNCZY poziom alertu — JEDEN głos, nie decyzję o reżimie.
    Orkiestrator dokłada go do licznika `RegimeDetector` (reguła ">= 2 głosy
    ELEVATED → RISK_OFF"), więc sama inwersja bez innych sygnałów nie wymusza
    RISK_OFF. To celowa semantyka "głos, nie weto": chroni przed
    wielomiesięcznymi inwersjami, które inaczej trzymałyby agenta w risk-off.
    """
    if state is YieldCurveState.INVERTED:
        return MacroAlertLevel.ELEVATED
    return MacroAlertLevel.NORMAL
