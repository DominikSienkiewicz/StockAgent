"""ALR-05 — powód reżimu dla czytelnika maila: ile głosów i jakiego poziomu
zdecydowało o klasyfikacji (`RegimeDetector.reason`)."""

from __future__ import annotations

from src.domain.macro_risk import MacroAlertLevel
from src.domain.regime import RegimeDetector


def test_no_macro_data_is_named() -> None:
    assert RegimeDetector().reason(None, []) == "brak danych makro"


def test_elevated_votes_are_counted_with_polish_plural() -> None:
    detector = RegimeDetector()

    one = detector.reason(MacroAlertLevel.ELEVATED, [])
    two = detector.reason(MacroAlertLevel.ELEVATED, [MacroAlertLevel.ELEVATED])
    five = detector.reason(MacroAlertLevel.ELEVATED, [MacroAlertLevel.ELEVATED] * 4)

    assert one == "1 głos ELEVATED"
    assert two == "2 głosy ELEVATED"
    assert five == "5 głosów ELEVATED"


def test_critical_wins_over_elevated_votes() -> None:
    reason = RegimeDetector().reason(MacroAlertLevel.CRITICAL, [MacroAlertLevel.ELEVATED])

    assert reason == "alert CRITICAL"


def test_calm_market_says_so() -> None:
    reason = RegimeDetector().reason(MacroAlertLevel.NORMAL, [MacroAlertLevel.NORMAL])

    assert reason == "wszystkie sygnały NORMAL"
