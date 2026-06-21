"""Domena detektora reżimu rynkowego.

Agreguje najgorszy alert makro oraz liczbę/severity sygnałów risk-off w jeden
z trzech reżimów (RISK_ON / NEUTRAL / RISK_OFF). Reżim steruje TYLKO
zacieśnianiem bramki volatility — `threshold_multiplier` nigdy nie schodzi
poniżej 1.0, więc reżim może co najwyżej podnieść próg (zmniejszyć liczbę
płatnych wywołań), nigdy go poluzować.

Wejściem są poziomy alertu na wspólnej skali NORMAL/ELEVATED/CRITICAL:
`MacroAlertLevel` (macro_risk, drawdown) oraz `MacroStressLevel`
(polish_macro). Domena nie importuje konkretnego enuma — porównuje po nazwie
członka, żeby pozostać odporną i wymienną między oboma skalami (stdlib only).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Protocol


class _AlertLevel(Protocol):
    """Strukturalny kontrakt poziomu alertu (NORMAL/ELEVATED/CRITICAL).

    `MacroAlertLevel` i `MacroStressLevel` spełniają go bez wspólnej bazy —
    liczy się tylko nazwa członka enuma.
    """

    @property
    def name(self) -> str: ...


class MarketRegime(Enum):
    RISK_ON = "RISK_ON"
    NEUTRAL = "NEUTRAL"
    RISK_OFF = "RISK_OFF"


# Mapowanie nazwy poziomu alertu na rangę severity (im wyżej, tym gorzej).
# Nieznana nazwa traktowana jak NORMAL (0) — fail-safe, nie zawyżamy alertu.
_SEVERITY = {"NORMAL": 0, "ELEVATED": 1, "CRITICAL": 2}

# Próg liczby sygnałów ELEVATED (bez żadnego CRITICAL), od którego sam ich
# nawał przeważa na risk-off — pojedynczy ELEVATED to jeszcze szum.
_ELEVATED_RISK_OFF_COUNT = 2


class RegimeDetector:
    """Czysty klasyfikator reżimu — bez stanu, bez I/O."""

    def classify(
        self,
        macro_alert: _AlertLevel | None,
        risk_signals: Sequence[_AlertLevel] | None,
    ) -> MarketRegime:
        """Mapuje najgorszy alert makro + sygnały risk-off na reżim.

        Reguły (od najostrzejszej):
        - brak danych makro (None) → NEUTRAL (brak informacji, nie risk-on),
        - CRITICAL makro LUB jakikolwiek CRITICAL sygnał risk-off → RISK_OFF,
        - >= 2 sygnały ELEVATED (bez CRITICAL) → RISK_OFF (nawał ostrzeżeń),
        - dokładnie jeden ELEVATED (makro lub sygnał) → NEUTRAL,
        - wszystko NORMAL → RISK_ON.
        """
        # Brak punktu odniesienia makro = nie ufamy "czystości" rynku.
        if macro_alert is None:
            return MarketRegime.NEUTRAL

        signals = list(risk_signals) if risk_signals is not None else []

        macro_severity = self._severity(macro_alert)
        signal_severities = [self._severity(s) for s in signals]
        worst_severity = max([macro_severity, *signal_severities], default=0)

        # Dowolny CRITICAL (makro lub sygnał) natychmiast → RISK_OFF.
        if worst_severity >= _SEVERITY["CRITICAL"]:
            return MarketRegime.RISK_OFF

        # Liczymy WSZYSTKIE poziomy ELEVATED — makro liczy się tak samo
        # jak pojedynczy sygnał risk-off.
        elevated_count = signal_severities.count(_SEVERITY["ELEVATED"])
        if macro_severity == _SEVERITY["ELEVATED"]:
            elevated_count += 1

        if elevated_count >= _ELEVATED_RISK_OFF_COUNT:
            return MarketRegime.RISK_OFF
        if elevated_count == 1:
            return MarketRegime.NEUTRAL
        return MarketRegime.RISK_ON

    def threshold_multiplier(
        self, regime: MarketRegime, max_multiplier: float
    ) -> float:
        """Mnożnik progu volatility dla danego reżimu.

        Niezmiennik FinOps: wynik jest ZAWSZE w [1.0, max(1.0, max_multiplier)].
        RISK_OFF zwraca górny kraniec (najmocniejsze zacieśnienie), NEUTRAL i
        RISK_ON zwracają 1.0. Reżim nigdy nie poluzuje bramki (< 1.0), nawet
        gdy podano absurdalny `max_multiplier` < 1.0.
        """
        ceiling = max(1.0, max_multiplier)
        if regime is MarketRegime.RISK_OFF:
            return ceiling
        return 1.0

    def label(self, regime: MarketRegime) -> str:
        """Krótka polska etykieta reżimu do bannera w raporcie."""
        return {
            MarketRegime.RISK_ON: "Apetyt na ryzyko",
            MarketRegime.NEUTRAL: "Neutralny",
            MarketRegime.RISK_OFF: "Ucieczka od ryzyka",
        }[regime]

    @staticmethod
    def _severity(level: _AlertLevel) -> int:
        """Ranga severity z nazwy członka enuma (nieznana → 0)."""
        return _SEVERITY.get(level.name, 0)
