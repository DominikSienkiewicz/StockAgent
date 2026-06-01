"""Risk Watch — równoległy use case do głównej pętli predykcyjnej.

Pobiera ceny instrumentów proxy ryzyka makro (inverse ETFs, gold, VIX,
EPOL dla PL), zapisuje snapshot ceny w każdym cyklu (cold-start guard),
porównuje vs poprzednia cena i wystawia poziom alertu per instrument.
Dodatkowo, jeśli skonfigurowany jest MacroIndicatorsPort, dorzuca snapshot
polskiego makro (kurs PLN) i jego poziom stresu.

Wyjątki per-symbol nie wywalają cyklu — analogicznie do main_agent.main(),
pojedynczy ticker padający na 403 (Finnhub free + LSE) jest logowany
i pomijany.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.application.ports import (
    MacroIndicatorsPort,
    MarketDataPort,
    RepositoryPort,
)
from src.domain.macro_risk import (
    MacroAlertLevel,
    MacroRiskInstrumentType,
    MacroRiskSignal,
)
from src.domain.polish_macro import MacroStressLevel, PolishMacroSnapshot

logger = logging.getLogger(__name__)

# Mapowanie poziomu stresu PL → MacroAlertLevel, by overall_alert miał
# jedną wspólną skalę.
_STRESS_TO_ALERT = {
    MacroStressLevel.NORMAL: MacroAlertLevel.NORMAL,
    MacroStressLevel.ELEVATED: MacroAlertLevel.ELEVATED,
    MacroStressLevel.CRITICAL: MacroAlertLevel.CRITICAL,
}

_ALERT_RANK = {
    MacroAlertLevel.NORMAL: 0,
    MacroAlertLevel.ELEVATED: 1,
    MacroAlertLevel.CRITICAL: 2,
}


@dataclass(frozen=True)
class MacroRiskReport:
    signals: list[MacroRiskSignal] = field(default_factory=list)
    polish_macro: PolishMacroSnapshot | None = None
    overall_alert: MacroAlertLevel = MacroAlertLevel.NORMAL


class MonitorMacroRiskUseCase:
    """Slow scanner — uruchamiany po głównej pętli AnalyzeMarketUseCase."""

    def __init__(
        self,
        *,
        market_port: MarketDataPort,
        repository_port: RepositoryPort,
        macro_port: MacroIndicatorsPort | None = None,
    ) -> None:
        self._market = market_port
        self._repository = repository_port
        self._macro = macro_port

    def run(
        self, symbol_types: dict[str, MacroRiskInstrumentType]
    ) -> MacroRiskReport:
        signals = self._collect_signals(symbol_types)
        polish_macro = self._fetch_polish_macro()
        overall = self._compute_overall_alert(signals, polish_macro)
        return MacroRiskReport(
            signals=signals,
            polish_macro=polish_macro,
            overall_alert=overall,
        )

    def _collect_signals(
        self, symbol_types: dict[str, MacroRiskInstrumentType]
    ) -> list[MacroRiskSignal]:
        out: list[MacroRiskSignal] = []
        for symbol, instrument_type in symbol_types.items():
            try:
                current = self._market.get_current_price(symbol)
                previous = self._repository.get_last_price(symbol)
                self._repository.save_price_snapshot(symbol, current)
                if previous is None:
                    # Cold start: brak referencji — nie da się policzyć zmiany.
                    logger.info(
                        "risk_watch cold-start for %s — snapshot saved, "
                        "signal skipped",
                        symbol,
                    )
                    continue
                out.append(
                    MacroRiskSignal(
                        symbol=symbol,
                        instrument_type=instrument_type,
                        current_price=current,
                        previous_price=previous,
                    )
                )
            except Exception:
                logger.exception("risk_watch signal failed for %s", symbol)
        return out

    def _fetch_polish_macro(self) -> PolishMacroSnapshot | None:
        if self._macro is None:
            return None
        try:
            return self._macro.fetch_polish_macro()
        except Exception:
            logger.exception("risk_watch polish macro fetch failed")
            return None

    @staticmethod
    def _compute_overall_alert(
        signals: list[MacroRiskSignal],
        polish_macro: PolishMacroSnapshot | None,
    ) -> MacroAlertLevel:
        levels: list[MacroAlertLevel] = [s.evaluate_alert() for s in signals]
        if polish_macro is not None:
            levels.append(_STRESS_TO_ALERT[polish_macro.evaluate_stress_level()])
        if not levels:
            return MacroAlertLevel.NORMAL
        return max(levels, key=lambda lvl: _ALERT_RANK[lvl])
