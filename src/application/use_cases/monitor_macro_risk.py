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
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from src.application.ports import (
    MacroIndicatorsPort,
    MarketDataPort,
    RepositoryPort,
)
from src.domain.correlation import returns_from_prices
from src.domain.drawdown import DrawdownSignal, peak_from_history
from src.domain.hedge import HedgeAssessment, assess_hedge
from src.domain.macro_risk import (
    MacroAlertLevel,
    MacroRiskInstrumentType,
    MacroRiskSignal,
)
from src.domain.polish_macro import MacroStressLevel, PolishMacroSnapshot
from src.domain.value_objects import Money

logger = logging.getLogger(__name__)

# Okno historii snapshotów do liczenia szczytu (darmowe odczyty z repo).
_DRAWDOWN_HISTORY_DAYS = 30

# Domyślne progi drawdownu (dodatnie ułamki głębokości spadku):
# ELEVATED przy -10% od szczytu, CRITICAL przy -20% — kalibracja pod
# swing instrumentów akcyjnych, nie dzienny szum.
_DRAWDOWN_ELEVATED_PCT = Decimal("0.10")
_DRAWDOWN_CRITICAL_PCT = Decimal("0.20")

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
    # Per-symbol drawdown od szczytu (liczony z darmowych snapshotów).
    # Default pusty → wsteczna kompatybilność starszych ścieżek raportu.
    drawdowns: list[DrawdownSignal] = field(default_factory=list)
    # R3: skuteczność hedge'y (realized corr instrumentów ryzyka vs book).
    # Default pusty → wsteczna kompatybilność.
    hedges: list[HedgeAssessment] = field(default_factory=list)


class MonitorMacroRiskUseCase:
    """Slow scanner — uruchamiany po głównej pętli AnalyzeMarketUseCase.

    Opcjonalnie (flaga `hedge_enabled` + `book_symbols`) liczy skuteczność
    hedge'y (R3): realized correlation instrumentów ryzyka vs portfel bazowy.
    Wszystko z DARMOWYCH snapshotów — zero płatnych portów.
    """

    def __init__(
        self,
        *,
        market_port: MarketDataPort,
        repository_port: RepositoryPort,
        macro_port: MacroIndicatorsPort | None = None,
        book_symbols: Sequence[str] = (),
        hedge_enabled: bool = False,
    ) -> None:
        self._market = market_port
        self._repository = repository_port
        self._macro = macro_port
        self._book_symbols = tuple(book_symbols)
        self._hedge_enabled = hedge_enabled

    def run(
        self, symbol_types: dict[str, MacroRiskInstrumentType]
    ) -> MacroRiskReport:
        signals = self._collect_signals(symbol_types)
        drawdowns = self._collect_drawdowns(symbol_types)
        hedges = self._collect_hedge_assessments(symbol_types)
        polish_macro = self._fetch_polish_macro()
        overall = self._compute_overall_alert(signals, drawdowns, polish_macro)
        return MacroRiskReport(
            signals=signals,
            polish_macro=polish_macro,
            overall_alert=overall,
            drawdowns=drawdowns,
            hedges=hedges,
        )

    def _collect_hedge_assessments(
        self, symbol_types: dict[str, MacroRiskInstrumentType]
    ) -> list[HedgeAssessment]:
        """Skuteczność każdego instrumentu ryzyka jako hedge'a portfela bazowego.

        Dla każdego symbolu ryzyka liczy realized correlation jego zwrotów ze
        zwrotami portfela równoważonego (book_symbols), wyrównanych po wspólnych
        timestampach. Instrumenty ryzyka traktujemy jak hedge ODWROTNY (mają
        rosnąć, gdy book spada → `expect_inverse=True`). Wyłączone / brak booka /
        brak historii → []. Pojedynczy błąd nie wywala cyklu."""
        if not self._hedge_enabled or not self._book_symbols:
            return []
        all_symbols = list(dict.fromkeys([*self._book_symbols, *symbol_types]))
        raw: dict[str, list[tuple[datetime, Money]]] = {}
        for symbol in all_symbols:
            try:
                history = self._repository.get_price_history(
                    symbol, days=_DRAWDOWN_HISTORY_DAYS
                )
            except Exception:
                logger.exception("hedge history fetch failed for %s", symbol)
                continue
            if len(history) >= 3:
                raw[symbol] = history

        out: list[HedgeAssessment] = []
        for hedge_symbol in symbol_types:
            if hedge_symbol not in raw:
                continue
            # Wyrównujemy book + ten hedge po WSPÓLNYCH timestampach (każdy hedge
            # może mieć inny przekrój dat z bookiem).
            subset = {b: raw[b] for b in self._book_symbols if b in raw}
            if not subset:
                continue
            subset[hedge_symbol] = raw[hedge_symbol]
            aligned = self._aligned_returns(subset)
            hedge_returns = aligned.get(hedge_symbol)
            book_series = [aligned[b] for b in self._book_symbols if b in aligned]
            if not hedge_returns or not book_series:
                continue
            count = len(book_series)
            length = len(hedge_returns)
            book_returns = [
                sum(series[t] for series in book_series) / count
                for t in range(length)
            ]
            out.append(
                assess_hedge(
                    hedge_symbol,
                    book_returns,
                    hedge_returns,
                    expect_inverse=True,
                )
            )
        return out

    @staticmethod
    def _aligned_returns(
        raw: dict[str, list[tuple[datetime, Money]]],
    ) -> dict[str, list[float]]:
        """Wyrównuje szeregi cen po WSPÓLNYCH timestampach i liczy zwroty.

        Analogiczne do PortfolioRiskUseCase — po wyrównaniu wszystkie szeregi
        mają tę samą długość, więc zwroty są sparowane okres-w-okres. Symbol
        z <2 zwrotami po wyrównaniu jest pomijany."""
        if not raw:
            return {}
        common: set[datetime] | None = None
        for series in raw.values():
            stamps = {ts for ts, _ in series}
            common = stamps if common is None else (common & stamps)
        if not common:
            return {}
        ordered = sorted(common)
        out: dict[str, list[float]] = {}
        for symbol, series in raw.items():
            by_ts = {ts: float(price.amount) for ts, price in series}
            rets = returns_from_prices([by_ts[ts] for ts in ordered])
            if len(rets) >= 2:
                out[symbol] = rets
        return out

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

    def _collect_drawdowns(
        self, symbol_types: dict[str, MacroRiskInstrumentType]
    ) -> list[DrawdownSignal]:
        """Drawdown per symbol z DARMOWEJ historii snapshotów.

        Dla każdego śledzonego symbolu pobiera 30-dniową historię cen,
        wyznacza szczyt (max) i porównuje z ostatnim snapshotem. Symbol bez
        historii (cold start) jest pomijany. Pojedynczy błąd nie wywala cyklu.
        """
        out: list[DrawdownSignal] = []
        for symbol in symbol_types:
            try:
                history = self._repository.get_price_history(
                    symbol, days=_DRAWDOWN_HISTORY_DAYS
                )
                if not history:
                    continue
                prices = [price.amount for _, price in history]
                peak = peak_from_history(prices)
                current = prices[-1]  # historia rosnąco — ostatni = najnowszy
                out.append(
                    DrawdownSignal(
                        symbol=symbol,
                        current_price=current,
                        peak_price=peak,
                    )
                )
            except Exception:
                logger.exception("risk_watch drawdown failed for %s", symbol)
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
        drawdowns: list[DrawdownSignal],
        polish_macro: PolishMacroSnapshot | None,
    ) -> MacroAlertLevel:
        levels: list[MacroAlertLevel] = [s.evaluate_alert() for s in signals]
        levels.extend(
            dd.evaluate_drawdown(
                elevated_pct=_DRAWDOWN_ELEVATED_PCT,
                critical_pct=_DRAWDOWN_CRITICAL_PCT,
            )
            for dd in drawdowns
        )
        if polish_macro is not None:
            levels.append(_STRESS_TO_ALERT[polish_macro.evaluate_stress_level()])
        if not levels:
            return MacroAlertLevel.NORMAL
        return max(levels, key=lambda lvl: _ALERT_RANK[lvl])
