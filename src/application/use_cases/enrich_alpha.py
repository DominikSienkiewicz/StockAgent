"""AlphaEnrichmentUseCase — agreguje nowe źródła alpha per symbol.

Spina pięć portów (insider, earnings, opcje, social, analitycy) w jeden
AlphaSignals. Każde źródło jest opcjonalne i ZAWSZE graceful: błąd lub brak
danych pojedynczego portu degraduje się do None, nie wywalając pozostałych.

W Fast Loop porty są domyślnie Null-adapterami (flagi off) → enrich zwraca
puste AlphaSignals bez żadnego wywołania sieciowego (zero płatnych)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TypeVar

from src.application.alpha_signals import AlphaSignals
from src.application.ports import (
    AnalystConsensusPort,
    EarningsCalendarPort,
    InsiderFlowPort,
    OptionsFlowPort,
    SocialVelocityPort,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class AlphaEnrichmentUseCase:
    """Zbiera sygnały alpha dla symbolu z pięciu niezależnych portów."""

    def __init__(
        self,
        *,
        insider_port: InsiderFlowPort,
        earnings_port: EarningsCalendarPort,
        options_port: OptionsFlowPort,
        social_port: SocialVelocityPort,
        analyst_port: AnalystConsensusPort,
    ) -> None:
        self._insider = insider_port
        self._earnings = earnings_port
        self._options = options_port
        self._social = social_port
        self._analyst = analyst_port

    def enrich(self, symbol: str) -> AlphaSignals:
        return AlphaSignals(
            symbol=symbol,
            insider=self._safe(self._insider.get_insider_flow, symbol, "insider"),
            earnings=self._safe(self._earnings.get_next_earnings, symbol, "earnings"),
            options=self._safe(self._options.get_options_flow, symbol, "options"),
            social=self._safe(self._social.get_social_velocity, symbol, "social"),
            analyst=self._safe(self._analyst.get_consensus, symbol, "analyst"),
        )

    @staticmethod
    def _safe(
        fetch: Callable[[str], _T | None], symbol: str, source: str
    ) -> _T | None:
        """Wywołuje port gracefully — błąd → None + log, nigdy nie propaguje."""
        try:
            return fetch(symbol)
        except Exception:
            logger.exception("alpha source %s failed for %s", source, symbol)
            return None
