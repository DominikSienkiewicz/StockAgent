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
    FundamentalsPort,
    InsiderFlowPort,
    OptionsFlowPort,
    SocialVelocityPort,
)
from src.domain.earnings import EarningsEvent
from src.domain.insider_flow import InsiderFlowSnapshot, has_insider_filings
from src.domain.value_objects import AssetType

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _stock(_symbol: str) -> AssetType:
    return AssetType.STOCK


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
        market_cap_port: FundamentalsPort | None = None,
        asset_type_of: Callable[[str], AssetType] = _stock,
    ) -> None:
        """`asset_type_of` klasyfikuje symbol (DI zna listy krypto/ETF); insider
        flow pytamy tylko o aktywa z Form 4 (`has_insider_filings`)."""
        self._insider = insider_port
        self._earnings = earnings_port
        self._options = options_port
        self._social = social_port
        self._analyst = analyst_port
        # DAT-21: fundamenty WYŁĄCZNIE z cache (DI) — kapitalizacja do progu
        # znaczącego zakupu insidera. None = próg stały, jak przed DAT-21.
        self._market_caps = market_cap_port
        self._asset_type_of = asset_type_of

    def fetch_earnings(self, symbol: str) -> EarningsEvent | None:
        """#6 — pobiera najbliższy raport kwartalny gracefully (błąd → None).

        Bramka earnings potrzebuje tego eventu PRZED grafem (żeby zacieśnić próg
        volatility), a `enrich` biegnie PO grafie. Bez tego wejścia orkiestrator
        musiałby wołać płatny port Alpha Vantage dwa razy na symbol.
        """
        return self._safe(self._earnings.get_next_earnings, symbol, "earnings")

    def enrich(
        self,
        symbol: str,
        *,
        earnings: EarningsEvent | None = None,
        earnings_prefetched: bool = False,
    ) -> AlphaSignals:
        """Agreguje snapshoty 5 portów alpha. Gdy `earnings_prefetched=True`,
        `earnings` jest brany wprost (także gdy to None — "brak raportu" to
        pełnoprawny wynik prefetchu) i port earnings NIE jest wołany ponownie.
        De-dup płatnego wywołania Alpha Vantage dla bramki earnings (#6)."""
        earnings_signal = (
            earnings
            if earnings_prefetched
            else self._safe(self._earnings.get_next_earnings, symbol, "earnings")
        )
        insider = (
            self._safe(self._insider.get_insider_flow, symbol, "insider")
            if has_insider_filings(self._asset_type_of(symbol))
            else None
        )
        return AlphaSignals(
            symbol=symbol,
            insider=insider,
            earnings=earnings_signal,
            options=self._safe(self._options.get_options_flow, symbol, "options"),
            social=self._safe(self._social.get_social_velocity, symbol, "social"),
            analyst=self._safe(self._analyst.get_consensus, symbol, "analyst"),
            market_cap=self._market_cap(symbol, insider),
        )

    def _market_cap(self, symbol: str, insider: InsiderFlowSnapshot | None) -> float | None:
        """DAT-21: kapitalizacja tylko, gdy jest zakup insidera do oceny — inaczej
        zero odczytów cache."""
        if self._market_caps is None or insider is None or insider.largest_buy is None:
            return None
        fundamentals = self._safe(self._market_caps.get_fundamentals, symbol, "market cap")
        return fundamentals.market_cap if fundamentals is not None else None

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
