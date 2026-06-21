"""Zagregowane sygnały alpha per symbol z nowych źródeł danych (kategoria „Dane").

Render-only kontener: zbiera opcjonalne snapshoty z portów (insider, earnings,
opcje, social, analitycy) w jeden obiekt, który raport prezentuje. Każde pole
może być None (źródło wyłączone / brak danych)."""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.analyst_consensus import AnalystConsensus
from src.domain.earnings import EarningsEvent
from src.domain.insider_flow import InsiderFlowSnapshot
from src.domain.options_flow import OptionsFlowSnapshot
from src.domain.social_velocity import SocialVelocitySnapshot


@dataclass(frozen=True)
class AlphaSignals:
    """Komplet sygnałów alpha dla jednego symbolu (wszystkie opcjonalne)."""

    symbol: str
    insider: InsiderFlowSnapshot | None = None
    earnings: EarningsEvent | None = None
    options: OptionsFlowSnapshot | None = None
    social: SocialVelocitySnapshot | None = None
    analyst: AnalystConsensus | None = None

    def has_any(self) -> bool:
        """Czy jest cokolwiek do pokazania (przynajmniej jedno źródło dało dane)."""
        return any(
            signal is not None
            for signal in (
                self.insider,
                self.earnings,
                self.options,
                self.social,
                self.analyst,
            )
        )
