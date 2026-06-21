"""Domena: przepływy opcji i zmienność implikowana (IV) — sygnał pozycjonowania.

Czysta logika: put/call ratio i IV klasyfikują nastawienie rynku opcji.
Niski P/C = przewaga calli (byczo), wysoki = przewaga putów (niedźwiedzio).
IV podbija lub obniża zaufanie do prognozy (wysoka IV = duża niepewność)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OptionsSentiment(Enum):
    """Nastawienie wyprowadzone z put/call ratio."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class OptionsFlowSnapshot:
    """Snapshot rynku opcji dla symbolu.

    `put_call_ratio` — wolumen putów / calli (None = brak danych). `implied_vol`
    — implikowana zmienność jako ułamek (np. 0.45 = 45% annualizowane), None gdy
    brak."""

    symbol: str
    put_call_ratio: float | None
    implied_vol: float | None = None

    def sentiment(
        self, bullish_below: float = 0.7, bearish_above: float = 1.0
    ) -> OptionsSentiment:
        """Klasyfikuje nastawienie. Niski P/C (< `bullish_below`) → BULLISH;
        wysoki (> `bearish_above`) → BEARISH; pośrodku / brak danych → NEUTRAL."""
        if self.put_call_ratio is None:
            return OptionsSentiment.NEUTRAL
        if self.put_call_ratio < bullish_below:
            return OptionsSentiment.BULLISH
        if self.put_call_ratio > bearish_above:
            return OptionsSentiment.BEARISH
        return OptionsSentiment.NEUTRAL

    def is_high_iv(self, threshold: float = 0.6) -> bool:
        """Czy IV jest podwyższona (sygnał dużej niepewności / drogich opcji)."""
        return self.implied_vol is not None and self.implied_vol >= threshold
