"""Domena: prędkość społeczna (Reddit/X) — tempo wzmianek vs baseline.

Czysta logika: porównuje wzmianki z ostatnich 24h z historycznym baseline'em
i klasyfikuje trend (surge / quiet). Skok wzmianek bywa wczesnym sygnałem
zainteresowania detalu (meme-stock dynamics) zanim ruch ujawni się w cenie."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SocialTrend(Enum):
    """Tempo wzmianek względem baseline'u."""

    SURGING = "SURGING"
    NORMAL = "NORMAL"
    QUIET = "QUIET"


@dataclass(frozen=True)
class SocialVelocitySnapshot:
    """Snapshot aktywności społecznej dla symbolu.

    `mentions_24h` — liczba wzmianek z ostatniej doby. `baseline_mentions` —
    typowa dzienna liczba wzmianek (średnia ruchoma). `avg_sentiment` —
    uśredniony sentyment wzmianek w [-1, 1] (None gdy brak danych)."""

    symbol: str
    mentions_24h: int
    baseline_mentions: float
    avg_sentiment: float | None = None

    def velocity_ratio(self) -> float:
        """Stosunek wzmianek 24h do baseline'u. Baseline ≤ 0 → 0.0 (brak
        odniesienia, nie dzielimy przez zero)."""
        if self.baseline_mentions <= 0:
            return 0.0
        return self.mentions_24h / self.baseline_mentions

    def trend(self, surge_ratio: float = 2.0, quiet_ratio: float = 0.5) -> SocialTrend:
        """Klasyfikuje trend. Ratio ≥ `surge_ratio` → SURGING; ≤ `quiet_ratio`
        → QUIET; pośrodku → NORMAL. Brak baseline'u (ratio 0.0) → QUIET."""
        ratio = self.velocity_ratio()
        if ratio >= surge_ratio:
            return SocialTrend.SURGING
        if ratio <= quiet_ratio:
            return SocialTrend.QUIET
        return SocialTrend.NORMAL
