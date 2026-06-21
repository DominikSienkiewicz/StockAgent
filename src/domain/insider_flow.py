"""Domena: przepływy insiderów (SEC Form 4 / 13F) — nowe źródło alpha.

Czysta logika: snapshot aktywności insiderów dla symbolu + klasyfikacja sygnału
(netto kupno/sprzedaż). Zero importów zewnętrznych — adapter EDGAR żyje w
infrastrukturze i mapuje surowe filingi na ten model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InsiderSignal(Enum):
    """Kierunek przepływu insiderów wyprowadzony ze snapshotu transakcji."""

    NET_BUYING = "NET_BUYING"
    NET_SELLING = "NET_SELLING"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class InsiderFlowSnapshot:
    """Zagregowana aktywność insiderów dla symbolu w oknie `window_days`.

    `net_shares` > 0 oznacza netto kupno (insiderzy dokupują), < 0 netto
    sprzedaż. `buy_count`/`sell_count` to liczba transakcji każdego typu.
    Adapter wypełnia te pola z Form 4 (transakcje insiderów)."""

    symbol: str
    net_shares: float
    buy_count: int
    sell_count: int
    window_days: int

    def evaluate_signal(self, min_net_shares: float = 0.0) -> InsiderSignal:
        """Klasyfikuje przepływ. Brak transakcji → NEUTRAL; netto powyżej progu
        `min_net_shares` → NET_BUYING; poniżej `-min_net_shares` → NET_SELLING;
        w martwej strefie wokół zera → NEUTRAL."""
        if self.buy_count == 0 and self.sell_count == 0:
            return InsiderSignal.NEUTRAL
        if self.net_shares > min_net_shares:
            return InsiderSignal.NET_BUYING
        if self.net_shares < -min_net_shares:
            return InsiderSignal.NET_SELLING
        return InsiderSignal.NEUTRAL
