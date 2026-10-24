"""Domena: przepływy insiderów (SEC Form 4 / 13F) — nowe źródło alpha.

Czysta logika: snapshot aktywności insiderów dla symbolu + klasyfikacja sygnału
(netto kupno/sprzedaż). Zero importów zewnętrznych — adapter EDGAR żyje w
infrastrukturze i mapuje surowe filingi na ten model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.domain.value_objects import AssetType


def has_insider_filings(asset_type: AssetType) -> bool:
    """Czy aktywo ma insiderów składających Form 4 — tylko spółki (STOCK).

    Krypto nie ma insiderów SEC, a ticker `BTC`/`ETH` w mapie SEC może należeć do
    trustu ETF, więc wiersz krypto dostałby cudze dane. ETF-y to fundusze otwarte
    bez Form 4: puste okno głosowałoby w Alpha Fusion neutralnie. Ta sama granica
    co przy fundamentach (`Asset.evaluate_valuation`)."""
    return asset_type is AssetType.STOCK


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
    Adapter wypełnia te pola z Form 4 (transakcje insiderów).

    `filings_seen`/`filings_parsed` (DAT-03): ile Form 4 z okna adapter widział
    i ile z nich sparsował. Bez nich niesparsowane filingi wyglądały jak okno bez
    transakcji. Domyślne zera = snapshot bez tej wiedzy, traktowany jak dowód."""

    symbol: str
    net_shares: float
    buy_count: int
    sell_count: int
    window_days: int
    filings_seen: int = 0
    filings_parsed: int = 0

    def has_evidence(self) -> bool:
        """Czy snapshot mówi coś o insiderach. Okno bez Form 4 to dowód braku
        transakcji; Form 4 widziane, a żaden niesparsowany, to brak danych EDGAR —
        zerowe liczniki nie mogą wtedy udawać spokojnego okna. Częściowo sparsowane
        okno zostaje dowodem (częściowa agregacja jak dotąd)."""
        return self.filings_seen == 0 or self.filings_parsed > 0

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
