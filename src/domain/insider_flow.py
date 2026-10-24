"""Domena: przepływy insiderów (SEC Form 4 / 13F) — nowe źródło alpha.

Czysta logika: snapshot aktywności insiderów dla symbolu + klasyfikacja sygnału
(netto kupno/sprzedaż). Zero importów zewnętrznych — adapter EDGAR żyje w
infrastrukturze i mapuje surowe filingi na ten model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import Enum

from src.domain.value_objects import AssetType

# DAT-21: próg znaczącego zakupu — kod P od officera za co najmniej tyle USD.
# Dolna granica przy każdej kapitalizacji i cały próg, gdy kapitalizacja nieznana.
SIGNIFICANT_BUY_MIN_USD = 100_000.0
# DAT-21: próg jako ułamek kapitalizacji — 4 mln USD przy spółce za 4 bln USD.
# Ta sama kwota znaczy co innego w małej i dużej spółce.
SIGNIFICANT_BUY_MIN_CAP_FRACTION = 1e-6


def significant_buy_threshold(market_cap: float | None) -> float:
    """DAT-21: próg znaczącego zakupu w USD — `max(1e-6 × kapitalizacja, 100k USD)`
    (decyzja właściciela: poniżej 100 mld USD kapitalizacji ułamek spadałby pod
    100k USD), a bez kapitalizacji (brak w cache, zero, NaN) samo 100k USD."""
    if market_cap is None or not math.isfinite(market_cap) or market_cap <= 0:
        return SIGNIFICANT_BUY_MIN_USD
    return max(market_cap * SIGNIFICANT_BUY_MIN_CAP_FRACTION, SIGNIFICANT_BUY_MIN_USD)


@dataclass(frozen=True)
class InsiderBuy:
    """Największy rynkowy zakup (kod P, poza planem 10b5-1) insidera w oknie.

    `role` — `officerTitle` z Form 4 (np. „CEO”; pusty, gdy brak tytułu albo
    kupuje członek rady); `is_officer` odróżnia zarząd od rady; `filing_url` —
    Form 4 tej transakcji. `value_usd` to suma zakupów P z jednego Form 4.
    `filed_on` — data filingu z EDGAR (okno leadu, `InsiderLeadWindow`); None =
    adapter jej nie zna."""

    role: str
    value_usd: float
    is_officer: bool = True
    filing_url: str | None = None
    filed_on: date | None = None


# DAT-21: tyle dni od daty filingu zakup insidera może otwierać maila (lead).
# 3 dni obejmują piątkowy filing, który pierwszy raz widzi poniedziałkowy cykl.
INSIDER_LEAD_MAX_AGE_DAYS = 3


@dataclass(frozen=True)
class InsiderLeadWindow:
    """DAT-21 (decyzja właściciela): zakup insidera prowadzi maila tylko przez
    `max_age_days` dni od filingu. Bez okna jeden Form 4 wracałby do leadu w każdym
    cyklu, dopóki jest w oknie insider flow (90 dni). `as_of` — dzień cyklu
    w kalendarzu NYSE, jak daty filingów EDGAR. Sekcji alfy okno nie dotyczy."""

    as_of: date
    max_age_days: int = INSIDER_LEAD_MAX_AGE_DAYS

    def admits(self, buy: InsiderBuy) -> bool:
        """Czy zakup może prowadzić maila w dniu `as_of` — filing najwyżej
        `max_age_days` dni wcześniej. Bez daty filingu brak dowodu wieku, więc
        zakup prowadzi jak przed oknem."""
        if buy.filed_on is None:
            return True
        return (self.as_of - buy.filed_on).days <= self.max_age_days


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
    transakcji. Domyślne zera = snapshot bez tej wiedzy, traktowany jak dowód.

    DAT-21 (pola opcjonalne, None = adapter ich nie policzył): `net_value_usd` —
    kupna P minus sprzedaże S w USD (akcje × cena), BEZ transakcji z planów
    10b5-1; `largest_buy` — największy zakup officera (albo członka rady, gdy
    officer nie kupował); `latest_filing_url` — najnowszy Form 4 okna.
    `net_shares` zostaje bez zmian: waży w Alpha Fusion, a zmiana wymaga A/B."""

    symbol: str
    net_shares: float
    buy_count: int
    sell_count: int
    window_days: int
    filings_seen: int = 0
    filings_parsed: int = 0
    net_value_usd: float | None = None
    largest_buy: InsiderBuy | None = None
    latest_filing_url: str | None = None

    def significant_buy(
        self, min_value_usd: float = SIGNIFICANT_BUY_MIN_USD
    ) -> InsiderBuy | None:
        """DAT-21: znaczący zakup = kod P od officera za ≥ `min_value_usd` USD.
        Liczba akcji nie mówi o skali (10 000 akcji po 5 USD i po 900 USD)."""
        buy = self.largest_buy
        if buy is None or not buy.is_officer or buy.value_usd < min_value_usd:
            return None
        return buy

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
