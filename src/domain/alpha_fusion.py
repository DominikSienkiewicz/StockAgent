"""Domena: Alpha Fusion Score — deterministyczna fuzja sygnałów alfa.

Największa zmarnowana inwestycja repo: pięć źródeł alfa (insider, options,
social, analyst, earnings) jest pobieranych i klasyfikowanych progami w
domenie… a potem renderowanych WYŁĄCZNIE jako tabelka. Predykcja LLM i cechy
XGBoost nic o nich nie wiedzą.

Ten moduł składa istniejące klasyfikacje domenowe w JEDEN, audytowalny
composite w [-1, 1] — smart-money score z ROZBICIEM WKŁADÓW obok każdej
predykcji ("fusion +0.42: insiderzy +0.3, opcje +0.2, social -0.08").

Zasady konstrukcji:

* REUŻYCIE, nie duplikacja progów. Kierunek każdego źródła wyprowadzamy z jego
  WŁASNEJ metody klasyfikującej:
    - `InsiderFlowSnapshot.evaluate_signal()`  → InsiderSignal
    - `OptionsFlowSnapshot.sentiment()`        → OptionsSentiment
    - `SocialVelocitySnapshot.trend()`         → SocialTrend (+ avg_sentiment na kierunek)
    - `AnalystConsensus.rating()`              → AnalystRating
    - `EarningsEvent.proximity()`              → EarningsProximity (dampener zaufania)

* Wagi to JAWNE stałe modułu (`ALPHA_WEIGHTS`), nie magic numbers rozsiane po
  kodzie. To świadoma decyzja: spec ostrzega, że ręcznie dobrane wagi mogą się
  okazać szumem — dlatego trzymamy je w jednym miejscu, żeby walk-forward gate
  mógł je ocenić/dostroić bez polowania po kodzie.

* Brakujące (None) źródło = wkład 0 ORAZ RENORMALIZACJA wag pozostałych.
  Bez renormalizacji symbol z jednym dostępnym źródłem dostałby sztucznie niski
  score (surową wagę zamiast pełnego znaku). Obecne-neutralne źródło ZOSTAJE w
  mianowniku — to realna informacja "brak sygnału", która słusznie rozcieńcza.

* Earnings NIE jest kierunkowym głosem (bliskość raportu jest bezkierunkowa),
  lecz DAMPENEREM zaufania: im bliżej raportu, tym mocniej ściągamy magnitudę
  ku zeru — ruch i tak zdomiuje zaskoczenie wynikami. Dampening wtapiamy w każdy
  wkład, więc niezmiennik "wkłady sumują się do score" trzyma się nadal.

Typ liczbowy: FLOAT (nie Decimal). Wszystkie sygnały to bezwymiarowe wartości w
[-1, 1]; nie ma tu pieniędzy ani cen wymagających dokładności Decimal, a źródła
(np. `avg_sentiment`, `earnings_threshold_multiplier`) już operują na float.
Świadomie NIE korzystamy z `AnalystConsensus.upside()` (Decimal), tylko z
`rating()` — brak mieszania float/Decimal, które mypy strict by wytknął.

Zero importów zewnętrznych — tylko stdlib + inne moduły domeny.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.domain.analyst_consensus import AnalystConsensus, AnalystRating
from src.domain.earnings import EarningsEvent, earnings_threshold_multiplier
from src.domain.insider_flow import InsiderFlowSnapshot, InsiderSignal
from src.domain.options_flow import OptionsFlowSnapshot, OptionsSentiment
from src.domain.social_velocity import SocialTrend, SocialVelocitySnapshot

# Wagi kierunkowych źródeł alfa. JAWNE stałe modułu (nie magic numbers) —
# jedyne miejsce, w którym żyją, żeby walk-forward gate mógł je dostroić.
# Pełny zestaw sumuje się do 1.0; renormalizacja liczy udziały właśnie od tego.
# Insider najwyżej (insiderzy znają własną spółkę), analyst i social niżej
# (sell-side jest opóźniony, retail bywa szumem).
# Tolerancja porównań z zerem. `avg_sentiment` bywa średnią z wielu wzmianek,
# więc „neutralny" wychodzi z arytmetyki jako ±1e-17, a nie jako dokładne 0.0 —
# porównanie przez `==` przepuszczałoby taki szum jako kierunek.
_ZERO_TOLERANCE = 1e-12

ALPHA_WEIGHTS: dict[str, float] = {
    "insider": 0.35,
    "options": 0.25,
    "analyst": 0.20,
    "social": 0.20,
}

# Mapy klasyfikacja → sygnał kierunkowy w [-1, 1]. Trzymane jako stałe modułu,
# żeby nie było magic numbers w ciele funkcji.
_INSIDER_SIGNAL: dict[InsiderSignal, float] = {
    InsiderSignal.NET_BUYING: 1.0,
    InsiderSignal.NET_SELLING: -1.0,
    InsiderSignal.NEUTRAL: 0.0,
}

_OPTIONS_SIGNAL: dict[OptionsSentiment, float] = {
    OptionsSentiment.BULLISH: 1.0,
    OptionsSentiment.BEARISH: -1.0,
    OptionsSentiment.NEUTRAL: 0.0,
}

_ANALYST_SIGNAL: dict[AnalystRating, float] = {
    AnalystRating.STRONG_BUY: 1.0,
    AnalystRating.BUY: 0.5,
    AnalystRating.HOLD: 0.0,
    AnalystRating.SELL: -0.5,
    AnalystRating.STRONG_SELL: -1.0,
}


@dataclass(frozen=True)
class AlphaFusionScore:
    """Wynik fuzji sygnałów alfa — composite z rozbiciem wkładów.

    `score` — złożony sygnał w [-1, 1] (> 0 byczo, < 0 niedźwiedzio). Wszystkie
    źródła None → 0.0 (zachowanie identyczne jak dziś).

    `contributions` — wkład KAŻDEGO dostępnego źródła (już po renormalizacji wag
    i po dampeningu earnings). NIEZMIENNIK: `sum(contributions.values()) ==
    score`. To sedno audytowalności — raport pokazuje rozbicie, nie samą sumę.

    `available_sources` — posortowane nazwy kierunkowych źródeł, które były
    obecne (weszły do renormalizacji). Earnings tu NIE występuje — to modyfikator
    zaufania, nie kierunkowy głos.

    `earnings_confidence` — mnożnik zaufania z bliskości earnings w (0, 1]:
    1.0 = brak/daleko (bez wpływu), < 1.0 = raport blisko (magnituda ściągnięta
    ku zeru). Odwrotność `earnings_threshold_multiplier` (który jest >= 1.0)."""

    score: float
    contributions: dict[str, float]
    available_sources: tuple[str, ...]
    earnings_confidence: float


def _earnings_confidence(event: EarningsEvent | None) -> float:
    """Mnożnik zaufania z bliskości earnings — odwrotność progu volatility.

    Brak zdarzenia → 1.0 (bez dampeningu). `earnings_threshold_multiplier` jest
    ZAWSZE >= 1.0 (może tylko zacieśnić bramkę), więc odwrotność jest w (0, 1] —
    dampening może tylko ściągnąć magnitudę ku zeru, nigdy jej rozdąć."""
    if event is None:
        return 1.0
    return 1.0 / earnings_threshold_multiplier(event.proximity())


def _social_signal(snapshot: SocialVelocitySnapshot) -> float:
    """Kierunek social: skok wzmianek daje sygnał tylko z sentymentem.

    SURGING + sentyment dodatni → +1, ujemny → -1. Skok bez kierunku sentymentu
    (None / dokładnie 0) oraz NORMAL/QUIET → 0. Reużywa `trend()` (progi surge/
    quiet żyją w social_velocity, nie tutaj)."""
    if snapshot.trend() is not SocialTrend.SURGING:
        return 0.0
    sentiment = snapshot.avg_sentiment
    if sentiment is None or math.isclose(sentiment, 0.0, abs_tol=_ZERO_TOLERANCE):
        return 0.0
    return 1.0 if sentiment > 0.0 else -1.0


def fuse_alpha_signals(
    *,
    insider: InsiderFlowSnapshot | None = None,
    options: OptionsFlowSnapshot | None = None,
    social: SocialVelocitySnapshot | None = None,
    analyst: AnalystConsensus | None = None,
    earnings: EarningsEvent | None = None,
) -> AlphaFusionScore:
    """Składa dostępne klasyfikacje domenowe w jeden composite w [-1, 1].

    Każde obecne źródło kierunkowe (insider/options/social/analyst) głosuje
    sygnałem w [-1, 1] wyprowadzonym z JEGO własnej metody klasyfikującej.
    Wagi z `ALPHA_WEIGHTS` są RENORMALIZOWANE po zbiorze OBECNYCH źródeł, więc
    brakujące (None) źródła nie zaniżają score. Earnings (jeśli podane) ściąga
    magnitudę ku zeru przez `earnings_confidence`, wtopione w każdy wkład.

    Wszystkie kierunkowe źródła None → `score` 0.0, `contributions` puste."""
    raw_signals: dict[str, float] = {}
    if insider is not None:
        raw_signals["insider"] = _INSIDER_SIGNAL[insider.evaluate_signal()]
    if options is not None:
        raw_signals["options"] = _OPTIONS_SIGNAL[options.sentiment()]
    if analyst is not None:
        raw_signals["analyst"] = _ANALYST_SIGNAL[analyst.rating()]
    if social is not None:
        raw_signals["social"] = _social_signal(social)

    if not raw_signals:
        # Wszystkie źródła kierunkowe nieobecne → neutralny composite (jak dziś).
        return AlphaFusionScore(
            score=0.0,
            contributions={},
            available_sources=(),
            earnings_confidence=_earnings_confidence(earnings),
        )

    # Renormalizacja: udział każdego źródła liczony TYLKO od obecnych wag.
    total_weight = sum(ALPHA_WEIGHTS[name] for name in raw_signals)
    confidence = _earnings_confidence(earnings)

    # Dampening wtopiony w każdy wkład → niezmiennik sumowania trzyma się nadal.
    contributions = {
        name: (ALPHA_WEIGHTS[name] / total_weight) * signal * confidence
        for name, signal in raw_signals.items()
    }
    score = sum(contributions.values())

    return AlphaFusionScore(
        score=score,
        contributions=contributions,
        available_sources=tuple(sorted(raw_signals)),
        earnings_confidence=confidence,
    )
