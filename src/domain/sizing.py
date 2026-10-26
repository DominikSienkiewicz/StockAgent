# src/domain/sizing.py
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

# Progi reguły Kelly-lite. Trzymane w domenie — to decyzja biznesowa o apetycie
# na ryzyko, nie szczegół prezentacji. Mocny konsensus rady, niski rozłam i
# przyzwoity track record agenta razem windują pozycję; słabość któregokolwiek
# czynnika (albo brak historii) ściąga sugestię w dół.
STRONG_CONSENSUS_THRESHOLD = 0.7
LOW_DISSENT_THRESHOLD = 0.34
GOOD_HIT_RATE_THRESHOLD = 0.55
# PORT-01 (G31): minimalna EFEKTYWNA próba (n_eff, `significance.effective_n`),
# od której hit-rate w ogóle coś mówi. Predykcje jednego dnia ruszają się razem,
# więc liczy się liczba niezależnych zakładów, nie wierszy. Poniżej progu
# hit-rate traktujemy jak cold-start — „Pozycja pełna” po jednym trafieniu to
# porada z szumu.
MIN_HIT_SAMPLE = 20

# R-15 faza 2 — istniejąca ekspozycja. Banda może zostać wyłącznie ZWĘŻONA:
# sizing nie ma prawa podbić słabego sygnału tylko dlatego, że portfel jest pusty.
# `CROWDED_*` = już sporo mamy → schodzimy o jeden stopień.
# `SATURATED_*` = mamy tak dużo, że jedyną rozsądną sugestią jest banda startowa.
CROWDED_WEIGHT_THRESHOLD = 0.15
SATURATED_WEIGHT_THRESHOLD = 0.30
# Klaster korelacji: trzy pozycje w jednym klastrze to jedna pozycja przebrana
# za trzy. Progi są luźniejsze niż dla pojedynczego symbolu, bo klaster agreguje.
CROWDED_CLUSTER_THRESHOLD = 0.40
SATURATED_CLUSTER_THRESHOLD = 0.60

# PORT-04 (decyzja 2026-10-19) — ryzyko całego portfela działa jak nadmierna
# ekspozycja i też może bandę wyłącznie zwęzić. Każdy sygnał to osobny stopień.
# 1-dniowy VaR na tym poziomie ufności (dodatnia magnituda straty) ≥ progu
# zdejmuje stopień.
RISK_VAR_CONFIDENCE = 0.95
VAR_TIGHTEN_THRESHOLD = 0.03
# Bieżące obsunięcie portfela od szczytu (dodatnia głębokość) ≥ progu → banda
# startowa: w głębokim dołku nie dokładamy ryzyka.
DEEP_DRAWDOWN_THRESHOLD = 0.15

# Bandy alokacji (% portfela). Świadomie konserwatywne — to sugestia magnitudy,
# nie rekomendacja inwestycyjna.
_STARTER = (1.0, 2.0)
_STANDARD = (2.0, 3.0)
_FULL = (4.0, 5.0)
_SPANS: dict[str, tuple[float, float]] = {
    "starter": _STARTER,
    "standard": _STANDARD,
    "full": _FULL,
}


@dataclass(frozen=True)
class SizeBand:
    """Sugerowana wielkość pozycji (banda procentowa) dla sygnału BUY/SELL.

    Czysty value object: `tier` to kategoria ("starter"/"standard"/"full"),
    `min_pct`/`max_pct` to zakres alokacji portfela w procentach, a `label`
    to gotowa, czytelna etykieta po polsku do maila.
    """

    tier: str
    min_pct: float
    max_pct: float
    label: str


def _clamp(value: float, low: float, high: float) -> float:
    """Przycina wartość do [low, high] — wejście spoza zakresu nie psuje reguły."""
    return max(low, min(high, value))


def suggest_band(
    consensus_strength: float,
    dissent_ratio: float,
    hit_rate: float | None,
    current_weight: float | None = None,
    cluster_exposure: float | None = None,
    hit_sample: float | None = None,
    portfolio_var: float | None = None,
    drawdown: float | None = None,
    hedge_effective: bool | None = None,
) -> SizeBand:
    """Mapuje konsensus rady + track record agenta na sugerowaną bandę pozycji.

    Reguła Kelly-lite, deterministyczna i czysta (stdlib only):

    - consensus/dissent są clampowane do [0,1] — żadne ekstremum nie wysadza
      klasyfikacji. Wartość niefinitna (NaN, ±inf) w consensus/dissent daje
      `starter`, a w hit_rate znaczy „nieznany” (PORT-12). hit_rate spoza [0,1]
      też znaczy „nieznany” (PORT-14, fail-closed): 2.0 to błąd skali albo danych,
      nie track record uprawniający do pełnej bandy.
    - `full` (4-5%) wymaga JEDNOCZEŚNIE: mocnego konsensusu, niskiego dissentu
      ORAZ znanego, przyzwoitego hit-rate. Brak track recordu (cold-start) lub
      słaby hit-rate nigdy nie sięgają po pełną bandę.
    - `standard` (2-3%): mocny konsensus + niski dissent, ale hit-rate nieznany
      lub poniżej progu (rada przekonana, agent jeszcze nieudowodniony).
    - `starter` (1-2%): słaby konsensus lub wysoki dissent — konserwatywnie.

    R-15 faza 2: `current_weight` (udział symbolu w portfelu) i `cluster_exposure`
    (udział jego klastra korelacji) mogą bandę wyłącznie ZWĘZIĆ. Sugerowanie
    „pełnej bandy 4-5%" komuś, kto ma już 40% kapitału w tym symbolu, było
    poradą liczoną na pustym portfelu. Oba argumenty domyślnie `None` →
    zachowanie identyczne jak przed fazą 2.

    PORT-01: `hit_sample` to efektywna liczność (n_eff) próby, z której policzono
    `hit_rate`. Przy `hit_sample < MIN_HIT_SAMPLE` hit-rate traktujemy jak
    cold-start (None), więc pełna banda jest wtedy nieosiągalna. `None` = wołający
    nie podaje próby (zachowanie sprzed PORT-01); raport podaje ją zawsze.

    PORT-04: ryzyko portfela z realnych pozycji, stosowane PO ekspozycji —
    `portfolio_var` (1-dniowy VaR, dodatnia strata) ≥ `VAR_TIGHTEN_THRESHOLD`
    zdejmuje stopień, `drawdown` (bieżąca głębokość od szczytu) ≥
    `DEEP_DRAWDOWN_THRESHOLD` wymusza bandę startową, a `hedge_effective=False`
    zdejmuje stopień. Ryzyko nigdy nie poszerza bandy; wszystkie trzy `None` →
    wynik identyczny jak przed PORT-04.
    """
    if not (math.isfinite(consensus_strength) and math.isfinite(dissent_ratio)):
        # PORT-12: NaN albo nieskończoność w konsensusie lub dissencie to brak odczytu
        # rady. `_clamp` czytał je jako 1.0 (mocny konsensus) i 0.0 (brak rozłamu).
        return _band("starter", _STARTER)
    consensus = _clamp(consensus_strength, 0.0, 1.0)
    dissent = _clamp(dissent_ratio, 0.0, 1.0)
    if hit_sample is not None and hit_sample < MIN_HIT_SAMPLE:
        hit_rate = None
    # TEST-12: NaN i nieskończoność to brak track recordu, nie hit-rate 100%
    # (`_clamp` zamieniał NaN i +inf na 1.0, czyli na pełną bandę).
    if hit_rate is not None and not math.isfinite(hit_rate):
        hit_rate = None
    # PORT-14: hit-rate spoza [0, 1] to cold-start (fail-closed), nie clamp do 1.0.
    if hit_rate is not None and not 0.0 <= hit_rate <= 1.0:
        hit_rate = None

    strong_consensus = consensus >= STRONG_CONSENSUS_THRESHOLD
    low_dissent = dissent <= LOW_DISSENT_THRESHOLD
    good_hit_rate = hit_rate is not None and hit_rate >= GOOD_HIT_RATE_THRESHOLD

    if not (strong_consensus and low_dissent):
        # Rada niepewna albo rozłam — zostajemy przy bandzie startowej.
        return _band("starter", _STARTER)
    # Wszystkie trzy czynniki sprzyjają → pełna banda; rada przekonana, ale
    # brak/słaby track record → banda standardowa.
    tier = "full" if good_hit_rate else "standard"

    # R-15 faza 2: dopiero TERAZ patrzymy na to, co już mamy. Ekspozycja może
    # bandę wyłącznie zwęzić — nigdy poszerzyć.
    tier = _apply_exposure(tier, current_weight, cluster_exposure)
    tier = _apply_risk(tier, portfolio_var, drawdown, hedge_effective)
    return _band(tier, _SPANS[tier])


_DOWNGRADE = {"full": "standard", "standard": "starter", "starter": "starter"}


def _apply_exposure(tier: str, current_weight: float | None, cluster_exposure: float | None) -> str:
    """Zwęża bandę o tyle stopni, ile każe istniejąca ekspozycja.

    Monotoniczność jest kontraktem: funkcja NIGDY nie podnosi bandy. Pusty
    portfel nie jest argumentem za większą pozycją — brak ekspozycji to brak
    informacji, a nie sygnał kupna.
    """
    weight = _clamp(current_weight, 0.0, 1.0) if current_weight is not None else 0.0
    cluster = _clamp(cluster_exposure, 0.0, 1.0) if cluster_exposure is not None else 0.0
    if _saturated(weight, cluster):
        return "starter"
    if weight >= CROWDED_WEIGHT_THRESHOLD or cluster >= CROWDED_CLUSTER_THRESHOLD:
        return _DOWNGRADE[tier]
    return tier


def _saturated(weight: float, cluster: float) -> bool:
    """Ekspozycja tak duża, że dokładanie nie ma sensu — symbol albo jego klaster."""
    return weight >= SATURATED_WEIGHT_THRESHOLD or cluster >= SATURATED_CLUSTER_THRESHOLD


def _apply_risk(
    tier: str,
    portfolio_var: float | None,
    drawdown: float | None,
    hedge_effective: bool | None,
) -> str:
    """PORT-04: zwęża bandę o stopień za każdy sygnał ryzyka portfela.

    Jak `_apply_exposure` — wyłącznie w dół. Brak danych (`None`) to brak
    informacji, nie zielone światło: nie zmienia bandy.

    PORT-24: wartość niefinitna (NaN, ±inf) to zepsuty odczyt ryzyka, nie brak
    danych — `NaN >= próg` daje False, więc bez tej reguły przechodziła jak niskie
    ryzyko. Obsunięcie liczymy wtedy jako głębokie, a VaR jako ponad progiem.
    """
    if drawdown is not None and (
        not math.isfinite(drawdown) or drawdown >= DEEP_DRAWDOWN_THRESHOLD
    ):
        return "starter"
    if portfolio_var is not None and (
        not math.isfinite(portfolio_var) or portfolio_var >= VAR_TIGHTEN_THRESHOLD
    ):
        tier = _DOWNGRADE[tier]
    if hedge_effective is False:
        tier = _DOWNGRADE[tier]
    return tier


_TIER_LABEL_PREFIX = {
    "starter": "Pozycja startowa",
    "standard": "Pozycja standardowa",
    "full": "Pozycja pełna",
}


def _band(tier: str, span: tuple[float, float]) -> SizeBand:
    """Składa SizeBand z gotową polską etykietą `Prefiks min-max% portfela`."""
    min_pct, max_pct = span
    label = f"{_TIER_LABEL_PREFIX[tier]} {_fmt_pct(min_pct)}-{_fmt_pct(max_pct)}% portfela"
    return SizeBand(tier=tier, min_pct=min_pct, max_pct=max_pct, label=label)


def _fmt_pct(value: float) -> str:
    """Formatuje procent bez zbędnego `.0` (4.0 → '4', 1.5 → '1.5')."""
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


class PositionAction(Enum):
    """PORT-08: kierunek sygnału modelu w kontekście pozycji czytelnika.

    Portfel jest long-only (`Position.quantity > 0`), więc spadek nie jest
    sugestią shorta: na trzymanym symbolu znaczy „zredukuj”, poza portfelem
    „unikaj”. Etykiety są ostrożne — to sygnał modelu, nie rekomendacja (MAIL-08).
    """

    BUY = "BUY"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    AVOID = "AVOID"
    WATCH = "WATCH"

    @property
    def label(self) -> str:
        return _POSITION_ACTION_LABELS[self]


_POSITION_ACTION_LABELS = {
    PositionAction.BUY: "KUP",
    PositionAction.HOLD: "TRZYMAJ, nie dokładaj",
    PositionAction.REDUCE: "ZREDUKUJ",
    PositionAction.AVOID: "UNIKAJ",
    PositionAction.WATCH: "OBSERWUJ",
}


def position_action(
    trend: str | None,
    current_weight: float | None,
    cluster_exposure: float | None = None,
) -> PositionAction:
    """PORT-08: trend modelu + udział symbolu w portfelu → kierunek dla czytelnika.

    - BEARISH na trzymanym symbolu (waga > 0) → REDUCE, na nietrzymanym → AVOID.
    - BULLISH przy nasyconej ekspozycji (symbol ≥ `SATURATED_WEIGHT_THRESHOLD` albo
      klaster ≥ `SATURATED_CLUSTER_THRESHOLD`, te same progi co sizing) → HOLD,
      inaczej BUY.
    - Brak kierunku (SIDEWAYS, brak albo nieznany trend) → WATCH.

    `current_weight` None albo 0 = symbolu nie ma w portfelu. Wołający podaje wagi
    tylko wtedy, gdy zna portfel czytelnika.
    """
    weight = _clamp(current_weight, 0.0, 1.0) if current_weight is not None else 0.0
    if trend == "BEARISH":
        return PositionAction.REDUCE if weight > 0.0 else PositionAction.AVOID
    if trend == "BULLISH":
        cluster = _clamp(cluster_exposure, 0.0, 1.0) if cluster_exposure is not None else 0.0
        return PositionAction.HOLD if _saturated(weight, cluster) else PositionAction.BUY
    return PositionAction.WATCH
