"""Domena niedzielnej retrospektywy „Tydzień StockAgenta" (roadmap #17).

Deterministyczny wybór bohaterów tygodnia z zamkniętych predykcji:
- **strzał tygodnia** — trafienie kierunkowe o najwyższym `accuracy_score`
  (WIA-07); tydzień bez trafień dostaje pudło najbliższe celu, oznaczone jako
  chybione,
- **wtopa tygodnia** — pudło kierunkowe o najniższym `accuracy_score` (lustro
  klucza strzału); tydzień bez pudeł dostaje trafienie najdalsze od celu; z
  lekcją z `correction_insight`,
- **wybroniony dysydent** — reużycie `vindicated_dissenters` z rady
  („Soros miał rację, gdy wszyscy mówili KUP"),
- **delta ECE** — reużycie krzywej kalibracji (`calibration_curve`),
- **Brier i BSS** (WIA-36) — główne metryki pewności; ECE zostaje pomocniczo,
  bo „zawsze 55%” przy 55% trafień ma ECE = 0 i zerową umiejętność.

REGUŁA DOMENOWA (roadmap #17, „wymaganie, nie opcja"): recap powstaje wyłącznie
przy N >= `MIN_CLOSED_PREDICTIONS_FOR_RECAP` zamkniętych predykcjach. Poniżej
progu funkcja zwraca `None` → mail NIE WYCHODZI. Powód: przy chudym tygodniu
recap robiłby dramat z szumu.

Czysta domena: stdlib + inne moduły domeny, ZERO importów zewnętrznych i ZERO
LLM. Narracyjną syntezę LLM dokłada warstwa application (use case) — poza tym
modułem. Wejście (`ClosedPrediction`) to WŁASNY typ domenowy; mapowanie z DTO
warstwy application (np. `get_resolved_predictions_detailed`) robi warstwa wyżej.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from src.domain.calibration_curve import (
    ProbabilityScores,
    expected_calibration_error,
    probability_scores,
    reliability_curve,
)
from src.domain.council import InvestorOpinion, vindicated_dissenters
from src.domain.prediction import Prediction, sideways_tolerance_for
from src.domain.value_objects import AssetType

# Twardy próg próbki jako jawna stała modułu (reguła domenowa, nie opcja).
# N >= 5 zamkniętych predykcji, inaczej recap nie powstaje i mail nie wychodzi.
MIN_CLOSED_PREDICTIONS_FOR_RECAP = 5


@dataclass(frozen=True)
class ClosedPrediction:
    """Zamknięta (rozwiązana) predykcja — domenowy input recapu.

    Świadomie NIE jest to `ResolvedPrediction` z warstwy application (to DTO
    warstwy wyżej). Trzymamy tu wyłącznie typy domenowe + prymitywy; mapowanie
    z rekordu repozytorium na `ClosedPrediction` należy do use case'u.

    - `prediction` — domenowa predykcja (dostarcza `accuracy_score`,
      `is_trend_correct`, symbol),
    - `actual_price` — cena zrealizowana po horyzoncie predykcji,
    - `confidence` — deklarowana pewność ∈ [0, 1] (zasila ECE),
    - `correction_insight` — tekstowa lekcja z self-reflection (pusta gdy brak),
    - `council_opinions` — głosy rady zapisane dla tej predykcji (do wybronień),
    - `council_recommendation` — finalna rekomendacja rady (do wybronień),
    - `recorded_trend_correct` — werdykt kierunku zapisany przez reflect_node
      (`is_trend_correct`; pasmo SIDEWAYS zależne od klasy aktywa, krypto
      ±2.5%). None = brak zapisu → liczymy z cen pasmem klasy aktywa,
    - `asset_type` — klasa aktywa wyznaczająca pasmo FLAT/SIDEWAYS, ta sama
      reguła co w reflect: CRYPTO → ±2.5%, reszta → ±0.5%. None = nieznana
      (pasmo akcyjne, zachowanie sprzed rozróżnienia).
    """

    prediction: Prediction
    actual_price: Decimal
    confidence: float
    correction_insight: str = ""
    council_opinions: tuple[InvestorOpinion, ...] = ()
    council_recommendation: Literal["BUY", "SELL", "HOLD"] = "HOLD"
    recorded_trend_correct: bool | None = None
    asset_type: AssetType | None = None

    def __post_init__(self) -> None:
        # Koercja do tuple — realna niezmienność frozen value objectu (frozen=True
        # blokuje rebinding atrybutu, nie mutację listy w miejscu).
        object.__setattr__(
            self, "council_opinions", tuple(self.council_opinions)
        )

    @property
    def symbol(self) -> str:
        return self.prediction.symbol

    def accuracy(self) -> Decimal:
        """Trafność cenowa predykcji — reużycie `Prediction.accuracy_score`."""
        return self.prediction.accuracy_score(self.actual_price)

    def is_correct(self) -> bool:
        """Czy kierunek trafiony — jedno źródło prawdy z mailem dziennym.

        Zapisany werdykt reflect_node wygrywa (zna pasmo SIDEWAYS dla krypto);
        bez zapisu reużywamy `Prediction.is_trend_correct` na cenach.
        """
        if self.recorded_trend_correct is not None:
            return self.recorded_trend_correct
        return self.prediction.is_trend_correct(
            self.actual_price, tolerance=self.flat_band()
        )

    def flat_band(self) -> Decimal:
        """Pasmo „bez ruchu” klasy aktywa — ta sama reguła co w reflect."""
        return sideways_tolerance_for(self.asset_type)

    def actual_direction(self) -> Literal["UP", "DOWN", "FLAT"]:
        """Rzeczywisty kierunek ruchu ceny (UP/DOWN/FLAT).

        Pasmo FLAT liczone tą samą tolerancją co predykcja SIDEWAYS
        (`flat_band`: krypto ±2.5%, reszta ±0.5%) — spójność z reflect, więc
        wybronieni dysydenci krypto nie zależą od pasma akcyjnego. Guard na
        cenę bazową 0 (skażony rekord): bez punktu odniesienia nie liczymy
        błędu względnego.
        """
        base = self.prediction.price_at_prediction
        delta = self.actual_price - base
        if base == 0:
            if delta == 0:
                return "FLAT"
            return "UP" if delta > 0 else "DOWN"
        if abs(delta / base) <= self.flat_band():
            return "FLAT"
        return "UP" if delta > 0 else "DOWN"

    def vindicated(self) -> tuple[InvestorOpinion, ...]:
        """Wybronieni dysydenci rady — reużycie `council.vindicated_dissenters`.

        Gdy rada miała rację (finalna rekomendacja zgodna z ruchem rynku) →
        pusta krotka.
        """
        return vindicated_dissenters(
            self.council_opinions,
            self.actual_direction(),
            self.council_recommendation,
        )


@dataclass(frozen=True)
class RecapHighlight:
    """Bohater tygodnia — strzał albo wtopa, z lekcją i wybronieniami.

    `accuracy` to bliskość ceny do celu (`accuracy_score`), NIE trafność
    kierunku — tę niesie `direction_correct` (None = nieznana, np. highlight
    zbudowany ręcznie bez werdyktu).
    """

    symbol: str
    accuracy: Decimal
    correction_insight: str
    vindicated_dissenters: tuple[InvestorOpinion, ...] = field(default=())
    direction_correct: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "vindicated_dissenters", tuple(self.vindicated_dissenters)
        )


@dataclass(frozen=True)
class WeeklyRecap:
    """Deterministyczny szkielet niedzielnej retrospektywy.

    Zawiera wyłącznie wybór faktów (bohaterowie, statystyki kalibracji).
    Narrację LLM dokleja warstwa application — tu jej NIE ma.

    `scores` (WIA-36) to Brier, Brier klimatologii i BSS z tych samych par
    `(confidence, is_correct)` co ECE; None = recap zbudowany bez nich.
    """

    sample_size: int
    best: RecapHighlight
    worst: RecapHighlight
    current_ece: float
    previous_ece: float | None
    ece_delta: float | None
    scores: ProbabilityScores | None = None


def _to_highlight(closed: ClosedPrediction) -> RecapHighlight:
    return RecapHighlight(
        symbol=closed.symbol,
        accuracy=closed.accuracy(),
        correction_insight=closed.correction_insight,
        vindicated_dissenters=closed.vindicated(),
        direction_correct=closed.is_correct(),
    )


def build_weekly_recap(
    closed_predictions: Sequence[ClosedPrediction],
    previous_ece: float | None = None,
    n_buckets: int = 10,
) -> WeeklyRecap | None:
    """Buduje recap tygodnia z zamkniętych predykcji albo zwraca `None`.

    Zwraca `None` (mail NIE WYCHODZI), gdy liczba zamkniętych predykcji jest
    poniżej `MIN_CLOSED_PREDICTIONS_FOR_RECAP` — twarda reguła domenowa.

    Wybór bohaterów jest deterministyczny. Strzał = klucz leksykograficzny
    (WIA-07): najpierw trafność kierunku, potem najwyższy `accuracy_score` —
    pudło kierunkowe bliskie celu nie wygrywa z trafieniem. Wtopa = lustro:
    najpierw pudło kierunkowe, potem najniższy `accuracy_score`. Remis
    rozstrzygany alfabetycznie po symbolu (stabilnie, niezależnie od kolejności
    wejścia).

    `current_ece` liczone z par `(confidence, is_correct)` przez krzywą
    kalibracji. `ece_delta = current_ece − previous_ece` (albo `None`, gdy brak
    poprzedniego punktu odniesienia). Z tych samych par liczone są Brier, Brier
    klimatologii i BSS (`scores`, WIA-36).
    """
    if len(closed_predictions) < MIN_CLOSED_PREDICTIONS_FOR_RECAP:
        return None

    # Deterministyczne rozstrzyganie remisów: przy równym accuracy wygrywa
    # symbol wcześniejszy alfabetycznie — dla strzału i wtopy tak samo.
    best = min(
        closed_predictions,
        key=lambda c: (not c.is_correct(), -c.accuracy(), c.symbol),
    )
    # Wtopa = lustro klucza strzału: najpierw pudła kierunkowe, potem najniższe
    # accuracy — jedyne trafienie nie bywa jednocześnie strzałem i wtopą.
    worst = min(
        closed_predictions,
        key=lambda c: (c.is_correct(), c.accuracy(), c.symbol),
    )

    samples = [(c.confidence, c.is_correct()) for c in closed_predictions]
    current_ece = expected_calibration_error(
        reliability_curve(samples, n_buckets)
    )
    ece_delta = None if previous_ece is None else current_ece - previous_ece

    return WeeklyRecap(
        sample_size=len(closed_predictions),
        best=_to_highlight(best),
        worst=_to_highlight(worst),
        current_ece=current_ece,
        previous_ece=previous_ece,
        ece_delta=ece_delta,
        scores=probability_scores(samples, n_buckets),
    )
