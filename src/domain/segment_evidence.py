"""Segmentacja dowodu trafności (WIA-55): ablacja komponentów i segmenty kontekstu.

Czysta domena (stdlib + inne moduły domeny). Jedna trafność dla całego okna
miesza komponenty (trend LLM, rekomendacja rady, kierunek XGBoost) i konteksty
(reżim, klasa aktywa, pewność, Alpha Fusion, edge vs opcje), więc nie wiadomo,
co niesie wynik, a co go rozcieńcza. `assess_segments` liczy dla każdego
segmentu ten sam dowód co Track Record (`assess_directional_hit_rate`: n zakładów
kierunkowych, Wilson na n_eff, naiwne baseline'y na tych samych wierszach,
p-wartość względem najlepszego z nich) i koryguje rodzinę segmentów jednej
segmentacji procedurą BH-FDR: przy kilku segmentach „najlepszy” jest
z konstrukcji selekcją szumu.

Segment z n < `MIN_SEGMENT_N` zakładów kierunkowych ma flagę „za mało danych”.
n jest zawsze podawane, bo mała próba kusi do nadinterpretacji.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.domain.equity_curve import signal_of
from src.domain.significance import DEFAULT_FDR_Q, benjamini_hochberg
from src.domain.track_record import (
    DEFAULT_TOLERANCE,
    HitRateEvidence,
    ScoredCall,
    assess_directional_hit_rate,
)

MIN_SEGMENT_N = 10
INSUFFICIENT_DATA_PL = "za mało danych"
UNKNOWN_SEGMENT = "unknown"

COMPONENT_LLM = "llm"
COMPONENT_COUNCIL = "council"
COMPONENT_MODEL = "xgboost"
COMPONENTS: tuple[str, ...] = (COMPONENT_LLM, COMPONENT_COUNCIL, COMPONENT_MODEL)

_COUNCIL_TRENDS = {"BUY": "BULLISH", "SELL": "BEARISH", "HOLD": "SIDEWAYS"}

# Kubełki deklarowanej pewności LLM: (etykieta, górna granica wyłączna).
# Szersze niż 10 kubełków krzywej kalibracji, bo segment ma mieć sensowne n.
# Brak pewności zapisuje się jako 0.5, więc ląduje w najniższym kubełku.
_CONFIDENCE_BUCKETS: tuple[tuple[str, float], ...] = (
    ("0.00-0.60", 0.6),
    ("0.60-0.70", 0.7),
    ("0.70-0.80", 0.8),
    ("0.80-1.00", math.inf),
)


def council_trend(final_recommendation: str | None) -> str | None:
    """Rekomendacja rady jako trend: BUY → BULLISH, SELL → BEARISH, HOLD →
    SIDEWAYS (brak zakładu). Brak werdyktu (rada pominięta progiem albo padła)
    i nieznana etykieta → None, czyli rada nie zabrała głosu."""
    if final_recommendation is None:
        return None
    return _COUNCIL_TRENDS.get(final_recommendation.strip().upper())


def model_trend(
    price_at_prediction: float | None, predicted_target_price: float | None
) -> str | None:
    """Kierunek XGBoost z jego celu cenowego.

    Cel równy cenie to baseline cold-startu „cena bez zmian” (model bez wag),
    więc SIDEWAYS, czyli brak zakładu kierunkowego. Brak ceny albo celu → None.
    """
    if price_at_prediction is None or predicted_target_price is None:
        return None
    if predicted_target_price > price_at_prediction:
        return "BULLISH"
    if predicted_target_price < price_at_prediction:
        return "BEARISH"
    return "SIDEWAYS"


def trend_hit(trend: str, realized_change: float, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """Trafienie trendu wg reguł `Prediction.is_trend_correct`: BULLISH wymaga
    wzrostu, BEARISH spadku, SIDEWAYS ruchu w paśmie ±`tolerance`."""
    signal = signal_of(trend)
    if signal != 0:
        return signal * realized_change > 0
    return abs(realized_change) <= tolerance


def confidence_bucket(score: float | None) -> str:
    """Kubełek deklarowanej pewności; brak, NaN albo wartość spoza [0, 1] →
    `UNKNOWN_SEGMENT`."""
    if score is None or math.isnan(score) or not 0.0 <= score <= 1.0:
        return UNKNOWN_SEGMENT
    for label, upper in _CONFIDENCE_BUCKETS:
        if score < upper:
            return label
    return _CONFIDENCE_BUCKETS[-1][0]


def sign_segment(value: float | None) -> str:
    """Znak wartości jako klucz segmentu (np. Alpha Fusion Score)."""
    if value is None or math.isnan(value):
        return UNKNOWN_SEGMENT
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


@dataclass(frozen=True)
class SegmentEvidence:
    """Dowód trafności jednego segmentu.

    `fdr_significant` — czy segment bije najlepszy naiwny baseline po korekcie
    BH-FDR w rodzinie segmentów tej samej segmentacji (False bez p-wartości).
    """

    key: str
    evidence: HitRateEvidence
    fdr_significant: bool

    @property
    def n(self) -> int:
        """Liczba zakładów kierunkowych (SIDEWAYS poza mianownikiem)."""
        return self.evidence.hit_rate.directional

    @property
    def rate(self) -> float | None:
        return self.evidence.hit_rate.rate

    @property
    def insufficient(self) -> bool:
        return self.n < MIN_SEGMENT_N


def assess_segments(
    keyed_calls: Iterable[tuple[str, ScoredCall]],
    *,
    expected_keys: Sequence[str] = (),
    q: float = DEFAULT_FDR_Q,
) -> tuple[SegmentEvidence, ...]:
    """Dowód trafności per segment z korektą BH-FDR na rodzinie segmentów.

    `keyed_calls` to pary (klucz segmentu, rozliczona predykcja). Segmenty
    z `expected_keys` idą pierwsze w podanej kolejności, także bez predykcji
    (np. komponent, który nie postawił zakładu); pozostałe alfabetycznie.
    BH obejmuje każdy segment z p-wartością, także ten z flagą „za mało
    danych” — był testowany, więc liczy się do rodziny.
    """
    grouped: dict[str, list[ScoredCall]] = {key: [] for key in expected_keys}
    for key, call in keyed_calls:
        grouped.setdefault(key, []).append(call)
    keys = [*expected_keys, *sorted(key for key in grouped if key not in expected_keys)]
    evidences = [assess_directional_hit_rate(grouped[key]) for key in keys]

    tested = [
        (index, evidence.p_value)
        for index, evidence in enumerate(evidences)
        if evidence.p_value is not None
    ]
    rejected = benjamini_hochberg([p for _, p in tested], q)
    significant = {index for (index, _), flag in zip(tested, rejected, strict=True) if flag}
    return tuple(
        SegmentEvidence(key=key, evidence=evidence, fdr_significant=index in significant)
        for index, (key, evidence) in enumerate(zip(keys, evidences, strict=True))
    )
