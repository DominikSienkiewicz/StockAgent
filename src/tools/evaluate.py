"""CLI ewaluacja jakości predykcji na podstawie `prediction_logs`.

Mierzy OFFLINE, czy agent faktycznie bije naiwny baseline — zanim zmiana
(próg, model rady, hiperparametry) trafi na produkcję. Read-only, bez kosztów
API: czyta tylko zamknięte (ocenione) predykcje z Supabase. Atrapy po awarii LLM
(`llm_analysis_failed`) odsiewa repozytorium, tak jak dla maila i Track Record.

Metryki:
    - hit_rate     — blended trafność (udział is_trend_correct=True po WSZYSTKICH
                     wierszach; SIDEWAYS liczy się jako trafienie na spokojnych dniach),
    - directional_hit_rate — trafność liczona TYLKO po BULLISH/BEARISH (bez „cichych"
                     trafień SIDEWAYS) + osobne pokrycie SIDEWAYS; definicja z domeny
                     (`directional_hit_rate`), ta sama co w mailu i scorecardzie,
    - evidence     — dowód trafności kierunkowej (WIA-22): naiwne baseline'y na tych
                     samych predykcjach, n_eff po cyklach (dniach UTC) i werdykt testu
                     dwumianowego względem lepszego baseline'u kierunkowego,
    - model_rmse_return / baseline_rmse_return — RMSE w przestrzeni zwrotów
                     ((cel − p0)/p0 vs (rzeczywista − p0)/p0) modelu i naiwnego
                     „brak zmiany” (WIA-40); niezależne od skali ceny, więc BTC nie
                     przytłacza akcji,
    - verdict      — bije / NIE bije / nieokreślony: cold-start (cel = cena wejścia)
                     daje identyczne RMSE, czyli brak pomiaru, nie porażkę,
    - model_rmse / baseline_rmse — to samo w surowych cenach (zgodność wsteczna,
                     nie podstawa werdyktu),
    - by_asset_class — RMSE w zwrotach per klasa aktywa (STOCK / ETF / CRYPTO),
    - confidence_scores — Brier i log loss deklarowanej pewności LLM (WIA-40).

Segmentacja (WIA-20): `--by-recipe` dokłada trafność kierunkową per wersja
przepisu predykcji (`prediction_logs.recipe_version`, WYMAGA migracji 030).
`--by-universe` (WIA-48) — per wersja uniwersum symboli z kwitu v2
(`decision_receipts`, bez nowej migracji). Wiersze bez wersji (sprzed WIA-20
albo zapisane bez flagi / bez kwitu v2) to segment `unversioned`.

Ablacja i segmentacja (WIA-55): `--by component` liczy na tych samych wierszach
trafność kierunkową trendu LLM, rekomendacji rady (`council_verdict`) i kierunku
XGBoost (cel ≠ cena; cel = cena to cold-start bez wag, czyli brak zakładu).
`--by regime|asset_class|confidence_bucket|alpha_fusion_sign|edge_label` dzieli
trafność LLM po kontekście. Każdy segment ma n, Wilsona na n_eff, naiwny
baseline i korektę BH-FDR w rodzinie segmentów; n < 10 → „za mało danych”.
Kolumny segmentu czytane są tylko na żądanie (każda istnieje od swojej migracji).

Heurystyki maila (WIA-47): `--by risk_signal` dzieli trafność LLM na wiersze
z sygnałem DIVERGENCE / AV_LLM_CONFLICT / LOW_SIGNAL i bez niego (reguły z
`report_signals`, brak wejścia = „brak danych”, DIVERGENCE wymaga `gate_delta`
z migracji 034), a `--by council_agreement` — na radę zgodną i niezgodną z LLM.
Dopiero sygnał, który przewiduje chybienie, uzasadnia wejście do kalibracji.

Test echa (EXP-01): `--echo` sprawdza, czy trend LLM to tylko znak delty, która
otworzyła bramkę (`prediction_logs.gate_delta`, WYMAGA migracji 034): odsetek
echa i odwrócenia, hit-rate baseline'ów „znak ruchu” / „przeciwny znak” na tych
samych wierszach i hit-rate LLM tam, gdzie nie zgadza się z momentum — z n,
n_eff i BH-FDR. Bez migracji drukuje komunikat zamiast wyjątku. Wynik żyje
WYŁĄCZNIE tu (decyzja właściciela) — nie w recapie ani w mailu.

Eksperymenty (EXP-05): `--experiment <id>` porównuje ramiona eksperymentu
z `config.toml` na wierszach z zapisanym ramieniem (kolumna `experiment_arms`
z migracji 037, a bez niej opcjonalny klucz kwitu v2). Dla każdej metryki części
promptu: średnia per ramię z n i n_eff, różnica zabieg − kontrola i 95% CI
z bootstrapu blokowego po dniach UTC; metryka główna oznaczona ★.

Rozróżnianie (WIA-38): `--discrimination` sprawdza, czy pewność coś znaczy — AUC
(Mann–Whitney) pewności LLM względem trafienia kierunku i siły konsensusu rady
względem trafienia BUY/SELL, z n, n_eff, 95% CI z bootstrapu po dniach UTC
i trafnością top 20% / 50% / 100% predykcji wg pewności. AUC ≈ 0.5 znaczy, że
kalibracja tylko przesuwa wszystkich do stopy bazowej, a ranking po pewności
jest szumem.

Skala i pewność (WIA-40): `--scores` dokłada RMSE w zwrotach per klasa aktywa
(klasa wg bieżącego configu) oraz Brier i log loss pewności LLM; czyta symbol
i `confidence_score` (016) tylko na żądanie. `--json` drukuje ten sam raport
(nagłówek, dowód trafności, RMSE, klasy, pewność, segmenty `--by`) jako JSON —
do porównywania przebiegów; komunikaty o brakujących kolumnach idą wtedy na stderr.

Uruchom (ARCH-39: składanie `Settings` i repozytorium żyje w `main_tools.py`,
tu zostają czyste funkcje):
    uv run python main_tools.py evaluate
    uv run python main_tools.py evaluate --days 60
    uv run python main_tools.py evaluate --by-recipe
    uv run python main_tools.py evaluate --by-universe
    uv run python main_tools.py evaluate --by component --by regime
    uv run python main_tools.py evaluate --by risk_signal --by council_agreement
    uv run python main_tools.py evaluate --echo
    uv run python main_tools.py evaluate --experiment EXP-02 --days 90
    uv run python main_tools.py evaluate --discrimination
    uv run python main_tools.py evaluate --scores
    uv run python main_tools.py evaluate --json > run.json

Exit code: 0 zawsze (narzędzie raportujące, nie bramka CI).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.application.report_signals import (
    divergence_direction,
    is_av_llm_conflict,
    is_low_signal,
)
from src.domain.calibration_curve import (
    DiscriminationEvidence,
    ProbabilityScores,
    discrimination_evidence,
    probability_scores,
    risk_coverage_curve,
)
from src.domain.equity_curve import DirectionalHitRate, directional_hit_rate, signal_of
from src.domain.experiment import (
    ArmComparison,
    Experiment,
    ExperimentCall,
    ExperimentMetric,
    compare_experiment,
)
from src.domain.implied_edge import edge_label_for_sigma
from src.domain.momentum_echo import EchoCall, EchoReport, HeadToHead, assess_momentum_echo
from src.domain.recipe import hit_rate_by_version
from src.domain.regime_key import canonical_regime_key
from src.domain.segment_evidence import (
    COMPONENT_COUNCIL,
    COMPONENT_LLM,
    COMPONENT_MODEL,
    COMPONENTS,
    INSUFFICIENT_DATA_PL,
    MIN_SEGMENT_N,
    UNKNOWN_SEGMENT,
    SegmentEvidence,
    assess_segments,
    confidence_bucket,
    council_trend,
    model_trend,
    sign_segment,
    trend_hit,
)
from src.domain.significance import DEFAULT_FDR_Q, MIN_BOOTSTRAP_CYCLES
from src.domain.track_record import (
    HitRateEvidence,
    ScoredCall,
    assess_directional_hit_rate,
    group_by_cycle,
)
from src.domain.value_objects import AssetType

# Prefiks wersji w raporcie CLI — pełny sha256 nic nie mówi człowiekowi.
_VERSION_PREFIX = 12

# WIA-55: segmentacje `--by` i pola repozytorium, których każda potrzebuje.
# Pola czytane są tylko na żądanie — domyślny przebieg działa na każdej bazie.
SEGMENT_FIELDS: dict[str, tuple[str, ...]] = {
    "component": ("council_recommendation",),
    "regime": ("regime",),
    "asset_class": ("symbol",),
    "confidence_bucket": ("confidence_score",),
    "alpha_fusion_sign": ("alpha_fusion_score",),
    "edge_label": ("edge_sigma",),
    # WIA-47: wejścia heurystyk RiskSignal z maila; delta bramki istnieje od 034.
    "risk_signal": ("sentiment_score", "av_llm_agreement", "news_volume_24h", "gate_delta"),
    "council_agreement": ("council_recommendation",),
}
SEGMENTATIONS: tuple[str, ...] = tuple(SEGMENT_FIELDS)

# WIA-47: segmenty heurystyk RiskSignal — każda osobno: z sygnałem, bez, brak danych.
RISK_SIGNAL_TYPES: tuple[str, ...] = ("DIVERGENCE", "AV_LLM_CONFLICT", "LOW_SIGNAL")
_WITH_SIGNAL = "z sygnałem"
_WITHOUT_SIGNAL = "bez"
_NO_SIGNAL_DATA = "brak danych"
RISK_SIGNAL_SEGMENTS: tuple[str, ...] = tuple(
    f"{kind}: {state}" for kind in RISK_SIGNAL_TYPES for state in (_WITH_SIGNAL, _WITHOUT_SIGNAL)
)
COUNCIL_AGREES = "rada zgodna z LLM"
COUNCIL_DISAGREES = "rada niezgodna z LLM"
NO_COUNCIL = "bez rady"

_SEGMENT_LABELS = {
    "component": "komponent",
    "regime": "reżim",
    "asset_class": "klasa aktywa (wg bieżącego configu)",
    "confidence_bucket": "kubełek pewności",
    "alpha_fusion_sign": "znak Alpha Fusion",
    "edge_label": "edge vs opcje",
    "risk_signal": "sygnał ryzyka z maila (trafność LLM z sygnałem i bez)",
    "council_agreement": "rada vs trend LLM",
}
_SEGMENT_KEY_WIDTH = 16

# EXP-01: pole delty bramki i komunikat, gdy migracja 034 jeszcze nie poszła.
GATE_DELTA_FIELD = "gate_delta"
GATE_DELTA_UNAVAILABLE = "kolumna gate_delta niedostępna — uruchom migrację 034"

# EXP-05: ramiona z kolumny (037) albo z kwitu v2 (020), pewność LLM (016)
# i werdykt rady (005) — pola czytane na żądanie `--experiment`.
EXPERIMENT_ARMS_FIELD = "experiment_arms"
RECEIPT_ARMS_FIELD = "receipt_experiment_arms"
EXPERIMENT_FIELDS: tuple[str, ...] = (
    EXPERIMENT_ARMS_FIELD,
    RECEIPT_ARMS_FIELD,
    "confidence_score",
    "council_recommendation",
    "council_consensus",
)
EXPERIMENT_ARMS_UNAVAILABLE = (
    "kolumna experiment_arms niedostępna — uruchom migrację 037; ramiona z kwitu decyzyjnego"
)
# WIA-38: pewność LLM (016) i werdykt rady (005) — pola czytane na żądanie
# `--discrimination`.
DISCRIMINATION_FIELDS: tuple[str, ...] = (
    "confidence_score",
    "council_recommendation",
    "council_consensus",
)
# WIA-40: symbol (klasa aktywa) i pewność LLM (016) — pola czytane na żądanie
# `--scores`, bo domyślny odczyt nie dotyka kolumn opcjonalnych.
SCORES_FIELDS: tuple[str, ...] = ("symbol", "confidence_score")
# Punkty krzywej ryzyko–pokrycie drukowane w CLI.
_DISCRIMINATION_COVERAGES: tuple[float, ...] = (0.2, 0.5, 1.0)
# Kolumny, bez których raport idzie dalej: komunikat i ponowny odczyt bez nich.
OPTIONAL_COLUMNS: Mapping[str, str] = {
    GATE_DELTA_FIELD: GATE_DELTA_UNAVAILABLE,
    EXPERIMENT_ARMS_FIELD: EXPERIMENT_ARMS_UNAVAILABLE,
}

_METRIC_LABELS: Mapping[ExperimentMetric, str] = {
    ExperimentMetric.HIT_RATE: "trafność LLM (kierunkowa)",
    ExperimentMetric.BRIER: "Brier pewności LLM",
    ExperimentMetric.COUNCIL_LLM_AGREEMENT: "zgodność rady z LLM",
    ExperimentMetric.COUNCIL_HIT_RATE: "trafność rady (BUY/SELL)",
    ExperimentMetric.COUNCIL_BRIER: "Brier konsensusu rady",
}


class BaselineVerdict(Enum):
    """WIA-40: czy model bije naiwny baseline „brak zmiany” w przestrzeni zwrotów.

    UNDETERMINED to trzeci stan, a nie „nie”: cold-start zwraca cel równy cenie
    wejścia, więc błąd modelu jest tożsamy z błędem baseline'u i nie ma czego
    porównywać (to samo bez kompletu cen)."""

    BEATS = "bije baseline"
    DOES_NOT_BEAT = "NIE bije baseline"
    UNDETERMINED = "nieokreślony"


@dataclass(frozen=True)
class ClassError:
    """WIA-40: błąd w zwrotach jednej klasy aktywa (STOCK / ETF / CRYPTO)."""

    n: int
    model_rmse: float
    baseline_rmse: float
    verdict: BaselineVerdict


@dataclass(frozen=True)
class EvalReport:
    sample_count: int
    hit_rate: float | None  # blended trafność (BULLISH/BEARISH/SIDEWAYS)
    model_rmse: float | None  # RMSE prognozy vs rzeczywistość
    baseline_rmse: float | None  # RMSE naiwnego "brak zmiany"
    beats_baseline: bool  # czy model bije baseline
    # Finding #24: blended hit_rate zalicza SIDEWAYS jako trafienie zawsze, gdy
    # ruch < ±0.5% — banking trafień na spokojnych dniach myli „trafiłem kierunek"
    # z „nic się nie ruszyło". Rozdzielamy directional-only od pokrycia SIDEWAYS.
    directional_hit_rate: float | None  # trafność tylko BULLISH/BEARISH
    directional_count: int  # liczba kierunkowych z flagą
    sideways_count: int  # liczba SIDEWAYS z flagą
    # WIA-22: przedział na n_eff, baseline'y i werdykt istotności (None bez wierszy).
    evidence: HitRateEvidence | None = None
    # WIA-55: segmenty per segmentacja z `segment_by` (puste bez segmentacji).
    segments: Mapping[str, tuple[SegmentEvidence, ...]] = field(default_factory=dict)
    # WIA-40: RMSE w przestrzeni zwrotów ((cel − p0)/p0 vs (rzeczywista − p0)/p0) i werdykt
    # z niego. `model_rmse` / `baseline_rmse` zostają w cenach dla zgodności wstecznej —
    # dominuje je najdroższe aktywo (BTC), więc werdyktu z nich się nie wyprowadza.
    model_rmse_return: float | None = None
    baseline_rmse_return: float | None = None
    verdict: BaselineVerdict = BaselineVerdict.UNDETERMINED
    # Per klasa aktywa — tylko dla wierszy z kluczem `asset_class` (`with_asset_class`).
    by_asset_class: Mapping[str, ClassError] = field(default_factory=dict)
    # Brier i log loss deklarowanej pewności LLM — tylko dla wierszy z `confidence_score`.
    confidence_scores: ProbabilityScores | None = None


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _cycle_key(value: Any) -> str:
    """Dzień UTC predykcji jako klucz cyklu; brak/śmieć → wspólny klucz ""."""
    try:
        stamp = datetime.fromisoformat(str(value))
    except ValueError:
        return ""
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(UTC)
    return stamp.date().isoformat()


def _realized_change(row: Mapping[str, Any]) -> float | None:
    """Zrealizowany ruch ceny (ułamek) albo None bez kompletu cen."""
    price_at = _to_float(row.get("price_at_prediction"))
    actual = _to_float(row.get("actual_price_after_12h"))
    if not price_at or actual is None:
        return None
    return (actual - price_at) / price_at


def _scored_call(row: dict[str, Any]) -> ScoredCall:
    return ScoredCall(
        cycle=_cycle_key(row.get("timestamp")),
        predicted_trend=str(row.get("predicted_trend")),
        is_correct=bool(row["is_trend_correct"]),
        realized_change=_realized_change(row),
    )


def _rmse(squared_errors: list[float]) -> float | None:
    if not squared_errors:
        return None
    return math.sqrt(sum(squared_errors) / len(squared_errors))


# WIA-40: różnica RMSE poniżej tej wartości (zwroty to ułamki) to remis, nie przewaga.
_RMSE_TIE_TOLERANCE = 1e-12


def _squared_return_errors(row: Mapping[str, Any]) -> tuple[float, float] | None:
    """(kwadrat błędu modelu, kwadrat błędu baseline'u „brak zmiany”) w zwrotach.

    Zwrot przewidziany to (cel − p0)/p0, zrealizowany (rzeczywista − p0)/p0; baseline
    przewiduje zwrot 0. None bez kompletu cen albo przy p0 = 0. Model i baseline
    liczymy zawsze na TYCH SAMYCH wierszach."""
    price_at = _to_float(row.get("price_at_prediction"))
    predicted = _to_float(row.get("predicted_target_price"))
    actual = _to_float(row.get("actual_price_after_12h"))
    if not price_at or predicted is None or actual is None:
        return None
    return ((predicted - actual) / price_at) ** 2, ((actual - price_at) / price_at) ** 2


def _baseline_verdict(model_rmse: float | None, baseline_rmse: float | None) -> BaselineVerdict:
    if (
        model_rmse is None
        or baseline_rmse is None
        or math.isclose(model_rmse, baseline_rmse, rel_tol=1e-9, abs_tol=_RMSE_TIE_TOLERANCE)
    ):
        return BaselineVerdict.UNDETERMINED
    return BaselineVerdict.BEATS if model_rmse < baseline_rmse else BaselineVerdict.DOES_NOT_BEAT


def _class_errors(rows: Sequence[Mapping[str, Any]]) -> dict[str, ClassError]:
    """RMSE w zwrotach per `asset_class` (klucz dopisuje `with_asset_class`)."""
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        asset_class = row.get("asset_class")
        errors = _squared_return_errors(row)
        if isinstance(asset_class, str) and errors is not None:
            grouped.setdefault(asset_class, []).append(errors)
    result: dict[str, ClassError] = {}
    for asset_class in sorted(grouped):
        pairs = grouped[asset_class]
        model_rmse = _rmse([model for model, _ in pairs])
        baseline_rmse = _rmse([baseline for _, baseline in pairs])
        if model_rmse is not None and baseline_rmse is not None:
            result[asset_class] = ClassError(
                n=len(pairs),
                model_rmse=model_rmse,
                baseline_rmse=baseline_rmse,
                verdict=_baseline_verdict(model_rmse, baseline_rmse),
            )
    return result


def _confidence_scores(flagged: Sequence[Mapping[str, Any]]) -> ProbabilityScores | None:
    """Brier i log loss pewności LLM na zakładach kierunkowych (None bez pewności)."""
    samples = [sample for row in flagged if (sample := _llm_sample(row)) is not None]
    return probability_scores(samples)


def summarize_evaluation(
    rows: list[dict[str, Any]], segment_by: str | Sequence[str] | None = None
) -> EvalReport:
    """Liczy metryki ewaluacji z zamkniętych predykcji (czysta funkcja).

    `rows` to rekordy `prediction_logs` z kluczami: `predicted_trend`,
    `is_trend_correct`, `price_at_prediction`, `predicted_target_price`,
    `actual_price_after_12h`. Wiersze bez kompletu cen są pomijane przy RMSE,
    ale wciąż liczą się do hit-rate (jeśli mają `is_trend_correct`).

    `segment_by` (WIA-55) — nazwa albo nazwy z `SEGMENTATIONS`; segment czyta
    pola z `SEGMENT_FIELDS` (`asset_class` to klucz dopisany przez wołającego
    z configu). Nieznana nazwa → ValueError.
    """
    segmentations = _segmentations(segment_by)
    flags = [bool(r["is_trend_correct"]) for r in rows if r.get("is_trend_correct") is not None]
    hit_rate = (sum(flags) / len(flags)) if flags else None

    # Directional-only hit-rate: tylko predykcje BULLISH/BEARISH. SIDEWAYS jest
    # raportowane jako osobne pokrycie (count), bo trafienie SIDEWAYS to często
    # „dzień bez ruchu", a nie realna trafność kierunku. Jedna definicja z domeny.
    flagged = [r for r in rows if r.get("is_trend_correct") is not None]
    directional = directional_hit_rate(
        (str(r.get("predicted_trend")), bool(r["is_trend_correct"])) for r in flagged
    )

    model_errors: list[float] = []
    baseline_errors: list[float] = []
    for r in rows:
        actual = _to_float(r.get("actual_price_after_12h"))
        if actual is None:
            continue
        predicted = _to_float(r.get("predicted_target_price"))
        price_at = _to_float(r.get("price_at_prediction"))
        if predicted is not None:
            model_errors.append((predicted - actual) ** 2)
        if price_at is not None:
            baseline_errors.append((price_at - actual) ** 2)

    model_rmse = _rmse(model_errors)
    baseline_rmse = _rmse(baseline_errors)
    return_errors = [errors for r in rows if (errors := _squared_return_errors(r)) is not None]
    model_rmse_return = _rmse([model for model, _ in return_errors])
    baseline_rmse_return = _rmse([baseline for _, baseline in return_errors])
    verdict = _baseline_verdict(model_rmse_return, baseline_rmse_return)
    return EvalReport(
        sample_count=len(rows),
        hit_rate=hit_rate,
        model_rmse=model_rmse,
        baseline_rmse=baseline_rmse,
        beats_baseline=verdict is BaselineVerdict.BEATS,
        directional_hit_rate=directional.rate,
        directional_count=directional.directional,
        sideways_count=directional.sideways,
        evidence=(
            assess_directional_hit_rate([_scored_call(r) for r in flagged]) if flagged else None
        ),
        segments={name: _assess_segmentation(flagged, name) for name in segmentations},
        model_rmse_return=model_rmse_return,
        baseline_rmse_return=baseline_rmse_return,
        verdict=verdict,
        by_asset_class=_class_errors(rows),
        confidence_scores=_confidence_scores(flagged),
    )


def _segmentations(segment_by: str | Sequence[str] | None) -> tuple[str, ...]:
    names = (segment_by,) if isinstance(segment_by, str) else tuple(segment_by or ())
    unknown = [name for name in names if name not in SEGMENT_FIELDS]
    if unknown:
        raise ValueError(f"nieznana segmentacja: {', '.join(unknown)}")
    return tuple(dict.fromkeys(names))


def _assess_segmentation(
    flagged: list[dict[str, Any]], segmentation: str
) -> tuple[SegmentEvidence, ...]:
    if segmentation == "component":
        return assess_segments(
            (pair for r in flagged for pair in _component_calls(r)), expected_keys=COMPONENTS
        )
    if segmentation == "risk_signal":
        return assess_segments(
            ((key, _scored_call(r)) for r in flagged for key in _risk_signal_keys(r)),
            expected_keys=RISK_SIGNAL_SEGMENTS,
        )
    if segmentation == "council_agreement":
        return assess_segments(
            ((_council_agreement_key(r), _scored_call(r)) for r in flagged),
            expected_keys=(COUNCIL_AGREES, COUNCIL_DISAGREES, NO_COUNCIL),
        )
    return assess_segments((_segment_key(r, segmentation), _scored_call(r)) for r in flagged)


def _risk_signal_keys(row: dict[str, Any]) -> Iterable[str]:
    """WIA-47: segment wiersza dla każdej heurystyki — reguły te same co w mailu
    (`report_signals`). Brak wejścia (np. `gate_delta` sprzed migracji 034) to
    „brak danych”, a nie „bez sygnału”."""
    delta = _to_float(row.get("gate_delta"))
    sentiment = _to_float(row.get("sentiment_score"))
    agreement = _to_float(row.get("av_llm_agreement"))
    news = _to_float(row.get("news_volume_24h"))
    divergence = (
        None if delta is None or sentiment is None else divergence_direction(delta, sentiment) != 0
    )
    yield f"DIVERGENCE: {_signal_state(divergence)}"
    conflict = None if agreement is None else is_av_llm_conflict(agreement)
    yield f"AV_LLM_CONFLICT: {_signal_state(conflict)}"
    low_signal = None if news is None else is_low_signal(int(news))
    yield f"LOW_SIGNAL: {_signal_state(low_signal)}"


def _signal_state(present: bool | None) -> str:
    if present is None:
        return _NO_SIGNAL_DATA
    return _WITH_SIGNAL if present else _WITHOUT_SIGNAL


def _council_agreement_key(row: dict[str, Any]) -> str:
    """WIA-47: czy rekomendacja rady (BUY/SELL/HOLD) mówi to samo co trend LLM."""
    trend = council_trend(_text(row.get("council_recommendation")))
    if trend is None:
        return NO_COUNCIL
    predicted = str(row.get("predicted_trend") or "").strip().upper()
    return COUNCIL_AGREES if trend == predicted else COUNCIL_DISAGREES


def _component_calls(row: dict[str, Any]) -> Iterable[tuple[str, ScoredCall]]:
    """Głosy komponentów jednego wiersza na tych samych cenach.

    LLM bierze zapisane `is_trend_correct` (jak cały raport); rada i XGBoost
    regułę `trend_hit` na zrealizowanym ruchu — dla BULLISH/BEARISH identyczną
    z `Prediction.is_trend_correct`. Wiersz bez ruchu ceny nie wchodzi wcale,
    a komponent bez głosu (brak werdyktu rady albo celu) jest pomijany."""
    call = _scored_call(row)
    change = call.realized_change
    if change is None:
        return
    yield COMPONENT_LLM, call
    votes = (
        (COMPONENT_COUNCIL, council_trend(_text(row.get("council_recommendation")))),
        (
            COMPONENT_MODEL,
            model_trend(
                _to_float(row.get("price_at_prediction")),
                _to_float(row.get("predicted_target_price")),
            ),
        ),
    )
    for component, trend in votes:
        if trend is not None:
            yield (
                component,
                ScoredCall(
                    cycle=call.cycle,
                    predicted_trend=trend,
                    is_correct=trend_hit(trend, change),
                    realized_change=change,
                ),
            )


def _segment_key(row: dict[str, Any], segmentation: str) -> str:
    if segmentation == "regime":
        return canonical_regime_key(_text(row.get("regime")))
    if segmentation == "asset_class":
        return _text(row.get("asset_class")) or UNKNOWN_SEGMENT
    if segmentation == "confidence_bucket":
        return confidence_bucket(_to_float(row.get("confidence_score")))
    if segmentation == "alpha_fusion_sign":
        return sign_segment(_to_float(row.get("alpha_fusion_score")))
    return edge_label_for_sigma(_to_float(row.get("edge_sigma"))).value


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


def with_asset_class(
    rows: list[dict[str, Any]], crypto_symbols: Iterable[str], etf_symbols: Iterable[str]
) -> list[dict[str, Any]]:
    """Dopisuje `asset_class` z BIEŻĄCEGO configu — `prediction_logs` nie trzyma
    klasy aktywa, a symbol przeniesiony między listami zmienia segment wstecz.
    Reguła jak `run_cycle.classify_asset`: CRYPTO, potem ETF, reszta STOCK."""
    crypto = {symbol.upper() for symbol in crypto_symbols}
    etf = {symbol.upper() for symbol in etf_symbols}

    def _classify(symbol: Any) -> str:
        if symbol is None:
            return UNKNOWN_SEGMENT
        ticker = str(symbol).upper()
        if ticker in crypto:
            return AssetType.CRYPTO.value
        if ticker in etf:
            return AssetType.ETF.value
        return AssetType.STOCK.value

    return [{**row, "asset_class": _classify(row.get("symbol"))} for row in rows]


def format_report(report: EvalReport, days: int) -> str:
    """Renderuje EvalReport do czytelnego tekstu (CLI)."""

    def _pct(v: float | None) -> str:
        return f"{v * 100:.1f}%" if v is not None else "—"

    def _num(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "—"

    directional = (
        f"{_pct(report.directional_hit_rate)} "
        f"(n={report.directional_count}, SIDEWAYS={report.sideways_count})"
    )
    lines = [
        f"StockAgent — ewaluacja predykcji (ostatnie {days} dni)",
        "=" * 56,
        f"  Próbek (zamkniętych predykcji): {report.sample_count}",
        f"  Hit-rate (blended):             {_pct(report.hit_rate)}",
        f"  Hit-rate (kierunkowy):          {directional}",
        *_evidence_lines(report.evidence),
        f"  RMSE modelu (zwroty):           {_num(report.model_rmse_return)}",
        f"  RMSE baseline (brak zmiany):    {_num(report.baseline_rmse_return)}",
        f"  Werdykt:                        {_verdict_text(report)}",
        *_class_error_lines(report.by_asset_class),
        *_confidence_score_lines(report.confidence_scores),
    ]
    return "\n".join(lines)


def evaluation_to_dict(report: EvalReport, days: int) -> dict[str, Any]:
    """Raport jako słownik JSON-owalny (WIA-40): liczby z nagłówka, dowód trafności,
    RMSE w zwrotach z kodem werdyktu, klasy aktywa, pewność i segmenty `--by`.
    Sekcje wersji, echa, eksperymentu i rozróżniania zostają tylko w tekście."""
    return {
        "days": days,
        "sample_count": report.sample_count,
        "hit_rate": report.hit_rate,
        "directional": {
            "hit_rate": report.directional_hit_rate,
            "n": report.directional_count,
            "sideways": report.sideways_count,
        },
        "evidence": _evidence_dict(report.evidence),
        "rmse_return": {
            "model": report.model_rmse_return,
            "baseline": report.baseline_rmse_return,
            "verdict": report.verdict.name.lower(),
        },
        "by_asset_class": {
            asset_class: {
                "n": error.n,
                "model_rmse": error.model_rmse,
                "baseline_rmse": error.baseline_rmse,
                "verdict": error.verdict.name.lower(),
            }
            for asset_class, error in report.by_asset_class.items()
        },
        "confidence": _scores_dict(report.confidence_scores),
        "segments": {
            name: [_segment_dict(segment) for segment in segments]
            for name, segments in report.segments.items()
        },
    }


def _scores_dict(scores: ProbabilityScores | None) -> dict[str, Any] | None:
    if scores is None:
        return None
    return {
        "n": scores.n,
        "brier": scores.brier,
        "log_loss": scores.log_loss,
        "climatology_brier": scores.climatology_brier,
        "skill": scores.skill,
    }


def _evidence_dict(evidence: HitRateEvidence | None) -> dict[str, Any] | None:
    if evidence is None:
        return None
    best = evidence.baselines.best_directional
    return {
        "directional_n": evidence.hit_rate.directional,
        "cycles": evidence.cycles,
        "n_eff": evidence.n_eff,
        "interval": None if evidence.interval is None else list(evidence.interval),
        "p_value": evidence.p_value,
        "best_baseline": None if best is None else {"name": best[0], "rate": best[1]},
    }


def _segment_dict(segment: SegmentEvidence) -> dict[str, Any]:
    evidence = segment.evidence
    return {
        "key": segment.key,
        "n": segment.n,
        "rate": segment.rate,
        "n_eff": evidence.n_eff,
        "interval": None if evidence.interval is None else list(evidence.interval),
        "p_value": evidence.p_value,
        "fdr_significant": segment.fdr_significant,
        "insufficient": segment.insufficient,
    }


def render_evaluation_json(
    rows: list[dict[str, Any]], *, days: int, segmentations: Sequence[str] = ()
) -> str:
    """JSON ewaluacji (posortowane klucze, stabilny diff między przebiegami)."""
    report = summarize_evaluation(rows, segment_by=segmentations)
    return json.dumps(
        evaluation_to_dict(report, days),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )


def _verdict_text(report: EvalReport) -> str:
    if report.verdict is BaselineVerdict.BEATS:
        return f"✅ {report.verdict.value}"
    if report.verdict is BaselineVerdict.DOES_NOT_BEAT:
        return f"⚠️ {report.verdict.value}"
    reason = (
        "brak kompletu cen"
        if report.model_rmse_return is None
        else "cel równy cenie wejścia, model bez zakładu"
    )
    return f"❔ {report.verdict.value} ({reason})"


def _class_error_lines(by_asset_class: Mapping[str, ClassError]) -> list[str]:
    if not by_asset_class:
        return []
    lines = ["  RMSE w zwrotach per klasa aktywa (klasa wg bieżącego configu):"]
    for asset_class, error in by_asset_class.items():
        lines.append(
            f"    {asset_class:<8} n={error.n:<4} model {error.model_rmse:.4f}  "
            f"baseline {error.baseline_rmse:.4f}  {error.verdict.value}"
        )
    return lines


def _confidence_score_lines(scores: ProbabilityScores | None) -> list[str]:
    if scores is None:
        return []
    skill = "—" if scores.skill is None else f"{scores.skill:+.3f}"
    return [
        f"  Brier pewności LLM:             {scores.brier:.4f} "
        f"(klimatologia {scores.climatology_brier:.4f}, BSS {skill}, n={scores.n})",
        f"  Log loss pewności LLM:          {scores.log_loss:.4f}",
    ]


def summarize_by_version(
    rows: list[dict[str, Any]], version_key: str
) -> dict[str, DirectionalHitRate]:
    """Trafność kierunkowa per wartość `version_key` (np. `recipe_version`).

    Bierze tylko wiersze z oceną kierunku; wiersz bez wersji trafia do segmentu
    `UNVERSIONED`. Definicja trafności ta sama co w całym raporcie."""
    return hit_rate_by_version(
        (
            None if r.get(version_key) is None else str(r[version_key]),
            str(r.get("predicted_trend")),
            bool(r["is_trend_correct"]),
        )
        for r in rows
        if r.get("is_trend_correct") is not None
    )


def format_segments(segments: dict[str, DirectionalHitRate], label: str) -> str:
    """Tabela segmentów do CLI: skrócona wersja, n kierunkowych, trafność."""
    lines = [f"  Trafność kierunkowa per {label}:"]
    for version, hit_rate in segments.items():
        rate = "—" if hit_rate.rate is None else f"{hit_rate.rate * 100:.1f}%"
        lines.append(
            f"    {version[:_VERSION_PREFIX]:<{_VERSION_PREFIX}}  {rate:>6}  "
            f"(n={hit_rate.directional}, SIDEWAYS={hit_rate.sideways})"
        )
    return "\n".join(lines)


def format_segment_evidence(segmentation: str, segments: Sequence[SegmentEvidence]) -> str:
    """Tabela segmentów do CLI: trafność kierunkowa, n i n_eff, Wilson 95% na
    n_eff, najlepszy naiwny baseline, p-wartość, werdykt BH i flaga małej próby."""
    label = _SEGMENT_LABELS.get(segmentation, segmentation)
    lines = [
        f"  Segmenty: {label} (n = zakłady kierunkowe, BH-FDR q={DEFAULT_FDR_Q:.2f}, "
        f"n < {MIN_SEGMENT_N} → {INSUFFICIENT_DATA_PL})"
    ]
    for segment in segments:
        lines.append(f"    {segment.key:<{_SEGMENT_KEY_WIDTH}}  {_segment_details(segment)}")
    return "\n".join(lines)


def _segment_details(segment: SegmentEvidence) -> str:
    evidence = segment.evidence
    parts = [
        f"{'—' if segment.rate is None else f'{segment.rate * 100:.1f}%':>6}",
        f"n={segment.n}",
    ]
    if segment.rate is not None:
        parts.append(f"n_eff≈{evidence.n_eff:.1f}")
    if evidence.interval is not None:
        low, high = evidence.interval
        parts.append(f"95% CI {low * 100:.0f}–{high * 100:.0f}%")
    best = evidence.baselines.best_directional
    if best is not None:
        parts.append(f"baseline zawsze-{best[0]} {best[1] * 100:.0f}%")
    if evidence.p_value is not None:
        parts.append(_p_text(evidence.p_value))
        parts.append("istotne po BH" if segment.fdr_significant else "nieistotne po BH")
    if segment.insufficient:
        parts.append(f"⚠️ {INSUFFICIENT_DATA_PL}")
    return "  ".join(parts)


def _p_text(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def summarize_echo(rows: list[dict[str, Any]]) -> EchoReport:
    """Test echa momentum (EXP-01) z zamkniętych predykcji z `gate_delta`.

    Bierze te same wiersze co reszta raportu (z oceną kierunku); wiersz bez
    delty bramki (sprzed backfillu albo cold-start) nie wchodzi wcale."""
    calls = []
    for row in rows:
        gate_delta = _to_float(row.get(GATE_DELTA_FIELD))
        if row.get("is_trend_correct") is None or gate_delta is None:
            continue
        scored = _scored_call(row)
        calls.append(
            EchoCall(
                cycle=scored.cycle,
                predicted_trend=scored.predicted_trend,
                gate_delta=gate_delta,
                realized_change=scored.realized_change,
            )
        )
    return assess_momentum_echo(calls)


def format_echo(report: EchoReport) -> str:
    """Sekcja CLI testu echa: (a) echo/odwrócenie, (b) baseline'y na tych samych
    wierszach, (c) LLM na rozbieżnych i sparowane testy znaków po BH-FDR."""

    def _share(rate: float | None, n: int) -> str:
        return f"{'—' if rate is None else f'{rate * 100:.1f}%'} (n={n})"

    lines = [
        f"  Echo momentum (delta bramki, EXP-01; BH-FDR q={DEFAULT_FDR_Q:.2f}):",
        f"    Echo (trend = znak delty):      {_share(report.echo_rate, report.with_delta)}",
        f"    Odwrócenie (trend = −znak):     {_share(report.reversal_rate, report.with_delta)}",
        f"    Porównanie na tych samych wierszach: n={report.compared}, n_eff≈{report.n_eff:.1f}",
        f"      LLM:                          {_share(report.llm_rate, report.compared)}",
        f"      Baseline „znak ruchu”:        {_share(report.momentum_rate, report.compared)}",
        f"      Baseline „przeciwny znak”:    {_share(report.contrarian_rate, report.compared)}",
        f"    LLM tam, gdzie odwraca momentum: "
        f"{_share(report.disagreement_rate, report.disagreement)}",
        f"    LLM vs „znak ruchu”:            {_head_to_head(report.vs_momentum)}",
        f"    LLM vs „przeciwny znak”:        {_head_to_head(report.vs_contrarian)}",
    ]
    if report.compared < MIN_SEGMENT_N:
        lines.append(f"    ⚠️ {INSUFFICIENT_DATA_PL} (n < {MIN_SEGMENT_N})")
    return "\n".join(lines)


def _head_to_head(result: HeadToHead) -> str:
    if result.p_value is None:
        return "brak rozbieżnych wierszy (n=0)"
    verdict = "istotne po BH" if result.fdr_significant else "nieistotne po BH"
    return (
        f"{result.wins}/{result.n} wygranych na rozbieżnych, n_eff≈{result.n_eff:.1f}, "
        f"{_p_text(result.p_value)}, {verdict}"
    )


def _evidence_lines(evidence: HitRateEvidence | None) -> list[str]:
    """Baseline'y i werdykt istotności kierunkowej (puste, gdy brak kierunkowych)."""
    if evidence is None or evidence.hit_rate.rate is None:
        return []
    b = evidence.baselines
    lines = [f"  Efektywna liczność (n_eff):     {evidence.n_eff:.1f} z {evidence.cycles} cykli"]
    if b.always_bullish is not None and b.always_bearish is not None:
        lines.append(
            f"  Naiwne baseline'y:              zawsze-BULLISH: {b.always_bullish * 100:.0f}%"
            f" · zawsze-BEARISH: {b.always_bearish * 100:.0f}%"
        )
    if evidence.verdict is not None:
        lines.append(f"  Istotność (kierunkowa):         {evidence.verdict}")
    return lines


def evaluation_reads(
    *,
    by_recipe: bool,
    by_universe: bool,
    segmentations: Sequence[str],
    echo: bool,
    experiment: bool = False,
    discrimination: bool = False,
    scores: bool = False,
) -> tuple[dict[str, Any], set[str]]:
    """Co odczytać z repozytorium: flagi segmentów wersji i dodatkowe kolumny.

    Kolumny segmentów tylko na żądanie: `recipe_version` istnieje dopiero po
    migracji 030, `gate_delta` po 034, `experiment_arms` po 037, a domyślny
    przebieg ma działać na każdej bazie. Zwraca (kwargs
    `with_recipe`/`with_universe`, pola `extra_fields`)."""
    reads: dict[str, Any] = {
        name: True
        for name, wanted in (("with_recipe", by_recipe), ("with_universe", by_universe))
        if wanted
    }
    fields = {name for s in segmentations for name in SEGMENT_FIELDS[s]}
    if echo:
        fields.add(GATE_DELTA_FIELD)
    if experiment:
        fields.update(EXPERIMENT_FIELDS)
    if discrimination:
        fields.update(DISCRIMINATION_FIELDS)
    if scores:
        fields.update(SCORES_FIELDS)
    return reads, fields


def fields_kwarg(fields: set[str]) -> dict[str, Any]:
    """Kwarg `extra_fields` tylko, gdy coś zamówiono — domyślny odczyt bez zmian."""
    return {"extra_fields": tuple(sorted(fields))} if fields else {}


def render_evaluation(
    rows: list[dict[str, Any]],
    *,
    days: int,
    segmentations: Sequence[str],
    by_recipe: bool,
    by_universe: bool,
    echo: bool,
    experiment: Experiment | None = None,
    discrimination: bool = False,
) -> str:
    """Pełny tekst raportu CLI: nagłówek, segmenty wersji, `--by`, test echa,
    porównanie ramion eksperymentu i rozróżnianie pewności (WIA-38)."""
    report = summarize_evaluation(rows, segment_by=segmentations)
    sections = [format_report(report, days)]
    if by_recipe:
        sections.append(format_segments(summarize_by_version(rows, "recipe_version"), "przepis"))
    if by_universe:
        sections.append(
            format_segments(summarize_by_version(rows, "universe_version"), "uniwersum")
        )
    sections.extend(format_segment_evidence(name, report.segments[name]) for name in segmentations)
    if echo:
        sections.append(format_echo(summarize_echo(rows)))
    if experiment is not None:
        sections.append(format_experiment(summarize_experiment(rows, experiment)))
    if discrimination:
        sections.append(format_discrimination(summarize_discrimination(rows)))
    return "\n".join(sections)


@dataclass(frozen=True)
class DiscriminationReport:
    """WIA-38: AUC i krzywa ryzyko–pokrycie pewności LLM i konsensusu rady.

    None = brak zakładów kierunkowych z pewnością (pewność LLM sprzed migracji
    016, rada pominięta progiem albo HOLD)."""

    llm: DiscriminationEvidence | None
    llm_coverage: tuple[tuple[float, float], ...]
    council: DiscriminationEvidence | None
    council_coverage: tuple[tuple[float, float], ...]


def _llm_sample(row: Mapping[str, Any]) -> tuple[float, bool] | None:
    """(pewność LLM, trafienie kierunku) dla predykcji BULLISH/BEARISH albo None."""
    confidence = _to_float(row.get("confidence_score"))
    correct = row.get("is_trend_correct")
    if confidence is None or correct is None or signal_of(str(row.get("predicted_trend"))) == 0:
        return None
    return confidence, bool(correct)


def _council_sample(row: Mapping[str, Any]) -> tuple[float, bool] | None:
    """(siła konsensusu rady, trafienie BUY/SELL) albo None — HOLD nie stawia zakładu."""
    trend = council_trend(_text(row.get("council_recommendation")))
    consensus = _to_float(row.get("council_consensus"))
    change = _realized_change(row)
    if trend is None or signal_of(trend) == 0 or consensus is None or change is None:
        return None
    return consensus, trend_hit(trend, change)


def _discrimination_of(
    keyed: list[tuple[str, tuple[float, bool]]],
) -> tuple[DiscriminationEvidence | None, tuple[tuple[float, float], ...]]:
    by_cycle = group_by_cycle(keyed)
    samples = [sample for _, sample in keyed]
    coverage = tuple(risk_coverage_curve(samples, _DISCRIMINATION_COVERAGES))
    return discrimination_evidence(by_cycle), coverage


def summarize_discrimination(rows: list[dict[str, Any]]) -> DiscriminationReport:
    """Rozróżnianie pewności LLM i rady na tych samych rozliczonych wierszach."""
    llm: list[tuple[str, tuple[float, bool]]] = []
    council: list[tuple[str, tuple[float, bool]]] = []
    for row in rows:
        cycle = _cycle_key(row.get("timestamp"))
        if (sample := _llm_sample(row)) is not None:
            llm.append((cycle, sample))
        if (sample := _council_sample(row)) is not None:
            council.append((cycle, sample))
    llm_evidence, llm_coverage = _discrimination_of(llm)
    council_evidence, council_coverage = _discrimination_of(council)
    return DiscriminationReport(
        llm=llm_evidence,
        llm_coverage=llm_coverage,
        council=council_evidence,
        council_coverage=council_coverage,
    )


def _discrimination_lines(
    label: str,
    evidence: DiscriminationEvidence | None,
    coverage: tuple[tuple[float, float], ...],
) -> list[str]:
    if evidence is None or evidence.auc is None:
        return [f"  {label}: {INSUFFICIENT_DATA_PL} (brak trafień albo pudeł z pewnością)"]
    spread = (
        f" [{evidence.interval[0]:.2f}; {evidence.interval[1]:.2f}]"
        if evidence.interval is not None
        else f" (za mało cykli na bootstrap, k < {MIN_BOOTSTRAP_CYCLES})"
    )
    points = " · ".join(
        f"trafność top {share * 100:.0f}%: {rate * 100:.1f}%" for share, rate in coverage
    )
    return [
        f"  {label}: AUC {evidence.auc:.2f}{spread}, n={evidence.n}, n_eff≈{evidence.n_eff:.0f}",
        f"    {points}",
    ]


def format_discrimination(report: DiscriminationReport) -> str:
    """Sekcja CLI „Czy pewność coś znaczy” — AUC 0.5 to pewność bez informacji."""
    lines = ["  Czy pewność coś znaczy (WIA-38; AUC 0.5 = losowo, 1.0 = idealnie):"]
    lines += _discrimination_lines(
        "LLM (confidence_score vs trafienie kierunku)", report.llm, report.llm_coverage
    )
    lines += _discrimination_lines(
        "rada (consensus_strength vs trafienie BUY/SELL)",
        report.council,
        report.council_coverage,
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class ExperimentReport:
    """Porównanie ramion jednego eksperymentu (EXP-05) na rozliczonych predykcjach.

    `arm_counts` — wiersze z ramieniem tego eksperymentu per ramię definicji
    (z oceną kierunku lub bez); `comparisons` — zabieg vs kontrola per metryka."""

    experiment: Experiment
    arm_counts: Mapping[str, int]
    comparisons: tuple[ArmComparison, ...]


def _arms_value(value: Any) -> Mapping[str, Any]:
    """JSONB z PostgREST: zwykle dict, czasem tekst JSON; reszta → brak ramion."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, Mapping) else {}


def experiment_arm(row: Mapping[str, Any], experiment_id: str) -> str | None:
    """Ramię wiersza: kolumna `experiment_arms` (037) ma pierwszeństwo przed kwitem v2."""
    for field_name in (EXPERIMENT_ARMS_FIELD, RECEIPT_ARMS_FIELD):
        arm = _arms_value(row.get(field_name)).get(experiment_id)
        if isinstance(arm, str):
            return arm
    return None


def _experiment_call(row: dict[str, Any], arm: str) -> ExperimentCall:
    correct = row.get("is_trend_correct")
    return ExperimentCall(
        cycle=_cycle_key(row.get("timestamp")),
        arm=arm,
        predicted_trend=str(row.get("predicted_trend")),
        is_correct=None if correct is None else bool(correct),
        confidence=_to_float(row.get("confidence_score")),
        realized_change=_realized_change(row),
        council_recommendation=_text(row.get("council_recommendation")),
        consensus_strength=_to_float(row.get("council_consensus")),
    )


def summarize_experiment(rows: list[dict[str, Any]], experiment: Experiment) -> ExperimentReport:
    """Ramiona eksperymentu na tych samych rozliczonych predykcjach (czysta funkcja).

    Wiersz bez ramienia tego eksperymentu (cykl bez aktywnego eksperymentu)
    nie wchodzi; ramię spoza definicji jest liczone, ale nie porównywane."""
    calls = [
        _experiment_call(row, arm)
        for row in rows
        if (arm := experiment_arm(row, experiment.id)) is not None
    ]
    counts = {arm: 0 for arm in experiment.arms}
    for call in calls:
        if call.arm in counts:
            counts[call.arm] += 1
    return ExperimentReport(
        experiment=experiment,
        arm_counts=counts,
        comparisons=compare_experiment(experiment, calls),
    )


def format_experiment(report: ExperimentReport) -> str:
    """Sekcja CLI eksperymentu: definicja, liczności ramion i różnica per metryka."""
    experiment = report.experiment
    counts = ", ".join(f"{arm} {count}" for arm, count in report.arm_counts.items())
    lines = [
        f"  Eksperyment {experiment.id} ({experiment.part.value}): {experiment.hypothesis}",
        f"    Okno: {experiment.start.isoformat()} – {experiment.end.isoformat()} · "
        f"włączony: {'tak' if experiment.enabled else 'nie'} · "
        f"metryka główna: {experiment.primary_metric.value}",
        f"    Wiersze z ramieniem: {counts} (CI blokowe 95% po dniach UTC, "
        f"k < {MIN_BOOTSTRAP_CYCLES} cykli → bez CI)",
    ]
    for comparison in report.comparisons:
        primary = "★" if comparison.metric is experiment.primary_metric else " "
        lines.append(f"    {primary} {_comparison_line(comparison)}")
    return "\n".join(lines)


def _comparison_line(comparison: ArmComparison) -> str:
    metric = comparison.metric
    control, treatment = comparison.control, comparison.treatment
    parts = [
        f"{_METRIC_LABELS[metric]}: {treatment.arm} − {control.arm} "
        f"{_metric_delta(metric, comparison.difference)}",
    ]
    if comparison.interval is not None:
        low, high = comparison.interval
        parts.append(f"95% CI blokowe {_metric_delta(metric, low)} … {_metric_delta(metric, high)}")
    else:
        parts.append(f"CI: za mało cykli (k={comparison.cycles})")
    parts.append(
        f"[{_arm_text(metric, control.arm, control.mean, control.n, control.n_eff)} · "
        f"{_arm_text(metric, treatment.arm, treatment.mean, treatment.n, treatment.n_eff)}]"
    )
    if metric.lower_is_better:
        parts.append("niżej = lepiej")
    if min(control.n, treatment.n) < MIN_SEGMENT_N:
        parts.append(f"⚠️ {INSUFFICIENT_DATA_PL}")
    return "  ".join(parts)


def _metric_delta(metric: ExperimentMetric, value: float | None) -> str:
    if value is None:
        return "—"
    if metric.lower_is_better:
        return f"{value:+.3f}"
    return f"{value * 100:+.1f} pp"


def _arm_text(metric: ExperimentMetric, arm: str, mean: float | None, n: int, n_eff: float) -> str:
    if mean is None:
        value = "—"
    elif metric.lower_is_better:
        value = f"{mean:.3f}"
    else:
        value = f"{mean * 100:.1f}%"
    return f"{arm} {value} n={n} n_eff≈{n_eff:.1f}"
