"""Wspólny „dowód trafności” dla publicznego scorecardu i sekcji Track Record.

Mapuje `ResolvedPrediction` na prymitywy domeny (`ScoredCall`, cykle trade'ów)
i formatuje wynik domeny jako polskie copy. Statystyka żyje w `src/domain`
(`track_record`, `significance`, `statistics`, `calibration_curve`) — tu jest
tylko mapowanie i prezentacja, żeby mail i strona publiczna liczyły trafność
jedną definicją i mówiły o niej tymi samymi słowami.

Kluczem cyklu jest dzień UTC znacznika `prediction_logs.timestamp`: Fast Loop
rusza najwyżej raz na dzień sesyjny, a predykcje jednego dnia są skorelowane.
Predykcje bez znacznika czasu lądują w jednym wspólnym cyklu — to zaniża n_eff
(ostrożnie), zamiast udawać, że są niezależne.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC

from src.application.report_models import ResolvedPrediction
from src.domain.calibration_curve import ProbabilityScores
from src.domain.equity_curve import EquityCurve, portfolio_equity_curve
from src.domain.significance import MIN_BOOTSTRAP_CYCLES
from src.domain.statistics import wilson_interval
from src.domain.track_record import (
    HitRateEvidence,
    ScoredCall,
    assess_directional_hit_rate,
    group_by_cycle,
)

# Poniżej tej efektywnej liczności przedział jest tak szeroki, że copy musi to
# powiedzieć wprost — inaczej „100% (n=2)” wygląda jak dowód.
SMALL_SAMPLE_N_EFF = 5

PORTFOLIO_METHOD_SINCE = "2026-10-18"
PORTFOLIO_ASSUMPTIONS_PL = (
    "Założenia: jeden punkt na cykl (dzień UTC predykcji), równa waga każdej "
    "predykcji w cyklu, prognoza „bez ruchu” to gotówka; bez kosztów "
    "transakcyjnych i poślizgu."
)
PORTFOLIO_METHOD_CHANGE_PL = (
    f"Metodologia zmieniona {PORTFOLIO_METHOD_SINCE}: wcześniej każdą transakcję "
    "składano szeregowo na 100% kapitału, więc starsze wyniki nie są porównywalne."
)


def cycle_key(prediction: ResolvedPrediction) -> str:
    """Klucz cyklu: dzień UTC predykcji w ISO; brak znacznika → wspólny klucz ""."""
    if prediction.predicted_at is None:
        return ""
    stamp = prediction.predicted_at
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(UTC)
    return stamp.date().isoformat()


def _realized_change(prediction: ResolvedPrediction) -> float | None:
    change = prediction.actual_change_pct
    return None if change is None else float(change)


def scored_calls(resolved: Sequence[ResolvedPrediction]) -> list[ScoredCall]:
    """Rozliczone predykcje jako prymitywy domeny (bez symboli i cen)."""
    return [
        ScoredCall(
            cycle=cycle_key(p),
            predicted_trend=p.predicted_trend,
            is_correct=p.is_correct,
            realized_change=_realized_change(p),
        )
        for p in resolved
    ]


def hit_rate_evidence(resolved: Sequence[ResolvedPrediction]) -> HitRateEvidence | None:
    """Dowód trafności kierunkowej z rozliczonych predykcji; None bez danych."""
    if not resolved:
        return None
    return assess_directional_hit_rate(scored_calls(resolved))


def portfolio_curve(resolved: Sequence[ResolvedPrediction]) -> EquityCurve | None:
    """Krzywa portfela o równej wadze per cykl (WIA-17); None bez ruchów cen."""
    keyed = [
        (cycle_key(p), (p.predicted_trend, change))
        for p in resolved
        if (change := _realized_change(p)) is not None
    ]
    if not keyed:
        return None
    return portfolio_equity_curve(group_by_cycle(keyed))


def interval_label(low: float, high: float) -> str:
    """„95% CI 41–63%” — przedział w pełnych procentach."""
    return f"95% CI {low * 100:.0f}–{high * 100:.0f}%"


def wilson_label(successes: float, n: int) -> str:
    """Przedział Wilsona z liczbą prób, np. „95% CI 41–63%, n=80”."""
    low, high = wilson_interval(successes, n)
    return f"{interval_label(low, high)}, n={n}"


def sample_label(n: int, n_eff: float) -> str:
    """„n=412, n_eff≈61”; n_eff pomijamy, gdy po zaokrągleniu równa się n."""
    if round(n_eff) >= n:
        return f"n={n}"
    return f"n={n}, n_eff≈{n_eff:.0f}"


def hit_rate_value(evidence: HitRateEvidence | None) -> str:
    """Hit-rate kierunkowy jako „52.0%” albo „—”, gdy nie było zakładów."""
    rate = None if evidence is None else evidence.hit_rate.rate
    return "—" if rate is None else f"{rate * 100:.1f}%"


def hit_rate_detail(evidence: HitRateEvidence) -> str:
    """„95% CI 41–63%, n=412, n_eff≈61” — przedział Wilsona liczony na n_eff."""
    if evidence.interval is None:
        return ""
    low, high = evidence.interval
    sample = sample_label(evidence.hit_rate.directional, evidence.n_eff)
    return f"{interval_label(low, high)}, {sample}"


def baselines_label(evidence: HitRateEvidence) -> str:
    """Naiwne baseline'y na tych samych predykcjach, np. „zawsze-BULLISH: 54% · …”."""
    b = evidence.baselines
    if b.always_bullish is None or b.always_bearish is None or b.always_sideways is None:
        return ""
    return (
        f"zawsze-BULLISH: {b.always_bullish * 100:.0f}% · "
        f"zawsze-BEARISH: {b.always_bearish * 100:.0f}% · "
        f"zawsze-SIDEWAYS: {b.always_sideways * 100:.0f}%"
    )


def uncertainty_note(evidence: HitRateEvidence) -> str:
    """Bootstrap po cyklach albo ostrzeżenie, że cykli jest za mało; plus mała próbka."""
    notes: list[str] = []
    if evidence.bootstrap is not None:
        low, high = evidence.bootstrap
        notes.append(f"bootstrap po cyklach: {low * 100:.0f}–{high * 100:.0f}%")
    elif evidence.hit_rate.directional:
        notes.append(
            f"za mało cykli ({evidence.cycles} < {MIN_BOOTSTRAP_CYCLES}) na bootstrap — "
            "przedział Wilsona na n_eff"
        )
    if evidence.hit_rate.directional and evidence.n_eff < SMALL_SAMPLE_N_EFF:
        notes.append(
            f"mała próbka (n_eff < {SMALL_SAMPLE_N_EFF}) — przedział jest bardzo szeroki"
        )
    return "; ".join(notes)


def _signed_pct(value: float) -> str:
    # `+ 0.0` zamienia ujemne zero z zaokrąglenia na dodatnie — „+0.0%”, nie „-0.0%”.
    return f"{round(value * 100, 1) + 0.0:+.1f}%"


def _score(value: float) -> str:
    """Wynik do 3 miejsc bez zer końcowych: 0.231 → „0.231”, 0.250 → „0.25”.

    Krótszy zapis jest też bezpieczniejszy dla strony publicznej: test anty-PII
    szuka w niej fragmentów kwot (np. „250”), a „0.250” byłoby fałszywym trafem.
    """
    return f"{value:.3f}".rstrip("0").rstrip(".")


def probability_label(scores: ProbabilityScores) -> str:
    """„Brier = 0.231 (klimatologia 0.245, BSS = +5.7%) · log loss = 0.66”."""
    skill = "n/d" if scores.skill is None else _signed_pct(scores.skill)
    return (
        f"Brier = {_score(scores.brier)} (klimatologia {_score(scores.climatology_brier)}, "
        f"BSS = {skill}) · log loss = {_score(scores.log_loss)}"
    )


def murphy_label(scores: ProbabilityScores) -> str:
    """Rozkład Murphy'ego: „REL 0.01 · RES 0.004 · UNC 0.245”."""
    return (
        f"REL {_score(scores.reliability)} · RES {_score(scores.resolution)} · "
        f"UNC {_score(scores.uncertainty)}"
    )
