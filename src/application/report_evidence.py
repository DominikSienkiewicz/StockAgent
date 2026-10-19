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

WIA-50: krzywa portfela może być liczona po kosztach transakcyjnych
(`PaperCosts`, koszt jednostronny od obrotu per symbol). Bez podanych kosztów
krzywa jest brutto, a copy założeń mówi to wprost.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from src.application.report_formatting import POLISH_NUMBERS, NumberStyle
from src.application.report_models import ResolvedPrediction
from src.domain.calibration_curve import ProbabilityScores
from src.domain.equity_curve import EquityCurve, PaperTrade, paper_portfolio_curve
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
_PORTFOLIO_RULES_PL = (
    "Założenia: jeden punkt na cykl (dzień UTC predykcji), równa waga każdej "
    "predykcji w cyklu, prognoza „bez ruchu” to gotówka"
)
PORTFOLIO_ASSUMPTIONS_PL = f"{_PORTFOLIO_RULES_PL}; bez kosztów transakcyjnych i poślizgu."
PORTFOLIO_METHOD_CHANGE_PL = (
    f"Metodologia zmieniona {PORTFOLIO_METHOD_SINCE}: wcześniej każdą transakcję "
    "składano szeregowo na 100% kapitału, więc starsze wyniki nie są porównywalne."
)
_EPOCH = datetime.min.replace(tzinfo=UTC)


@dataclass(frozen=True)
class PaperCosts:
    """WIA-50: założenia kosztowe paper-PnL — koszt jednostronny w bps od obrotu.

    `crypto_cost_bps` dotyczy symboli z `crypto_symbols`, `cost_bps` — reszty
    (akcje i ETF-y). Symbole służą tylko do doboru kosztu; nie trafiają do copy.
    """

    cost_bps: float = 0.0
    crypto_cost_bps: float = 0.0
    crypto_symbols: frozenset[str] = frozenset()

    def for_symbol(self, symbol: str) -> float:
        """Koszt jednostronny dla symbolu (krypto ma własny)."""
        return self.crypto_cost_bps if symbol in self.crypto_symbols else self.cost_bps


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


def _moment(prediction: ResolvedPrediction) -> datetime:
    """Znacznik predykcji w UTC do sortowania; brak znacznika → najwcześniej."""
    stamp = prediction.predicted_at
    if stamp is None:
        return _EPOCH
    return stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)


def paper_cycles(
    resolved: Sequence[ResolvedPrediction], costs: PaperCosts | None = None
) -> list[list[PaperTrade]]:
    """Predykcje z ruchem ceny jako cykle trade'ów, chronologicznie.

    Repozytorium oddaje predykcje od najnowszej, a obrót per symbol (WIA-50)
    wymaga kolejności czasu także wewnątrz cyklu. Bez `costs` koszt jest 0.
    """
    costs = costs or PaperCosts()
    keyed = [
        (
            cycle_key(p),
            PaperTrade(p.symbol, p.predicted_trend, change, costs.for_symbol(p.symbol)),
        )
        for p in sorted(resolved, key=_moment)
        if (change := _realized_change(p)) is not None
    ]
    return group_by_cycle(keyed)


def portfolio_curve(
    resolved: Sequence[ResolvedPrediction], costs: PaperCosts | None = None
) -> EquityCurve | None:
    """Krzywa portfela o równej wadze per cykl (WIA-17), po kosztach z `costs`
    (WIA-50, bez nich brutto); None bez ruchów cen."""
    cycles = paper_cycles(resolved, costs)
    if not cycles:
        return None
    return paper_portfolio_curve(cycles)


def bps_label(value: float, *, numbers: NumberStyle = POLISH_NUMBERS) -> str:
    """Koszt w bps: „10”, a ułamkowy w zapisie stylu — „12,5” / „12.5”."""
    if float(value).is_integer():
        return str(int(value))
    return numbers.number(value, decimals=1)


def paper_pnl_label(costs: PaperCosts, *, numbers: NumberStyle = POLISH_NUMBERS) -> str:
    """Etykieta krzywej: „paper-PnL (koszt 10 bps)”."""
    return f"paper-PnL (koszt {bps_label(costs.cost_bps, numbers=numbers)} bps)"


def portfolio_assumptions(
    costs: PaperCosts | None, *, numbers: NumberStyle = POLISH_NUMBERS
) -> str:
    """Copy założeń krzywej portfela; bez `costs` — wariant brutto (bez kosztów)."""
    if costs is None:
        return PORTFOLIO_ASSUMPTIONS_PL
    crypto = ""
    if costs.crypto_symbols and costs.crypto_cost_bps != costs.cost_bps:
        crypto = f" (krypto {bps_label(costs.crypto_cost_bps, numbers=numbers)} bps)"
    return (
        f"{_PORTFOLIO_RULES_PL}; koszt transakcyjny "
        f"{bps_label(costs.cost_bps, numbers=numbers)} bps{crypto} od obrotu, "
        "naliczany tylko przy zmianie pozycji symbolu: wejście i wyjście do gotówki "
        "po 1×, odwrócenie 2×; bez poślizgu."
    )


def interval_label(low: float, high: float, *, numbers: NumberStyle = POLISH_NUMBERS) -> str:
    """„95% CI 41–63%” — przedział w pełnych procentach."""
    return f"95% CI {numbers.pct_range(low, high)}"


def wilson_label(successes: float, n: int, *, numbers: NumberStyle = POLISH_NUMBERS) -> str:
    """Przedział Wilsona z liczbą prób, np. „95% CI 41–63%, n=80”."""
    low, high = wilson_interval(successes, n)
    return f"{interval_label(low, high, numbers=numbers)}, n={n}"


def sample_label(n: int, n_eff: float) -> str:
    """„n=412, n_eff≈61”; n_eff pomijamy, gdy po zaokrągleniu równa się n."""
    if round(n_eff) >= n:
        return f"n={n}"
    return f"n={n}, n_eff≈{n_eff:.0f}"


def hit_rate_value(
    evidence: HitRateEvidence | None, *, numbers: NumberStyle = POLISH_NUMBERS
) -> str:
    """Hit-rate kierunkowy jako „52,0%” albo „—”, gdy nie było zakładów."""
    rate = None if evidence is None else evidence.hit_rate.rate
    return "—" if rate is None else numbers.pct(rate, decimals=1)


def hit_rate_detail(evidence: HitRateEvidence, *, numbers: NumberStyle = POLISH_NUMBERS) -> str:
    """„95% CI 41–63%, n=412, n_eff≈61” — przedział Wilsona liczony na n_eff."""
    if evidence.interval is None:
        return ""
    low, high = evidence.interval
    sample = sample_label(evidence.hit_rate.directional, evidence.n_eff)
    return f"{interval_label(low, high, numbers=numbers)}, {sample}"


def baselines_label(evidence: HitRateEvidence, *, numbers: NumberStyle = POLISH_NUMBERS) -> str:
    """Naiwne baseline'y na tych samych predykcjach, np. „zawsze-BULLISH: 54% · …”."""
    b = evidence.baselines
    if b.always_bullish is None or b.always_bearish is None or b.always_sideways is None:
        return ""
    return (
        f"zawsze-BULLISH: {numbers.pct(b.always_bullish, decimals=0)} · "
        f"zawsze-BEARISH: {numbers.pct(b.always_bearish, decimals=0)} · "
        f"zawsze-SIDEWAYS: {numbers.pct(b.always_sideways, decimals=0)}"
    )


def uncertainty_note(evidence: HitRateEvidence, *, numbers: NumberStyle = POLISH_NUMBERS) -> str:
    """Bootstrap po cyklach albo ostrzeżenie, że cykli jest za mało; plus mała próbka."""
    notes: list[str] = []
    if evidence.bootstrap is not None:
        low, high = evidence.bootstrap
        notes.append(f"bootstrap po cyklach: {numbers.pct_range(low, high)}")
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


def _score(value: float, numbers: NumberStyle) -> str:
    """Wynik do 3 miejsc bez zer końcowych: 0.231 → „0,231”, 0.250 → „0,25”.

    Krótszy zapis jest też bezpieczniejszy dla strony publicznej: test anty-PII
    szuka w niej fragmentów kwot (np. „250”), a „0.250” byłoby fałszywym trafem.
    """
    text = numbers.number(value, decimals=3)
    return text.rstrip("0").rstrip(numbers.decimal_separator)


def probability_label(scores: ProbabilityScores, *, numbers: NumberStyle = POLISH_NUMBERS) -> str:
    """„Brier = 0,231 (klimatologia 0,245, BSS = +5,7%) · log loss = 0,66”."""
    skill = "n/d" if scores.skill is None else numbers.pct(scores.skill, decimals=1, signed=True)
    return (
        f"Brier = {_score(scores.brier, numbers)} "
        f"(klimatologia {_score(scores.climatology_brier, numbers)}, "
        f"BSS = {skill}) · log loss = {_score(scores.log_loss, numbers)}"
    )


def murphy_label(scores: ProbabilityScores, *, numbers: NumberStyle = POLISH_NUMBERS) -> str:
    """Rozkład Murphy'ego: „REL 0,01 · RES 0,004 · UNC 0,245”."""
    return (
        f"REL {_score(scores.reliability, numbers)} · RES {_score(scores.resolution, numbers)} · "
        f"UNC {_score(scores.uncertainty, numbers)}"
    )
