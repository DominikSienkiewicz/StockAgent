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

WIA-21: `benchmark_comparison` stawia krzywą agenta obok koszyka (buy&hold tych
samych symboli w tych samych oknach) i IVV (buy&hold w oknach predykcji, ze
snapshotów `price_snapshots`), z tymi samymi kosztami. Brak danych IVV to jawne
„brak danych benchmarku”, nie pusta liczba.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from src.application.report_formatting import POLISH_NUMBERS, NumberStyle
from src.application.report_models import ResolvedPrediction
from src.domain.benchmark import BenchmarkComparison, compare_with_benchmarks, window_change
from src.domain.calibration_curve import ProbabilityScores
from src.domain.change_detection import AccuracyDrop, detect_accuracy_drop
from src.domain.equity_curve import EquityCurve, PaperTrade, paper_portfolio_curve
from src.domain.paper_ledger import ExitReason, LedgerTrade, trend_of
from src.domain.quota import QuotaAlert, QuotaSeverity
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
# WIA-34: wejście to cena z chwili predykcji. Fast Loop chodzi przed sesją, więc
# to ostatnie zamknięcie — trafność i paper-PnL liczą lukę na otwarciu.
ENTRY_PRICE_NOTE_PL = (
    "Cena wejścia to cena z chwili predykcji (poza sesją: ostatnie zamknięcie), "
    "więc wynik obejmuje lukę na otwarciu, której przed sesją nie da się kupić."
)
_EPOCH = datetime.min.replace(tzinfo=UTC)

# WIA-21: benchmark rynku — ETF S&P 500 z watchlisty, więc ma snapshot w każdym
# cyklu (`price_snapshots`, odczyt darmowy).
BENCHMARK_SYMBOL = "IVV"
# Strona publiczna nie podaje symboli watchlisty, także tickera benchmarku —
# nazywa indeks, który IVV odwzorowuje (decyzja 2026-10-19).
PUBLIC_BENCHMARK_LABEL = "S&P 500"
NO_BENCHMARK_PL = "brak danych benchmarku"


def benchmark_method(index_label: str = BENCHMARK_SYMBOL) -> str:
    """Copy metody porównania z benchmarkami; `index_label` nazywa indeks."""
    return (
        "Koszyk = buy&hold tych samych symboli w tych samych oknach (long-only), "
        f"{index_label} = buy&hold w oknach predykcji; agent i benchmarki płacą "
        "ten sam koszt od obrotu. Okna z tych samych snapshotów cyklu co predykcje "
        "(snapshot 05:30 UTC to zamknięcie poprzedniej sesji)."
    )


BENCHMARK_METHOD_PL = benchmark_method()


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


def executable_change(prediction: ResolvedPrediction) -> float | None:
    """WIA-54: zmiana od ceny wykonalnej (otwarcie sesji) jako float; brak albo
    poza zakresem floata → None, jak `_realized_change`."""
    change = prediction.executable_change
    if change is None:
        return None
    value = float(change)
    return value if math.isfinite(value) else None


def _realized_change(prediction: ResolvedPrediction) -> float | None:
    """Zmiana ceny jako float; poza zakresem floata (inf) → None (PORT-09):
    krzywe domeny odrzucają NaN/inf, więc taka predykcja wypada sama."""
    change = prediction.actual_change_ratio
    if change is None:
        return None
    value = float(change)
    return value if math.isfinite(value) else None


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


def _benchmark_change(
    prediction: ResolvedPrediction,
    history: Sequence[tuple[datetime, float]],
    settle_after: timedelta | None,
) -> float | None:
    """Zmiana benchmarku w oknie predykcji; None bez historii albo znacznika."""
    if settle_after is None or not history or prediction.predicted_at is None:
        return None
    return window_change(history, prediction.predicted_at, settle_after)


def paper_cycles(
    resolved: Sequence[ResolvedPrediction],
    costs: PaperCosts | None = None,
    *,
    benchmark_history: Sequence[tuple[datetime, float]] = (),
    settle_after: timedelta | None = None,
    change_of: Callable[[ResolvedPrediction], float | None] | None = None,
) -> list[list[PaperTrade]]:
    """Predykcje z ruchem ceny jako cykle trade'ów, chronologicznie.

    Repozytorium oddaje predykcje od najnowszej, a obrót per symbol (WIA-50)
    wymaga kolejności czasu także wewnątrz cyklu. Bez `costs` koszt jest 0.
    Z `benchmark_history` i `settle_after` (WIA-21) każdy trade niesie zmianę
    benchmarku w swoim oknie. `change_of` (WIA-54) — zmiana trade'u; domyślnie od
    ceny z chwili predykcji (`actual_change_ratio`).
    """
    costs = costs or PaperCosts()
    keyed = [
        (
            cycle_key(p),
            PaperTrade(
                p.symbol,
                p.predicted_trend,
                change,
                costs.for_symbol(p.symbol),
                benchmark_change=_benchmark_change(p, benchmark_history, settle_after),
            ),
        )
        for p in sorted(resolved, key=_moment)
        if (change := (change_of or _realized_change)(p)) is not None
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


def ledger_cycles(trades: Sequence[LedgerTrade]) -> list[list[PaperTrade]]:
    """WIA-59: zamknięte pozycje ledgera jako cykle trade'ów (dzień UTC wejścia).

    Na krzywą idą wyłącznie pozycje rozliczone (`ExitReason.SETTLED`): otwarta nie ma
    wyniku, a zamknięta na nieaktualnym notowaniu (DAT-14) wypada, jak predykcja
    z trafności. Zmiana z cen ledgera, koszt z wiersza (zapisany przy wejściu),
    kolejność cyklu — kolejność łańcucha."""
    keyed: list[tuple[str, PaperTrade]] = []
    for trade in trades:
        close = trade.close
        entry = trade.row.entry
        if close is None or close.exit_reason is not ExitReason.SETTLED or entry.entry_price <= 0:
            continue
        change = float((close.exit_price - entry.entry_price) / entry.entry_price)
        stamp = entry.entry_ts.astimezone(UTC) if entry.entry_ts.tzinfo else entry.entry_ts
        keyed.append(
            (
                stamp.date().isoformat(),
                PaperTrade(entry.symbol, trend_of(entry.side), change, float(entry.cost_bps)),
            )
        )
    return group_by_cycle(keyed)


def ledger_curve(trades: Sequence[LedgerTrade]) -> EquityCurve | None:
    """WIA-59: krzywa portfela o równej wadze per cykl z paper ledgera (te same
    reguły co `portfolio_curve`); None bez zamkniętej, rozliczonej pozycji."""
    cycles = ledger_cycles(trades)
    if not cycles:
        return None
    return paper_portfolio_curve(cycles)


def benchmark_comparison(
    resolved: Sequence[ResolvedPrediction],
    costs: PaperCosts,
    benchmark_history: Sequence[tuple[datetime, float]],
    *,
    settle_after: timedelta,
    index_symbol: str = BENCHMARK_SYMBOL,
) -> BenchmarkComparison | None:
    """WIA-21: agent, koszyk i indeks z tych samych okien i kosztów; None bez ruchów cen.

    `benchmark_history` to chronologiczne `(znacznik, cena)` indeksu, a
    `settle_after` — wiek, po którym `reflect_node` rozlicza predykcję. Indeks
    płaci koszt akcji/ETF (`costs.for_symbol`).
    """
    cycles = paper_cycles(
        resolved, costs, benchmark_history=benchmark_history, settle_after=settle_after
    )
    if not cycles:
        return None
    return compare_with_benchmarks(
        cycles, index_symbol=index_symbol, index_cost_bps=costs.for_symbol(index_symbol)
    )


def benchmark_line(
    comparison: BenchmarkComparison,
    *,
    numbers: NumberStyle = POLISH_NUMBERS,
    index_label: str | None = None,
) -> str:
    """„agent +1,2% · koszyk +3,4% · IVV +2,1%”; bez danych IVV — jawny brak.

    `index_label` zastępuje ticker indeksu (strona publiczna: „S&P 500”)."""
    label = index_label or comparison.index_symbol
    index = (
        f"{label}: {NO_BENCHMARK_PL}"
        if comparison.index is None
        else f"{label} {numbers.pct(comparison.index.total_return, decimals=1, signed=True)}"
    )
    return (
        f"agent {numbers.pct(comparison.agent.total_return, decimals=1, signed=True)} · "
        f"koszyk {numbers.pct(comparison.basket.total_return, decimals=1, signed=True)} · "
        f"{index}"
    )


def benchmark_detail(
    comparison: BenchmarkComparison,
    *,
    numbers: NumberStyle = POLISH_NUMBERS,
    index_label: str | None = None,
) -> str:
    """Zwrot nadwyżkowy vs koszyk (p.p.) i hit-rate względem rynku z liczbą zakładów."""
    excess = numbers.pct(comparison.excess_vs_basket, decimals=1, signed=True)
    hit_rate = comparison.market_hit_rate
    market = (
        NO_BENCHMARK_PL
        if hit_rate.rate is None
        else f"{numbers.pct(hit_rate.rate, decimals=1)} (n={hit_rate.directional})"
    )
    return (
        f"Zwrot nadwyżkowy vs koszyk: {excess.removesuffix('%')} p.p. · "
        f"hit-rate względem rynku ({index_label or comparison.index_symbol}): {market}"
    )


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


# NIEZ-20: źródło alertu bannera, gdy detektor zmiany zobaczy spadek trafności.
ACCURACY_DROP_ALERT_SOURCE = "Detektor trafności"
_ACCURACY_DROP_ACTION_PL = (
    "Porównaj przepisy i dostawcę modelu z okresu przed i po dacie zmiany "
    "(`main_tools.py evaluate --by-recipe`); po pauzie crona sprawdź, czy dane wejściowe "
    "wróciły do normy."
)


def accuracy_drop_evidence(resolved: Sequence[ResolvedPrediction]) -> AccuracyDrop | None:
    """NIEZ-20: data istotnego spadku trafności kierunkowej (Page-Hinkley per
    cykl) albo None — brak danych, za krótka historia albo brak zmiany."""
    if not resolved:
        return None
    return detect_accuracy_drop(scored_calls(resolved))


def accuracy_drop_alert(drop: AccuracyDrop, now: datetime) -> QuotaAlert:
    """NIEZ-20: WARNING do bannera maila z tym samym zdaniem co Track Record."""
    return QuotaAlert(
        source=ACCURACY_DROP_ALERT_SOURCE,
        severity=QuotaSeverity.WARNING,
        message=drop.describe_pl(),
        action=_ACCURACY_DROP_ACTION_PL,
        occurred_at=now,
    )
