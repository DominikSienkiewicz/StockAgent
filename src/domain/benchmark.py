"""Domena benchmarku krzywej kapitału (WIA-21) — agent na tle koszyka i indeksu.

Czysta domena (stdlib only). W hossie „zawsze BULLISH” trafia i krzywa rośnie
bez żadnej umiejętności, więc paper-PnL bez benchmarku nie odpowiada na pytanie
„czy warto było słuchać”. Dwa benchmarki, oba buy&hold i oba z tym samym
kosztem co agent (WIA-50):

  - koszyk — long-only te same symbole w tych samych oknach
    (`equity_curve.buy_and_hold_curve`),
  - indeks (IVV) — long-only w oknach predykcji, jeden punkt na cykl agenta.

Konwencja znaczników czasu: okno benchmarku zaczyna się snapshotem benchmarku
z TEGO SAMEGO cyklu co predykcja (dzień UTC, snapshot najbliższy chwili
predykcji) i kończy pierwszym snapshotem co najmniej `settle_after` później —
tak jak `reflect_node` rozlicza predykcję w pierwszym cyklu po
`reflection_min_age_hours`. Snapshot cyklu o 05:30 UTC to zamknięcie
POPRZEDNIEJ sesji, więc obie strony mierzą zamknięcie(D-1) → zamknięcie(D);
data kalendarza giełdy nie wchodzi do gry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import fmean

from src.domain.equity_curve import (
    LONG_TREND,
    DirectionalHitRate,
    EquityCurve,
    PaperTrade,
    buy_and_hold_curve,
    excess_return,
    market_relative_hit_rate,
    paper_portfolio_curve,
)


def _utc(moment: datetime) -> datetime:
    """Znacznik w UTC; naiwny traktujemy jak UTC (tak zapisuje baza)."""
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def window_change(
    history: Sequence[tuple[datetime, float]],
    start: datetime,
    settle_after: timedelta,
) -> float | None:
    """Zmiana ceny benchmarku w oknie predykcji z chwili `start` (ułamek).

    Start = snapshot z tego samego dnia UTC najbliższy `start` (benchmark bywa
    zapisany kilka minut przed albo po symbolu w tym samym cyklu), koniec =
    pierwszy snapshot nie wcześniej niż `start + settle_after`. None, gdy
    brakuje któregoś końca okna albo cena startu wynosi 0.
    """
    moment = _utc(start)
    series = sorted((_utc(stamp), price) for stamp, price in history)
    same_cycle = [
        (abs(stamp - moment), price) for stamp, price in series if stamp.date() == moment.date()
    ]
    if not same_cycle:
        return None
    opening = min(same_cycle, key=lambda point: point[0])[1]
    settle_at = moment + settle_after
    closing = next((price for stamp, price in series if stamp >= settle_at), None)
    if closing is None or opening == 0:
        return None
    return closing / opening - 1.0


def index_curve(
    cycles: Sequence[Sequence[PaperTrade]],
    *,
    symbol: str,
    cost_bps: float,
) -> EquityCurve | None:
    """Buy&hold indeksu w oknach predykcji — jeden punkt na KAŻDY cykl agenta.

    Zmiana cyklu to średnia `benchmark_change` trade'ów cyklu, które ją mają
    (okna jednego cyklu są zwykle identyczne). Cykl bez żadnej zmiany
    benchmarku → None: krzywa z dziurą nie byłaby wyrównana do krzywej agenta.
    Koszt jednostronny `cost_bps` płaci się raz, przy wejściu.
    """
    changes: list[float] = []
    for cycle in cycles:
        if not cycle:
            continue
        known = [t.benchmark_change for t in cycle if t.benchmark_change is not None]
        if not known:
            return None
        changes.append(fmean(known))
    if not changes:
        return None
    return paper_portfolio_curve(
        [[PaperTrade(symbol, LONG_TREND, change, cost_bps)] for change in changes]
    )


@dataclass(frozen=True)
class BenchmarkComparison:
    """Krzywa agenta obok koszyka i indeksu — te same cykle, te same koszty.

    `index` = None, gdy benchmark nie ma snapshotów któregoś cyklu.
    `market_hit_rate` liczy się z trade'ów, dla których znamy zmianę indeksu
    w tym samym oknie (bez nich mianownik jest pusty).
    """

    agent: EquityCurve
    basket: EquityCurve
    index: EquityCurve | None
    index_symbol: str
    market_hit_rate: DirectionalHitRate

    @property
    def excess_vs_basket(self) -> float:
        """Zwrot nadwyżkowy agenta nad koszykiem (różnica wyników łącznych)."""
        return excess_return(self.agent.total_return, self.basket.total_return)

    @property
    def excess_vs_index(self) -> float | None:
        """Zwrot nadwyżkowy agenta nad indeksem; None bez danych indeksu."""
        if self.index is None:
            return None
        return excess_return(self.agent.total_return, self.index.total_return)


def compare_with_benchmarks(
    cycles: Sequence[Sequence[PaperTrade]],
    *,
    index_symbol: str,
    index_cost_bps: float,
) -> BenchmarkComparison:
    """Agent, koszyk i indeks z tych samych trade'ów (cykle chronologicznie).

    Predykcje samego indeksu nie wchodzą do hit-rate względem rynku: ich zwrot
    nadwyżkowy nad sobą jest z definicji zerowy, więc liczyłyby się jak pudła.
    """
    calls = [
        (trade.predicted_trend, trade.change, trade.benchmark_change)
        for cycle in cycles
        for trade in cycle
        if trade.benchmark_change is not None and trade.symbol != index_symbol
    ]
    return BenchmarkComparison(
        agent=paper_portfolio_curve(cycles),
        basket=buy_and_hold_curve(cycles),
        index=index_curve(cycles, symbol=index_symbol, cost_bps=index_cost_bps),
        index_symbol=index_symbol,
        market_hit_rate=market_relative_hit_rate(calls),
    )
