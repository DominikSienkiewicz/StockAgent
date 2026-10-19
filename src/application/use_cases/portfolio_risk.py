"""Portfolio Risk — radar korelacji i koncentracji watchlisty.

Osobny pass (analogiczny do MonitorMacroRisk), uruchamiany po głównej
pętli predykcyjnej. Pobiera DARMOWĄ historię cen per symbol z
`RepositoryPort.get_price_history`, wyrównuje szeregi po wspólnych
dniach UTC (`align_by_utc_day`), liczy zwroty, buduje macierz korelacji parami i wykrywa
klastry symboli poruszających się razem ("43 symbole = 3 zakłady").

ZERO płatnych portów — całość liczona ze snapshotów cen. Pojedynczy
błąd per-symbol nie wywala passu; <2 symbole z użyteczną historią dają
pusty, łagodny raport.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from src.application.ports import RepositoryPort
from src.domain.correlation import (
    align_by_utc_day,
    correlation_matrix,
    detect_clusters,
    returns_from_prices,
)
from src.domain.drawdown import drawdown_from_returns
from src.domain.hedge import assess_hedge
from src.domain.number_format import format_number_pl
from src.domain.portfolio import Portfolio
from src.domain.risk_sample import RiskStat, sample_sufficiency
from src.domain.scenarios import Scenario, ScenarioImpact, stress_test
from src.domain.sizing import RISK_VAR_CONFIDENCE
from src.domain.value_objects import Money
from src.domain.var import conditional_var, historical_var

logger = logging.getLogger(__name__)

# Okno historii cen czytane z price_snapshots (dni). PORT-22: VaR wymaga
# `MIN_VAR_OBSERVATIONS` (60) dziennych zwrotów, a 30 dni kalendarzowych daje
# ~21 sesji. 90 dni to ~63 sesje — minimum, przy którym VaR w ogóle może powstać.
PRICE_HISTORY_DAYS = 90

# Próg korelacji, powyżej którego para symboli trafia do jednego klastra.
# 0.7 łapie silne współzależności (np. mega-cap tech + BTC w risk-on), nie
# zlewając przy tym wszystkiego w jeden blob.
CLUSTER_THRESHOLD = 0.7


@dataclass(frozen=True)
class PortfolioRiskReport:
    """Zamrożony wynik passu korelacji.

    - matrix: korelacje parami (klucz `(min, max)` symboli, górny trójkąt)
    - clusters: grupy symboli o pairwise korelacji ≥ progu (≥2 elementy)
    - verdict: jednolinijkowy werdykt (największy klaster + udział w watchliście)
    - watchlist_size: liczba symboli z użyteczną historią (mianownik udziału)
    - threshold: użyty próg korelacji (dla prezentacji)
    """

    matrix: dict[tuple[str, str], float] = field(default_factory=dict)
    clusters: list[tuple[str, ...]] = field(default_factory=list)
    verdict: str = ""
    watchlist_size: int = 0
    threshold: float = CLUSTER_THRESHOLD
    # R1: VaR/CVaR portfela równoważonego (empiryczny percentyl ogona strat),
    # liczone z tych samych darmowych zwrotów. None = wyłączone / brak danych.
    var: float | None = None
    cvar: float | None = None
    var_confidence: float = 0.95
    # R2: wyniki scenariuszy stress-test (β-to-portfel × szok). Pusta krotka =
    # wyłączone / brak scenariuszy.
    scenarios: tuple[ScenarioImpact, ...] = ()
    # PORT-22: statystyki pominięte z braku historii jako `(stat, n, wymagane)`,
    # np. `("var", 9, 60)`. Raport mówi wtedy „za mało historii (9/60)”.
    insufficient: tuple[tuple[RiskStat, int, int], ...] = ()


@dataclass(frozen=True)
class PositionRisk:
    """PORT-04: ryzyko REALNEGO portfela, wejście sizingu (nie sekcja maila).

    Wagi to wartość rynkowa pozycji z ostatniego snapshotu, renormalizowana po
    trzymanych symbolach z historią — nigdy równe wagi watchlisty.

    - var: 1-dniowy VaR (`RISK_VAR_CONFIDENCE`) jako dodatnia strata; None
      poniżej `MIN_VAR_OBSERVATIONS` zwrotów.
    - drawdown: bieżąca głębokość pod szczytem okna `PRICE_HISTORY_DAYS`.
    - hedge_effective: czy trzymane instrumenty ryzyka poruszają się przeciw
      reszcie portfela; None bez trzymanego hedge'a albo za krótkiej próby.
    """

    var: float | None = None
    drawdown: float | None = None
    hedge_effective: bool | None = None


class PortfolioRiskUseCase:
    """Slow scanner korelacji — uruchamiany po AnalyzeMarketUseCase.

    Opcjonalnie (flagi) dokłada VaR/CVaR (R1) i stress-test (R2) — oba liczone
    z tych SAMYCH darmowych zwrotów co macierz korelacji, zero nowych odczytów.

    PORT-22: domyślnie (`require_min_samples=True`) VaR/CVaR powstaje dopiero
    od `MIN_VAR_OBSERVATIONS` zwrotów, a korelacje, klastry i bety od
    `MIN_PAIRED_RETURNS` sparowanych zwrotów; pominięte statystyki lądują w
    `PortfolioRiskReport.insufficient`. `False` wyłącza bramkę (sama matematyka
    na dowolnie krótkiej próbie) — produkcja z tego nie korzysta.
    """

    def __init__(
        self,
        *,
        repository_port: RepositoryPort,
        var_enabled: bool = False,
        stress_enabled: bool = False,
        var_confidence: float = 0.95,
        scenarios: Sequence[Scenario] = (),
        require_min_samples: bool = True,
    ) -> None:
        self._repository = repository_port
        self._require_min_samples = require_min_samples
        self._var_enabled = var_enabled
        self._stress_enabled = stress_enabled
        self._var_confidence = var_confidence
        self._scenarios = tuple(scenarios)

    def run(self, symbols: list[str]) -> PortfolioRiskReport:
        returns_by_symbol = self._collect_returns(symbols)
        if len(returns_by_symbol) < 2:
            # <2 symbole z historią — nie ma czego korelować ani liczyć VaR.
            return PortfolioRiskReport()

        usable_symbols = sorted(returns_by_symbol.keys())
        watchlist_size = len(usable_symbols)
        # Po wyrównaniu po dniu UTC wszystkie szeregi mają tę samą długość.
        paired = min(len(rets) for rets in returns_by_symbol.values())
        shortfalls: list[tuple[RiskStat, int, int]] = []

        matrix: dict[tuple[str, str], float] = {}
        clusters: list[tuple[str, ...]] = []
        verdict = ""
        correlation_gap = self._shortfall("correlation", paired)
        if correlation_gap is not None:
            shortfalls.append(correlation_gap)
        else:
            matrix = correlation_matrix(returns_by_symbol)
            clusters = detect_clusters(matrix, usable_symbols, CLUSTER_THRESHOLD)
            verdict = self._build_verdict(matrix, clusters, watchlist_size)

        # R1/R2 — VaR/CVaR i stress-test z równoważonych zwrotów portfela.
        # Liczone tylko gdy włączone (flagi); ten sam szereg zwrotów napędza oba.
        var: float | None = None
        cvar: float | None = None
        scenarios_out: tuple[ScenarioImpact, ...] = ()
        stress_wanted = self._stress_enabled and bool(self._scenarios)
        if self._var_enabled or stress_wanted:
            market_returns = self._equal_weight_returns(returns_by_symbol)
            if self._var_enabled and market_returns:
                var_gap = self._shortfall("var", len(market_returns))
                if var_gap is None:
                    var = historical_var(market_returns, self._var_confidence)
                    cvar = conditional_var(market_returns, self._var_confidence)
                else:
                    shortfalls.append(var_gap)
            if stress_wanted and market_returns:
                beta_gap = self._shortfall("beta", paired)
                if beta_gap is None:
                    scenarios_out = tuple(
                        stress_test(returns_by_symbol, market_returns, self._scenarios)
                    )
                else:
                    shortfalls.append(beta_gap)

        return PortfolioRiskReport(
            matrix=matrix,
            clusters=clusters,
            verdict=verdict,
            watchlist_size=watchlist_size,
            threshold=CLUSTER_THRESHOLD,
            var=var,
            cvar=cvar,
            var_confidence=self._var_confidence,
            scenarios=scenarios_out,
            insufficient=tuple(shortfalls),
        )

    def position_risk(
        self, portfolio: Portfolio, *, hedge_symbols: Collection[str] = ()
    ) -> PositionRisk:
        """PORT-04: VaR, obsunięcie i skuteczność hedge'a trzymanych pozycji.

        Czyta wyłącznie historię trzymanych symboli (darmowe `price_snapshots`).
        Symbol bez historii albo z błędem odczytu wypada, a wagi pozostałych się
        renormalizują. `hedge_symbols` to instrumenty ryzyka: trzymane liczą się
        jako hedge reszty portfela (oczekiwany ruch przeciwny)."""
        held = sorted({position.symbol for position in portfolio.positions})
        history = self._collect_history(held)
        returns_by_symbol = self._aligned_returns(history)
        last_prices = {
            symbol: max(history[symbol], key=lambda point: point[0])[1].amount
            for symbol in returns_by_symbol
        }
        # Granica Decimal → float: matematyka ryzyka liczy na floatach.
        weights = {
            symbol: float(weight)
            for symbol, weight in portfolio.weights(last_prices).items()
        }
        portfolio_returns = self._weighted_returns(returns_by_symbol, weights)
        if not portfolio_returns:
            return PositionRisk()
        var_gap = self._shortfall("var", len(portfolio_returns))
        return PositionRisk(
            var=(
                historical_var(portfolio_returns, RISK_VAR_CONFIDENCE)
                if var_gap is None
                else None
            ),
            drawdown=drawdown_from_returns(portfolio_returns),
            hedge_effective=self._hedge_effective(
                returns_by_symbol, weights, frozenset(hedge_symbols)
            ),
        )

    def _hedge_effective(
        self,
        returns_by_symbol: dict[str, list[float]],
        weights: Mapping[str, float],
        hedge_symbols: frozenset[str],
    ) -> bool | None:
        """Noga hedge'y (trzymane instrumenty ryzyka) vs noga książki (reszta),
        obie ważone wartością pozycji. None, gdy brakuje którejś nogi albo
        sparowanych zwrotów jest mniej niż `MIN_PAIRED_RETURNS`."""
        hedges = {s: r for s, r in returns_by_symbol.items() if s in hedge_symbols}
        book = {s: r for s, r in returns_by_symbol.items() if s not in hedge_symbols}
        hedge_returns = self._weighted_returns(hedges, weights)
        book_returns = self._weighted_returns(book, weights)
        paired = min(len(hedge_returns), len(book_returns))
        if paired == 0 or self._shortfall("correlation", paired) is not None:
            return None
        assessment = assess_hedge(
            "+".join(sorted(hedges)), book_returns, hedge_returns, expect_inverse=True
        )
        return assessment.label == "skuteczny"

    @staticmethod
    def _weighted_returns(
        returns_by_symbol: dict[str, list[float]], weights: Mapping[str, float]
    ) -> list[float]:
        """Zwroty koszyka ważone `weights` (renormalizowanymi po symbolach
        koszyka). Pusty koszyk albo zerowa suma wag → []."""
        series = {s: r for s, r in returns_by_symbol.items() if weights.get(s, 0.0) > 0}
        total = sum(weights[s] for s in series)
        if not series or total <= 0:
            return []
        n = min(len(r) for r in series.values())
        return [
            sum(weights[s] * r[t] for s, r in series.items()) / total for t in range(n)
        ]

    def _shortfall(self, stat: RiskStat, n: int) -> tuple[RiskStat, int, int] | None:
        """`(stat, n, wymagane)` gdy próba jest za mała; None gdy wystarcza albo
        bramka minimalnych prób jest wyłączona."""
        if not self._require_min_samples:
            return None
        sufficiency = sample_sufficiency(n, stat)
        if sufficiency.ok:
            return None
        return stat, sufficiency.n, sufficiency.required

    @staticmethod
    def _equal_weight_returns(
        returns_by_symbol: dict[str, list[float]],
    ) -> list[float]:
        """Szereg zwrotów portfela równoważonego: średnia zwrotów po symbolach
        w każdym okresie. Szeregi są już wyrównane (równe długości) po
        `align_by_utc_day`; `min` zabezpiecza na wszelki wypadek."""
        series = list(returns_by_symbol.values())
        if not series:
            return []
        n = min(len(s) for s in series)
        count = len(series)
        return [sum(s[t] for s in series) / count for t in range(n)]

    def _collect_returns(self, symbols: list[str]) -> dict[str, list[float]]:
        """Pobiera historię per symbol, wyrównuje po dniu UTC i liczy zwroty.

        Symbole z za krótką/pustą historią (mniej niż 2 zwroty) są pomijane.
        Pojedynczy błąd odczytu nie wywala całego passu.
        """
        return self._aligned_returns(self._collect_history(symbols))

    def _collect_history(
        self, symbols: list[str]
    ) -> dict[str, list[tuple[datetime, Money]]]:
        """Historia cen per symbol z ≥3 snapshotami; błąd odczytu → pominięty."""
        raw: dict[str, list[tuple[datetime, Money]]] = {}
        for symbol in symbols:
            try:
                history = self._repository.get_price_history(
                    symbol, PRICE_HISTORY_DAYS
                )
            except Exception:
                logger.exception("portfolio_risk history fetch failed for %s", symbol)
                continue
            # Symbol musi mieć ≥3 snapshoty, by dać ≥2 zwroty. Krótsze szeregi
            # odsiewamy TU, przed wyrównaniem — inaczej ich nieliczne timestampy
            # zawęziłyby wspólny przekrój i wyzerowały korelacje pozostałych.
            if len(history) >= 3:
                raw[symbol] = history
        return raw

    @staticmethod
    def _aligned_returns(
        raw: dict[str, list[tuple[datetime, Money]]],
    ) -> dict[str, list[float]]:
        """Wyrównuje historię po wspólnych dniach UTC i liczy zwroty (≥2)."""
        aligned = align_by_utc_day(
            {
                symbol: [(ts, float(price.amount)) for ts, price in series]
                for symbol, series in raw.items()
            }
        )

        returns_by_symbol: dict[str, list[float]] = {}
        for symbol, prices in aligned.items():
            rets = returns_from_prices(prices)
            if len(rets) >= 2:
                returns_by_symbol[symbol] = rets
        return returns_by_symbol

    @staticmethod
    def _build_verdict(
        matrix: dict[tuple[str, str], float],
        clusters: list[tuple[str, ...]],
        watchlist_size: int,
    ) -> str:
        """Jednolinijkowy werdykt o koncentracji ("...to jeden zakład").

        Bierze największy klaster, jego średnią korelację wewnętrzną i udział
        w watchliście. Pusty string, gdy brak klastrów (portfel naprawdę
        zdywersyfikowany).
        """
        if not clusters or watchlist_size == 0:
            return ""
        biggest = max(clusters, key=len)
        members = "/".join(biggest)
        avg_corr = PortfolioRiskUseCase._avg_intra_corr(matrix, biggest)
        share_pct = round(len(biggest) / watchlist_size * 100)
        return (
            f"{members} poruszają się razem {format_number_pl(Decimal(str(avg_corr)), False, 2)} — "
            f"{share_pct}% watchlisty to jeden zakład"
        )

    @staticmethod
    def _avg_intra_corr(
        matrix: dict[tuple[str, str], float], cluster: tuple[str, ...]
    ) -> float:
        """Średnia korelacja parami wewnątrz klastra (do werdyktu)."""
        pairs: list[float] = []
        for i, left in enumerate(cluster):
            for right in cluster[i + 1 :]:
                key = (left, right) if left < right else (right, left)
                if key in matrix:
                    pairs.append(matrix[key])
        if not pairs:
            return 0.0
        return sum(pairs) / len(pairs)
