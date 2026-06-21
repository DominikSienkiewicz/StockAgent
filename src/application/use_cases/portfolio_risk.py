"""Portfolio Risk — radar korelacji i koncentracji watchlisty.

Osobny pass (analogiczny do MonitorMacroRisk), uruchamiany po głównej
pętli predykcyjnej. Pobiera DARMOWĄ historię cen per symbol z
`RepositoryPort.get_price_history`, wyrównuje szeregi po wspólnych
timestampach, liczy zwroty, buduje macierz korelacji parami i wykrywa
klastry symboli poruszających się razem ("43 symbole = 3 zakłady").

ZERO płatnych portów — całość liczona ze snapshotów cen. Pojedynczy
błąd per-symbol nie wywala passu; <2 symbole z użyteczną historią dają
pusty, łagodny raport.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from src.application.ports import RepositoryPort
from src.domain.correlation import (
    correlation_matrix,
    detect_clusters,
    returns_from_prices,
)
from src.domain.scenarios import Scenario, ScenarioImpact, stress_test
from src.domain.value_objects import Money
from src.domain.var import conditional_var, historical_var

logger = logging.getLogger(__name__)

# Okno historii cen czytane z price_snapshots (dni). Dłuższe = stabilniejsza
# korelacja, ale wolniejszy odczyt. 30 dni daje sensowną próbkę dziennych
# snapshotów bez sięgania po stare, nieaktualne reżimy rynkowe.
PRICE_HISTORY_DAYS = 30

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


class PortfolioRiskUseCase:
    """Slow scanner korelacji — uruchamiany po AnalyzeMarketUseCase.

    Opcjonalnie (flagi) dokłada VaR/CVaR (R1) i stress-test (R2) — oba liczone
    z tych SAMYCH darmowych zwrotów co macierz korelacji, zero nowych odczytów.
    """

    def __init__(
        self,
        *,
        repository_port: RepositoryPort,
        var_enabled: bool = False,
        stress_enabled: bool = False,
        var_confidence: float = 0.95,
        scenarios: Sequence[Scenario] = (),
    ) -> None:
        self._repository = repository_port
        self._var_enabled = var_enabled
        self._stress_enabled = stress_enabled
        self._var_confidence = var_confidence
        self._scenarios = tuple(scenarios)

    def run(self, symbols: list[str]) -> PortfolioRiskReport:
        returns_by_symbol = self._collect_returns(symbols)
        if len(returns_by_symbol) < 2:
            # <2 symbole z historią — nie ma czego korelować ani liczyć VaR.
            return PortfolioRiskReport()

        matrix = correlation_matrix(returns_by_symbol)
        usable_symbols = sorted(returns_by_symbol.keys())
        clusters = detect_clusters(matrix, usable_symbols, CLUSTER_THRESHOLD)
        watchlist_size = len(usable_symbols)
        verdict = self._build_verdict(matrix, clusters, watchlist_size)

        # R1/R2 — VaR/CVaR i stress-test z równoważonych zwrotów portfela.
        # Liczone tylko gdy włączone (flagi); ten sam szereg zwrotów napędza oba.
        var: float | None = None
        cvar: float | None = None
        scenarios_out: tuple[ScenarioImpact, ...] = ()
        if self._var_enabled or (self._stress_enabled and self._scenarios):
            market_returns = self._equal_weight_returns(returns_by_symbol)
            if self._var_enabled and market_returns:
                var = historical_var(market_returns, self._var_confidence)
                cvar = conditional_var(market_returns, self._var_confidence)
            if self._stress_enabled and self._scenarios and market_returns:
                scenarios_out = tuple(
                    stress_test(returns_by_symbol, market_returns, self._scenarios)
                )

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
        )

    @staticmethod
    def _equal_weight_returns(
        returns_by_symbol: dict[str, list[float]],
    ) -> list[float]:
        """Szereg zwrotów portfela równoważonego: średnia zwrotów po symbolach
        w każdym okresie. Szeregi są już wyrównane (równe długości) po
        `_align_by_timestamp`; `min` zabezpiecza na wszelki wypadek."""
        series = list(returns_by_symbol.values())
        if not series:
            return []
        n = min(len(s) for s in series)
        count = len(series)
        return [sum(s[t] for s in series) / count for t in range(n)]

    def _collect_returns(self, symbols: list[str]) -> dict[str, list[float]]:
        """Pobiera historię per symbol, wyrównuje po timestampie i liczy zwroty.

        Symbole z za krótką/pustą historią (mniej niż 2 zwroty) są pomijane.
        Pojedynczy błąd odczytu nie wywala całego passu.
        """
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

        aligned = self._align_by_timestamp(raw)

        returns_by_symbol: dict[str, list[float]] = {}
        for symbol, prices in aligned.items():
            rets = returns_from_prices(prices)
            if len(rets) >= 2:
                returns_by_symbol[symbol] = rets
        return returns_by_symbol

    @staticmethod
    def _align_by_timestamp(
        raw: dict[str, list[tuple[datetime, Money]]],
    ) -> dict[str, list[float]]:
        """Wyrównuje szeregi cen po WSPÓLNYCH timestampach.

        Snapshoty różnych symboli mogą startować w innych dniach (różny moment
        dodania do watchlisty). Korelacja wymaga sparowanych obserwacji, więc
        bierzemy tylko timestampy obecne we WSZYSTKICH szeregach i czytamy ceny
        w rosnącym porządku czasu. Mniej niż 2 wspólne punkty → pusta lista.
        """
        if not raw:
            return {}
        common: set[datetime] | None = None
        for series in raw.values():
            stamps = {ts for ts, _ in series}
            common = stamps if common is None else (common & stamps)
        if not common:
            return {sym: [] for sym in raw}
        ordered = sorted(common)
        out: dict[str, list[float]] = {}
        for symbol, series in raw.items():
            by_ts = {ts: float(price.amount) for ts, price in series}
            out[symbol] = [by_ts[ts] for ts in ordered]
        return out

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
            f"{members} poruszają się razem {avg_corr:.2f} — "
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
