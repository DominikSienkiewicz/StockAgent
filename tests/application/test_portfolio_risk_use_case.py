"""Testy PortfolioRiskUseCase — radar korelacji/koncentracji portfela.

Use case to osobny pass (jak MonitorMacroRisk): ciągnie DARMOWĄ historię
cen per symbol (`get_price_history`), wyrównuje po timestampie, buduje
macierz korelacji i klastry, zwraca zamrożony PortfolioRiskReport.
Wyłącznie RepositoryPort (darmowy odczyt) — zero płatnych portów.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application.ports import RepositoryPort
from src.application.use_cases.portfolio_risk import (
    PortfolioRiskReport,
    PortfolioRiskUseCase,
)
from src.domain.value_objects import Money

_BASE = datetime(2026, 6, 1, tzinfo=UTC)


def _series(values: list[str]) -> list[tuple[datetime, Money]]:
    """Buduje chronologiczną historię (rosnąco) z listy cen."""
    return [
        (_BASE + timedelta(days=i), Money(Decimal(v))) for i, v in enumerate(values)
    ]


@pytest.fixture
def repository_port() -> Mock:
    return Mock(spec=RepositoryPort)


def _make_uc(repo: Mock) -> PortfolioRiskUseCase:
    return PortfolioRiskUseCase(repository_port=repo)


class TestBuildsReport:
    def test_builds_matrix_and_clusters_from_history(
        self, repository_port: Mock
    ) -> None:
        # NVDA: zwroty zmienne (+10%, -5%, +20%); MSFT identyczne (perfect corr);
        # GLD lustrzane (anti-corr). Zwroty MUSZĄ się różnić, inaczej zerowa
        # wariancja zerowałaby korelację.
        history = {
            "NVDA": _series(["100", "110", "104.5", "125.4"]),
            "MSFT": _series(["50", "55", "52.25", "62.7"]),  # = 0.5 * NVDA → corr 1.0
            "GLD": _series(["100", "90", "94.5", "75.6"]),  # lustro NVDA → corr -1.0
        }
        repository_port.get_price_history.side_effect = lambda sym, days: history[sym]

        report = _make_uc(repository_port).run(["NVDA", "MSFT", "GLD"])

        assert isinstance(report, PortfolioRiskReport)
        assert report.matrix[("MSFT", "NVDA")] == 1.0
        assert report.matrix[("GLD", "NVDA")] == -1.0
        # NVDA i MSFT poruszają się razem.
        assert any(
            "NVDA" in cluster and "MSFT" in cluster for cluster in report.clusters
        )

    def test_pulls_history_for_each_symbol(self, repository_port: Mock) -> None:
        repository_port.get_price_history.return_value = _series(
            ["100", "110", "121"]
        )

        _make_uc(repository_port).run(["A", "B"])

        assert repository_port.get_price_history.call_count == 2

    def test_verdict_mentions_cluster_and_share(self, repository_port: Mock) -> None:
        history = {
            "NVDA": _series(["100", "110", "104.5", "125.4"]),
            "MSFT": _series(["50", "55", "52.25", "62.7"]),
            "AAPL": _series(["10", "11", "10.45", "12.54"]),
        }
        repository_port.get_price_history.side_effect = lambda sym, days: history[sym]

        report = _make_uc(repository_port).run(["NVDA", "MSFT", "AAPL"])

        # Werdykt jednolinijkowy — wymienia symbole klastra i udział w watchliście.
        assert report.verdict != ""
        assert "%" in report.verdict
        assert "NVDA" in report.verdict


class TestVarAndStress:
    """R1/R2 — VaR/CVaR i stress-test liczone z tych samych darmowych zwrotów
    co macierz korelacji. Domyślnie wyłączone (wsteczna kompatybilność)."""

    def _history(self) -> dict[str, list[tuple[datetime, Money]]]:
        return {
            "NVDA": _series(["100", "110", "104.5", "125.4", "118.0"]),
            "MSFT": _series(["50", "55", "52.25", "62.7", "59.0"]),
            "GLD": _series(["100", "90", "94.5", "75.6", "80.0"]),
        }

    def test_var_disabled_by_default(self, repository_port: Mock) -> None:
        repository_port.get_price_history.side_effect = (
            lambda sym, days: self._history()[sym]
        )
        report = PortfolioRiskUseCase(repository_port=repository_port).run(
            ["NVDA", "MSFT", "GLD"]
        )
        assert report.var is None
        assert report.cvar is None
        assert report.scenarios == ()

    def test_var_and_cvar_populated_when_enabled(
        self, repository_port: Mock
    ) -> None:
        repository_port.get_price_history.side_effect = (
            lambda sym, days: self._history()[sym]
        )
        report = PortfolioRiskUseCase(
            repository_port=repository_port,
            var_enabled=True,
            var_confidence=0.95,
        ).run(["NVDA", "MSFT", "GLD"])
        assert report.var is not None and report.var >= 0.0
        assert report.cvar is not None
        # CVaR (expected shortfall) nigdy nie jest mniejszy niż VaR.
        assert report.cvar >= report.var
        assert report.var_confidence == 0.95

    def test_stress_scenarios_populated_when_enabled(
        self, repository_port: Mock
    ) -> None:
        from src.domain.scenarios import Scenario

        repository_port.get_price_history.side_effect = (
            lambda sym, days: self._history()[sym]
        )
        scenarios = (
            Scenario(name="Korekta -10%", market_shock=-0.10),
            Scenario(name="Krach -20%", market_shock=-0.20),
        )
        report = PortfolioRiskUseCase(
            repository_port=repository_port,
            stress_enabled=True,
            scenarios=scenarios,
        ).run(["NVDA", "MSFT", "GLD"])
        assert len(report.scenarios) == 2
        names = {s.scenario_name for s in report.scenarios}
        assert names == {"Korekta -10%", "Krach -20%"}
        # Każdy scenariusz wycenia per-symbol wpływ dla całej watchlisty.
        assert all(impact.per_symbol for impact in report.scenarios)

    def test_var_skipped_when_too_few_symbols(self, repository_port: Mock) -> None:
        repository_port.get_price_history.return_value = _series(["100", "110"])
        report = PortfolioRiskUseCase(
            repository_port=repository_port, var_enabled=True
        ).run(["AAPL"])
        # <2 symbole z historią → pusty raport, bez VaR.
        assert report.var is None


class TestGracefulDegradation:
    def test_single_symbol_yields_empty_report(self, repository_port: Mock) -> None:
        repository_port.get_price_history.return_value = _series(["100", "110"])

        report = _make_uc(repository_port).run(["AAPL"])

        assert report.matrix == {}
        assert report.clusters == []
        assert report.verdict == ""

    def test_empty_symbol_list_yields_empty_report(
        self, repository_port: Mock
    ) -> None:
        report = _make_uc(repository_port).run([])

        assert report.matrix == {}
        assert report.clusters == []

    def test_symbols_with_too_short_history_skipped(
        self, repository_port: Mock
    ) -> None:
        history = {
            "A": _series(["100", "110", "121"]),
            "B": _series(["50", "55", "60.5"]),
            "C": _series(["10"]),  # za krótka — brak zwrotów, pomijana
        }
        repository_port.get_price_history.side_effect = lambda sym, days: history[sym]

        report = _make_uc(repository_port).run(["A", "B", "C"])

        assert ("A", "C") not in report.matrix
        assert ("B", "C") not in report.matrix
        assert ("A", "B") in report.matrix

    def test_per_symbol_history_error_does_not_kill_pass(
        self, repository_port: Mock
    ) -> None:
        def get_history(sym: str, days: int) -> list[tuple[datetime, Money]]:
            if sym == "BROKEN":
                raise RuntimeError("DB timeout")
            return _series(["100", "110", "121"])

        repository_port.get_price_history.side_effect = get_history

        report = _make_uc(repository_port).run(["A", "B", "BROKEN"])

        # A i B przeżywają mimo padniętego BROKEN.
        assert ("A", "B") in report.matrix

    def test_fewer_than_two_with_history_is_graceful(
        self, repository_port: Mock
    ) -> None:
        history = {
            "A": _series(["100", "110", "121"]),
            "B": _series([]),  # brak historii
        }
        repository_port.get_price_history.side_effect = lambda sym, days: history[sym]

        report = _make_uc(repository_port).run(["A", "B"])

        assert report.matrix == {}
        assert report.clusters == []
        assert report.verdict == ""


class TestTimestampAlignment:
    def test_aligns_on_common_timestamps(self, repository_port: Mock) -> None:
        # B ma jeden timestamp więcej z przodu — wyrównanie po wspólnych datach
        # musi dać identyczny kształt zwrotów, więc korelacja = 1.0.
        a = _series(["100", "110", "104.5", "125.4"])
        b_full = [(_BASE - timedelta(days=1), Money(Decimal("999")))] + _series(
            ["100", "110", "104.5", "125.4"]
        )
        history = {"A": a, "B": b_full}
        repository_port.get_price_history.side_effect = lambda sym, days: history[sym]

        report = _make_uc(repository_port).run(["A", "B"])

        assert report.matrix[("A", "B")] == 1.0


class TestReportImmutability:
    def test_report_is_frozen(self, repository_port: Mock) -> None:
        repository_port.get_price_history.return_value = _series(["100", "110"])
        report = _make_uc(repository_port).run(["A"])

        with pytest.raises(FrozenInstanceError):
            report.verdict = "tampered"  # type: ignore[misc]
