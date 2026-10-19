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
        assert report.matrix[("MSFT", "NVDA")] == pytest.approx(1.0)
        assert report.matrix[("GLD", "NVDA")] == pytest.approx(-1.0)
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
        assert report.var is not None
        assert report.var >= 0.0
        assert report.cvar is not None
        # CVaR (expected shortfall) nigdy nie jest mniejszy niż VaR.
        assert report.cvar >= report.var
        assert report.var_confidence == pytest.approx(0.95)

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

        assert report.matrix[("A", "B")] == pytest.approx(1.0)


class TestUtcDayAlignment:
    """PORT-02 (G19): snapshoty z jednego cyklu różnią się o sekundy, więc
    wyrównanie po dokładnym timestampie dawało puste przecięcie, a radar, VaR,
    stress i hedge znikały. Wyrównujemy po dniu UTC."""

    @staticmethod
    def _daily(prices: list[float], *, second: int) -> list[tuple[datetime, Money]]:
        return [
            (
                _BASE + timedelta(days=i, hours=5, minutes=30, seconds=second),
                Money(Decimal(str(round(p, 4)))),
            )
            for i, p in enumerate(prices)
        ]

    @staticmethod
    def _zigzag(days: int) -> list[float]:
        prices = [100.0]
        for i in range(1, days):
            prices.append(prices[-1] * (1.03 if i % 3 else 0.96))
        return prices

    def test_aligns_snapshots_differing_by_seconds_within_same_day(
        self, repository_port: Mock
    ) -> None:
        nvda = self._zigzag(25)
        history = {
            "NVDA": self._daily(nvda, second=1),
            "AMD": self._daily([p * 0.5 for p in nvda], second=8),
        }
        repository_port.get_price_history.side_effect = lambda sym, days: history[sym]

        report = _make_uc(repository_port).run(["NVDA", "AMD"])

        assert report.clusters == [("AMD", "NVDA")]
        assert report.matrix[("AMD", "NVDA")] == pytest.approx(1.0)

    def test_last_snapshot_of_a_day_is_the_one_that_counts(
        self, repository_port: Mock
    ) -> None:
        nvda = self._zigzag(25)
        noise = [
            (_BASE + timedelta(days=i, hours=1), Money(Decimal("1")))
            for i in range(25)
        ]
        history = {
            "NVDA": sorted(noise + self._daily(nvda, second=1)),
            "AMD": self._daily([p * 0.5 for p in nvda], second=8),
        }
        repository_port.get_price_history.side_effect = lambda sym, days: history[sym]

        report = _make_uc(repository_port).run(["NVDA", "AMD"])

        assert report.matrix[("AMD", "NVDA")] == pytest.approx(1.0)


class TestMinimumSamples:
    """PORT-22: VaR z kilku obserwacji to najgorszy dzień próbki, a korelacja
    z dwóch zwrotów to ±1. Poniżej progów z `risk_sample` statystyka nie
    powstaje, a raport mówi, ile historii brakuje."""

    @staticmethod
    def _history(days: int) -> dict[str, list[tuple[datetime, Money]]]:
        nvda = [100.0]
        for i in range(1, days):
            nvda.append(nvda[-1] * (1.03 if i % 3 else 0.96))

        def series(scale: float) -> list[tuple[datetime, Money]]:
            return [
                (_BASE + timedelta(days=i), Money(Decimal(str(round(p * scale, 4)))))
                for i, p in enumerate(nvda)
            ]

        gld = [
            (_BASE + timedelta(days=i), Money(Decimal(str(100 + (i % 4)))))
            for i in range(days)
        ]
        return {"NVDA": series(1.0), "MSFT": series(0.5), "GLD": gld}

    def _run(self, repo: Mock, days: int, **kwargs: object) -> PortfolioRiskReport:
        history = self._history(days)
        repo.get_price_history.side_effect = lambda sym, days: history[sym]
        return PortfolioRiskUseCase(
            repository_port=repo, require_min_samples=True, **kwargs
        ).run(["NVDA", "MSFT", "GLD"])

    def test_var_is_none_below_min_observations(self, repository_port: Mock) -> None:
        report = self._run(repository_port, 10, var_enabled=True)

        assert report.var is None
        assert report.cvar is None
        assert ("var", 9, 60) in report.insufficient

    def test_var_is_computed_from_60_observations(self, repository_port: Mock) -> None:
        report = self._run(repository_port, 61, var_enabled=True)

        assert report.var is not None
        assert all(stat != "var" for stat, _, _ in report.insufficient)

    def test_correlations_are_skipped_below_min_paired_returns(
        self, repository_port: Mock
    ) -> None:
        report = self._run(repository_port, 10)

        assert report.matrix == {}
        assert report.clusters == []
        assert report.verdict == ""
        assert ("correlation", 9, 20) in report.insufficient

    def test_correlations_are_kept_on_20_paired_returns(
        self, repository_port: Mock
    ) -> None:
        report = self._run(repository_port, 21)

        assert report.clusters == [("MSFT", "NVDA")]
        assert report.insufficient == ()

    def test_stress_betas_are_skipped_below_min_paired_returns(
        self, repository_port: Mock
    ) -> None:
        from src.domain.scenarios import Scenario

        report = self._run(
            repository_port, 10,
            stress_enabled=True,
            scenarios=(Scenario(name="Krach -20%", market_shock=-0.20),),
        )

        assert report.scenarios == ()
        assert ("beta", 9, 20) in report.insufficient

    def test_disabled_var_is_not_reported_as_insufficient(
        self, repository_port: Mock
    ) -> None:
        report = self._run(repository_port, 10)

        assert all(stat != "var" for stat, _, _ in report.insufficient)

    def test_history_window_covers_60_trading_days(self, repository_port: Mock) -> None:
        from src.application.use_cases.portfolio_risk import PRICE_HISTORY_DAYS

        self._run(repository_port, 10)

        assert PRICE_HISTORY_DAYS == 90
        assert {c.args[1] for c in repository_port.get_price_history.call_args_list} == {90}


class TestReportImmutability:
    def test_report_is_frozen(self, repository_port: Mock) -> None:
        repository_port.get_price_history.return_value = _series(["100", "110"])
        report = _make_uc(repository_port).run(["A"])

        with pytest.raises(FrozenInstanceError):
            report.verdict = "tampered"  # type: ignore[misc]
