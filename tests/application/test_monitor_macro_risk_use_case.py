"""Testy MonitorMacroRisk — use case Risk Watch.

Spina MarketDataPort (ceny instrumentów proxy), RepositoryPort (snapshot
poprzedniej ceny + zapis bieżącej) i opcjonalnie MacroIndicatorsPort
(polskie makro). Wynik: MacroRiskReport z listą sygnałów per instrument,
opcjonalnym snapshotem polskiego makro i ogólnym poziomem alertu.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application.ports import (
    MacroIndicatorsPort,
    MarketDataPort,
    RepositoryPort,
)
from src.application.use_cases.monitor_macro_risk import (
    MacroRiskReport,
    MonitorMacroRiskUseCase,
)
from src.domain.macro_risk import MacroAlertLevel, MacroRiskInstrumentType
from src.domain.polish_macro import MacroStressLevel, PolishMacroSnapshot
from src.domain.value_objects import Money

SYMBOLS = {
    "SH": MacroRiskInstrumentType.INVERSE_EQUITY,
    "EPOL": MacroRiskInstrumentType.SOVEREIGN_PROXY,
    "VIXY": MacroRiskInstrumentType.VOLATILITY,
}


@pytest.fixture
def market_port() -> Mock:
    return Mock(spec=MarketDataPort)


@pytest.fixture
def repository_port() -> Mock:
    return Mock(spec=RepositoryPort)


@pytest.fixture
def macro_port() -> Mock:
    return Mock(spec=MacroIndicatorsPort)


def _make_uc(market: Mock, repo: Mock, macro: Mock | None = None) -> MonitorMacroRiskUseCase:
    return MonitorMacroRiskUseCase(
        market_port=market,
        repository_port=repo,
        macro_port=macro,
    )


class TestSignalCollection:
    def test_fetches_current_and_previous_for_each_symbol(
        self, market_port: Mock, repository_port: Mock
    ) -> None:
        market_port.get_current_price.return_value = Money(Decimal("50"))
        repository_port.get_last_price.return_value = Money(Decimal("49"))

        _make_uc(market_port, repository_port).run(SYMBOLS)

        assert market_port.get_current_price.call_count == 3
        assert repository_port.get_last_price.call_count == 3

    def test_saves_price_snapshot_every_cycle(
        self, market_port: Mock, repository_port: Mock
    ) -> None:
        """Cold-start guard — snapshot zapisywany ZAWSZE, nawet przy first run."""
        market_port.get_current_price.return_value = Money(Decimal("50"))
        repository_port.get_last_price.return_value = None  # first cycle

        _make_uc(market_port, repository_port).run({"SH": MacroRiskInstrumentType.INVERSE_EQUITY})

        repository_port.save_price_snapshot.assert_called_once_with(
            "SH", Money(Decimal("50"))
        )

    def test_skips_signal_on_cold_start(
        self, market_port: Mock, repository_port: Mock
    ) -> None:
        market_port.get_current_price.return_value = Money(Decimal("50"))
        repository_port.get_last_price.return_value = None

        report = _make_uc(market_port, repository_port).run(
            {"SH": MacroRiskInstrumentType.INVERSE_EQUITY}
        )

        assert report.signals == []

    def test_per_symbol_error_does_not_kill_cycle(
        self, market_port: Mock, repository_port: Mock
    ) -> None:
        """SH rzuca, EPOL+VIXY działają — raport ma 2 sygnały, nie 0."""
        def get_price(sym: str) -> Money:
            if sym == "SH":
                raise RuntimeError("Finnhub 403")
            return Money(Decimal("100"))

        market_port.get_current_price.side_effect = get_price
        repository_port.get_last_price.return_value = Money(Decimal("99"))

        report = _make_uc(market_port, repository_port).run(SYMBOLS)

        assert len(report.signals) == 2
        assert {s.symbol for s in report.signals} == {"EPOL", "VIXY"}


class TestOverallAlert:
    def test_overall_alert_is_max_across_signals(
        self, market_port: Mock, repository_port: Mock
    ) -> None:
        def get_price(sym: str) -> Money:
            return {
                "SH": Money(Decimal("104")),     # +4% → CRITICAL
                "EPOL": Money(Decimal("99.8")),  # -0.2% → NORMAL
                "VIXY": Money(Decimal("102")),   # +2% → ELEVATED
            }[sym]

        market_port.get_current_price.side_effect = get_price
        repository_port.get_last_price.return_value = Money(Decimal("100"))

        report = _make_uc(market_port, repository_port).run(SYMBOLS)

        assert report.overall_alert is MacroAlertLevel.CRITICAL

    def test_overall_alert_normal_when_no_signals(
        self, market_port: Mock, repository_port: Mock
    ) -> None:
        market_port.get_current_price.return_value = Money(Decimal("100"))
        repository_port.get_last_price.return_value = None  # cold start

        report = _make_uc(market_port, repository_port).run(SYMBOLS)

        assert report.overall_alert is MacroAlertLevel.NORMAL

    def test_polish_macro_critical_lifts_overall(
        self, market_port: Mock, repository_port: Mock, macro_port: Mock
    ) -> None:
        """Critical PL macro = critical overall, nawet gdy żaden ticker nie alarmuje."""
        market_port.get_current_price.return_value = Money(Decimal("100"))
        repository_port.get_last_price.return_value = Money(Decimal("99.9"))  # +0.1% → NORMAL
        macro_port.fetch_polish_macro.return_value = PolishMacroSnapshot(
            eur_pln=Decimal("4.55"),
            usd_pln=Decimal("4.20"),
            eur_pln_30d_change_pct=Decimal("0.06"),  # +6% → CRITICAL stress
            usd_pln_30d_change_pct=Decimal("0.04"),
            fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
        )

        report = _make_uc(market_port, repository_port, macro_port).run(SYMBOLS)

        assert report.overall_alert is MacroAlertLevel.CRITICAL


class TestPolishMacroIntegration:
    def test_macro_port_none_means_no_polish_macro(
        self, market_port: Mock, repository_port: Mock
    ) -> None:
        market_port.get_current_price.return_value = Money(Decimal("100"))
        repository_port.get_last_price.return_value = Money(Decimal("100"))

        report = _make_uc(market_port, repository_port).run(SYMBOLS)

        assert report.polish_macro is None

    def test_macro_port_error_does_not_kill_report(
        self, market_port: Mock, repository_port: Mock, macro_port: Mock
    ) -> None:
        market_port.get_current_price.return_value = Money(Decimal("100"))
        repository_port.get_last_price.return_value = Money(Decimal("100"))
        macro_port.fetch_polish_macro.side_effect = RuntimeError("NBP timeout")

        report = _make_uc(market_port, repository_port, macro_port).run(SYMBOLS)

        assert report.polish_macro is None
        # Reszta raportu mimo to dostarczona.
        assert isinstance(report, MacroRiskReport)

    def test_macro_port_returning_none_passes_through(
        self, market_port: Mock, repository_port: Mock, macro_port: Mock
    ) -> None:
        market_port.get_current_price.return_value = Money(Decimal("100"))
        repository_port.get_last_price.return_value = Money(Decimal("100"))
        macro_port.fetch_polish_macro.return_value = None

        report = _make_uc(market_port, repository_port, macro_port).run(SYMBOLS)

        assert report.polish_macro is None

    def test_macro_stress_level_exposed_via_snapshot(
        self, market_port: Mock, repository_port: Mock, macro_port: Mock
    ) -> None:
        market_port.get_current_price.return_value = Money(Decimal("100"))
        repository_port.get_last_price.return_value = Money(Decimal("100"))
        snap = PolishMacroSnapshot(
            eur_pln=Decimal("4.40"),
            usd_pln=Decimal("4.05"),
            eur_pln_30d_change_pct=Decimal("0.025"),
            usd_pln_30d_change_pct=Decimal("0.01"),
            fetched_at=datetime(2026, 6, 1, tzinfo=UTC),
        )
        macro_port.fetch_polish_macro.return_value = snap

        report = _make_uc(market_port, repository_port, macro_port).run(SYMBOLS)

        assert report.polish_macro is snap
        assert (
            report.polish_macro.evaluate_stress_level() is MacroStressLevel.ELEVATED
        )
