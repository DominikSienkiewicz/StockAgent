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
from src.domain.drawdown import DrawdownSignal
from src.domain.macro_risk import MacroAlertLevel, MacroRiskInstrumentType
from src.domain.polish_macro import MacroStressLevel, PolishMacroSnapshot
from src.domain.value_objects import Money


def _history(*prices: str) -> list[tuple[datetime, Money]]:
    """Chronologiczna (rosnąco) historia snapshotów cen do mocka repo."""
    base = datetime(2026, 5, 1, tzinfo=UTC)
    return [
        (base.replace(day=i + 1), Money(Decimal(p)))
        for i, p in enumerate(prices)
    ]

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


class TestDrawdowns:
    def test_builds_drawdown_signal_per_symbol_from_history(
        self, market_port: Mock, repository_port: Mock
    ) -> None:
        market_port.get_current_price.return_value = Money(Decimal("90"))
        repository_port.get_last_price.return_value = Money(Decimal("89"))
        # Szczyt 120 w historii, ostatni snapshot 90 → drawdown -25%.
        repository_port.get_price_history.return_value = _history("100", "120", "90")

        report = _make_uc(market_port, repository_port).run(
            {"TSLA": MacroRiskInstrumentType.INVERSE_EQUITY}
        )

        assert len(report.drawdowns) == 1
        dd = report.drawdowns[0]
        assert isinstance(dd, DrawdownSignal)
        assert dd.symbol == "TSLA"
        assert dd.peak_price == Decimal("120")
        assert dd.current_price == Decimal("90")

    def test_uses_latest_history_price_as_current(
        self, market_port: Mock, repository_port: Mock
    ) -> None:
        """Current = ostatni snapshot z historii (rosnąco), nie pierwszy."""
        market_port.get_current_price.return_value = Money(Decimal("0"))
        repository_port.get_last_price.return_value = Money(Decimal("99"))
        repository_port.get_price_history.return_value = _history("100", "120", "80")

        report = _make_uc(market_port, repository_port).run(
            {"TSLA": MacroRiskInstrumentType.INVERSE_EQUITY}
        )

        assert report.drawdowns[0].current_price == Decimal("80")

    def test_empty_history_yields_no_drawdown_signal(
        self, market_port: Mock, repository_port: Mock
    ) -> None:
        market_port.get_current_price.return_value = Money(Decimal("100"))
        repository_port.get_last_price.return_value = Money(Decimal("99"))
        repository_port.get_price_history.return_value = []

        report = _make_uc(market_port, repository_port).run(
            {"TSLA": MacroRiskInstrumentType.INVERSE_EQUITY}
        )

        assert report.drawdowns == []

    def test_drawdown_error_does_not_kill_cycle(
        self, market_port: Mock, repository_port: Mock
    ) -> None:
        market_port.get_current_price.return_value = Money(Decimal("100"))
        repository_port.get_last_price.return_value = Money(Decimal("99"))
        repository_port.get_price_history.side_effect = RuntimeError("DB down")

        report = _make_uc(market_port, repository_port).run(
            {"TSLA": MacroRiskInstrumentType.INVERSE_EQUITY}
        )

        # Sygnał makro nadal zbudowany mimo padniętej historii.
        assert report.drawdowns == []
        assert len(report.signals) == 1

    def test_critical_drawdown_lifts_overall_alert(
        self, market_port: Mock, repository_port: Mock
    ) -> None:
        """Głęboki drawdown sam w sobie podnosi overall_alert do CRITICAL."""
        market_port.get_current_price.return_value = Money(Decimal("100"))
        # Sygnał makro: NORMAL (cena płaska vs poprzednia).
        repository_port.get_last_price.return_value = Money(Decimal("100"))
        # Historia: szczyt 200, teraz 100 → -50% → CRITICAL drawdown.
        repository_port.get_price_history.return_value = _history("200", "100")

        report = _make_uc(market_port, repository_port).run(
            {"TSLA": MacroRiskInstrumentType.INVERSE_EQUITY}
        )

        assert report.overall_alert is MacroAlertLevel.CRITICAL

    def test_default_drawdowns_empty_for_backward_compat(self) -> None:
        assert MacroRiskReport().drawdowns == []


class TestHedgeEffectiveness:
    """R3 — realized correlation hedge vs book. Inverse ETF, który faktycznie
    porusza się przeciwnie do portfela, jest 'skuteczny'; jeśli koreluje
    dodatnio (zepsuty hedge) — 'odwrócony'. Domyślnie wyłączone."""

    def _wire(
        self, market: Mock, repo: Mock, histories: dict[str, list]
    ) -> None:
        repo.get_price_history.side_effect = (
            lambda sym, days: histories.get(sym, [])
        )
        market.get_current_price.return_value = Money(Decimal("80"))
        repo.get_last_price.return_value = Money(Decimal("82"))

    def test_hedges_empty_by_default(
        self, market_port: Mock, repository_port: Mock
    ) -> None:
        self._wire(
            market_port, repository_port,
            {"AAPL": _history("100", "110", "104.5", "125.4", "118.0"),
             "SH": _history("100", "90", "94.5", "75.6", "80.06")},
        )
        report = _make_uc(market_port, repository_port).run(
            {"SH": MacroRiskInstrumentType.INVERSE_EQUITY}
        )
        assert report.hedges == []

    def test_effective_inverse_hedge_labeled_skuteczny(
        self, market_port: Mock, repository_port: Mock
    ) -> None:
        # SH to dokładne lustro zwrotów AAPL → korelacja -1 → effectiveness +1.
        self._wire(
            market_port, repository_port,
            {"AAPL": _history("100", "110", "104.5", "125.4", "118.0"),
             "SH": _history("100", "90", "94.5", "75.6", "80.06")},
        )
        uc = MonitorMacroRiskUseCase(
            market_port=market_port,
            repository_port=repository_port,
            book_symbols=["AAPL"],
            hedge_enabled=True,
        )
        report = uc.run({"SH": MacroRiskInstrumentType.INVERSE_EQUITY})

        assert len(report.hedges) == 1
        h = report.hedges[0]
        assert h.hedge_symbol == "SH"
        assert h.correlation < 0          # porusza się przeciwnie do booka
        assert h.effectiveness > 0.5      # dobry inverse hedge
        assert h.label == "skuteczny"

    def test_broken_hedge_labeled_odwrocony(
        self, market_port: Mock, repository_port: Mock
    ) -> None:
        # "Hedge" porusza się TAK SAMO jak book (korelacja +1) → odwrócony.
        self._wire(
            market_port, repository_port,
            {"AAPL": _history("100", "110", "104.5", "125.4", "118.0"),
             "BAD": _history("100", "110", "104.5", "125.4", "118.0")},
        )
        uc = MonitorMacroRiskUseCase(
            market_port=market_port,
            repository_port=repository_port,
            book_symbols=["AAPL"],
            hedge_enabled=True,
        )
        report = uc.run({"BAD": MacroRiskInstrumentType.INVERSE_EQUITY})

        assert report.hedges[0].label == "odwrócony"

    def test_no_book_symbols_yields_no_hedges(
        self, market_port: Mock, repository_port: Mock
    ) -> None:
        self._wire(
            market_port, repository_port,
            {"SH": _history("100", "90", "94.5", "75.6", "80.06")},
        )
        uc = MonitorMacroRiskUseCase(
            market_port=market_port,
            repository_port=repository_port,
            book_symbols=[],
            hedge_enabled=True,
        )
        report = uc.run({"SH": MacroRiskInstrumentType.INVERSE_EQUITY})
        assert report.hedges == []

    def test_default_hedges_empty_for_backward_compat(self) -> None:
        assert MacroRiskReport().hedges == []
