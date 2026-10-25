"""MAIL-35: zwarty mail cichego dnia (`build_html_report(..., compact=True)`).

Kilka linijek zamiast pełnego raportu: największy ruch z progiem, ile symboli
zostało pod progiem i stan portfela (albo watchlisty), a pod nimi pełna stopka —
zastrzeżenie, źródła danych z linkami atrybucji i notka o warunkach dostawców
(MAIL-07), bo wymagają ich warunki licencji.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from src.application.report_builder import ReportContext, SymbolResult, build_html_report
from src.application.report_formatting import (
    DATA_SOURCE_LINKS,
    DATA_SOURCES_FOOTER_PL,
    DATA_TERMS_NOTICE,
    INVESTMENT_DISCLAIMER,
)
from src.domain.cycle_maturity import SkipReason
from src.domain.portfolio import Portfolio, Position

STARTED = datetime(2026, 10, 15, 10, 30, tzinfo=UTC)


def _ignored(symbol: str, delta: str, threshold: str = "0.02") -> SymbolResult:
    return SymbolResult(
        symbol=symbol,
        status="ignored",
        delta=Decimal(delta),
        current_price=Decimal("100") * (1 + Decimal(delta)),
        skip_reason=SkipReason.BELOW_THRESHOLD,
        effective_threshold=Decimal(threshold),
    )


RESULTS = [
    _ignored("NVDA", "0.004"),
    _ignored("AMD", "-0.007"),
    _ignored("ETH", "-0.011", threshold="0.05"),
]


def _compact(context: ReportContext | None = None) -> tuple[str, str]:
    return build_html_report(RESULTS, STARTED, 12.5, context, compact=True)


def test_compact_mail_names_the_largest_move_and_its_threshold() -> None:
    html, text = _compact()

    for body in (html, text):
        assert "cichy dzień" in body.lower()
        assert "Największy ruch:" in body
        assert "ETH" in body
        assert "-1,10%" in body
        assert "próg 5,0%" in body
        assert "Żaden z 3 symboli nie przekroczył progu" in body


def test_compact_mail_without_a_portfolio_reports_the_watchlist() -> None:
    _, text = _compact()

    # (0,4 − 0,7 − 1,1) / 3 = −0,47% — równe wagi, więc „watchlista”, nie „portfel”.
    assert "Watchlista (równe wagi): średnio -0,47%" in text
    assert "Twój portfel" not in text


def test_compact_mail_reports_the_owners_portfolio_change() -> None:
    portfolio = Portfolio(
        positions=(
            Position(symbol="NVDA", quantity=Decimal("10"), purchase_price=Decimal("90")),
        ),
        as_of=STARTED - timedelta(days=1),
    )

    html, text = _compact(ReportContext(user_portfolio=portfolio))

    assert "Twój portfel od ostatniego maila: +0,4% (+$4,00)" in text
    assert "Twój portfel od ostatniego maila" in html
    assert "Watchlista" not in text


def test_compact_mail_keeps_the_full_footer_with_sources_and_terms() -> None:
    html, text = _compact()

    for body in (html, text):
        assert INVESTMENT_DISCLAIMER in body
        assert DATA_TERMS_NOTICE in body
    assert DATA_SOURCES_FOOTER_PL in text
    for _phrase, url in DATA_SOURCE_LINKS:
        assert f'href="{url}"' in html


def test_compact_mail_drops_the_full_report_sections() -> None:
    html, text = _compact()
    full_html, full_text = build_html_report(RESULTS, STARTED, 12.5)

    assert "ZAPLECZE" in full_text
    assert "ZAPLECZE" not in text
    assert "⏸ Pominięte" in full_html
    assert "⏸ Pominięte" not in html
    assert len(text.splitlines()) < len(full_text.splitlines())


def test_full_mail_has_no_quiet_day_block() -> None:
    html, text = build_html_report(RESULTS, STARTED, 12.5)

    assert "cichy dzień" not in html.lower()
    assert "cichy dzień" not in text.lower()
