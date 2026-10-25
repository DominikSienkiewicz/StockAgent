"""MAIL-21: sekcja 📡 Stan źródeł danych — linijka na dostawcę, ukryta bez problemów."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from src.application.report_builder import ReportContext, SymbolResult, build_html_report
from src.application.report_source_health import (
    render_source_health_html,
    render_source_health_text,
)
from src.application.source_health import SOURCE_ORDER, SourceHealth, SourceStatus

AT = datetime(2026, 10, 15, 10, 30, tzinfo=UTC)


def _rows(**overrides: SourceHealth) -> tuple[SourceHealth, ...]:
    return tuple(
        overrides.get(source, SourceHealth(source, SourceStatus.OK)) for source in SOURCE_ORDER
    )


def test_open_circuit_renders_unavailable_row() -> None:
    rows = _rows(Finnhub=SourceHealth("Finnhub", SourceStatus.UNAVAILABLE, affected=43))

    html = render_source_health_html(rows)
    text = "\n".join(render_source_health_text(rows))

    assert "📡 Stan źródeł danych" in html
    assert "Finnhub" in html
    assert "niedostępne (otwarty obwód, 43 symbole pominięte)" in html
    assert "Finnhub: niedostępne (otwarty obwód, 43 symbole pominięte)" in text


def test_every_provider_gets_one_line_with_its_state() -> None:
    rows = _rows(
        **{
            "Alpha Vantage": SourceHealth("Alpha Vantage", SourceStatus.DEGRADED, affected=3),
            "EDGAR": SourceHealth("EDGAR", SourceStatus.DEGRADED, failed_requests=4),
            "Reddit": SourceHealth("Reddit", SourceStatus.DISABLED),
        }
    )

    text = render_source_health_text(rows)

    assert "  Alpha Vantage: zdegradowane (3 symbole)" in text
    assert "  EDGAR: zdegradowane (4 błędne odpowiedzi)" in text
    assert "  Reddit: wyłączone flagą" in text
    assert "  CoinGecko: OK" in text
    assert sum(1 for line in text if line.startswith("  ")) == len(SOURCE_ORDER)


def test_section_is_hidden_when_nothing_is_wrong() -> None:
    rows = _rows(Reddit=SourceHealth("Reddit", SourceStatus.DISABLED))

    assert render_source_health_html(rows) == ""
    assert render_source_health_text(rows) == []


def test_report_shows_the_section_under_the_quota_banner() -> None:
    context = ReportContext(
        source_health=_rows(Finnhub=SourceHealth("Finnhub", SourceStatus.UNAVAILABLE, affected=2))
    )

    html, text = build_html_report(
        [SymbolResult(symbol="MSFT", status="ignored", delta=Decimal("0.005"))], AT, 1.0, context
    )

    assert "📡 Stan źródeł danych" in html
    assert text.index("STAN ŹRÓDEŁ DANYCH") < text.index("STOCKAGENT — RAPORT CYKLU")
