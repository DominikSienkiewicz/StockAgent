"""ALR-06: sekcja ⚡ Między mailami — szoki z watcha od ostatniego maila."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from src.application.ports import StoredShockAlert
from src.application.report_builder import ReportContext, SymbolResult, build_html_report

AT = datetime(2026, 10, 15, 5, 30, tzinfo=UTC)
NVDA_SHOCK = StoredShockAlert(
    symbol="NVDA",
    delta=-0.062,
    direction="DOWN",
    created_at=datetime(2026, 10, 14, 22, 7, tzinfo=UTC),
)


def _report(alerts: list[StoredShockAlert] | None) -> tuple[str, str]:
    results = [
        SymbolResult(
            symbol="NVDA",
            status="ignored",
            delta=Decimal("-0.004"),
            current_price=Decimal("131.20"),
        )
    ]
    context = ReportContext(shock_alerts=alerts, display_timezone="Europe/Warsaw")
    return build_html_report(results, AT, 1.0, context)


def test_shock_section_lists_alerts_since_last_digest() -> None:
    html, text = _report([NVDA_SHOCK])

    assert "⚡ Między mailami" in html
    section = html.split("⚡ Między mailami")[1]
    assert "NVDA" in section
    assert "-6,20%" in section
    assert "spadek" in section
    # 22:07 UTC = 00:07 w Warszawie następnego dnia (czwartek).
    assert "czw 15.10 00:07" in section
    assert "teraz $131,20" in section
    assert "  NVDA -6,20% (spadek) · czw 15.10 00:07 · teraz $131,20" in text


def test_shock_section_hides_without_alerts() -> None:
    for alerts in (None, []):
        html, text = _report(alerts)

        assert "Między mailami" not in html
        assert "MIĘDZY MAILAMI" not in text
