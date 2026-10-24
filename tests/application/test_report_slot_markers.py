"""MAIL-21 (decyzja właściciela 2026-10-25): puste sekcje nie zostawiają w HTML
znaczników slotów — `<!-- QUOTA_BANNER_SLOT -->` i `<!-- RISK_WATCH_SLOT -->`
były jedynymi resztkami szablonu sprzed ARCH-12."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from src.application.report_builder import SymbolResult, build_html_report


def test_html_without_quota_alerts_and_risk_watch_has_no_slot_markers() -> None:
    html, _ = build_html_report(
        [SymbolResult(symbol="MSFT", status="ignored", delta=Decimal("0.005"))],
        datetime(2026, 10, 15, 10, 30, tzinfo=UTC),
        1.0,
    )

    assert "QUOTA_BANNER_SLOT" not in html
    assert "RISK_WATCH_SLOT" not in html
