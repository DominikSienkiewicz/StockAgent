"""Dostępność maila (MAIL-57): język dokumentu, semantyka tabel, kierunek nie
tylko kolorem i kontrast kolorów tekstu (WCAG AA, ≥ 4.5:1 dla tekstu 11–13 px)."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.application import report_builder
from src.application.report_builder import (
    ReportContext,
    ResolvedPrediction,
    SymbolResult,
    build_delta_chart_html,
    build_html_report,
)
from src.application.report_formatting import PALETTE, contrast_ratio, direction_pct
from src.domain.quota import QuotaAlert, QuotaSeverity
from tests.application.test_report_builder_snapshot import _full_cycle

STARTED = datetime(2026, 10, 19, 5, 30, tzinfo=UTC)


def _saved(symbol: str, delta: str, target: str) -> SymbolResult:
    return SymbolResult(
        symbol=symbol, status="saved", delta=Decimal(delta), current_price=Decimal("100"),
        trend="BULLISH" if Decimal(target) > 100 else "BEARISH",
        target_price=Decimal(target), confidence_score=0.7, reasoning="Teza.",
    )


# ---------------------------------------------------------------------------
# (1) Dokument: język polski, kodowanie, arkusz w <head>
# ---------------------------------------------------------------------------


def test_mail_declares_polish_language() -> None:
    html, _ = build_html_report([_saved("AAPL", "0.03", "103")], STARTED, 1.0)

    assert html.startswith('<!DOCTYPE html><html lang="pl"><head><meta charset="utf-8">')
    assert html.endswith("</body></html>")


def test_stylesheet_sits_in_the_head() -> None:
    html, _ = build_html_report([_saved("AAPL", "0.03", "103")], STARTED, 1.0)

    head = html[: html.index("</head>")]
    assert "<style>" in head
    assert "<style>" not in html[html.index("<body>"):]


# ---------------------------------------------------------------------------
# (2) Tabele: layoutowe jako presentation, nagłówki danych z zakresem kolumny
# ---------------------------------------------------------------------------


def test_bar_chart_table_is_presentation() -> None:
    chart = build_delta_chart_html([_saved("AAPL", "0.03", "103"), _saved("MSFT", "-0.02", "98")])

    assert "<table role='presentation'" in chart


def test_quota_banner_table_is_presentation() -> None:
    alert = QuotaAlert(
        source="OpenAI", severity=QuotaSeverity.CRITICAL, message="429",
        action="Doładuj konto.", occurred_at=STARTED,
    )
    html, _ = build_html_report([], STARTED, 1.0, ReportContext(quota_alerts=[alert]))

    banner = html[html.index("ALERTY SUBSKRYPCJI"):]
    assert "<table role='presentation'" in banner[: banner.index("</table>")]


def test_every_data_table_header_scopes_its_column() -> None:
    html, _ = _full_cycle()

    headers = re.findall(r"<th\b[^>]*>", html)
    assert headers
    assert all("scope=" in header and "col" in header for header in headers)


# ---------------------------------------------------------------------------
# (3) Kierunek nie tylko kolorem: strzałka i znak, słowo obok ✅/❌
# ---------------------------------------------------------------------------


def test_direction_not_conveyed_by_color_only() -> None:
    up = SymbolResult(symbol="AAPL", status="ignored", delta=Decimal("0.0042"))
    down = SymbolResult(symbol="MSFT", status="ignored", delta=Decimal("-0.0042"))

    html = report_builder._render_ignored_html([up, down])

    assert "▲ +0,42%" in html
    assert "▼ -0,42%" in html


def test_predictions_table_shows_arrow_and_sign_for_the_move_and_the_forecast() -> None:
    html, _ = build_html_report([_saved("AAPL", "0.025", "98")], STARTED, 1.0)

    table = html[html.index("🔮 Wygenerowane predykcje"):html.index("</table>")]
    assert "▲ +2,50%" in table
    assert "▼ -2,00%" in table


def test_resolved_verdict_has_a_word_next_to_the_mark() -> None:
    resolved = [
        ResolvedPrediction(symbol="AAPL", predicted_trend="BULLISH", is_correct=True),
        ResolvedPrediction(symbol="MSFT", predicted_trend="BEARISH", is_correct=False),
    ]
    html, _ = build_html_report([], STARTED, 1.0, ReportContext(resolved_predictions=resolved))

    section = html[html.index("📊 Zamknięte predykcje"):]
    assert re.search(r"✅.{0,200}Trafiona", section)
    assert re.search(r"❌.{0,200}Błędna", section)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("0.0042"), "▲ +0,42%"),
        (Decimal("-0.0042"), "▼ -0,42%"),
        (Decimal("0"), "+0,00%"),
        (Decimal("0.00001"), "+0,00%"),
        (None, "—"),
    ],
)
def test_direction_pct_pairs_the_arrow_with_the_polish_sign(
    value: Decimal | None, expected: str
) -> None:
    # Ruch zaokrąglony do zera nie dostaje strzałki — „▲ +0,00%” byłoby sprzeczne.
    assert direction_pct(value) == expected


# ---------------------------------------------------------------------------
# (4) Kontrast: czysta funkcja i paleta raportu
# ---------------------------------------------------------------------------


def test_contrast_ratio_spans_one_to_twenty_one() -> None:
    assert contrast_ratio("#000000", "#ffffff") == pytest.approx(21.0)
    assert contrast_ratio("#ffffff", "#000000") == pytest.approx(21.0)
    assert contrast_ratio("#6b7280", "#6b7280") == pytest.approx(1.0)


def test_contrast_ratio_matches_the_wcag_reference_values() -> None:
    # Szarość #767676 to najjaśniejszy szary na białym, który przechodzi AA (4.54:1).
    assert contrast_ratio("#767676", "#ffffff") == pytest.approx(4.54, abs=0.01)
    assert contrast_ratio("#16a34a", "#fff") == pytest.approx(3.30, abs=0.01)


_TEXT_ROLES = ("positive", "negative", "neutral", "caution", "text", "text_secondary",
               "muted", "subtle")
_BACKGROUNDS = ("#ffffff", PALETTE.surface)


@pytest.mark.parametrize(
    ("role", "background"),
    [(role, background) for role in _TEXT_ROLES for background in _BACKGROUNDS],
)
def test_palette_meets_wcag_aa(role: str, background: str) -> None:
    assert contrast_ratio(getattr(PALETTE, role), background) >= 4.5


def test_footer_text_meets_wcag_aa() -> None:
    html, _ = _full_cycle()

    footer = html[html.rindex("margin-top: 32px; padding-top: 16px;"):]
    colors = re.findall(r"color: (#[0-9a-f]{6})", footer)
    assert colors
    assert all(contrast_ratio(color, "#ffffff") >= 4.5 for color in colors)
