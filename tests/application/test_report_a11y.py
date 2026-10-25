"""Dostępność maila (MAIL-57): język dokumentu, semantyka tabel, kierunek nie
tylko kolorem i kontrast kolorów tekstu (WCAG AA, ≥ 4.5:1 dla tekstu 11–13 px)."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path

import pytest

from src.application import report_skipped
from src.application.report_builder import (
    ReportContext,
    ResolvedPrediction,
    SymbolResult,
    build_delta_chart_html,
    build_html_report,
)
from src.application.report_formatting import PALETTE, contrast_ratio, direction_pct
from src.domain.quota import QuotaAlert, QuotaSeverity
from tests.application.test_report_builder_snapshot import SCENARIOS, _full_cycle

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

    html = report_skipped._render_ignored_html([up, down])

    assert "▲ +0,42%" in html
    assert "▼ -0,42%" in html


def test_predictions_table_shows_arrow_and_sign_for_the_move_and_the_forecast() -> None:
    html, _ = build_html_report([_saved("AAPL", "0.025", "98")], STARTED, 1.0)

    start = html.index("🔮 Wygenerowane predykcje")
    table = html[start:html.index("</table>", start)]
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


# ---------------------------------------------------------------------------
# (5) MAIL-63: kontrast każdego tekstu na JEGO tle, nie tylko kolorów palety
# ---------------------------------------------------------------------------

_APPLICATION = Path(__file__).resolve().parents[2] / "src" / "application"
_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")
_VOID_TAGS = frozenset({"br", "hr", "img", "meta", "input", "link", "col", "wbr"})
_NON_VISIBLE_TAGS = frozenset({"style", "head", "title", "script"})
_TEXT_COLOR = re.compile(r"(?<![\w-])color:\s*(#[0-9a-fA-F]{3,6})\b")
_AA = 4.5


def _declarations(style: str) -> dict[str, str]:
    pairs = (part.split(":", 1) for part in style.split(";") if ":" in part)
    return {name.strip().lower(): value.strip() for name, value in pairs}


def _stylesheet_classes(html: str) -> dict[str, dict[str, str]]:
    css = " ".join(re.findall(r"<style>(.*?)</style>", html, re.S))
    rules: dict[str, dict[str, str]] = {}
    for selector, body in re.findall(r"\.([A-Za-z0-9_-]+)\{([^}]*)\}", css):
        rules.setdefault(selector, {}).update(_declarations(body))
    return rules


class _ContrastAudit(HTMLParser):
    """Dla każdego tekstu: kolor z `style` albo klasy arkusza (dziedziczony) na tle
    najbliższego przodka z kolorem tła (domyślnie biel). Zbiera pary poniżej AA."""

    def __init__(self, classes: dict[str, dict[str, str]]) -> None:
        super().__init__(convert_charrefs=True)
        self._classes = classes
        self._stack: list[tuple[str, str]] = [("#000000", "#ffffff")]
        self._hidden = 0
        self.violations: list[tuple[float, str, str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _NON_VISIBLE_TAGS:
            self._hidden += 1
            return
        if tag in _VOID_TAGS:
            return
        attributes = dict(attrs)
        declared: dict[str, str] = {}
        for name in (attributes.get("class") or "").split():
            declared.update(self._classes.get(name, {}))
        declared.update(_declarations(attributes.get("style") or ""))
        foreground, background = self._stack[-1]
        if (match := _HEX_COLOR.match(declared.get("color", ""))) is not None:
            foreground = match.group(0)
        for key in ("background", "background-color"):
            if (match := _HEX_COLOR.match(declared.get(key, ""))) is not None:
                background = match.group(0)
        self._stack.append((foreground, background))

    def handle_endtag(self, tag: str) -> None:
        if tag in _NON_VISIBLE_TAGS:
            self._hidden = max(0, self._hidden - 1)
        elif tag not in _VOID_TAGS and len(self._stack) > 1:
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        if self._hidden or not re.search(r"\w", data):
            return
        foreground, background = self._stack[-1]
        ratio = contrast_ratio(foreground, background)
        if ratio < _AA:
            self.violations.append((round(ratio, 2), foreground, background, data.strip()[:40]))


def _contrast_violations(html: str) -> list[tuple[float, str, str, str]]:
    audit = _ContrastAudit(_stylesheet_classes(html))
    audit.feed(html)
    return audit.violations


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_no_text_in_the_mail_is_below_aa_on_its_own_background(scenario: str) -> None:
    html, _ = SCENARIOS[scenario]()

    assert _contrast_violations(html) == []


def test_the_audit_flags_a_light_warning_yellow_on_white() -> None:
    # Straż nie może być pusta: #f59e0b na bieli to ok. 2,1:1 (przykład z MAIL-63).
    html = "<div><span style='color: #f59e0b;'>Słaby sygnał</span></div>"

    assert _contrast_violations(html) == [(2.15, "#f59e0b", "#ffffff", "Słaby sygnał")]


def test_the_audit_uses_the_background_of_the_nearest_ancestor_and_the_stylesheet() -> None:
    html = (
        "<style>.dn{color:#dc2626}.gd{background:#fef2f2}</style>"
        "<table><tr class='gd'><td><span class='dn'>spadek</span></td></tr></table>"
    )

    assert _contrast_violations(html) == [(4.41, "#dc2626", "#fef2f2", "spadek")]


def test_states_missing_from_the_golden_scenarios_meet_aa_too() -> None:
    # Czerwone i żółte warianty, których żaden scenariusz golden nie renderuje:
    # baner kwot (oba poziomy), sygnały ostrzegawcze każdej wagi, ujemny wynik
    # krzywej kapitału i alert CRITICAL w Risk Watch.
    from src.application.report_models import RiskSignal
    from src.application.report_quota_banner import render_quota_banner_html
    from src.application.report_risk_signals import _render_risk_signals_html
    from src.application.report_risk_watch import render_risk_watch_html
    from src.application.report_track_record import _render_equity_summary_html
    from src.application.use_cases.monitor_macro_risk import MacroRiskReport
    from src.domain.drawdown import DrawdownSignal
    from src.domain.equity_curve import EquityCurve
    from src.domain.macro_risk import MacroAlertLevel

    alerts = [
        QuotaAlert(
            source=source, severity=severity, message="m", action="a", occurred_at=STARTED
        )
        for source, severity in (("A", QuotaSeverity.CRITICAL), ("B", QuotaSeverity.WARNING))
    ]
    signals = [
        RiskSignal(symbol="AAPL", type="LOW_SIGNAL", severity=severity, description="d")
        for severity in ("high", "medium", "low")
    ]
    losing = EquityCurve(
        points=(), total_return=-0.12, win_rate=0.4, max_drawdown=0.2, n_trades=10
    )
    critical = MacroRiskReport(
        drawdowns=[
            DrawdownSignal(symbol="TSLA", current_price=Decimal("75"), peak_price=Decimal("100"))
        ],
        overall_alert=MacroAlertLevel.CRITICAL,
    )

    fragments = [
        render_quota_banner_html(alerts),
        _render_risk_signals_html(signals),
        _render_equity_summary_html(losing, None),
        render_risk_watch_html(critical),
    ]

    assert all(fragment for fragment in fragments)
    assert [_contrast_violations(fragment) for fragment in fragments] == [[]] * len(fragments)


def _text_color_literals() -> list[tuple[str, str]]:
    sources = [
        path
        for path in sorted(_APPLICATION.glob("report_*.py"))
        # Publiczna strona scorecardu to osobny dokument www, nie mail.
        if path.name != "report_public_scorecard.py"
    ] + sorted((_APPLICATION / "templates").glob("*.j2"))
    return [
        (path.name, color)
        for path in sources
        for color in _TEXT_COLOR.findall(path.read_text("utf-8"))
        # Biały tekst jest na ciemnym wypełnieniu (pigułka werdyktu); jego tło
        # sprawdza audyt powyżej, bo biel na bieli nie ma sensu.
        if color.lower() not in {"#fff", "#ffffff"}
    ]


def test_the_scan_of_text_color_literals_sees_the_known_ones() -> None:
    names = {name for name, _ in _text_color_literals()}

    assert {"report_skipped.py", "report_user_portfolio.py", "valuation_section.html.j2"} <= names


@pytest.mark.parametrize(("module", "color"), sorted(set(_text_color_literals())))
def test_no_text_color_literal_in_a_report_module_is_below_aa_on_white(
    module: str, color: str
) -> None:
    # Siatka na sekcje, których nie ma w żadnym scenariuszu golden (kolory dobierane
    # z danych też przechodzą przez te literały): biel to najmniej wymagające tło.
    assert contrast_ratio(color, "#ffffff") >= _AA, f"{module}: color {color}"


def test_budget_cut_chips_meet_aa() -> None:
    # FIN-12/FIN-25: chip symbolu odciętego budżetem niesie czerwony albo zielony
    # ruch — na bursztynowym tle #fef3c7 czerwień miała 4,34:1.
    from src.domain.cycle_maturity import SkipReason

    cut = [
        SymbolResult(
            symbol=symbol, status="ignored", delta=Decimal(delta),
            skip_reason=SkipReason.BUDGET_EXHAUSTED,
        )
        for symbol, delta in (("NVDA", "-0.035"), ("AMD", "0.031"))
    ]

    fragment = report_skipped._render_budget_exhausted_html(cut)

    assert fragment
    assert _contrast_violations(fragment) == []
