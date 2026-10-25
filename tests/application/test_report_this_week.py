"""DAT-20: poniedziałkowa sekcja „📅 Ten tydzień” — wyniki spółek z watchlisty do
niedzieli: dzień, pora względem sesji i szacunek EPS. Bez danych sekcji nie ma."""

from __future__ import annotations

from datetime import UTC, datetime

from src.application.report_builder import ReportContext, build_html_report
from src.domain.earnings import EarningsEvent, EarningsTiming

MONDAY = datetime(2026, 10, 26, 6, 0, tzinfo=UTC)


def _week() -> tuple[EarningsEvent, ...]:
    return (
        EarningsEvent(
            "MSFT", 2, "2026-10-28", timing=EarningsTiming.AFTER_CLOSE, eps_estimate=3.1
        ),
        EarningsEvent("AAPL", 4, "2026-10-30", timing=EarningsTiming.BEFORE_OPEN),
        EarningsEvent("NVDA", 5, "2026-10-31"),
    )


def _report(week: tuple[EarningsEvent, ...] | None) -> tuple[str, str]:
    return build_html_report([], MONDAY, 1.0, context=ReportContext(earnings_this_week=week))


def test_section_lists_the_week_with_day_timing_and_estimate() -> None:
    html, _ = _report(_week())

    assert "📅 Ten tydzień" in html
    assert "śr 28.10 — MSFT (Microsoft Corporation) · po sesji · szac. EPS 3,10" in html
    assert "pt 30.10 — AAPL (Apple Inc.) · przed sesją" in html
    assert html.index("MSFT (Microsoft") < html.index("AAPL (Apple")


def test_unknown_timing_and_estimate_are_left_out() -> None:
    html, _ = _report(_week())

    assert "sob 31.10 — NVDA (NVIDIA Corporation)</li>" in html


def test_section_is_in_the_plain_text() -> None:
    _, text = _report(_week())

    assert "TEN TYDZIEŃ" in text
    assert "śr 28.10 — MSFT (Microsoft Corporation) · po sesji · szac. EPS 3,10" in text


def test_no_week_data_means_no_section() -> None:
    for week in (None, ()):
        html, text = _report(week)

        assert "Ten tydzień" not in html
        assert "TEN TYDZIEŃ" not in text
