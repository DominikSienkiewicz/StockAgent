"""Sekcja 📅 Ten tydzień — wyniki spółek z watchlisty do niedzieli (DAT-20).

Raport kwartalny to najczęstsza przyczyna szoku na pojedynczym tickerze, a mail
dotąd o nim milczał (bramka earnings jest off). W poniedziałek, z flagą
`finnhub_earnings_calendar_enabled`, DI dokłada `ReportContext.earnings_this_week`
z kalendarza Finnhub (jedno zapytanie na watchlistę): dzień, pora względem sesji
i szacunek EPS. Pusta lista albo None — sekcji nie ma.
"""

from __future__ import annotations

import html as _html
from datetime import date

from src.application.report_formatting import company_label, number, weekday_day_label
from src.application.report_formatting import section_h2 as _section_h2
from src.application.reporting.section import ReportSection, SectionInput
from src.domain.earnings import EarningsEvent, earnings_timing_label


def _event_line(event: EarningsEvent) -> str | None:
    """„śr 28.10 — MSFT (Microsoft Corporation) · po sesji · szac. EPS 3,10”;
    nieznana pora i brak szacunku po prostu nie wchodzą do linii."""
    if event.report_date is None:
        return None
    parts = [f"{weekday_day_label(date.fromisoformat(event.report_date))} — "
             f"{company_label(event.symbol)}"]
    if event.timing is not None:
        parts.append(earnings_timing_label(event.timing))
    if event.eps_estimate is not None:
        parts.append(f"szac. EPS {number(event.eps_estimate, decimals=2)}")
    return " · ".join(parts)


def _lines(data: SectionInput) -> list[str]:
    events = data.context.earnings_this_week or ()
    return [line for line in map(_event_line, events) if line is not None]


def _section_html(data: SectionInput) -> str:
    lines = _lines(data)
    if not lines:
        return ""
    items = "".join(
        f"<li style='margin: 2px 0;'>{_html.escape(line)}</li>" for line in lines
    )
    return (
        _section_h2("📅 Ten tydzień")
        + "<p style='font-size: 13px; color: #4b5563; margin: 0 0 6px 0;'>"
        "Wyniki spółek z watchlisty do niedzieli (kalendarz Finnhub).</p>"
        f"<ul style='font-size: 13px; margin: 0 0 12px 0; padding-left: 18px;'>{items}</ul>"
    )


def _section_text(data: SectionInput) -> list[str]:
    lines = _lines(data)
    if not lines:
        return []
    return ["TEN TYDZIEŃ — WYNIKI SPÓŁEK", "-" * 64, *(f"  {line}" for line in lines), ""]


THIS_WEEK_SECTION = ReportSection(
    key="this_week",
    render_html=_section_html,
    render_text=_section_text,
    context_fields=frozenset({"earnings_this_week"}),
)
