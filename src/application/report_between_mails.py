"""ALR-06: sekcja ⚡ Między mailami — szoki z watcha od ostatniego maila.

Szok idzie pushem (`main_watch.py`), a `shock_alerts` czytał dotąd tylko watch do
debounce'u — kto nie ma Telegrama albo przegapił push, nie dowiadywał się z maila
o nocnym -6% na NVDA. Sekcja drukuje każdy alert od ostatniego dostarczonego maila:
symbol, ruch, kierunek, godzinę lokalną i bieżącą cenę z cyklu. Przy zapauzowanym
watchu alertów nie ma, więc sekcja chowa się sama.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

from src.application.ports import StoredShockAlert
from src.application.report_formatting import (
    delta_color,
    direction_pct,
    format_local,
    money,
    pct,
)
from src.application.report_formatting import escape as _html
from src.application.report_formatting import section_h2 as _section_h2
from src.application.reporting.section import ReportSection, SectionInput

_DIRECTION_PL = {"UP": "wzrost", "DOWN": "spadek"}
_NOTE = "Alerty szoku wysłane pushem od ostatniego maila (watch co godzinę)."


def _direction(alert: StoredShockAlert) -> str:
    return _DIRECTION_PL.get(alert.direction.upper(), alert.direction)


def _now_label(alert: StoredShockAlert, current_prices: Mapping[str, Decimal]) -> str:
    price = current_prices.get(alert.symbol)
    return f"teraz {money(price)}" if price is not None else ""


def _line(
    alert: StoredShockAlert, current_prices: Mapping[str, Decimal], display_timezone: str
) -> str:
    parts = [
        f"{alert.symbol} {pct(alert.delta, signed=True)} ({_direction(alert)})",
        format_local(alert.created_at, display_timezone),
        _now_label(alert, current_prices),
    ]
    return " · ".join(part for part in parts if part)


def render_between_mails_html(
    alerts: Sequence[StoredShockAlert],
    current_prices: Mapping[str, Decimal],
    display_timezone: str,
) -> str:
    """Tabela alertów szoku od ostatniego maila. "" bez alertów."""
    if not alerts:
        return ""
    rows = "".join(
        "<tr>"
        f"<td style='padding: 3px 8px 3px 0;'><strong>{_html(alert.symbol)}</strong></td>"
        f"<td style='padding: 3px 8px 3px 0; color: {delta_color(Decimal(str(alert.delta)))};'>"
        f"{direction_pct(alert.delta)} ({_html(_direction(alert))})</td>"
        f"<td style='padding: 3px 8px 3px 0; color: #6b7280;'>"
        f"{_html(format_local(alert.created_at, display_timezone))}</td>"
        f"<td style='padding: 3px 0;'>{_html(_now_label(alert, current_prices))}</td>"
        "</tr>"
        for alert in alerts
    )
    return (
        _section_h2("⚡ Między mailami")
        + f"<div style='font-size: 12px; color: #6b7280; margin: 0 0 6px 0;'>{_NOTE}</div>"
        + "<table role='presentation' style='border-collapse: collapse; font-size: 13px; "
        f"margin-bottom: 12px;'><tbody>{rows}</tbody></table>"
    )


def render_between_mails_text(
    alerts: Sequence[StoredShockAlert],
    current_prices: Mapping[str, Decimal],
    display_timezone: str,
) -> list[str]:
    """Te same alerty w wersji tekstowej. [] bez alertów."""
    if not alerts:
        return []
    return [
        "MIĘDZY MAILAMI",
        "-" * 64,
        *(f"  {_line(alert, current_prices, display_timezone)}" for alert in alerts),
        "",
    ]


def _section_html(data: SectionInput) -> str:
    return render_between_mails_html(
        data.context.shock_alerts or (), data.current_prices, data.context.display_timezone
    )


def _section_text(data: SectionInput) -> list[str]:
    return render_between_mails_text(
        data.context.shock_alerts or (), data.current_prices, data.context.display_timezone
    )


# ALR-06: tuż po pierwszym ekranie — to też „co się stało od ostatniego maila”.
BETWEEN_MAILS_SECTION = ReportSection(
    key="between_mails", render_html=_section_html, render_text=_section_text
)
