"""MAIL-21: sekcja 📡 Stan źródeł danych — jedna linijka na dostawcę.

Stoi tuż pod bannerem kwot i mówi, ile ufać dzisiejszemu mailowi: które źródło
działało, które się zdegradowało (ile symboli), które było niedostępne (otwarty
obwód HTTP) i które jest wyłączone flagą. Werdykty liczy `source_health`; sekcja
tylko je drukuje i chowa się, gdy żadne źródło nie ma problemu — wyłączenie
flagą to stan zamierzony, nie problem.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.application.report_formatting import PALETTE, plural_pl
from src.application.report_formatting import escape as _html
from src.application.report_formatting import section_h2 as _section_h2
from src.application.reporting.section import ReportSection, SectionInput
from src.application.source_health import SourceHealth, SourceStatus

__all__ = [
    "SOURCE_HEALTH_SECTION",
    "SourceHealth",
    "SourceStatus",
    "render_source_health_html",
    "render_source_health_text",
]

_TEXT_TITLE = "STAN ŹRÓDEŁ DANYCH"
_STATUS_COLOR = {
    SourceStatus.OK: PALETTE.positive,
    SourceStatus.DEGRADED: PALETTE.caution,
    SourceStatus.UNAVAILABLE: PALETTE.negative,
    SourceStatus.DISABLED: PALETTE.muted,
}


def _symbols(count: int) -> str:
    return f"{count} {plural_pl(count, 'symbol', 'symbole', 'symboli')}"


def _requests(count: int) -> str:
    noun = plural_pl(count, "błędna odpowiedź", "błędne odpowiedzi", "błędnych odpowiedzi")
    return f"{count} {noun}"


def status_label(row: SourceHealth) -> str:
    """„OK”, „zdegradowane (3 symbole)”, „niedostępne (otwarty obwód, …)”,
    „wyłączone flagą”."""
    if row.status is SourceStatus.UNAVAILABLE:
        skipped = plural_pl(row.affected, "pominięty", "pominięte", "pominiętych")
        return f"niedostępne (otwarty obwód, {_symbols(row.affected)} {skipped})"
    if row.status is SourceStatus.DEGRADED:
        details = [
            detail
            for detail, count in (
                (_symbols(row.affected), row.affected),
                (_requests(row.failed_requests), row.failed_requests),
            )
            if count
        ]
        return f"zdegradowane ({', '.join(details)})" if details else "zdegradowane"
    if row.status is SourceStatus.DISABLED:
        return "wyłączone flagą"
    return "OK"


def render_source_health_html(rows: Sequence[SourceHealth]) -> str:
    """Tabela dwóch kolumn (dostawca, stan). "" gdy żadne źródło nie ma problemu."""
    if not any(row.is_problem for row in rows):
        return ""
    body = "".join(
        "<tr>"
        f"<td style='padding: 2px 8px 2px 0; font-weight: 600;'>{_html(row.source)}</td>"
        f"<td style='padding: 2px 0; color: {_STATUS_COLOR[row.status]};'>"
        f"{_html(status_label(row))}</td>"
        "</tr>"
        for row in rows
    )
    return (
        _section_h2("📡 Stan źródeł danych")
        + "<table role='presentation' style='border-collapse: collapse; font-size: 12px; "
        f"margin-bottom: 12px;'><tbody>{body}</tbody></table>"
    )


def render_source_health_text(rows: Sequence[SourceHealth]) -> list[str]:
    """Te same linijki w wersji tekstowej. [] gdy żadne źródło nie ma problemu."""
    if not any(row.is_problem for row in rows):
        return []
    return [
        _TEXT_TITLE,
        "-" * 64,
        *(f"  {row.source}: {status_label(row)}" for row in rows),
        "",
    ]


def _section_html(data: SectionInput) -> str:
    return render_source_health_html(data.context.source_health or ())


def _section_text(data: SectionInput) -> list[str]:
    return render_source_health_text(data.context.source_health or ())


# MAIL-21: tuż pod bannerem kwot — awaria źródła waży tyle, co wyczerpany klucz.
SOURCE_HEALTH_SECTION = ReportSection(
    key="source_health",
    render_html=_section_html,
    render_text=_section_text,
    context_fields=frozenset({"source_health"}),
)
