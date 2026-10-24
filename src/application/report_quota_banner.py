"""Render bannera 🚨 Alerty subskrypcji / limitów na samej górze maila.

Banner pojawia się ZAWSZE przed sekcją sesji NYSE, gdy są jakiekolwiek
alerty (z bieżącego cyklu lub history 24h). Czas alertu jest w strefie wyświetlania
maila (`display_timezone`, MAIL-59), jak nagłówek i temat. Kolory:
- CRITICAL → czerwony banner
- WARNING  → żółty banner
"""

from __future__ import annotations

import html as _html

from src.application.report_formatting import DEFAULT_DISPLAY_TIMEZONE, format_local
from src.application.reporting.section import ReportSection, SectionInput, text_block
from src.domain.quota import QuotaAlert, QuotaSeverity

# MAIL-15: banner jest treścią maila, więc po polsku; nazwy poziomów w logach
# i w bazie (`QuotaSeverity.value`) zostają angielskie.
_SEVERITY_LABEL = {
    QuotaSeverity.WARNING: "⚠️ OSTRZEŻENIE",
    QuotaSeverity.CRITICAL: "🔴 KRYTYCZNY",
}
_SEVERITY_COLOR = {
    QuotaSeverity.WARNING: "#ca8a04",
    QuotaSeverity.CRITICAL: "#dc2626",
}
_SEVERITY_BG = {
    QuotaSeverity.WARNING: "#fefce8",
    QuotaSeverity.CRITICAL: "#fef2f2",
}


def render_quota_banner_html(
    alerts: list[QuotaAlert], display_timezone: str = DEFAULT_DISPLAY_TIMEZONE
) -> str:
    if not alerts:
        return ""

    # Dedup po (source, severity, message) — gdy ten sam alert wystawiony
    # i w bieżącym cyklu, i w history z bazy.
    seen: set[tuple[str, str, str]] = set()
    deduped: list[QuotaAlert] = []
    for a in alerts:
        key = (a.source, a.severity.value, a.message)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(a)

    # Sortuj: CRITICAL przed WARNING; w obrębie severity malejąco po dacie.
    deduped.sort(key=lambda a: (-a.severity.rank, -a.occurred_at.timestamp()))

    highest = max(deduped, key=lambda a: a.severity.rank).severity
    main_color = _SEVERITY_COLOR[highest]
    main_bg = _SEVERITY_BG[highest]

    rows: list[str] = []
    for a in deduped:
        rows.append(
            f"<tr>"
            f"<td style='padding: 6px 8px; white-space: nowrap; "
            f"color: {_SEVERITY_COLOR[a.severity]}; font-weight: 600;'>"
            f"{_SEVERITY_LABEL[a.severity]}</td>"
            f"<td style='padding: 6px 8px; white-space: nowrap; "
            f"font-weight: 600;'>{_html.escape(a.source)}</td>"
            f"<td style='padding: 6px 8px; color: #6b7280;'>"
            f"{format_local(a.occurred_at, display_timezone)}</td>"
            f"<td style='padding: 6px 8px;'>"
            f"<div style='font-weight: 500;'>{_html.escape(a.message)}</div>"
            f"<div style='color: #6b7280; font-size: 12px; margin-top: 2px;'>"
            f"→ {_html.escape(a.action)}</div></td>"
            f"</tr>"
        )

    return (
        f"<div style='padding: 12px 16px; background: {main_bg}; "
        f"border-left: 4px solid {main_color}; border-radius: 4px; "
        f"margin-bottom: 16px;'>"
        f"<div style='font-size: 14px; font-weight: 600; color: {main_color}; "
        f"margin-bottom: 8px;'>"
        f"🚨 ALERTY SUBSKRYPCJI / LIMITÓW ({len(deduped)})"
        f"</div>"
        f"<table role='presentation' style='width: 100%; border-collapse: collapse; "
        f"font-size: 12px;'>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table>"
        f"</div>"
    )


def render_quota_banner_text(
    alerts: list[QuotaAlert], display_timezone: str = DEFAULT_DISPLAY_TIMEZONE
) -> str:
    if not alerts:
        return ""

    seen: set[tuple[str, str, str]] = set()
    deduped: list[QuotaAlert] = []
    for a in alerts:
        key = (a.source, a.severity.value, a.message)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(a)
    deduped.sort(key=lambda a: (-a.severity.rank, -a.occurred_at.timestamp()))

    lines = [
        f"=== 🚨 ALERTY SUBSKRYPCJI / LIMITÓW ({len(deduped)}) ===",
    ]
    for a in deduped:
        lines.append(
            f"  [{_SEVERITY_LABEL[a.severity]}] {a.source}  "
            f"({format_local(a.occurred_at, display_timezone)})"
        )
        lines.append(f"    {a.message}")
        lines.append(f"    → {a.action}")
    lines.append("")
    return "\n".join(lines)


def _section_html(data: SectionInput) -> str:
    return render_quota_banner_html(data.context.quota_alerts or [], data.context.display_timezone)


def _section_text(data: SectionInput) -> list[str]:
    return text_block(
        render_quota_banner_text(data.context.quota_alerts or [], data.context.display_timezone)
    )


# Banner idzie NAJWYŻEJ: wyczerpany płatny klucz waży więcej niż ruch ceny.
QUOTA_BANNER_SECTION = ReportSection(
    key="quota_banner", render_html=_section_html, render_text=_section_text
)
