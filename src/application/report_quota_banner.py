"""Render bannera 🚨 Alerty subskrypcji / limitów na samej górze maila.

Banner pojawia się ZAWSZE przed sekcją sesji NYSE, gdy są jakiekolwiek
alerty (z bieżącego cyklu lub history 24h). Kolory:
- CRITICAL → czerwony banner
- WARNING  → żółty banner
"""

from __future__ import annotations

import html as _html

from src.domain.quota import QuotaAlert, QuotaSeverity

_SEVERITY_LABEL = {
    QuotaSeverity.WARNING: "⚠️ WARNING",
    QuotaSeverity.CRITICAL: "🔴 CRITICAL",
}
_SEVERITY_COLOR = {
    QuotaSeverity.WARNING: "#ca8a04",
    QuotaSeverity.CRITICAL: "#dc2626",
}
_SEVERITY_BG = {
    QuotaSeverity.WARNING: "#fefce8",
    QuotaSeverity.CRITICAL: "#fef2f2",
}


def render_quota_banner_html(alerts: list[QuotaAlert]) -> str:
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
            f"{a.occurred_at.strftime('%Y-%m-%d %H:%M UTC')}</td>"
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
        f"<table style='width: 100%; border-collapse: collapse; "
        f"font-size: 12px;'>"
        f"<tbody>{''.join(rows)}</tbody>"
        f"</table>"
        f"</div>"
    )


def render_quota_banner_text(alerts: list[QuotaAlert]) -> str:
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
            f"({a.occurred_at.strftime('%Y-%m-%d %H:%M UTC')})"
        )
        lines.append(f"    {a.message}")
        lines.append(f"    → {a.action}")
    lines.append("")
    return "\n".join(lines)
