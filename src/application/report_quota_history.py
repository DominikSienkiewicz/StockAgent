"""Neutralny zapis wcześniejszych alertów, bez wnioskowania o obecnej awarii."""

from __future__ import annotations

import html
from dataclasses import dataclass

from src.application.report_formatting import DEFAULT_DISPLAY_TIMEZONE, PALETTE, format_local
from src.domain.quota import QuotaAlert, QuotaSeverity

# NIEZ-01: tydzień + 3 h zachowuje niedzielne alerty Slow Loopa do kolejnego maila.
QUOTA_ALERT_LOOKBACK_HOURS = 7 * 24 + 3
_TITLE = f"HISTORIA ALERTÓW (ostatnie {QUOTA_ALERT_LOOKBACK_HOURS} h)"
_NOTICE = "Wpisy spoza bieżącego cyklu. Status tych zdarzeń wymaga osobnej weryfikacji."
_SEVERITY_LABEL = {
    QuotaSeverity.WARNING: "OSTRZEŻENIE",
    QuotaSeverity.CRITICAL: "KRYTYCZNY",
}


@dataclass(frozen=True)
class _AlertGroup:
    latest: QuotaAlert
    count: int

    def summary(self, display_timezone: str) -> str:
        return (
            f"Poziom w chwili zdarzenia: {_SEVERITY_LABEL[self.latest.severity]} · "
            f"Liczba zdarzeń: {self.count} · "
            f"ostatnio {format_local(self.latest.occurred_at, display_timezone)}"
        )


def _groups(alerts: list[QuotaAlert]) -> list[_AlertGroup]:
    grouped: dict[tuple[str, QuotaSeverity, str], list[QuotaAlert]] = {}
    for alert in alerts:
        grouped.setdefault((alert.source, alert.severity, alert.message), []).append(alert)
    result = [
        _AlertGroup(latest=max(items, key=lambda alert: alert.occurred_at), count=len(items))
        for items in grouped.values()
    ]
    return sorted(result, key=lambda group: group.latest.occurred_at, reverse=True)


def render_quota_history_html(
    alerts: list[QuotaAlert], display_timezone: str = DEFAULT_DISPLAY_TIMEZONE
) -> str:
    if not alerts:
        return ""
    rows = []
    for group in _groups(alerts):
        alert = group.latest
        rows.append(
            "<div style='margin-top: 8px;'>"
            f"<div style='font-weight: 600;'>{html.escape(alert.source)}: "
            f"{html.escape(alert.message)}</div>"
            f"<div>{html.escape(group.summary(display_timezone))}</div>"
            f"<div>Zalecenie zapisane przy zdarzeniu: {html.escape(alert.action)}</div>"
            "</div>"
        )
    return (
        f"<div style='padding: 12px 16px; background: {PALETTE.surface}; "
        f"color: {PALETTE.muted}; border-left: 4px solid {PALETTE.muted}; "
        "border-radius: 4px; margin-bottom: 16px; font-size: 12px;'>"
        f"<div style='font-size: 14px; font-weight: 600;'>{_TITLE}</div>"
        f"<div style='margin-top: 4px;'>{_NOTICE}</div>{''.join(rows)}</div>"
    )


def render_quota_history_text(
    alerts: list[QuotaAlert], display_timezone: str = DEFAULT_DISPLAY_TIMEZONE
) -> str:
    if not alerts:
        return ""
    lines = [f"=== {_TITLE} ===", _NOTICE]
    for group in _groups(alerts):
        alert = group.latest
        lines.extend(
            [
                f"  {alert.source}: {alert.message}",
                f"    {group.summary(display_timezone)}",
                f"    Zalecenie zapisane przy zdarzeniu: {alert.action}",
            ]
        )
    return "\n".join([*lines, ""])
