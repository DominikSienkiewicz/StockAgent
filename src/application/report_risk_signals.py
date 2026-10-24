"""Sekcja 🚨 Sygnały ostrzegawcze — rozbieżność cena ↔ sentyment, niezgodność
AV ↔ LLM i słaby sygnał (`detect_risk_signals`)."""

from __future__ import annotations

from src.application.report_formatting import (
    PALETTE,
)
from src.application.report_formatting import (
    company_label as _company_label,
)
from src.application.report_formatting import (
    escape as _html,
)
from src.application.report_models import RiskSignal
from src.application.report_signals import detect_risk_signals
from src.application.reporting.section import ReportSection, SectionInput


def _render_risk_signals_html(risk_signals: list[RiskSignal]) -> str:
    """🚨 Sygnały ostrzegawcze. Pusty string, gdy brak sygnałów."""
    if not risk_signals:
        return ""
    parts = [
        "<h2 style='font-size: 16px; margin: 20px 0 8px 0; color: #991b1b;'>"
        "🚨 Sygnały ostrzegawcze</h2>"
    ]
    type_label = {
        "DIVERGENCE": "Rozbieżność cena ↔ sentyment",
        "AV_LLM_CONFLICT": "Niezgodność AV ↔ LLM",
        "LOW_SIGNAL": "Słaby sygnał",
    }
    for rs in risk_signals:
        sev_color = {
            "high": PALETTE.negative,
            "medium": PALETTE.caution,
            "low": PALETTE.muted,
        }[rs.severity]
        parts.append(f"""
              <div style="margin-bottom: 6px; padding: 8px 12px; background: #fffbeb;
                          border-left: 3px solid {sev_color}; border-radius: 4px;
                          font-size: 12px;">
                <strong>{_html(_company_label(rs.symbol))}</strong>
                <span style="color: {sev_color}; font-weight: 600;">
                  · {type_label.get(rs.type, _html(rs.type))}
                </span><br/>
                <span style="color: #4b5563;">{_html(rs.description)}</span>
              </div>
            """)
    return "".join(parts)


def _render_risk_signals_text(risk_signals: list[RiskSignal]) -> list[str]:
    """Sekcja 'Sygnały ostrzegawcze' (plain). Pusta lista, gdy brak."""
    if not risk_signals:
        return []
    lines = ["SYGNAŁY OSTRZEGAWCZE", "-" * 64]
    type_label = {
        "DIVERGENCE": "Rozbieżność",
        "AV_LLM_CONFLICT": "Niezgodność AV↔LLM",
        "LOW_SIGNAL": "Słaby sygnał",
    }
    for rs in risk_signals:
        lines.append(
            f"  ⚠ {_company_label(rs.symbol)}: "
            f"[{type_label.get(rs.type, rs.type)}] {rs.description}"
        )
    lines.append("")
    return lines


def _section_html(data: SectionInput) -> str:
    return _render_risk_signals_html(detect_risk_signals(data.results))


def _section_text(data: SectionInput) -> list[str]:
    return _render_risk_signals_text(detect_risk_signals(data.results))


RISK_SIGNALS_SECTION = ReportSection(
    key="risk_signals", render_html=_section_html, render_text=_section_text
)
