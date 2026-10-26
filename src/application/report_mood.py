"""Sekcja 📊 Nastroje watchlisty — sentyment cyklu w Zapleczu.

Średni sentyment, skrajne symbole, predykcje z wysoką pewnością i liczba newsów.
MAIL-10: cykl bez zapisanych predykcji nie ma sekcji, a średnia bez ani jednego
sentymentu to „brak danych”, nie „+0,00 (Neutralny)”.
"""

from __future__ import annotations

from src.application.report_formatting import (
    company_label as _company_label,
)
from src.application.report_formatting import (
    escape as _html,
)
from src.application.report_formatting import (
    number as _number,
)
from src.application.report_models import PortfolioMood
from src.application.report_signals import build_portfolio_mood
from src.application.reporting.section import ReportSection, SectionInput

# MAIL-10: średnia bez ani jednego sentymentu to brak danych, nie „+0,00 (Neutralny)”.
_NO_SENTIMENT_LABEL = "brak danych"


def _mood_average(mood: PortfolioMood) -> tuple[str, str]:
    """(średni sentyment, etykieta) — „—” i „brak danych”, gdy cykl nie ma sentymentu."""
    if mood.avg_sentiment is None or mood.avg_sentiment_label is None:
        return "—", _NO_SENTIMENT_LABEL
    average = _number(mood.avg_sentiment, decimals=2, signed=True)
    return average, mood.avg_sentiment_label


def _render_mood_html(mood: PortfolioMood) -> str:
    """📊 Nastroje watchlisty — sentyment cyklu w Zapleczu. MAIL-10: cykl bez
    zapisanych predykcji (dzień zdegradowany, same pominięte) nie ma boxu."""
    if mood.saved_count == 0:
        return ""
    average, average_label = _mood_average(mood)
    bullish_part = (
        f"<strong>{_html(_company_label(mood.most_bullish.symbol))}</strong> "
        f"({_number(mood.most_bullish.sentiment_score, decimals=2, signed=True)})"
        if mood.most_bullish
        else "—"
    )
    bearish_part = (
        f"<strong>{_html(_company_label(mood.most_bearish.symbol))}</strong> "
        f"({_number(mood.most_bearish.sentiment_score, decimals=2, signed=True)})"
        if mood.most_bearish
        else "—"
    )
    high_confidence = f"{mood.high_confidence_count} / {mood.saved_count}"
    return f"""
      <div style="padding: 12px 14px; background: #eff6ff; border-left: 3px solid #2563eb;
                  border-radius: 4px; margin-bottom: 20px; font-size: 13px;">
        <div style="font-weight: 600; margin-bottom: 6px; color: #1e3a8a;">
          📊 Nastroje watchlisty
        </div>
        <div style="color: #1e40af; line-height: 1.7;">
          Średni sentyment: <strong>{average}</strong>
          ({average_label})<br/>
          Najbardziej pozytywny: {bullish_part} · Najbardziej negatywny: {bearish_part}<br/>
          Predykcje z wysoką pewnością (≥70%): {high_confidence}
          · Łącznie newsów: {mood.total_news}
        </div>
      </div>
    """


def _render_mood_text(mood: PortfolioMood) -> list[str]:
    """Sekcja 'Nastroje watchlisty' (plain) — w Zapleczu, jak w HTML (MAIL-10: bez
    zapisanych predykcji sekcji nie ma)."""
    if mood.saved_count == 0:
        return []
    average, average_label = _mood_average(mood)
    lines = [
        "NASTROJE WATCHLISTY",
        "-" * 64,
        f"  Średni sentyment: {average} ({average_label})",
    ]
    if mood.most_bullish:
        mb = mood.most_bullish
        lines.append(
            f"  Najbardziej pozytywny: {_company_label(mb.symbol)} "
            f"({_number(mb.sentiment_score, decimals=2, signed=True)})"
        )
    if mood.most_bearish:
        mbe = mood.most_bearish
        lines.append(
            f"  Najbardziej negatywny: {_company_label(mbe.symbol)} "
            f"({_number(mbe.sentiment_score, decimals=2, signed=True)})"
        )
    lines.append(
        f"  Wysoka pewność (≥70%): {mood.high_confidence_count} / {mood.saved_count}"
        f"  ·  Newsów: {mood.total_news}"
    )
    lines.append("")
    return lines


def _section_html(data: SectionInput) -> str:
    return _render_mood_html(build_portfolio_mood(data.results))


def _section_text(data: SectionInput) -> list[str]:
    return _render_mood_text(build_portfolio_mood(data.results))


MOOD_SECTION = ReportSection(key="mood", render_html=_section_html, render_text=_section_text)
