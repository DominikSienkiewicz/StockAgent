"""Buduje raport HTML+text z wyników cyklu Fast Loop.

Czysta logika prezentacji — bez I/O, łatwo testowalne.
"""

from __future__ import annotations

import html
import urllib.parse
from datetime import datetime
from decimal import Decimal
from typing import Any

from src.application.report_charts import (
    build_chart_url,
    build_correlation_chart_url,
    build_forecast_chart_url,
)
from src.application.report_formatting import (
    company_label as _company_label,
)
from src.application.report_formatting import (
    delta_color as _delta_color,
)
from src.application.report_formatting import (
    money as _money,
)
from src.application.report_formatting import (
    pct as _pct,
)
from src.application.report_formatting import (
    sentiment_label as _sentiment_label,
)
from src.application.report_formatting import (
    trend_color as _trend_color,
)
from src.application.report_formatting import (
    trend_label as _trend_label,
)
from src.application.report_models import (
    ResolvedPrediction,
    RiskSignal,
    SymbolResult,
    TopNewsItem,
    TradeSignal,
)
from src.application.report_signals import (
    build_portfolio_mood,
    build_trade_signals,
    detect_risk_signals,
    market_status,
    parse_resolved_predictions,
)
from src.domain.council import CouncilVerdict

__all__ = [
    "ResolvedPrediction",
    "RiskSignal",
    "SymbolResult",
    "TopNewsItem",
    "TradeSignal",
    "build_chart_url",
    "build_correlation_chart_url",
    "build_forecast_chart_url",
    "build_html_report",
    "build_portfolio_mood",
    "build_trade_signals",
    "detect_risk_signals",
    "market_status",
    "parse_resolved_predictions",
    "to_symbol_result",
]

_SAFE_URL_SCHEMES = {"http", "https"}

_DIRECTION_LABEL = {
    "BULLISH": "KUP",
    "BEARISH": "SPRZEDAJ",
    "SIDEWAYS": "WSTRZYMAJ",
}

_DIRECTION_COLOR = {
    "KUP": "#16a34a",
    "SPRZEDAJ": "#dc2626",
    "WSTRZYMAJ": "#737373",
}

_DIRECTION_BG = {
    "KUP": "#f0fdf4",
    "SPRZEDAJ": "#fef2f2",
    "WSTRZYMAJ": "#f9fafb",
}

_COUNCIL_REC_LABEL = {"BUY": "KUP", "SELL": "SPRZEDAJ", "HOLD": "TRZYMAJ"}
_COUNCIL_REC_COLOR = {"BUY": "#16a34a", "SELL": "#dc2626", "HOLD": "#ca8a04"}
_COUNCIL_REC_BG = {"BUY": "#f0fdf4", "SELL": "#fef2f2", "HOLD": "#fefce8"}
_COUNCIL_REC_EMOJI = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}


def _recommendation(r: SymbolResult) -> str:
    return _DIRECTION_LABEL.get(r.trend or "", "WSTRZYMAJ")


def _recommendation_reason_html(r: SymbolResult) -> str:
    """Zwraca krótki HTML-owy powód rekomendacji (1–2 zdania)."""
    direction = _recommendation(r)
    color = _DIRECTION_COLOR.get(direction, "#737373")
    bg = _DIRECTION_BG.get(direction, "#f9fafb")

    conf_part = (
        f"Pewność: <strong>{r.confidence_score * 100:.0f}%</strong>."
        if r.confidence_score is not None
        else ""
    )
    change_part = ""
    if r.expected_change is not None:
        change_part = (
            f" Prognozowana zmiana: "
            f"<strong style='color: {_delta_color(r.expected_change)};'>"
            f"{_pct(r.expected_change, signed=True)}</strong>."
        )
    sentiment_part = ""
    if r.sentiment_score is not None:
        if r.sentiment_score > 0.2:
            sentiment_part = " Sentyment rynkowy pozytywny."
        elif r.sentiment_score < -0.2:
            sentiment_part = " Sentyment rynkowy negatywny."
        else:
            sentiment_part = " Sentyment neutralny."

    return (
        f"<div style='margin-top: 6px; padding: 6px 10px; background: {bg}; "
        f"border-radius: 3px; font-size: 12px;'>"
        f"<span style='font-weight: 700; color: {color};'>➤ {direction}</span> — "
        f"{conf_part}{change_part}{sentiment_part}"
        f"</div>"
    )


def _recommendation_reason_text(r: SymbolResult) -> str:
    direction = _recommendation(r)
    conf_part = f"pewność {r.confidence_score * 100:.0f}%" if r.confidence_score is not None else ""
    change_part = (
        f"prognoza {_pct(r.expected_change, signed=True)}"
        if r.expected_change is not None
        else ""
    )
    parts = [p for p in [conf_part, change_part] if p]
    detail = ", ".join(parts)
    return f"Rekomendacja: {direction}" + (f" ({detail})" if detail else "")


def _html(value: object) -> str:
    return html.escape(str(value), quote=True)


def _safe_href(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in _SAFE_URL_SCHEMES or not parsed.netloc:
        return None
    return _html(url)


def _render_council_section(verdict: CouncilVerdict) -> str:
    final = verdict.final_recommendation
    label = _COUNCIL_REC_LABEL.get(final, final)
    color = _COUNCIL_REC_COLOR.get(final, "#737373")
    bg = _COUNCIL_REC_BG.get(final, "#f9fafb")
    strength_pct = int(verdict.consensus_strength * 100)

    rows = ""
    for op in verdict.investor_opinions:
        op_label = _COUNCIL_REC_LABEL.get(op.recommendation, op.recommendation)
        op_emoji = _COUNCIL_REC_EMOJI.get(op.recommendation, "⬜")
        op_color = _COUNCIL_REC_COLOR.get(op.recommendation, "#737373")
        op_bg = _COUNCIL_REC_BG.get(op.recommendation, "#f9fafb")
        factors = ", ".join(op.key_factors[:3])
        rows += (
            f'<tr style="background:{op_bg}">'
            f'<td style="padding:6px 10px;font-weight:500">{html.escape(op.investor_name)}</td>'
            f'<td style="padding:6px 10px;color:{op_color};font-weight:700">'
            f'{op_emoji} {html.escape(op_label)}</td>'
            f'<td style="padding:6px 10px;text-align:center">{int(op.confidence * 100)}%</td>'
            f'<td style="padding:6px 10px;font-size:0.85em;color:#555">{html.escape(factors)}</td>'
            f'</tr>'
        )

    dissent = ""
    if verdict.dissenting_views:
        views = "; ".join(html.escape(v) for v in verdict.dissenting_views)
        dissent = (
            f'<tr><td colspan="4" style="padding:8px 10px;font-size:0.85em;'
            f'color:#92400e;background:#fffbeb">'
            f'⚠️ Głosy niezgodne: {views}</td></tr>'
        )

    return (
        f'<div style="margin-top:16px;border:1px solid #e5e7eb;border-radius:8px;overflow:hidden">'
        f'<div style="background:{bg};padding:12px 16px;border-bottom:1px solid #e5e7eb">'
        f'<span style="font-weight:700;font-size:1.05em">RADA DORADCZA</span>'
        f'&nbsp;&nbsp;'
        f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:12px;'
        f'font-weight:700">{label}</span>'
        f'&nbsp;&nbsp;'
        f'<span style="color:#555;font-size:0.9em">Zgodność: {strength_pct}%</span>'
        f'<div style="margin-top:6px;font-size:0.9em;color:#374151;font-style:italic">'
        f'&ldquo;{html.escape(verdict.summary)}&rdquo;</div>'
        f'</div>'
        f'<table style="width:100%;border-collapse:collapse;font-size:0.9em">'
        f'<thead><tr style="background:#f3f4f6">'
        f'<th style="padding:6px 10px;text-align:left;font-weight:600">Inwestor</th>'
        f'<th style="padding:6px 10px;text-align:left;font-weight:600">Opinia</th>'
        f'<th style="padding:6px 10px;text-align:center;font-weight:600">Pewność</th>'
        f'<th style="padding:6px 10px;text-align:left;font-weight:600">Kluczowe czynniki</th>'
        f'</tr></thead>'
        f'<tbody>{rows}{dissent}</tbody>'
        f'</table>'
        f'</div>'
    )


def build_html_report(
    results: list[SymbolResult],
    started_at: datetime,
    duration_seconds: float,
    accuracy_stats: dict[str, Any] | None = None,
    resolved_predictions: list[ResolvedPrediction] | None = None,
) -> tuple[str, str]:
    """Zwraca (html_body, plain_text) — oba reprezentacje raportu.

    Parametry opcjonalne:
        accuracy_stats: wynik `RepositoryPort.get_accuracy_stats(days)`.
        resolved_predictions: zamknięte predykcje z ostatnich N godzin
            (do sekcji day-over-day).
    """
    saved = [r for r in results if r.status == "saved"]
    ignored = [r for r in results if r.status == "ignored"]
    errors = [r for r in results if r.status == "error"]
    mood = build_portfolio_mood(results)
    session = market_status(started_at)
    trade_signals = build_trade_signals(results)
    risk_signals = detect_risk_signals(results)

    html = _render_html(
        results, saved, ignored, errors, started_at, duration_seconds,
        mood, session, accuracy_stats, trade_signals, risk_signals,
        resolved_predictions or [],
    )
    text = _render_plain(
        results, saved, ignored, errors, started_at, duration_seconds,
        mood, session, accuracy_stats, trade_signals, risk_signals,
        resolved_predictions or [],
    )
    return html, text


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


def _render_html(
    results: list[SymbolResult],
    saved: list[SymbolResult],
    ignored: list[SymbolResult],
    errors: list[SymbolResult],
    started_at: datetime,
    duration_seconds: float,
    mood: dict[str, Any],
    session: dict[str, str],
    accuracy_stats: dict[str, Any] | None,
    trade_signals: list[TradeSignal],
    risk_signals: list[RiskSignal],
    resolved_predictions: list[ResolvedPrediction],
) -> str:
    sections = []

    # Header
    sections.append(f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; max-width: 720px; margin: 0 auto; color: #1f2937;">
      <h1 style="font-size: 22px; margin: 0 0 4px 0;">📊 StockAgent — raport cyklu</h1>
      <div style="font-size: 13px; color: #6b7280; margin-bottom: 24px;">
        Czas rozpoczęcia: {started_at.strftime("%Y-%m-%d %H:%M:%S UTC")}
        · Czas trwania: {duration_seconds:.1f}s
        · Liczba symboli: {len(results)}
      </div>
    """)

    # Status sesji giełdowej
    sections.append(f"""
      <div style="padding: 10px 14px; background: #f9fafb; border-radius: 4px;
                  margin-bottom: 16px; font-size: 13px;">
        <strong>Sesja NYSE:</strong> {session["label"]}
        <span style="color: #6b7280;">· {session["detail"]}</span>
      </div>
    """)

    # Summary boxes
    sections.append(f"""
      <div style="display: flex; gap: 8px; margin-bottom: 16px;">
        <div style="flex: 1; padding: 12px; background: #f0fdf4; border-left: 3px solid #16a34a; border-radius: 4px;">
          <div style="font-size: 11px; color: #15803d; text-transform: uppercase;">Predykcje</div>
          <div style="font-size: 24px; font-weight: 600; color: #14532d;">{len(saved)}</div>
        </div>
        <div style="flex: 1; padding: 12px; background: #f9fafb; border-left: 3px solid #9ca3af; border-radius: 4px;">
          <div style="font-size: 11px; color: #4b5563; text-transform: uppercase;">Pominięte</div>
          <div style="font-size: 24px; font-weight: 600; color: #1f2937;">{len(ignored)}</div>
        </div>
        <div style="flex: 1; padding: 12px; background: {'#fef2f2' if errors else '#f9fafb'}; border-left: 3px solid {'#dc2626' if errors else '#9ca3af'}; border-radius: 4px;">
          <div style="font-size: 11px; color: {'#991b1b' if errors else '#4b5563'}; text-transform: uppercase;">Błędy</div>
          <div style="font-size: 24px; font-weight: 600; color: {'#7f1d1d' if errors else '#1f2937'};">{len(errors)}</div>
        </div>
      </div>
    """)

    # 🎯 Trade ideas (najsilniejsze sygnały transakcyjne)
    if trade_signals:
        sections.append(
            "<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>"
            "🎯 Najsilniejsze sygnały</h2>"
        )
        for sig in trade_signals:
            dir_color = {
                "KUP": "#16a34a",
                "SPRZEDAJ": "#dc2626",
                "OBSERWUJ": "#737373",
            }[sig.direction]
            dir_bg = {
                "KUP": "#f0fdf4",
                "SPRZEDAJ": "#fef2f2",
                "OBSERWUJ": "#f9fafb",
            }[sig.direction]
            change_color = _delta_color(sig.expected_change)
            sections.append(f"""
              <div style="margin-bottom: 8px; padding: 10px 14px; background: {dir_bg};
                          border-left: 4px solid {dir_color}; border-radius: 4px;
                          font-size: 13px; display: flex; gap: 12px; align-items: center;">
                <span style="font-weight: 700; color: {dir_color}; min-width: 90px;">
                  {sig.direction}
                </span>
                <span style="font-weight: 600; min-width: 60px;">{_html(_company_label(sig.symbol))}</span>
                <span style="color: #4b5563;">
                  pewność <strong>{sig.confidence * 100:.0f}%</strong>
                </span>
                <span style="color: {change_color};">
                  prognoza <strong>{_pct(sig.expected_change, signed=True)}</strong>
                </span>
                <span style="color: #6b7280; margin-left: auto; font-size: 12px;">
                  siła sygnału: <strong>{sig.strength:.2f}</strong>
                </span>
              </div>
            """)

    # 🚨 Risk signals (anomalie / niespójności)
    if risk_signals:
        sections.append(
            "<h2 style='font-size: 16px; margin: 20px 0 8px 0; color: #991b1b;'>"
            "🚨 Sygnały ostrzegawcze</h2>"
        )
        type_label = {
            "DIVERGENCE": "Rozbieżność cena ↔ sentyment",
            "AV_LLM_CONFLICT": "Niezgodność AV ↔ LLM",
            "LOW_SIGNAL": "Słaby sygnał",
        }
        for rs in risk_signals:
            sev_color = {"high": "#dc2626", "medium": "#f59e0b", "low": "#6b7280"}[rs.severity]
            sections.append(f"""
              <div style="margin-bottom: 6px; padding: 8px 12px; background: #fffbeb;
                          border-left: 3px solid {sev_color}; border-radius: 4px;
                          font-size: 12px;">
                <strong>{_html(_company_label(rs.symbol))}</strong>
                <span style="color: {sev_color}; font-weight: 600;">
                  · {type_label.get(rs.type, rs.type)}
                </span><br/>
                <span style="color: #4b5563;">{_html(rs.description)}</span>
              </div>
            """)

    # Portfolio mood box
    bullish_part = (
        f"<strong>{_html(_company_label(mood['most_bullish'].symbol))}</strong> "
        f"({mood['most_bullish'].sentiment_score:+.2f})"
        if mood["most_bullish"] else "—"
    )
    bearish_part = (
        f"<strong>{_html(_company_label(mood['most_bearish'].symbol))}</strong> "
        f"({mood['most_bearish'].sentiment_score:+.2f})"
        if mood["most_bearish"] else "—"
    )
    sections.append(f"""
      <div style="padding: 12px 14px; background: #eff6ff; border-left: 3px solid #2563eb;
                  border-radius: 4px; margin-bottom: 20px; font-size: 13px;">
        <div style="font-weight: 600; margin-bottom: 6px; color: #1e3a8a;">
          📊 Nastroje portfela
        </div>
        <div style="color: #1e40af; line-height: 1.7;">
          Średni sentyment: <strong>{mood['avg_sentiment']:+.2f}</strong>
          ({mood['avg_sentiment_label']})<br/>
          Najbardziej pozytywny: {bullish_part} · Najbardziej negatywny: {bearish_part}<br/>
          Predykcje z wysoką pewnością (≥70%): {mood['high_confidence_count']} / {mood['saved_count']}
          · Łącznie newsów: {mood['total_news']}
        </div>
      </div>
    """)

    # 📊 Day-over-day — zamknięte predykcje z ostatnich 24h
    if resolved_predictions:
        correct = [p for p in resolved_predictions if p.is_correct]
        wrong = [p for p in resolved_predictions if not p.is_correct]
        sections.append(
            "<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>"
            "📊 Zamknięte predykcje (ostatnie 24h)</h2>"
        )
        for p in resolved_predictions:
            mark = "✅" if p.is_correct else "❌"
            color = "#16a34a" if p.is_correct else "#dc2626"
            sections.append(f"""
              <div style="margin-bottom: 4px; padding: 6px 10px; background: #fafafa;
                          border-left: 3px solid {color}; border-radius: 4px;
                          font-size: 12px;">
                {mark} <strong>{_html(_company_label(p.symbol))}</strong>
                <span style="color: #6b7280;">·
                  prognoza {_html(_trend_label(p.predicted_trend))} ·
                  trafność <strong style="color: {color};">
                    {p.accuracy_score * 100:.0f}%
                  </strong>
                </span>
              </div>
            """)
        sections.append(
            "<div style='font-size: 11px; color: #6b7280; margin: 4px 0 16px 0;'>"
            f"Suma: {len(correct)} trafionych / {len(wrong)} błędnych "
            f"({len(correct) / max(1, len(resolved_predictions)) * 100:.0f}% accuracy)"
            "</div>"
        )

    # Historia trafności
    if accuracy_stats and accuracy_stats.get("mean_accuracy") is not None:
        acc = accuracy_stats["mean_accuracy"]
        n = accuracy_stats["sample_count"]
        correct = accuracy_stats["correct_count"]
        days = accuracy_stats["days_window"]
        sections.append(f"""
      <div style="padding: 12px 14px; background: #fef3c7; border-left: 3px solid #f59e0b;
                  border-radius: 4px; margin-bottom: 20px; font-size: 13px;">
        <div style="font-weight: 600; margin-bottom: 6px; color: #92400e;">
          🎯 Historia trafności (ostatnie {days} dni)
        </div>
        <div style="color: #78350f;">
          Średnia trafność: <strong>{acc * 100:.1f}%</strong>
          · Predykcji ocenionych: {n}
          · Poprawnych (accuracy > 0.5): {correct}
        </div>
      </div>
        """)
    elif accuracy_stats is not None:
        sections.append("""
      <div style="padding: 10px 14px; background: #f9fafb; border-radius: 4px;
                  margin-bottom: 20px; font-size: 12px; color: #6b7280;">
        🎯 Historia trafności: brak ocenionych predykcji (potrzeba ≥1 zamkniętego cyklu).
      </div>
        """)

    # Wykres zmiany cen
    chart_url = build_chart_url(results)
    if chart_url:
        sections.append(f"""
      <div style="margin-bottom: 16px; text-align: center;">
        <img src="{_html(chart_url)}" alt="Wykres zmiany cen 12h"
             style="max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 4px;" />
      </div>
        """)

    # Self-Reflection — wnioski z poprzednich cykli
    reflections = [r for r in saved if r.reflection_insight]
    if reflections:
        sections.append(
            "<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>"
            "🧠 Wnioski z poprzednich cykli (Self-Reflection)</h2>"
        )
        for r in reflections:
            sections.append(f"""
              <div style="margin-bottom: 8px; padding: 10px 12px; background: #faf5ff;
                          border-left: 3px solid #9333ea; border-radius: 4px;
                          font-size: 12px;">
                <strong>{_html(_company_label(r.symbol))}:</strong>
                <span style="color: #581c87;">{_html(r.reflection_insight)}</span>
              </div>
            """)

    # Predykcje (saved)
    if saved:
        sections.append("<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>🔮 Wygenerowane predykcje</h2>")
        sections.append("<table style='width: 100%; border-collapse: collapse; font-size: 13px;'>")
        sections.append("""
          <tr style="background: #f3f4f6; text-align: left;">
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Symbol</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Cena</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Zmiana 12h</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Trend</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Prognoza (12h)</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Pewność</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Sentyment</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Newsy</th>
            <th style="padding: 6px 8px; border-bottom: 1px solid #e5e7eb;">Rekomendacja</th>
          </tr>
        """)
        for r in saved:
            sentiment_text = (
                f"{_html(_sentiment_label(r.sentiment_label))} ({r.sentiment_score:+.2f})"
                if r.sentiment_score is not None
                else "—"
            )
            confidence_text = (
                f"{r.confidence_score * 100:.0f}%"
                if r.confidence_score is not None
                else "—"
            )
            forecast_text = (
                f"{_money(r.target_price)} "
                f"<span style='color: {_delta_color(r.expected_change)};'>"
                f"({_pct(r.expected_change, signed=True)})</span>"
                if r.target_price is not None
                else "—"
            )
            rec = _recommendation(r)
            rec_color = _DIRECTION_COLOR.get(rec, "#737373")
            sections.append(f"""
              <tr>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6; font-weight: 600;">{_html(_company_label(r.symbol))}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6;">{_money(r.current_price)}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6; color: {_delta_color(r.delta)};">{_pct(r.delta)}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6; color: {_trend_color(r.trend)}; font-weight: 600;">{_html(_trend_label(r.trend))}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6;">{forecast_text}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6;">{confidence_text}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6;">{sentiment_text}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6;">{r.news_volume or 0}</td>
                <td style="padding: 6px 8px; border-bottom: 1px solid #f3f4f6; font-weight: 700; color: {rec_color};">{rec}</td>
              </tr>
            """)
        sections.append("</table>")

        # Wykres prognozy
        forecast_chart_url = build_forecast_chart_url(saved)
        if forecast_chart_url:
            sections.append(f"""
      <div style="margin: 16px 0; text-align: center;">
        <img src="{_html(forecast_chart_url)}" alt="Prognoza zmian cen"
             style="max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 4px;" />
      </div>
            """)

        # Uzasadnienia
        sections.append("<h3 style='font-size: 14px; margin: 16px 0 8px 0;'>💡 Uzasadnienia</h3>")
        for r in saved:
            if not r.reasoning and r.council_verdict is None:
                continue
            move_line = ""
            if r.current_price is not None and r.target_price is not None:
                conf_part = (
                    f" · Pewność {r.confidence_score * 100:.0f}%"
                    if r.confidence_score is not None
                    else ""
                )
                move_line = (
                    f"<div style='font-size: 11px; color: #6b7280; margin-top: 2px;'>"
                    f"Oczekiwany ruch: {_money(r.current_price)} → {_money(r.target_price)} "
                    f"<span style='color: {_delta_color(r.expected_change)};'>"
                    f"({_pct(r.expected_change, signed=True)})</span>"
                    f"{conf_part}"
                    f"</div>"
                )
            news_block = ""
            if r.top_news:
                items = []
                for n in r.top_news:
                    src = f"<strong>[{_html(n.source or 'unknown')}]</strong>"
                    href = _safe_href(n.url)
                    title_html = _html(n.title)
                    if href:
                        title_html = (
                            f"<a href='{href}' target='_blank' "
                            f"style='color: #2563eb; text-decoration: none;'>"
                            f"{title_html}</a>"
                        )
                    items.append(
                        f"<li style='margin: 2px 0;'>{src} {title_html}"
                        f" <span style='color: #6b7280;'>"
                        f"(relevance {n.relevance:.2f}, sentyment {n.sentiment:+.2f})"
                        f"</span></li>"
                    )
                news_block = (
                    "<div style='font-size: 11px; color: #4b5563; margin-top: 8px;'>"
                    "📰 <strong>Top newsy:</strong>"
                    f"<ul style='margin: 4px 0 0 20px; padding: 0;'>{''.join(items)}</ul>"
                    "</div>"
                )
            rec_block = _recommendation_reason_html(r)
            council_block = (
                _render_council_section(r.council_verdict)
                if r.council_verdict is not None
                else ""
            )
            sections.append(f"""
              <div style="margin-bottom: 10px; padding: 10px 12px; background: #fafafa; border-left: 3px solid {_trend_color(r.trend)}; border-radius: 4px;">
                <div style="font-weight: 600; font-size: 13px;">{_html(_company_label(r.symbol))} <span style="color: {_trend_color(r.trend)};">{_html(_trend_label(r.trend))}</span></div>
                {move_line}
                {rec_block}
                {f'<div style="font-size: 12px; color: #4b5563; margin-top: 6px;">{_html(r.reasoning)}</div>' if r.reasoning else ''}
                {news_block}
                {council_block}
              </div>
            """)

    # Scatter: sentyment vs cena (jeśli ≥3 punkty)
    scatter_url = build_correlation_chart_url(results)
    if scatter_url:
        sections.append(f"""
      <h3 style="font-size: 14px; margin: 20px 0 8px 0;">📈 Korelacja sentymentu z ruchem ceny</h3>
      <div style="margin-bottom: 16px; text-align: center;">
        <img src="{_html(scatter_url)}" alt="Scatter sentyment vs cena"
             style="max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 4px;" />
      </div>
        """)

    # Pominięte
    if ignored:
        sections.append("<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>⏸ Pominięte (poniżej progu zmienności)</h2>")
        rows = "".join(
            f"<span style='display: inline-block; padding: 2px 8px; margin: 2px; background: #f3f4f6; border-radius: 3px; font-size: 12px;'>"
            f"<strong>{_html(r.symbol)}</strong> <span style='color: {_delta_color(r.delta)}'>{_pct(r.delta)}</span></span>"
            for r in ignored
        )
        sections.append(f"<div>{rows}</div>")

    # Błędy
    if errors:
        sections.append("<h2 style='font-size: 16px; margin: 20px 0 8px 0; color: #991b1b;'>⚠️ Błędy</h2>")
        for r in errors:
            sections.append(f"""
              <div style="margin-bottom: 8px; padding: 10px; background: #fef2f2; border-left: 3px solid #dc2626; border-radius: 4px; font-size: 12px;">
                <strong>{_html(_company_label(r.symbol))}</strong>: {_html(r.error_message)}
              </div>
            """)

    sections.append("""
      <div style="margin-top: 32px; padding-top: 16px; border-top: 1px solid #e5e7eb; font-size: 11px; color: #9ca3af;">
        StockAgent · Agentic RAG + Continual Learning · raport automatyczny
      </div>
    </div>
    """)

    return "".join(sections)


# ---------------------------------------------------------------------------
# Plain text fallback
# ---------------------------------------------------------------------------


def _render_plain(
    results: list[SymbolResult],
    saved: list[SymbolResult],
    ignored: list[SymbolResult],
    errors: list[SymbolResult],
    started_at: datetime,
    duration_seconds: float,
    mood: dict[str, Any],
    session: dict[str, str],
    accuracy_stats: dict[str, Any] | None,
    trade_signals: list[TradeSignal],
    risk_signals: list[RiskSignal],
    resolved_predictions: list[ResolvedPrediction],
) -> str:
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("STOCKAGENT — RAPORT CYKLU")
    lines.append("=" * 64)
    lines.append(f"Czas rozpoczęcia:  {started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Czas trwania:      {duration_seconds:.1f}s")
    lines.append(f"Sesja NYSE:        {session['label']} — {session['detail']}")
    lines.append(
        f"Symboli:           {len(results)} "
        f"(predykcji={len(saved)}, pominiętych={len(ignored)}, błędów={len(errors)})"
    )
    lines.append("")

    # Trade ideas
    if trade_signals:
        lines.append("NAJSILNIEJSZE SYGNAŁY")
        lines.append("-" * 64)
        for sig in trade_signals:
            lines.append(
                f"  {sig.direction:9s} {_company_label(sig.symbol):40s} "
                f"pewność {sig.confidence * 100:.0f}%  "
                f"prognoza {_pct(sig.expected_change, signed=True):>8s}  "
                f"siła {sig.strength:.2f}"
            )
        lines.append("")

    # Risk signals
    if risk_signals:
        lines.append("SYGNAŁY OSTRZEGAWCZE")
        lines.append("-" * 64)
        type_label = {
            "DIVERGENCE": "Rozbieżność",
            "AV_LLM_CONFLICT": "Niezgodność AV↔LLM",
            "LOW_SIGNAL": "Słaby sygnał",
        }
        for rs in risk_signals:
            lines.append(
                f"  ⚠ {_company_label(rs.symbol)}: [{type_label.get(rs.type, rs.type)}] {rs.description}"
            )
        lines.append("")

    # Zamknięte predykcje
    if resolved_predictions:
        correct = sum(1 for p in resolved_predictions if p.is_correct)
        lines.append("ZAMKNIĘTE PREDYKCJE (ostatnie 24h)")
        lines.append("-" * 64)
        for p in resolved_predictions:
            mark = "✅" if p.is_correct else "❌"
            lines.append(
                f"  {mark} {_company_label(p.symbol):40s}  trend {_trend_label(p.predicted_trend):11s} "
                f"trafność {p.accuracy_score * 100:.0f}%"
            )
        lines.append(
            f"  Suma: {correct}/{len(resolved_predictions)} "
            f"({correct / max(1, len(resolved_predictions)) * 100:.0f}% accuracy)"
        )
        lines.append("")

    # Nastroje portfela
    lines.append("NASTROJE PORTFELA")
    lines.append("-" * 64)
    lines.append(
        f"  Średni sentyment: {mood['avg_sentiment']:+.2f} ({mood['avg_sentiment_label']})"
    )
    if mood["most_bullish"]:
        mb = mood["most_bullish"]
        lines.append(f"  Najbardziej pozytywny: {_company_label(mb.symbol)} ({mb.sentiment_score:+.2f})")
    if mood["most_bearish"]:
        mbe = mood["most_bearish"]
        lines.append(f"  Najbardziej negatywny: {_company_label(mbe.symbol)} ({mbe.sentiment_score:+.2f})")
    lines.append(
        f"  Wysoka pewność (≥70%): {mood['high_confidence_count']} / {mood['saved_count']}"
        f"  ·  Newsów: {mood['total_news']}"
    )
    lines.append("")

    if accuracy_stats and accuracy_stats.get("mean_accuracy") is not None:
        lines.append(f"HISTORIA TRAFNOŚCI (ostatnie {accuracy_stats['days_window']} dni)")
        lines.append("-" * 64)
        lines.append(
            f"  Średnia trafność: {accuracy_stats['mean_accuracy'] * 100:.1f}%"
            f"  ·  Predykcji ocenionych: {accuracy_stats['sample_count']}"
            f"  ·  Poprawnych: {accuracy_stats['correct_count']}"
        )
        lines.append("")

    # Wnioski z poprzednich cykli
    reflections = [r for r in saved if r.reflection_insight]
    if reflections:
        lines.append("WNIOSKI Z POPRZEDNICH CYKLI (Self-Reflection)")
        lines.append("-" * 64)
        for r in reflections:
            lines.append(f"  {_company_label(r.symbol)}: {r.reflection_insight}")
        lines.append("")

    if saved:
        lines.append("PREDYKCJE")
        lines.append("-" * 64)
        for r in saved:
            conf_part = (
                f" · pewność {r.confidence_score * 100:.0f}%"
                if r.confidence_score is not None
                else ""
            )
            forecast_part = (
                f"prognoza {_money(r.target_price)} "
                f"({_pct(r.expected_change, signed=True)})"
                if r.target_price is not None and r.expected_change is not None
                else "prognoza —"
            )
            sentiment_part = (
                f"sentyment {r.sentiment_score:+.2f}"
                if r.sentiment_score is not None
                else "sentyment —"
            )
            rec_text = _recommendation_reason_text(r)
            lines.append(
                f"  {_company_label(r.symbol):40s} {_money(r.current_price):>10s}  "
                f"Δ12h {_pct(r.delta, signed=True):>8s}  →  "
                f"{_trend_label(r.trend):11s} {forecast_part}{conf_part}"
            )
            lines.append(f"         {sentiment_part}")
            lines.append(f"         {rec_text}")
            if r.reasoning:
                lines.append(f"        └ {r.reasoning}")
            for n in r.top_news:
                lines.append(
                    f"        📰 [{n.source or '?'}] {n.title[:78]}"
                    f" (rel={n.relevance:.2f}, sent={n.sentiment:+.2f})"
                )
        lines.append("")

    if ignored:
        lines.append("POMINIĘTE (poniżej progu zmienności)")
        lines.append("-" * 64)
        ignored_str = ", ".join(f"{r.symbol}({_pct(r.delta)})" for r in ignored)
        lines.append(f"  {ignored_str}")
        lines.append("")

    if errors:
        lines.append("BŁĘDY")
        lines.append("-" * 64)
        for r in errors:
            lines.append(f"  {_company_label(r.symbol)}: {r.error_message}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: convert use case output → SymbolResult
# ---------------------------------------------------------------------------


def to_symbol_result(symbol: str, raw: dict[str, Any] | None, error: str | None = None) -> SymbolResult:
    """Mapper z wyjścia AnalyzeMarketUseCase.run() na SymbolResult."""
    if error is not None:
        return SymbolResult(symbol=symbol, status="error", error_message=error)
    if raw is None:
        return SymbolResult(symbol=symbol, status="error", error_message="No result")

    status = str(raw.get("status", "unknown"))
    sentiment = raw.get("sentiment") or {}
    llm = raw.get("llm_analysis") or {}
    confidence = llm.get("confidence_score")
    av_agreement = llm.get("av_agreement")

    council_verdict = raw.get("council_verdict")

    return SymbolResult(
        symbol=symbol,
        status=status,
        delta=_as_decimal(raw.get("delta")),
        current_price=_as_decimal(raw.get("current_price")),
        trend=llm.get("trend_direction"),
        target_price=_as_decimal(raw.get("ml_target_price")),
        confidence_score=float(confidence) if confidence is not None else None,
        reasoning=llm.get("reasoning"),
        sentiment_score=sentiment.get("av_sentiment_score"),
        sentiment_label=sentiment.get("av_sentiment_label"),
        news_volume=sentiment.get("news_volume_24h"),
        av_llm_agreement=float(av_agreement) if av_agreement is not None else None,
        reflection_insight=_clean_reflection(raw.get("reflection_context")),
        top_news=_extract_top_news(raw.get("news") or []),
        council_verdict=council_verdict if isinstance(council_verdict, CouncilVerdict) else None,
    )


def _clean_reflection(raw_ctx: Any) -> str | None:
    """Filtruje cold-start placeholdery — pokazujemy reflection tylko gdy
    jest realny wniosek (LLM analyze_mistake lub potwierdzenie trafności)."""
    if not raw_ctx or not isinstance(raw_ctx, str):
        return None
    txt = raw_ctx.strip()
    if not txt:
        return None
    # Skip cold-start placeholder messages
    if "brak danych historycznych" in txt.lower():
        return None
    return txt


def _extract_top_news(news_items: list[dict[str, Any]], top_n: int = 2) -> list[TopNewsItem]:
    """Top N artykułów po relevance_score (z bonusem dla high-sentiment)."""
    valid = [
        item for item in news_items
        if item.get("relevance_score") is not None and item.get("title")
    ]
    sorted_news = sorted(valid, key=lambda x: float(x["relevance_score"] or 0), reverse=True)
    return [
        TopNewsItem(
            title=str(item["title"]),
            source=item.get("source"),
            url=item.get("url"),
            relevance=float(item.get("relevance_score") or 0),
            sentiment=float(item.get("ticker_sentiment_score") or 0),
        )
        for item in sorted_news[:top_n]
    ]


def _as_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None
