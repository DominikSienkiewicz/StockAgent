"""Sekcja 🎯 Historia trafności — Zaplecze, z `RepositoryPort.get_accuracy_stats`.

Trafność kierunkowa obok łącznej (WIA-29), przedział Wilsona przy procencie
(WIA-32) i — w okresie przejściowym WIA-61 — trafność na oficjalnym zamknięciu
sesji. Te same etykiety niesie suma sekcji zamkniętych predykcji
(`report_resolved.py`).
"""

from __future__ import annotations

import html
from typing import Any

from src.application.report_evidence import wilson_label
from src.application.report_formatting import (
    pct as _pct,
)
from src.application.reporting.section import ReportSection, SectionInput

# WIA-29 (c): dwie definicje trafności obok siebie, każda nazwana tym, co liczy.
DIRECTIONAL_ACCURACY_LABEL = "Trafność kierunkowa (bez prognoz „bez ruchu”)"
TOTAL_ACCURACY_LABEL = "Trafność łączna (z prognozami „bez ruchu”)"

# WIA-61 (okres przejściowy, opcja „oba”): druga cena rozliczenia obok /quote.
_SESSION_CLOSE_ACCURACY_LABEL = "Na oficjalnym zamknięciu sesji"


def _session_close_accuracy(accuracy_stats: dict[str, Any]) -> str | None:
    """WIA-61: linia trafności na zamknięciu sesji z trafnością /quote tych samych
    predykcji; None, gdy statystyki jej nie niosą (flaga off) albo próba pusta."""
    n = accuracy_stats.get("session_close_count")
    if not n:
        return None
    hits = accuracy_stats["session_close_correct_count"]
    quote_hits = accuracy_stats["session_close_quote_correct_count"]
    return (
        f"{_SESSION_CLOSE_ACCURACY_LABEL}: {_pct(hits / n, decimals=1)}"
        f" ({wilson_label(hits, n)}) · na cenie /quote tych samych predykcji:"
        f" {_pct(quote_hits / n, decimals=1)}"
    )


def _directional_accuracy(accuracy_stats: dict[str, Any]) -> tuple[str, str] | None:
    """(wartość, dopisek) trafności kierunkowej; None, gdy statystyki jej nie niosą."""
    n = accuracy_stats.get("directional_count")
    hits = accuracy_stats.get("directional_correct_count")
    if n is None or hits is None:
        return None
    if n == 0:
        return "—", "brak prognoz kierunkowych"
    return _pct(hits / n, decimals=1), wilson_label(hits, n)


def _render_accuracy_history_html(accuracy_stats: dict[str, Any] | None) -> str:
    """🎯 Historia trafności. Pusty string, gdy brak danych statystyk."""
    if accuracy_stats and accuracy_stats.get("mean_accuracy") is not None:
        acc = accuracy_stats["mean_accuracy"]
        n = accuracy_stats["sample_count"]
        correct = accuracy_stats["correct_count"]
        days = accuracy_stats["days_window"]
        # WIA-32: przedział Wilsona obok procentu — goły % jest nieweryfikowalny.
        interval = wilson_label(correct, n)
        directional = _directional_accuracy(accuracy_stats)
        directional_html = (
            f"{DIRECTIONAL_ACCURACY_LABEL}: <strong>{directional[0]}</strong>"
            f" ({directional[1]})<br/>"
            if directional is not None
            else ""
        )
        session_close = _session_close_accuracy(accuracy_stats)
        session_close_html = (
            f"<br/>{html.escape(session_close)}" if session_close is not None else ""
        )
        return f"""
      <div style="padding: 12px 14px; background: #fef3c7; border-left: 3px solid #f59e0b;
                  border-radius: 4px; margin-bottom: 20px; font-size: 13px;">
        <div style="font-weight: 600; margin-bottom: 6px; color: #92400e;">
          🎯 Historia trafności (ostatnie {days} dni)
        </div>
        <div style="color: #78350f;">
          {directional_html}
          {TOTAL_ACCURACY_LABEL}: <strong>{_pct(acc, decimals=1)}</strong> ({interval})
          · Predykcji ocenionych: {n}
          · Poprawnych: {correct}{session_close_html}
        </div>
      </div>
        """
    if accuracy_stats is not None:
        return """
      <div style="padding: 10px 14px; background: #f9fafb; border-radius: 4px;
                  margin-bottom: 20px; font-size: 12px; color: #6b7280;">
        🎯 Historia trafności: brak ocenionych predykcji (potrzeba ≥1 zamkniętego cyklu).
      </div>
        """
    return ""


def _render_accuracy_history_text(accuracy_stats: dict[str, Any] | None) -> list[str]:
    """Sekcja 'Historia trafności' (plain). Pusta lista, gdy brak danych."""
    if not (accuracy_stats and accuracy_stats.get("mean_accuracy") is not None):
        return []
    lines = [
        f"HISTORIA TRAFNOŚCI (ostatnie {accuracy_stats['days_window']} dni)",
        "-" * 64,
    ]
    directional = _directional_accuracy(accuracy_stats)
    if directional is not None:
        lines.append(f"  {DIRECTIONAL_ACCURACY_LABEL}: {directional[0]} ({directional[1]})")
    lines.extend(
        [
            f"  {TOTAL_ACCURACY_LABEL}: {_pct(accuracy_stats['mean_accuracy'], decimals=1)}"
            f" ({wilson_label(accuracy_stats['correct_count'], accuracy_stats['sample_count'])})"
            f"  ·  Predykcji ocenionych: {accuracy_stats['sample_count']}"
            f"  ·  Poprawnych: {accuracy_stats['correct_count']}",
        ]
    )
    session_close = _session_close_accuracy(accuracy_stats)
    if session_close is not None:
        lines.append(f"  {session_close}")
    lines.append("")
    return lines


def _section_html(data: SectionInput) -> str:
    return _render_accuracy_history_html(data.context.accuracy_stats)


def _section_text(data: SectionInput) -> list[str]:
    return _render_accuracy_history_text(data.context.accuracy_stats)


ACCURACY_HISTORY_SECTION = ReportSection(
    key="accuracy_history", render_html=_section_html, render_text=_section_text
)
