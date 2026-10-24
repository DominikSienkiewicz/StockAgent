"""Stopka maila: zastrzeżenie, liczniki pipeline'u, źródła danych i przebieg.

MAIL-08: zastrzeżenie stoi w stałej stopce, nie w sekcji warunkowej — sekcja,
która się chowa, zabrałaby je ze sobą. MAIL-16: liczniki pracy pipeline'u (błąd
nadal czerwony). MAIL-07: źródła danych. MAIL-42: który przebieg i commit
wygenerował mail. Stopka zamyka ramkę maila otwartą przez nagłówek
(`report_header.py`).
"""

from __future__ import annotations

from src.application.report_formatting import (
    DATA_SOURCES_FOOTER_PL,
    INVESTMENT_DISCLAIMER,
    PALETTE,
    safe_http_url,
)
from src.application.report_formatting import (
    escape as _html,
)
from src.application.report_formatting import (
    number as _number,
)
from src.application.report_models import SymbolResult
from src.application.reporting.section import ReportSection, SectionInput
from src.application.run_context import RunContext

# ---------------------------------------------------------------------------
# MAIL-42: proweniencja przebiegu
# ---------------------------------------------------------------------------


def _run_footer_label(run_context: RunContext) -> str:
    """„przebieg #77 · commit abc1234” — logi repo są publiczne, więc stopka nie
    niesie nic poza numerem przebiegu, krótkim SHA i linkiem."""
    return f"przebieg #{run_context.run_id} · commit {run_context.short_sha}"


def _render_run_footer_html(run_context: RunContext | None) -> str:
    if run_context is None:
        return ""
    label = _html(_run_footer_label(run_context))
    url = safe_http_url(run_context.run_url)
    logs = (
        f' · <a href="{_html(url)}" style="color: {PALETTE.muted};">logi</a>'
        if url is not None
        else ""
    )
    return f"<br/>{label}{logs}"


def _render_run_footer_text(run_context: RunContext | None) -> str:
    if run_context is None:
        return ""
    url = safe_http_url(run_context.run_url)
    logs = f" · logi: {url}" if url is not None else ""
    return "\n".join(["-" * 64, f"{_run_footer_label(run_context)}{logs}"])


# ---------------------------------------------------------------------------
# MAIL-16: liczniki pipeline'u (dawniej pierwszy ekran)
# ---------------------------------------------------------------------------


def _render_technical_footer_html(
    results: list[SymbolResult],
    saved: list[SymbolResult],
    ignored: list[SymbolResult],
    errors: list[SymbolResult],
    duration_seconds: float,
) -> str:
    """MAIL-16: liczniki pracy pipeline'u w stopce — błąd nadal jest czerwony."""
    errors_color = PALETTE.negative if errors else PALETTE.muted
    return (
        '<div style="margin-bottom: 8px; color: #6b7280;">'
        f"Symboli: <strong>{len(results)}</strong>"
        f" · Predykcje: <strong>{len(saved)}</strong>"
        f" · Pominięte: <strong>{len(ignored)}</strong>"
        f' · <span style="color: {errors_color};">Błędy: <strong>{len(errors)}</strong></span>'
        f" · Czas trwania: {_number(duration_seconds, decimals=1)}s"
        "</div>"
    )


def _technical_footer_text(
    results: list[SymbolResult],
    saved: list[SymbolResult],
    ignored: list[SymbolResult],
    errors: list[SymbolResult],
    duration_seconds: float,
) -> str:
    return (
        f"Symboli: {len(results)} · predykcji: {len(saved)} · "
        f"pominiętych: {len(ignored)} · błędów: {len(errors)} · "
        f"czas trwania: {_number(duration_seconds, decimals=1)}s"
    )


# ---------------------------------------------------------------------------
# Sekcja rejestru (ARCH-12)
# ---------------------------------------------------------------------------

_FOOTER_STYLE = (
    "margin-top: 32px; padding-top: 16px; border-top: 1px solid #e5e7eb; "
    f"font-size: 11px; color: {PALETTE.muted};"
)


def _section_html(data: SectionInput) -> str:
    counters = _render_technical_footer_html(
        data.results, data.saved, data.ignored, data.errors, data.duration_seconds
    )
    run_footer = _render_run_footer_html(data.context.run_context)
    return f"""
      <div style="{_FOOTER_STYLE}">
        <div style="margin-bottom: 8px; color: #6b7280;">{_html(INVESTMENT_DISCLAIMER)}</div>
        {counters}
        <div style="margin-bottom: 8px;">{_html(DATA_SOURCES_FOOTER_PL)}</div>
        StockAgent · Agentic RAG + Continual Learning · raport automatyczny
        {run_footer}
      </div>
    </div>
    """


def _section_text(data: SectionInput) -> list[str]:
    """Zastrzeżenie zamyka treść; liczniki (MAIL-16) i przebieg (MAIL-42) za nim."""
    run_footer = _render_run_footer_text(data.context.run_context)
    return [
        "-" * 64,
        INVESTMENT_DISCLAIMER,
        _technical_footer_text(
            data.results, data.saved, data.ignored, data.errors, data.duration_seconds
        ),
        DATA_SOURCES_FOOTER_PL,
        *([run_footer] if run_footer else []),
    ]


FOOTER_SECTION = ReportSection(
    key="footer",
    render_html=_section_html,
    render_text=_section_text,
    context_fields=frozenset({"run_context"}),
)
