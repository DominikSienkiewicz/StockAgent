"""Render sekcji "Koszt cyklu" (FinOps) w raporcie e-mail.

"Agent, który raportuje własny rachunek": tania, charakterystyczna sekcja,
która pokazuje ile płatnych wywołań zewnętrznych poszło w tym cyklu i ile to
mniej więcej kosztowało. Czysta logika prezentacji — bierze gotowy `CycleCost`
z domeny (`src.domain.finops`) i formatuje go. Zero I/O, zero płatnych wywołań.

Dwie świadome decyzje o samosupresji, celowo RÓŻNE od reszty raportu:

  - `None` → "". Brak licznika kosztu (np. cykl bez instrumentacji) chowa sekcję.
  - Cykl DARMOWY (0 wywołań) → NIE chowamy. To najciekawsza informacja tej
    sekcji: bramka volatility odcięła wszystkie symbole i cykl nie kosztował nic.
    Sekcja renderuje to jawnie, zamiast milczeć.

Uczciwość jest wymogiem, nie ozdobą: kwoty to SZACUNEK rzędu wielkości (patrz
`UNIT_COST_USD` w domenie), nie faktura — więc przy każdej kwocie stoi słowo
"szacunkowy", a na końcu sekcji jawne zastrzeżenie.

FIN-13: źródło z wycenionymi tokenami ma kwotę „z tokenów” (bez tyldy), a obok
niego źródła na stałej są oznaczone „(szacunek)”; nagłówek mówi, co jest czym,
i podaje datę cennika. Cykl bez danych o tokenach renderuje się jak przed FIN-13.
"""

from __future__ import annotations

import html as _html
from datetime import date

from src.application.report_formatting import PALETTE, number, plural_pl
from src.application.reporting.section import ReportSection, SectionInput, text_block
from src.domain.finops import CycleCost, SourceCost

_HEADING = "💸 Koszt cyklu"

# Zastrzeżenie doklejane pod kwotą — bez niego prezentowalibyśmy szacunek jak
# rachunek, co byłoby nieuczciwe.
_DISCLAIMER = (
    "Kwoty to szacunek rzędu wielkości na podstawie cennika jednostkowego, nie faktura od dostawcy."
)

# FIN-13: zastrzeżenie sekcji z kwotami z tokenów — to ceny katalogowe, nie faktura.
_TOKEN_DISCLAIMER = (
    "Kwoty „z tokenów” to tokeny zwrócone przez dostawcę wycenione cennikiem "
    "katalogowym z config.toml (standard tier) — to wciąż nie faktura od dostawcy."
)
_MIXED_DISCLAIMER_SUFFIX = " Pozycje oznaczone „(szacunek)” liczy cennik jednostkowy na wywołanie."
_ESTIMATE_MARKER = "(szacunek)"
_MIXED_NOTE = "część z tokenów, reszta szacunkiem"

# Komunikat cyklu darmowego — sedno FinOps repo, renderowany jawnie.
_FREE_CYCLE = "0 płatnych wywołań — bramka volatility odcięła wszystkie symbole (cykl darmowy)."

# Czytelne etykiety źródeł. Domena trzyma techniczne klucze, prezentacja —
# ludzkie nazwy po polsku. Nieznany klucz renderujemy dosłownie (po escape).
_SOURCE_LABELS: dict[str, str] = {
    "llm": "Model predykcyjny (LLM)",
    "council_llm": "Rada doradcza (LLM)",
    "sentiment": "Sentyment (Alpha Vantage)",
    "news": "Newsy",
    "embedding": "Embeddingi (RAG)",
    # EXP-26: wariant promptu albo modelu w cieniu (`SHADOW_LLM_SOURCE`).
    "shadow_llm": "Wariant promptu w cieniu (LLM)",
}

_COLOR_MUTED = "#6b7280"
_COLOR_TEXT = "#374151"
_COLOR_FREE = PALETTE.positive


def _source_label(source: str) -> str:
    """Czytelna etykieta źródła albo surowy klucz, gdy nieznany."""
    return _SOURCE_LABELS.get(source, source)


def _fmt_usd(amount: float) -> str:
    """Kwota USD z 4 miejscami po przecinku („$0,0123”) — koszty są rzędu centów."""
    return f"${number(amount, decimals=4)}"


def _calls_label(calls: int) -> str:
    return f"{calls} {plural_pl(calls, 'wywołanie', 'wywołania', 'wywołań')}"


def _fmt_date(day: date) -> str:
    return day.strftime("%d.%m.%Y")


def _is_mixed(cost: CycleCost) -> bool:
    """Część kwoty z tokenów, część ze stałej (albo źródła spoza cennika)."""
    return bool(cost.unknown_sources) or not all(line.from_tokens for line in cost.lines)


def _token_header_details(cost: CycleCost) -> str:
    """Nawias nagłówka sekcji z tokenami: liczba wywołań, mieszanka, data cennika."""
    parts = [f"{cost.total_calls} płatnych wywołań"]
    if _is_mixed(cost):
        parts.append(_MIXED_NOTE)
    if cost.priced_as_of is not None:
        parts.append(f"cennik z {_fmt_date(cost.priced_as_of)}")
    return "; ".join(parts)


def _token_total(cost: CycleCost) -> tuple[str, str]:
    """(prefiks nagłówka, kwota) — tylda tylko, gdy w sumie jest szacunek."""
    if _is_mixed(cost):
        return "Koszt cyklu:", f"~{_fmt_usd(cost.total_usd)}"
    return "Koszt cyklu z tokenów:", _fmt_usd(cost.total_usd)


def _token_disclaimer(cost: CycleCost) -> str:
    return _TOKEN_DISCLAIMER + (_MIXED_DISCLAIMER_SUFFIX if _is_mixed(cost) else "")


def _line_parts(line: SourceCost) -> tuple[str, str, str]:
    """(opis wywołań, kwota, dopisek) linii rozbicia w sekcji z tokenami — wspólne
    dla HTML i tekstu. Obok kwot z tokenów szacunek ze stałej jest oznaczony."""
    if line.from_tokens:
        return f"{_calls_label(line.calls)} z tokenów =", _fmt_usd(line.subtotal_usd), ""
    return (
        f"{line.calls} × ~{_fmt_usd(line.unit_cost_usd)} =",
        f"~{_fmt_usd(line.subtotal_usd)}",
        f" {_ESTIMATE_MARKER}",
    )


def _token_rows_html(cost: CycleCost) -> list[str]:
    prefix, total = _token_total(cost)
    rows = [
        f"<div style='font-size: 13px; color: {_COLOR_TEXT}; "
        f"margin-bottom: 6px;'>{_html.escape(prefix)} "
        f"<strong>{_html.escape(total)}</strong> "
        f"({_html.escape(_token_header_details(cost))})</div>"
    ]
    for line in cost.lines:
        calls, amount, marker = _line_parts(line)
        rows.append(
            f"<div style='font-size: 12px; color: {_COLOR_TEXT}; "
            f"margin-bottom: 2px;'>{_html.escape(_source_label(line.source))}: "
            f"{_html.escape(calls)} <strong>{_html.escape(amount)}</strong>"
            f"{_html.escape(marker)}</div>"
        )
    return rows


def render_finops_html(cost: CycleCost | None) -> str:
    """Renderuje sekcję HTML kosztu cyklu lub "" gdy brak licznika (None).

    Cykl darmowy (0 wywołań) renderuje się JAWNIE — nie chowamy go.
    """
    if cost is None:
        return ""

    rows: list[str] = []

    if cost.total_calls > 0 and cost.from_tokens:
        rows.extend(_token_rows_html(cost))
    elif cost.total_calls == 0:
        rows.append(
            f"<div style='font-size: 13px; color: {_COLOR_FREE}; "
            f"font-weight: 600;'>✅ {_html.escape(_FREE_CYCLE)}</div>"
        )
    else:
        rows.append(
            f"<div style='font-size: 13px; color: {_COLOR_TEXT}; "
            f"margin-bottom: 6px;'>Szacunkowy koszt cyklu: "
            f"<strong>~{_html.escape(_fmt_usd(cost.total_usd))}</strong> "
            f"({cost.total_calls} płatnych wywołań)</div>"
        )
        for line in cost.lines:
            label = _html.escape(_source_label(line.source))
            rows.append(
                f"<div style='font-size: 12px; color: {_COLOR_TEXT}; "
                f"margin-bottom: 2px;'>{label}: {line.calls} × "
                f"~{_html.escape(_fmt_usd(line.unit_cost_usd))} = "
                f"<strong>~{_html.escape(_fmt_usd(line.subtotal_usd))}</strong>"
                f"</div>"
            )

    if cost.unknown_sources:
        joined = ", ".join(_html.escape(src) for src in cost.unknown_sources)
        rows.append(
            f"<div style='font-size: 12px; color: {_COLOR_MUTED}; "
            f"margin-top: 4px;'>Źródła spoza cennika (koszt nieoszacowany): "
            f"{joined}</div>"
        )

    disclaimer = _token_disclaimer(cost) if cost.from_tokens else _DISCLAIMER
    rows.append(
        f"<div style='font-size: 11px; color: {_COLOR_MUTED}; "
        f"margin-top: 6px; font-style: italic;'>{_html.escape(disclaimer)}</div>"
    )

    return (
        f"<h2 style='font-size: 16px; margin: 20px 0 8px 0;'>{_HEADING}</h2>"
        "<div style='padding: 10px 14px; background: #f8fafc; "
        "border-left: 3px solid #0ea5e9; border-radius: 4px; margin-bottom: 20px;'>"
        + "".join(rows)
        + "</div>"
    )


def render_finops_text(cost: CycleCost | None) -> str:
    """Renderuje sekcję plain-text kosztu cyklu lub "" gdy brak licznika (None)."""
    if cost is None:
        return ""

    lines: list[str] = ["KOSZT CYKLU", "-" * 64]

    if cost.total_calls > 0 and cost.from_tokens:
        prefix, total = _token_total(cost)
        lines.append(f"  {prefix} {total} ({_token_header_details(cost)})")
        for line in cost.lines:
            calls, amount, marker = _line_parts(line)
            lines.append(f"    {_source_label(line.source)}: {calls} {amount}{marker}")
    elif cost.total_calls == 0:
        lines.append(f"  {_FREE_CYCLE}")
    else:
        lines.append(
            f"  Szacunkowy koszt cyklu: ~{_fmt_usd(cost.total_usd)} "
            f"({cost.total_calls} płatnych wywołań)"
        )
        for line in cost.lines:
            lines.append(
                f"    {_source_label(line.source)}: {line.calls} × "
                f"~{_fmt_usd(line.unit_cost_usd)} = ~{_fmt_usd(line.subtotal_usd)}"
            )

    if cost.unknown_sources:
        joined = ", ".join(cost.unknown_sources)
        lines.append(f"  Źródła spoza cennika (koszt nieoszacowany): {joined}")

    lines.append(f"  Uwaga: {_token_disclaimer(cost) if cost.from_tokens else _DISCLAIMER}")
    lines.append("")
    return "\n".join(lines)


def _section_html(data: SectionInput) -> str:
    return render_finops_html(data.context.cycle_cost)


def _section_text(data: SectionInput) -> list[str]:
    return text_block(render_finops_text(data.context.cycle_cost))


# FinOps: „agent, który raportuje własny rachunek”. Cykl darmowy (0 wywołań)
# renderuje się JAWNIE — to najciekawsza informacja w tej sekcji, nie brak danych.
FINOPS_SECTION = ReportSection(
    key="finops",
    render_html=_section_html,
    render_text=_section_text,
    context_fields=frozenset({"cycle_cost"}),
)
