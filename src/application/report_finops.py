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
"""

from __future__ import annotations

import html as _html

from src.domain.finops import CycleCost

_HEADING = "💸 Koszt cyklu"

# Zastrzeżenie doklejane pod kwotą — bez niego prezentowalibyśmy szacunek jak
# rachunek, co byłoby nieuczciwe.
_DISCLAIMER = (
    "Kwoty to szacunek rzędu wielkości na podstawie cennika jednostkowego, "
    "nie faktura od dostawcy."
)

# Komunikat cyklu darmowego — sedno FinOps repo, renderowany jawnie.
_FREE_CYCLE = (
    "0 płatnych wywołań — bramka volatility odcięła wszystkie symbole "
    "(cykl darmowy)."
)

# Czytelne etykiety źródeł. Domena trzyma techniczne klucze, prezentacja —
# ludzkie nazwy po polsku. Nieznany klucz renderujemy dosłownie (po escape).
_SOURCE_LABELS: dict[str, str] = {
    "llm": "Model predykcyjny (LLM)",
    "council_llm": "Rada doradcza (LLM)",
    "sentiment": "Sentyment (Alpha Vantage)",
    "news": "Newsy",
    "embedding": "Embeddingi (RAG)",
}

_COLOR_MUTED = "#6b7280"
_COLOR_TEXT = "#374151"
_COLOR_FREE = "#16a34a"


def _source_label(source: str) -> str:
    """Czytelna etykieta źródła albo surowy klucz, gdy nieznany."""
    return _SOURCE_LABELS.get(source, source)


def _fmt_usd(amount: float) -> str:
    """Formatuje kwotę USD z 4 miejscami — kwoty są rzędu centów lub mniej."""
    return f"${amount:.4f}"


def render_finops_html(cost: CycleCost | None) -> str:
    """Renderuje sekcję HTML kosztu cyklu lub "" gdy brak licznika (None).

    Cykl darmowy (0 wywołań) renderuje się JAWNIE — nie chowamy go.
    """
    if cost is None:
        return ""

    rows: list[str] = []

    if cost.total_calls == 0:
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

    rows.append(
        f"<div style='font-size: 11px; color: {_COLOR_MUTED}; "
        f"margin-top: 6px; font-style: italic;'>{_html.escape(_DISCLAIMER)}</div>"
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

    if cost.total_calls == 0:
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

    lines.append(f"  Uwaga: {_DISCLAIMER}")
    lines.append("")
    return "\n".join(lines)
