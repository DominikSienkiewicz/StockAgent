"""Sekcja maila „Od otwarcia sesji” — druga krzywa kapitału z ceny wykonalnej (WIA-54).

Fast Loop chodzi przed sesją, więc cena z chwili predykcji to ostatnie zamknięcie,
a krzywa Track Recordu liczy od niej także lukę na otwarciu, której czytelnik maila
przed sesją nie kupi. Za `executable_entry_enabled` reflect zapisuje cenę wykonalną
(`entry_price`, otwarcie pierwszej sesji po predykcji), a ta sekcja stawia obok
siebie dwie krzywe portfela na TYCH SAMYCH transakcjach: od otwarcia i od ceny
predykcji. Różnica to luka na otwarciu. Render-only, zero płatnych wywołań; bez
krzywej (flaga off, brak cen otwarcia) sekcja znika i mail zostaje bajt w bajt.
"""

from __future__ import annotations

import html as _html
from collections.abc import Sequence
from dataclasses import dataclass

from src.application.report_evidence import PaperCosts, executable_change, paper_cycles
from src.application.report_formatting import pct
from src.application.report_models import ResolvedPrediction
from src.application.reporting.section import ReportSection, SectionInput, text_block
from src.domain.equity_curve import EquityCurve, paper_portfolio_curve

_HEADER = "Od otwarcia sesji (cena wykonalna)"
_NOTE_STYLE = "font-size: 12px; color: #6b7280; margin: 0 0 10px 0;"


@dataclass(frozen=True)
class OpenEntryCurve:
    """Dwie krzywe portfela na tych samych transakcjach (WIA-54).

    `from_open` — wejście po cenie wykonalnej, `from_prediction` — po cenie z chwili
    predykcji; `with_entry` — transakcje z ceną otwarcia, `settled` — wszystkie
    rozliczone z ruchem ceny w oknie (krypto i zaległe rozliczenia jej nie mają)."""

    from_open: EquityCurve
    from_prediction: EquityCurve
    with_entry: int
    settled: int
    costs: PaperCosts | None = None


def build_open_entry_curve(
    resolved: Sequence[ResolvedPrediction], costs: PaperCosts | None = None
) -> OpenEntryCurve | None:
    """Krzywe od otwarcia i od ceny predykcji z tych samych rozliczeń (te same reguły
    portfela co Track Record: jeden punkt na cykl, równa waga, koszt od obrotu).
    None, gdy żadne rozliczenie nie ma ceny wykonalnej."""
    priced = [p for p in resolved if p.actual_change_pct is not None]
    executable = [p for p in priced if executable_change(p) is not None]
    if not executable:
        return None
    return OpenEntryCurve(
        from_open=paper_portfolio_curve(
            paper_cycles(executable, costs, change_of=executable_change)
        ),
        from_prediction=paper_portfolio_curve(paper_cycles(executable, costs)),
        with_entry=len(executable),
        settled=len(priced),
        costs=costs,
    )


def _summary(curve: OpenEntryCurve) -> str:
    return (
        f"{_HEADER}: wynik łączny {pct(curve.from_open.total_return, signed=True, decimals=1)}"
        f" · trafność {pct(curve.from_open.win_rate, decimals=1)}"
        f" · transakcji: {curve.from_open.n_trades}"
    )


def _comparison(curve: OpenEntryCurve) -> str:
    gross = " (przed kosztami)" if curve.costs is None else ""
    return (
        f"Te same transakcje od ceny z chwili predykcji: "
        f"{pct(curve.from_prediction.total_return, signed=True, decimals=1)}{gross} — "
        "różnica to luka na otwarciu, której przed sesją nie da się kupić. "
        f"Cenę otwarcia ma {curve.with_entry} z {curve.settled} rozliczonych predykcji "
        "(krypto i zaległe rozliczenia jej nie mają)."
    )


def render_open_entry_html(curve: OpenEntryCurve | None) -> str:
    """Fragment HTML sekcji; "" bez krzywej."""
    if curve is None:
        return ""
    total_color = "#059669" if curve.from_open.total_return >= 0 else "#dc2626"
    summary = _html.escape(_summary(curve))
    head, _, numbers = summary.partition(": ")
    return (
        "<div style='padding: 10px 14px; background: #f0fdf4; "
        "border-left: 3px solid #059669; border-radius: 4px; "
        "margin-bottom: 6px; font-size: 13px;'>"
        f"<strong>📈 {head}:</strong> <span style='color: {total_color};'>{numbers}</span>"
        "</div>"
        f"<p style='{_NOTE_STYLE}'>{_html.escape(_comparison(curve))}</p>"
    )


def render_open_entry_text(curve: OpenEntryCurve | None) -> list[str]:
    """Linie wersji tekstowej; [] bez krzywej."""
    if curve is None:
        return []
    return text_block(f"📈 {_summary(curve)}\n   {_comparison(curve)}", spaced=True)


def _section_html(data: SectionInput) -> str:
    return render_open_entry_html(data.context.open_entry_curve)


def _section_text(data: SectionInput) -> list[str]:
    return render_open_entry_text(data.context.open_entry_curve)


OPEN_ENTRY_SECTION = ReportSection(
    key="open_entry",
    render_html=_section_html,
    render_text=_section_text,
    context_fields=frozenset({"open_entry_curve"}),
)
