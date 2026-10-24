"""Render sekcji „Tydzień StockAgenta" (#17) — HTML i plain-text.

Niedzielna retrospektywa jako mail-opowieść: strzał tygodnia, wtopa tygodnia
z lekcją, wybronieni dysydenci rady, Brier z BSS (WIA-36) i delta ECE
(pomocniczo). Selekcję robi domena
(`build_weekly_recap`), narrację dokłada use case (LLM) — tu tylko prezentacja.

Sekcja samosupresująca: `recap is None` (poniżej progu / brak danych) →
`render_*` zwraca "". Zero płatnych wywołań, zero I/O.
"""

from __future__ import annotations

import html as _html
from decimal import Decimal

from src.application.report_evidence import probability_label
from src.application.report_formatting import INVESTMENT_DISCLAIMER, number, pct
from src.domain.council import InvestorOpinion
from src.domain.council_redundancy import CouncilRedundancy
from src.domain.number_format import format_number_pl
from src.domain.polish_labels import council_recommendation_label
from src.domain.weekly_recap import RecapHighlight, WeeklyRecap

_COLOR_GOOD = "#16a34a"
_COLOR_POOR = "#dc2626"
_COLOR_MUTED = "#6b7280"

_BEST_TITLE = "Strzał tygodnia"
_BEST_MISS_TITLE = "Najbliżej celu (kierunek chybiony)"
_WORST_TITLE = "Wtopa tygodnia"
_WORST_HIT_TITLE = "Najdalej od celu (kierunek trafiony)"


def _ece_delta_label(recap: WeeklyRecap) -> str:
    """Opis zmiany kalibracji (ECE) w ludzkim języku."""
    if recap.ece_delta is None:
        return "delta ECE: brak punktu odniesienia (pierwszy tydzień)"
    direction = "poprawa" if recap.ece_delta < 0 else "pogorszenie"
    if recap.ece_delta == 0:
        direction = "bez zmian"
    return (
        f"ECE {number(recap.current_ece, decimals=3)} "
        f"({direction} o {number(abs(recap.ece_delta), decimals=3)} tydzień do tygodnia)"
    )


def _scores_label(recap: WeeklyRecap) -> str:
    """„Brier = 0,168 (klimatologia 0,24, BSS = +30,0%) · log loss = 0,5” albo ""."""
    return "" if recap.scores is None else probability_label(recap.scores)


def _scores_html(recap: WeeklyRecap) -> str:
    label = _scores_label(recap)
    if not label:
        return ""
    return (
        "<div style='font-size: 12px; color: #6b7280; margin-top: 8px;'>"
        f"🎯 Pewność: {_html.escape(label)} — BSS 0% to brak umiejętności "
        "względem prognozy „zawsze stopa bazowa”.</div>"
    )


def _redundancy_label(redundancy: CouncilRedundancy | None) -> str:
    """„7 person ≈ 2,1 niezależnego głosu (średnia kappa 0,52, zgodność 81%)” (WIA-45).

    Gdy N_eff nie przekracza progu jednego głosu, dopisuje wprost, że rada mówi
    w praktyce jednym głosem; najbardziej zgodna para person idzie na końcu."""
    if redundancy is None or redundancy.n_eff is None or redundancy.mean_kappa is None:
        return ""
    n_eff = format_number_pl(Decimal(str(redundancy.n_eff)), signed=False, decimals=1)
    kappa = format_number_pl(Decimal(str(redundancy.mean_kappa)), signed=False, decimals=2)
    agreement = format_number_pl(
        Decimal(str((redundancy.mean_agreement or 0.0) * 100)), signed=False, decimals=0
    )
    label = (
        f"{redundancy.personas} person ≈ {n_eff} niezależnego głosu "
        f"(średnia kappa {kappa}, surowa zgodność {agreement}%, "
        f"{redundancy.predictions} predykcji)"
    )
    if redundancy.is_effectively_single_voter:
        label += " — rada mówi w praktyce jeden głos"
    top = redundancy.pairs[0] if redundancy.pairs else None
    if top is not None:
        top_kappa = format_number_pl(Decimal(str(top.kappa)), signed=False, decimals=2)
        label += f"; najbardziej zgodne: {top.first} i {top.second} (kappa {top_kappa})"
    return label


def _redundancy_html(recap: WeeklyRecap) -> str:
    label = _redundancy_label(recap.redundancy)
    if not label:
        return ""
    return (
        "<div style='font-size: 12px; color: #6b7280; margin-top: 8px;'>"
        f"🧮 Rada: {_html.escape(label)}.</div>"
    )


def _dissenters_html(dissenters: tuple[InvestorOpinion, ...]) -> str:
    if not dissenters:
        return ""
    items = "".join(
        f"<li><strong>{_html.escape(op.investor_name)}</strong> "
        f"({_html.escape(council_recommendation_label(op.recommendation))}) — "
        f"{_html.escape(op.reasoning)}</li>"
        for op in dissenters
    )
    return (
        "<h3 style='font-size: 14px; margin: 16px 0 4px 0;'>"
        "🛡️ Wybronieni dysydenci rady</h3>"
        "<div style='font-size: 11px; color: #6b7280; margin-bottom: 4px;'>"
        "Głosowali zgodnie z rynkiem, wbrew finalnej rekomendacji rady.</div>"
        f"<ul style='margin: 0 0 8px 20px; font-size: 13px;'>{items}</ul>"
    )


def hero_verdict(highlight: RecapHighlight) -> str:
    """„✅ kierunek trafiony · błąd ceny docelowej 1,5%” (WIA-07).

    `accuracy` to bliskość ceny do celu, nie trafność — dlatego mail pokazuje
    błąd ceny docelowej (1 − accuracy, w % ceny wejścia), a trafność kierunku
    osobno. Werdykt pomijamy, gdy highlight go nie niesie.
    """
    error = Decimal("1") - highlight.accuracy
    error_label = (
        "błąd ceny docelowej ≥100%"
        if highlight.accuracy <= 0
        else f"błąd ceny docelowej {pct(error, decimals=1)}"
    )
    if highlight.direction_correct is None:
        return error_label
    direction = (
        "✅ kierunek trafiony" if highlight.direction_correct else "❌ kierunek chybiony"
    )
    return f"{direction} · {error_label}"


def best_title(recap: WeeklyRecap) -> str:
    """Tydzień bez trafień kierunkowych nie ma „strzału” — jest „najbliżej celu”."""
    if recap.best.direction_correct is False:
        return _BEST_MISS_TITLE
    return _BEST_TITLE


def worst_title(recap: WeeklyRecap) -> str:
    """Lustro `best_title`: tydzień bez pudeł kierunkowych nie ma „wtopy”."""
    if recap.worst.direction_correct is True:
        return _WORST_HIT_TITLE
    return _WORST_TITLE


def _hero_html(title: str, emoji: str, highlight: RecapHighlight, color: str) -> str:
    lesson = (
        f"<div style='font-size: 12px; color: #374151; margin-top: 4px;'>"
        f"Lekcja: {_html.escape(highlight.correction_insight)}</div>"
        if highlight.correction_insight
        else ""
    )
    return (
        f"<div style='margin-bottom: 12px;'>"
        f"<div style='font-size: 13px; color: {_COLOR_MUTED};'>{emoji} {title}</div>"
        f"<div style='font-size: 15px; font-weight: 600; color: {color};'>"
        f"{_html.escape(highlight.symbol)} "
        f"<span style='color: {_COLOR_MUTED}; font-weight: 400;'>"
        f"({hero_verdict(highlight)})</span></div>"
        f"{lesson}</div>"
    )


def _best_hero_html(recap: WeeklyRecap) -> str:
    if recap.best.direction_correct is False:
        return _hero_html(best_title(recap), "📏", recap.best, _COLOR_MUTED)
    return _hero_html(best_title(recap), "🎯", recap.best, _COLOR_GOOD)


def _worst_hero_html(recap: WeeklyRecap) -> str:
    if recap.worst.direction_correct is True:
        return _hero_html(worst_title(recap), "📏", recap.worst, _COLOR_MUTED)
    return _hero_html(worst_title(recap), "💥", recap.worst, _COLOR_POOR)


def render_weekly_recap_html(recap: WeeklyRecap | None, narrative: str = "") -> str:
    """Renderuje sekcję HTML retrospektywy lub "" gdy `recap is None`."""
    if recap is None:
        return ""

    narrative_html = (
        f"<div style='font-size: 13px; font-style: italic; color: #374151; "
        f"margin-bottom: 12px;'>{_html.escape(narrative)}</div>"
        if narrative
        else ""
    )

    return (
        "<h2 style='font-size: 16px; margin: 24px 0 8px 0;'>"
        "📅 Tydzień StockAgenta</h2>"
        "<div style='font-size: 11px; color: #6b7280; margin-bottom: 8px;'>"
        f"Retrospektywa z {recap.sample_size} rozliczonych predykcji.</div>"
        f"{narrative_html}"
        + _best_hero_html(recap)
        + _worst_hero_html(recap)
        + _dissenters_html(recap.worst.vindicated_dissenters)
        + _dissenters_html(recap.best.vindicated_dissenters)
        + _scores_html(recap)
        + _redundancy_html(recap)
        + "<div style='font-size: 12px; color: #6b7280; margin-top: 8px;'>"
        f"📐 Kalibracja: {_html.escape(_ece_delta_label(recap))}</div>"
        # MAIL-08: recap to osobny mail, więc zamyka go własne zastrzeżenie.
        + "<div style='font-size: 11px; color: #6b7280; margin-top: 16px; "
        "padding-top: 8px; border-top: 1px solid #e5e7eb;'>"
        f"{_html.escape(INVESTMENT_DISCLAIMER)}</div>"
    )


def _hero_text(title: str, highlight: RecapHighlight) -> list[str]:
    lines = [f"  {title}: {highlight.symbol} ({hero_verdict(highlight)})"]
    if highlight.correction_insight:
        lines.append(f"    Lekcja: {highlight.correction_insight}")
    for op in highlight.vindicated_dissenters:
        lines.append(
            f"    Wybroniony: {op.investor_name} "
            f"({council_recommendation_label(op.recommendation)}) — {op.reasoning}"
        )
    return lines


def render_weekly_recap_text(recap: WeeklyRecap | None, narrative: str = "") -> str:
    """Renderuje sekcję plain-text retrospektywy lub "" gdy `recap is None`."""
    if recap is None:
        return ""

    lines: list[str] = [
        "TYDZIEŃ STOCKAGENTA",
        "-" * 64,
        f"  Rozliczone predykcje: {recap.sample_size}",
    ]
    if narrative:
        lines.append(f"  {narrative}")
    lines.extend(_hero_text(best_title(recap), recap.best))
    lines.extend(_hero_text(worst_title(recap), recap.worst))
    scores = _scores_label(recap)
    if scores:
        lines.append(f"  Pewność: {scores}")
    redundancy = _redundancy_label(recap.redundancy)
    if redundancy:
        lines.append(f"  Rada: {redundancy}")
    lines.append(f"  Kalibracja: {_ece_delta_label(recap)}")
    lines.extend(["-" * 64, INVESTMENT_DISCLAIMER])
    return "\n".join(lines)
