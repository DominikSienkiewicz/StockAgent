"""Render sekcji powitalnej "Dzień 1" w raporcie e-mail (roadmap #4).

Problem: pierwszy mail nowego deploymentu to ściana "⏸ Pominięte" (wszystkie
symbole to cold-start — brak punktu odniesienia dla bramki volatility). Bez
kontekstu wygląda nieodróżnialnie od cyklu, w którym padły wszystkie źródła
cen. Ta sekcja zamienia tę ścianę w czytelne powitanie: "buduję punkt
odniesienia dla N instrumentów, pierwsze predykcje jutro".

Render-only, zero płatnych wywołań. Renderer NIE decyduje sam o tonie — bierze
gotową klasyfikację `CycleMaturity` z domeny (`classify_cycle`). To zamraża
największe ryzyko: cykl zdominowany przez awarie źródeł nigdy nie zostanie
ubrany w powitalny ton, bo domena zwróci wtedy `STEADY_STATE`.

Sekcja samosupresująca: `CycleMaturity.STEADY_STATE` → obie `render_*` zwracają
"", a `onboarding_subject` zwraca `None` (orkiestrator zostawia zwykły temat).
"""

from __future__ import annotations

import html as _html
from collections.abc import Mapping, Sequence

from src.application.reporting.section import ReportSection, SectionInput, text_block
from src.domain.cycle_maturity import CycleMaturity


def _instruments_label(count: int) -> str:
    """Polska odmiana rzeczownika "instrument" przez liczebnik.

    1 instrument / 2-4 instrumenty / 5+ instrumentów, z wyjątkiem nastek
    (12-14 → "instrumentów"). Odmiana idzie po KOŃCÓWCE liczby, więc 22 →
    "instrumenty", a 25 → "instrumentów".
    """
    if count == 1:
        return "1 instrument"
    last_two = count % 100
    last_one = count % 10
    if 2 <= last_one <= 4 and not 12 <= last_two <= 14:
        return f"{count} instrumenty"
    return f"{count} instrumentów"


def _ordered_sectors(
    sectors: Mapping[str, int], sector_order: Sequence[str] = ()
) -> list[tuple[str, int]]:
    """Porządkuje sektory portfela wg kanonicznej kolejności `sector_order`.

    `sector_order` to unikalne etykiety mapy sektorów z configu w kolejności
    wstawienia (PORT-05). Sektory obecne w portfelu, ale spoza tej kolejności
    (defensywnie) dostawiamy na końcu w kolejności podanej przez wołającego.
    Sektory z zerową liczebnością pomijamy.
    """
    canonical = list(dict.fromkeys(sector_order))
    ordered_names = [name for name in canonical if name in sectors]
    ordered_names += [name for name in sectors if name not in canonical]
    return [(name, sectors[name]) for name in ordered_names if sectors[name] > 0]


# MAIL-22: przewodnik „jak czytać ten mail” — tylko w pierwszym mailu deploymentu.
READING_GUIDE_TITLE = "📖 Jak czytać ten mail"
READING_GUIDE = (
    "Pewność przy prognozie to deklarowana siła sygnału modelu, nie prawdopodobieństwo "
    "trafienia — ile jest warta, pokazuje kalibracja w Track Recordzie.",
    "KUP / SPRZEDAJ / TRZYMAJ to sygnał modelu i rady person, nie rekomendacja "
    "inwestycyjna.",
    "Wyniki portfela to paper-PnL: symulacja na papierze, bez realnych zleceń, "
    "poślizgu i podatków.",
    "Ile model naprawdę trafia, mówi sekcja 📈 Track Record w Zapleczu (trafność "
    "kierunkowa, Brier, krzywa kapitału) — pierwsze liczby po kilku rozliczonych cyklach.",
)


def _reading_guide_html() -> str:
    items = "".join(
        f"<li style='margin: 2px 0;'>{_html.escape(sentence)}</li>" for sentence in READING_GUIDE
    )
    return (
        "<h3 style='font-size: 14px; margin: 12px 0 4px 0; color: #1e3a8a;'>"
        "📖 Jak czytać ten mail</h3>"
        "<ul style='margin: 0; padding-left: 18px; font-size: 13px; "
        f"color: #374151;'>{items}</ul>"
    )


def render_onboarding_html(
    maturity: CycleMaturity,
    *,
    instrument_count: int,
    sectors: Mapping[str, int],
    sector_order: Sequence[str] = (),
) -> str:
    """Renderuje sekcję HTML powitania "Dzień 1" lub "" dla STEADY_STATE.

    `sectors` to gotowe zliczenie sektor → liczba instrumentów (agregat
    orkiestratora), `sector_order` — kanoniczna kolejność sektorów z configu.
    Wszystkie etykiety idące do HTML są escapowane.
    """
    if maturity is CycleMaturity.STEADY_STATE:
        return ""

    instruments = _html.escape(_instruments_label(instrument_count))
    rows = "".join(
        "<li style='margin: 2px 0;'>"
        f"<strong>{_html.escape(name)}</strong>"
        f" — {count}</li>"
        for name, count in _ordered_sectors(sectors, sector_order)
    )
    composition = (
        "<ul style='margin: 8px 0 0 0; padding-left: 18px; "
        f"color: #374151;'>{rows}</ul>"
        if rows
        else ""
    )

    return (
        "<div style='background: #eff6ff; border: 1px solid #bfdbfe; "
        "border-radius: 8px; padding: 12px 16px; margin: 0 0 16px 0;'>"
        "<h2 style='font-size: 16px; margin: 0 0 6px 0; color: #1e3a8a;'>"
        "👋 Dzień 1 — buduję punkt odniesienia</h2>"
        "<div style='font-size: 13px; color: #374151;'>"
        f"Startuję dla {instruments}. Ten cykl zapisuje pierwsze ceny jako "
        "punkt odniesienia — <strong>pierwsze predykcje pojawią się "
        "jutro</strong>, gdy będzie z czym porównać ruch. Pominięcia poniżej "
        "to nie awaria, tylko brak historii."
        "</div>"
        f"{composition}"
        f"{_reading_guide_html()}"
        "</div>"
    )


def render_onboarding_text(
    maturity: CycleMaturity,
    *,
    instrument_count: int,
    sectors: Mapping[str, int],
    sector_order: Sequence[str] = (),
) -> str:
    """Renderuje tekstowy wariant powitania "Dzień 1" lub "" dla STEADY_STATE."""
    if maturity is CycleMaturity.STEADY_STATE:
        return ""

    lines = [
        "👋 Dzień 1 — buduję punkt odniesienia",
        (
            f"Startuję dla {_instruments_label(instrument_count)}. Ten cykl "
            "zapisuje pierwsze ceny jako punkt odniesienia — pierwsze predykcje "
            "pojawią się jutro. Pominięcia poniżej to nie awaria, tylko brak "
            "historii."
        ),
    ]
    lines += [
        f"  • {name} — {count}" for name, count in _ordered_sectors(sectors, sector_order)
    ]
    lines += ["", READING_GUIDE_TITLE, *(f"  • {sentence}" for sentence in READING_GUIDE)]
    return "\n".join(lines)


def onboarding_subject(
    maturity: CycleMaturity,
    *,
    instrument_count: int,
) -> str | None:
    """Temat maila dla "Dnia 1" lub `None` dla STEADY_STATE.

    `None` sygnalizuje orkiestratorowi, że ma zostawić zwykły temat raportu.
    """
    if maturity is CycleMaturity.STEADY_STATE:
        return None
    return (
        f"👋 Dzień 1: startuję — buduję punkt odniesienia dla "
        f"{_instruments_label(instrument_count)}"
    )


# Renderer nigdy nie decyduje o tonie — dostaje gotową klasyfikację z domeny
# (brak = STEADY_STATE, czyli sekcji nie ma). Kolejność sektorów: kolejność
# wstawienia mapy sektorów z configu.


def _section_html(data: SectionInput) -> str:
    context = data.context
    return render_onboarding_html(
        context.cycle_maturity or CycleMaturity.STEADY_STATE,
        instrument_count=len(data.results),
        sectors=context.portfolio_sectors or {},
        sector_order=tuple(dict.fromkeys(data.sector_map.values())),
    )


def _section_text(data: SectionInput) -> list[str]:
    context = data.context
    text = render_onboarding_text(
        context.cycle_maturity or CycleMaturity.STEADY_STATE,
        instrument_count=len(data.results),
        sectors=context.portfolio_sectors or {},
        sector_order=tuple(dict.fromkeys(data.sector_map.values())),
    )
    return text_block(text, spaced=True)


# #4: powitanie pierwszego cyklu („Dzień 1”) zamiast ściany „Pominięte”.
ONBOARDING_SECTION = ReportSection(
    key="onboarding",
    render_html=_section_html,
    render_text=_section_text,
    context_fields=frozenset({"cycle_maturity", "portfolio_sectors"}),
)
