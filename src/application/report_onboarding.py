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
from collections.abc import Mapping

from src.application.report_formatting import SECTORS
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


def _ordered_sectors(sectors: Mapping[str, int]) -> list[tuple[str, int]]:
    """Porządkuje sektory portfela wg kanonicznej kolejności z `SECTORS`.

    `SECTORS` (single source of truth: symbol → sektor) wyznacza też kolejność
    sektorów — bierzemy unikalne etykiety w kolejności wstawienia. Sektory
    obecne w portfelu, ale spoza mapy (defensywnie) dostawiamy na końcu w
    kolejności podanej przez wołającego. Sektory z zerową liczebnością pomijamy.
    """
    canonical = list(dict.fromkeys(SECTORS.values()))
    ordered_names = [name for name in canonical if name in sectors]
    ordered_names += [name for name in sectors if name not in canonical]
    return [(name, sectors[name]) for name in ordered_names if sectors[name] > 0]


def render_onboarding_html(
    maturity: CycleMaturity,
    *,
    instrument_count: int,
    sectors: Mapping[str, int],
) -> str:
    """Renderuje sekcję HTML powitania "Dzień 1" lub "" dla STEADY_STATE.

    `sectors` to gotowe zliczenie sektor → liczba instrumentów (agregat
    orkiestratora). Wszystkie etykiety idące do HTML są escapowane.
    """
    if maturity is CycleMaturity.STEADY_STATE:
        return ""

    instruments = _html.escape(_instruments_label(instrument_count))
    rows = "".join(
        "<li style='margin: 2px 0;'>"
        f"<strong>{_html.escape(name)}</strong>"
        f" — {count}</li>"
        for name, count in _ordered_sectors(sectors)
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
        "</div>"
    )


def render_onboarding_text(
    maturity: CycleMaturity,
    *,
    instrument_count: int,
    sectors: Mapping[str, int],
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
        f"  • {name} — {count}" for name, count in _ordered_sectors(sectors)
    ]
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
