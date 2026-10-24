"""ARCH-12: kontrakt sekcji maila — `ReportSection` i wspólne wejście `SectionInput`.

Mail to uporządkowana krotka sekcji z `registry.py`. `build_html_report` liczy
raz `SectionInput`, a każda sekcja oddaje z niego fragment HTML i linie wersji
tekstowej. Pusty fragment i pusta lista znaczą „sekcji nie ma”, więc sekcja bez
danych chowa się sama.

Nowa sekcja = moduł `src/application/report_<nazwa>.py` z rendererem i obiektem
`ReportSection`, wpis w `REPORT_SECTIONS` i pole `ReportContext`, które sekcja
wymienia w `context_fields`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING

from src.application.report_models import ResolvedPrediction, SymbolResult
from src.domain.digest_lead import LeadItem

if TYPE_CHECKING:
    # Tylko typ: report_builder importuje moduły sekcji, a te ten plik — import
    # w czasie wykonania zamknąłby cykl.
    from src.application.report_builder import ReportContext


@dataclasses.dataclass(frozen=True)
class SectionInput:
    """Wejście każdej sekcji: wyniki cyklu i pochodne liczone RAZ w `build_html_report`.

    Pochodna trafia tutaj, gdy łączy kilka pól `ReportContext` albo czyta ją
    więcej niż jedna sekcja (lead → preheader i lead). Pole kontekstu, które czyta
    jedna sekcja, sekcja bierze z `context` sama.
    """

    context: ReportContext
    # U2: wyniki przycięte do watchlisty subskrybenta i ich podział po statusie.
    results: list[SymbolResult]
    saved: list[SymbolResult]
    ignored: list[SymbolResult]
    errors: list[SymbolResult]
    started_at: datetime
    duration_seconds: float
    # PORT-05: `ReportContext.sector_map` albo pusta mapa.
    sector_map: Mapping[str, str]
    # #1: lead cyklu — jedno źródło treści maila, preheadera i tematu.
    lead_items: list[LeadItem]
    # `ReportContext.resolved_predictions` albo pusta lista.
    resolved: list[ResolvedPrediction]


@dataclasses.dataclass(frozen=True)
class ReportSection:
    """Jedna sekcja maila.

    `render_html` oddaje fragment HTML ("" = brak sekcji), `render_text` linie
    wersji tekstowej, które składanie łączy znakiem nowej linii ([] = brak
    sekcji). Sekcja tylko w HTML (wykresy, tabele) bierze `no_text`.
    `context_fields` to pola `ReportContext`, których ta sekcja jest jedynym
    konsumentem.
    """

    key: str
    render_html: Callable[[SectionInput], str]
    render_text: Callable[[SectionInput], list[str]]
    context_fields: frozenset[str] = frozenset()


def no_text(data: SectionInput) -> list[str]:
    """Sekcja świadomie nieobecna w wersji tekstowej (wykres, tabela inline)."""
    return []


def text_block(text: str, *, spaced: bool = False) -> list[str]:
    """Gotowy tekst sekcji jako linie maila: [] dla pustego, z `spaced` także
    pusta linia odstępu za nim (renderer, którego tekst sam jej nie ma)."""
    if not text:
        return []
    return [text, ""] if spaced else [text]
