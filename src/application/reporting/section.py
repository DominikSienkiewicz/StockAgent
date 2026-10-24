"""ARCH-12: kontrakt sekcji maila — `ReportSection` i wspólne wejście `SectionInput`.

Mail to uporządkowana krotka sekcji z `registry.py`. `build_html_report` liczy
raz `SectionInput`, a każda sekcja oddaje z niego fragment HTML i linie wersji
tekstowej. Pusty fragment i pusta lista znaczą „sekcji nie ma”, więc sekcja bez
danych chowa się sama.

Nowa sekcja = moduł `src/application/report_<nazwa>.py` z rendererem i obiektem
`ReportSection`, wpis w `REPORT_SECTIONS` i pole `ReportContext`, które sekcja
wymienia w `context_fields`. Strażnik `tests/application/test_report_registry.py`
pilnuje, że każde pole `ReportContext` ma dokładnie jednego konsumenta: jedną
sekcję albo wspólne wejście (`SHARED_CONTEXT_FIELDS`).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from src.application.report_models import ResolvedPrediction, SymbolResult, TradeSignal
from src.domain.digest_lead import LeadItem
from src.domain.portfolio import PlnRates

if TYPE_CHECKING:
    # Tylko typy: report_builder i moduły sekcji importują ten plik — import
    # w czasie wykonania zamknąłby cykl.
    from src.application.report_builder import ReportContext
    from src.application.report_user_portfolio import PortfolioChangeView


# Pola `ReportContext`, których konsumentem jest wspólne wejście, a nie jedna
# sekcja: `build_html_report` liczy z nich `SectionInput` (filtr subskrybenta,
# ceny i kursy, lead, ranking sygnałów, okno rozliczeń, chipy trafności) albo
# czyta je kilka sekcji. Każda sekcja może je czytać; pole spoza tej listy ma
# dokładnie jedną sekcję w jej `context_fields`.
SHARED_CONTEXT_FIELDS: frozenset[str] = frozenset(
    {
        # U2: watchlista subskrybenta tnie wyniki przed wszystkimi sekcjami.
        "symbols_filter",
        # Ceny raportu, kursy NBP i zmiana portfela (PORT-06/07/11).
        "user_portfolio",
        "held_prices",
        "portfolio_fx",
        "macro_risk_report",
        "user_portfolio_in_text",
        # Lead (#1): też temat maila i push.
        "quota_alerts",
        "consensus_shifts",
        "alpha_signals",
        "insider_lead_window",
        "regime",
        # Ranking sygnałów (MAIL-20) i sizing (Q7, PORT-04/08, #9).
        "accuracy_stats",
        "signal_calibration_buckets",
        "position_risk",
        "portfolio_clusters",
        "symbol_track_record",
        # Rozliczenia: pierwszy ekran, sekcja zamkniętych predykcji i preheader.
        "resolved_predictions",
        "resolved_this_cycle",
        "resolved_unavailable",
        # Etykiety sektorów i strefa czasu — wiele sekcji.
        "sector_map",
        "display_timezone",
        # ALR-06: szoki od ostatniego maila — sekcja ⚡ i kandydat leadu.
        "shock_alerts",
    }
)


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
    # MAIL-14: start cyklu w strefie wyświetlania — nagłówek HTML i tekstu.
    started_local: str
    # PORT-05: `ReportContext.sector_map` albo pusta mapa.
    sector_map: Mapping[str, str]
    # #1: lead cyklu — jedno źródło treści maila, preheadera i tematu.
    lead_items: list[LeadItem]
    # `ReportContext.resolved_predictions` albo pusta lista i okno ich rozliczeń
    # (WIA-19) — sekcja zamkniętych predykcji i pierwszy ekran.
    resolved: list[ResolvedPrediction]
    resolved_window: str
    # #15/PORT-07: ceny raportu — z wyników cyklu, uzupełnione `held_prices`.
    current_prices: Mapping[str, Decimal]
    # PORT-11: kursy NBP do „P/L zł”, wag i zmiany portfela; USD/PLN z Risk Watch.
    pln_rates: PlnRates | None
    usd_pln: Decimal | None
    # PORT-06: zmiana portfela od ostatniego maila — pierwszy ekran i sekcja 💼.
    portfolio_change: PortfolioChangeView | None
    # MAIL-20: czołówka rankingu sygnałów (🎯) — ten sam ranking co push.
    trade_signals: list[TradeSignal]
    # MAIL-06: symbole z pełnym blokiem uzasadnień (top N i lead); None = każdy.
    full_detail: frozenset[str] | None
    # WIA-35: trafność kierunkowa per symbol — chip przy sygnale i predykcji.
    symbol_chips: Mapping[str, str]

    @property
    def large_day(self) -> bool:
        """MAIL-06: cykl ponad próg pełnych uzasadnień (`full_detail` nie jest
        None) — Track Record też idzie w wersji zwartej, żeby zastrzeżenie
        zmieściło się w ~100 KB."""
        return self.full_detail is not None


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
