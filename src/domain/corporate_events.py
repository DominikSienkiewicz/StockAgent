"""Domena: zdarzenia korporacyjne z raportów bieżących SEC 8-K (DAT-17).

Duży ruch bez newsów często ma przyczynę w 8-K: wyniki (item 2.02), zmiana
w zarządzie (5.02), istotna umowa (1.01). Adapter EDGAR wyciąga je z tej samej
odpowiedzi submissions, z której czyta Form 4 — zero nowych requestów — a domena
mówi, czy w oknie jest świeży 8-K i co oznaczają jego itemy.

Dotyczy wyłącznie emitentów US z CIK: ADR-y składają 6-K, a ETF-y w ogóle nie
składają 8-K, więc brak zdarzeń nie znaczy „nic się nie stało”.

Zero importów zewnętrznych — tylko stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# Formularz raportu bieżącego i jego poprawka.
CURRENT_REPORT_FORMS = frozenset({"8-K", "8-K/A"})

# Świeży 8-K: złożony najwyżej tyle dni kalendarzowych przed dniem cyklu
# (weekend między piątkowym filingiem a poniedziałkowym ruchem się mieści).
FRESH_EVENT_MAX_AGE_DAYS = 3

# Itemy 8-K, które najczęściej tłumaczą ruch ceny pojedynczej spółki.
ITEM_LABELS_PL: dict[str, str] = {
    "1.01": "istotna umowa",
    "1.02": "rozwiązanie istotnej umowy",
    "2.01": "przejęcie lub zbycie aktywów",
    "2.02": "wyniki finansowe",
    "2.05": "restrukturyzacja",
    "2.06": "odpis aktywów",
    "3.01": "ryzyko wycofania z giełdy",
    "4.02": "korekta sprawozdań",
    "5.02": "zmiana w zarządzie",
    "7.01": "komunikat Reg FD",
    "8.01": "inne istotne zdarzenie",
}
# Item 9.01 to same załączniki (exhibits) — nie mówi, co się stało.
_EXHIBITS_ONLY = "9.01"


@dataclass(frozen=True)
class CorporateEvent:
    """Jeden raport bieżący: formularz, itemy (np. ("2.02", "9.01")), dzień filingu
    i link do dokumentu dla człowieka (None, gdy nieznany)."""

    form: str
    items: tuple[str, ...]
    filed_on: date
    filing_url: str | None = None

    def material_items(self) -> tuple[str, ...]:
        """Itemy bez samych załączników (9.01)."""
        return tuple(item for item in self.items if item != _EXHIBITS_ONLY)

    def labels_pl(self) -> tuple[str, ...]:
        """Polskie etykiety znanych itemów; nieznany item zostaje swoim numerem."""
        return tuple(ITEM_LABELS_PL.get(item, f"item {item}") for item in self.material_items())

    def is_earnings(self) -> bool:
        """Item 2.02 — publikacja wyników."""
        return "2.02" in self.items


@dataclass(frozen=True)
class CorporateEventSnapshot:
    """Raporty bieżące symbolu z okna `window_days`, od najnowszego."""

    symbol: str
    events: tuple[CorporateEvent, ...]
    window_days: int

    def latest(self) -> CorporateEvent | None:
        return self.events[0] if self.events else None

    def fresh_events(
        self, as_of: date, max_age_days: int = FRESH_EVENT_MAX_AGE_DAYS
    ) -> tuple[CorporateEvent, ...]:
        """Raporty złożone w ostatnich `max_age_days` dniach przed `as_of`
        (włącznie z tym dniem); filing z przyszłości (zegar SEC vs cyklu) też się liczy."""
        return tuple(
            event for event in self.events if (as_of - event.filed_on).days <= max_age_days
        )

    def has_fresh_event(
        self, as_of: date, max_age_days: int = FRESH_EVENT_MAX_AGE_DAYS
    ) -> bool:
        """Flaga świeżego 8-K do promptu, cech i lessons (konsumenci za flagą)."""
        return bool(self.fresh_events(as_of, max_age_days))
