"""Domena: oficjalne zamknięcie sesji NYSE jako druga cena rozliczenia (WIA-61).

Dziś predykcję rozlicza cena /quote z chwili cyklu (~05:30 UTC) — nikt z zewnątrz
nie odtworzy jej co do centa. Okres przejściowy (decyzja właściciela, opcja
„oba”): obok niej zapisujemy zamknięcie sesji z datą i źródłem, które każdy
sprawdzi w publicznym serwisie, i raportujemy obie trafności.

Zamknięcie bierzemy z TEGO SAMEGO notowania, bez nowego źródła: po zamknięciu
sesji cena notowania (`c` w Finnhub /quote) jest jej zamknięciem, a w trakcie
następnej sesji zamknięcie poprzedniej niesie `prev_close` (`pc`). Krypto nie ma
sesji, więc nie dostaje tej ceny. Zero importów zewnętrznych.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum

from src.domain.market_calendar import last_started_session, next_open, to_nyse_time
from src.domain.value_objects import Money, Quote


class SessionCloseSource(StrEnum):
    """Pole notowania, z którego pochodzi zamknięcie (`prediction_logs.settlement_source`)."""

    # Notowanie wzięte po zamknięciu sesji — jego cena to zamknięcie tej sesji.
    QUOTE_CLOSE = "quote_close"
    # Notowanie z trwającej sesji — zamknięcie poprzedniej to jego `prev_close`.
    QUOTE_PREVIOUS_CLOSE = "quote_previous_close"


@dataclass(frozen=True)
class SessionClose:
    """Zamknięcie sesji NYSE z dnia `session_day` (kalendarz NYSE)."""

    price: Money
    session_day: date
    source: SessionCloseSource

    def settles(self, predicted_at: datetime) -> bool:
        """Czy zamknięcie może rozliczyć predykcję z `predicted_at` — pochodzi
        z pierwszej sesji otwartej po niej albo z późniejszej. Zamknięcie sprzed
        horyzontu (weekend, święto, drugi przebieg) jest ceną znaną w chwili
        predykcji. Naiwny `datetime` to UTC."""
        return self.session_day >= next_open(predicted_at).date()


@dataclass(frozen=True)
class SessionCloseSettlement:
    """Rozliczenie predykcji na zamknięciu sesji: cena z datą i źródłem oraz
    trafność kierunku na tej cenie (to samo pasmo SIDEWAYS co na /quote)."""

    close: SessionClose
    is_trend_correct: bool


def session_close_of(quote: Quote, now: datetime) -> SessionClose | None:
    """Zamknięcie ostatniej zakończonej sesji NYSE z notowania wziętego w `now`.

    Po zamknięciu ostatniej rozpoczętej sesji (także w nocy, weekend i święto)
    cena notowania jest jej zamknięciem; w trakcie sesji zamknięciem poprzedniej
    jest `prev_close`. Notowanie sprzed otwarcia ostatniej rozpoczętej sesji nie
    mówi, której sesji dotyczy — wtedy, jak bez `prev_close` w trakcie sesji, None.
    Naiwny `datetime` to UTC."""
    session = last_started_session(now)
    if to_nyse_time(quote.quoted_at) < session.opens_at:
        return None
    if to_nyse_time(now) >= session.closes_at:
        return SessionClose(quote.price, session.day, SessionCloseSource.QUOTE_CLOSE)
    if quote.prev_close is None:
        return None
    previous = last_started_session(session.opens_at - timedelta(seconds=1))
    return SessionClose(quote.prev_close, previous.day, SessionCloseSource.QUOTE_PREVIOUS_CLOSE)
