"""Kalendarz sesji NYSE (ALR-15): święta, skrócone sesje i dni handlowe.

Status sesji w nagłówku maila był „uproszczony — bez świąt”, więc w Thanksgiving
mail pokazywał „Otwarta”, a reguła deadmana liczyła weekend jak dzień pracy. Tu
żyje statyczna lista dni zamknięcia i skróconych sesji NYSE oraz arytmetyka dni
handlowych. Wyłącznie stdlib (`zoneinfo`), zero I/O.

Lista wymaga corocznego dopisania kolejnego roku. `COVERED_YEARS` mówi, które lata
są znane, a strażnik testowy pada, gdy kalendarz nie zna bieżącego roku. Poza
znanymi latami kalendarz zna tylko weekendy — tak działał status sesji przed ALR-15.

Naiwny `datetime` traktujemy jako UTC, jak całe repo (`format_local`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from types import MappingProxyType
from zoneinfo import ZoneInfo

NYSE_TZ = ZoneInfo("America/New_York")
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
# Skrócona sesja akcji kończy się o 13:00 ET (opcje kwalifikowane 13:15 — nie dotyczy).
EARLY_CLOSE = time(13, 0)

# Źródło: https://www.nyse.com/markets/hours-calendars (odczyt 2026-10-19), lista
# „Holidays & Trading Hours” na lata 2026, 2027 i 2028, czas wschodni (ET). Daty
# przepisane 1:1; nazwy po polsku, oryginalna nazwa NYSE w komentarzu wiersza.
# 2028: 1 stycznia wypada w sobotę i NYSE nie obchodzi Nowego Roku w innym dniu.
NYSE_HOLIDAYS: Mapping[date, str] = MappingProxyType(
    {
        date(2026, 1, 1): "Nowy Rok",  # New Year's Day
        date(2026, 1, 19): "Dzień Martina Luthera Kinga",  # Martin Luther King, Jr. Day
        date(2026, 2, 16): "Dzień Prezydenta",  # Washington's Birthday
        date(2026, 4, 3): "Wielki Piątek",  # Good Friday
        date(2026, 5, 25): "Memorial Day",  # Memorial Day
        date(2026, 6, 19): "Juneteenth",  # Juneteenth
        date(2026, 7, 3): "Dzień Niepodległości",  # Independence Day (observed)
        date(2026, 9, 7): "Święto Pracy",  # Labor Day
        date(2026, 11, 26): "Święto Dziękczynienia",  # Thanksgiving Day
        date(2026, 12, 25): "Boże Narodzenie",  # Christmas Day
        date(2027, 1, 1): "Nowy Rok",  # New Year's Day
        date(2027, 1, 18): "Dzień Martina Luthera Kinga",  # Martin Luther King, Jr. Day
        date(2027, 2, 15): "Dzień Prezydenta",  # Washington's Birthday
        date(2027, 3, 26): "Wielki Piątek",  # Good Friday
        date(2027, 5, 31): "Memorial Day",  # Memorial Day
        date(2027, 6, 18): "Juneteenth",  # Juneteenth (observed)
        date(2027, 7, 5): "Dzień Niepodległości",  # Independence Day (observed)
        date(2027, 9, 6): "Święto Pracy",  # Labor Day
        date(2027, 11, 25): "Święto Dziękczynienia",  # Thanksgiving Day
        date(2027, 12, 24): "Boże Narodzenie",  # Christmas Day (observed)
        date(2028, 1, 17): "Dzień Martina Luthera Kinga",  # Martin Luther King, Jr. Day
        date(2028, 2, 21): "Dzień Prezydenta",  # Washington's Birthday
        date(2028, 4, 14): "Wielki Piątek",  # Good Friday
        date(2028, 5, 29): "Memorial Day",  # Memorial Day
        date(2028, 6, 19): "Juneteenth",  # Juneteenth
        date(2028, 7, 4): "Dzień Niepodległości",  # Independence Day
        date(2028, 9, 4): "Święto Pracy",  # Labor Day
        date(2028, 11, 23): "Święto Dziękczynienia",  # Thanksgiving Day
        date(2028, 12, 25): "Boże Narodzenie",  # Christmas Day
    }
)

# To samo źródło: skrócone sesje (zamknięcie 13:00 ET). Lista NYSE na 2026–2028
# wymienia dokładnie te pięć dni.
NYSE_EARLY_CLOSES: Mapping[date, str] = MappingProxyType(
    {
        date(2026, 11, 27): "dzień po Święcie Dziękczynienia",  # Day after Thanksgiving
        date(2026, 12, 24): "Wigilia",  # Christmas Eve
        date(2027, 11, 26): "dzień po Święcie Dziękczynienia",  # Day after Thanksgiving
        date(2028, 7, 3): "dzień przed Dniem Niepodległości",  # Day before Independence Day
        date(2028, 11, 24): "dzień po Święcie Dziękczynienia",  # Day after Thanksgiving
    }
)

COVERED_YEARS: frozenset[int] = frozenset({2026, 2027, 2028})


@dataclass(frozen=True)
class TradingSession:
    """Sesja regularna NYSE jednego dnia handlowego; godziny jako aware `datetime` w ET.
    `early_close_reason` jest ustawione wyłącznie dla skróconej sesji."""

    day: date
    opens_at: datetime
    closes_at: datetime
    early_close_reason: str | None = None

    @property
    def is_early_close(self) -> bool:
        return self.early_close_reason is not None


def is_calendar_covered(year: int) -> bool:
    """Czy kalendarz zna święta i skrócone sesje roku `year`."""
    return year in COVERED_YEARS


def holiday_name(day: date) -> str | None:
    """Nazwa święta NYSE w dniu `day` (dzień zamknięcia giełdy) albo None."""
    return NYSE_HOLIDAYS.get(day)


def is_trading_day(day: date) -> bool:
    """Dzień handlowy = dzień roboczy, który nie jest świętem NYSE. Poza znanymi
    latami (`COVERED_YEARS`) święta są nieznane i liczą się tylko weekendy."""
    return day.weekday() < 5 and day not in NYSE_HOLIDAYS


def session_on(day: date) -> TradingSession | None:
    """Sesja dnia `day` z godzinami otwarcia i zamknięcia; None w dzień bez sesji."""
    if not is_trading_day(day):
        return None
    reason = NYSE_EARLY_CLOSES.get(day)
    close = EARLY_CLOSE if reason is not None else REGULAR_CLOSE
    return TradingSession(
        day=day,
        opens_at=datetime.combine(day, REGULAR_OPEN, tzinfo=NYSE_TZ),
        closes_at=datetime.combine(day, close, tzinfo=NYSE_TZ),
        early_close_reason=reason,
    )


def trading_days_between(start: date, end: date) -> int:
    """Liczba dni handlowych `d`, dla których `start < d <= end` — ile sesji minęło
    od dnia `start` do dnia `end`. `end <= start` → 0."""
    count = 0
    day = start + timedelta(days=1)
    while day <= end:
        if is_trading_day(day):
            count += 1
        day += timedelta(days=1)
    return count


def upcoming_session(moment: datetime) -> TradingSession:
    """Sesja, której zamknięcie jako pierwsze następuje po `moment`: trwająca albo
    najbliższa przyszła."""
    nyc = to_nyse_time(moment)
    day = nyc.date()
    while True:
        session = session_on(day)
        if session is not None and session.closes_at > nyc:
            return session
        day += timedelta(days=1)


def next_open(moment: datetime) -> datetime:
    """Najbliższe otwarcie sesji ściśle po `moment` (aware, w ET). W trakcie sesji
    to otwarcie następnego dnia handlowego."""
    nyc = to_nyse_time(moment)
    day = nyc.date()
    while True:
        session = session_on(day)
        if session is not None and session.opens_at > nyc:
            return session.opens_at
        day += timedelta(days=1)


def to_nyse_time(moment: datetime) -> datetime:
    """`moment` w strefie NYSE; naiwny `datetime` to UTC."""
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
    return aware.astimezone(NYSE_TZ)
