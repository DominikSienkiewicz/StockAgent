"""Kalendarz sesji NYSE (ALR-15): święta, skrócone sesje i dni handlowe 2026–2028.

Daty pochodzą z oficjalnej listy NYSE (https://www.nyse.com/markets/hours-calendars,
odczyt 2026-10-19). Testy przypinają tę listę, a strażnik roku pada, gdy kalendarz
nie zna bieżącego roku z zegara testu — lista wymaga corocznego dopisania.
"""

from __future__ import annotations

import warnings
from datetime import UTC, date, datetime

from src.domain.market_calendar import (
    COVERED_YEARS,
    NYSE_EARLY_CLOSES,
    NYSE_HOLIDAYS,
    NYSE_TZ,
    holiday_name,
    is_calendar_covered,
    is_trading_day,
    next_open,
    session_on,
    trading_days_between,
    upcoming_session,
)


def _et(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=NYSE_TZ)


class TestCoverageGuard:
    def test_calendar_covers_the_current_year(self) -> None:
        # Strażnik ALR-15: rok z zegara testu. 1 stycznia roku spoza listy CI robi
        # się czerwone, zanim mail pokaże „Otwarta” w święto nowego roku.
        current_year = date.today().year
        assert is_calendar_covered(current_year), (
            f"Kalendarz NYSE nie zna roku {current_year} — dopisz święta i skrócone "
            "sesje z https://www.nyse.com/markets/hours-calendars do "
            "src/domain/market_calendar.py"
        )

    def test_calendar_covers_2026_through_2028(self) -> None:
        assert set(COVERED_YEARS) == {2026, 2027, 2028}

    def test_warns_in_the_last_quarter_when_next_year_is_missing(self) -> None:
        # Wczesne ostrzeżenie (bez czerwieni): od 1 października ostatniego znanego
        # roku podsumowanie pytesta przypomina o dopisaniu kolejnego roku.
        today = date.today()
        if today.month >= 10 and not is_calendar_covered(today.year + 1):
            warnings.warn(
                f"Kalendarz NYSE kończy się na {today.year} — dopisz {today.year + 1}.",
                UserWarning,
                stacklevel=1,
            )

    def test_year_outside_the_list_is_not_covered(self) -> None:
        assert not is_calendar_covered(2025)
        assert not is_calendar_covered(2029)


class TestOfficialList:
    def test_full_closes_per_year_match_the_nyse_list(self) -> None:
        per_year = {year: sum(1 for d in NYSE_HOLIDAYS if d.year == year) for year in COVERED_YEARS}
        # 2028: Nowy Rok wypada w sobotę i NYSE nie obchodzi go w zamian.
        assert per_year == {2026: 10, 2027: 10, 2028: 9}

    def test_new_year_2028_falls_on_saturday_without_an_observed_day(self) -> None:
        assert date(2028, 1, 1).weekday() == 5
        assert date(2028, 1, 1) not in NYSE_HOLIDAYS
        assert is_trading_day(date(2027, 12, 31))

    def test_observed_holidays_are_the_weekday_ones(self) -> None:
        assert holiday_name(date(2026, 7, 3)) is not None  # Independence Day (observed)
        assert holiday_name(date(2027, 6, 18)) is not None  # Juneteenth (observed)
        assert holiday_name(date(2027, 7, 5)) is not None  # Independence Day (observed)
        assert holiday_name(date(2027, 12, 24)) is not None  # Christmas Day (observed)

    def test_every_closed_day_is_a_weekday_in_a_covered_year(self) -> None:
        for day in (*NYSE_HOLIDAYS, *NYSE_EARLY_CLOSES):
            assert day.weekday() < 5, f"{day} wypada w weekend — błąd przepisania listy"
            assert day.year in COVERED_YEARS

    def test_early_closes_are_trading_days(self) -> None:
        assert sorted(NYSE_EARLY_CLOSES) == [
            date(2026, 11, 27),
            date(2026, 12, 24),
            date(2027, 11, 26),
            date(2028, 7, 3),
            date(2028, 11, 24),
        ]
        for day in NYSE_EARLY_CLOSES:
            assert is_trading_day(day)


class TestTradingDays:
    def test_thanksgiving_2026_is_not_a_trading_day(self) -> None:
        assert not is_trading_day(date(2026, 11, 26))

    def test_weekend_is_not_a_trading_day(self) -> None:
        assert not is_trading_day(date(2026, 10, 17))
        assert not is_trading_day(date(2026, 10, 18))

    def test_regular_weekday_is_a_trading_day(self) -> None:
        assert is_trading_day(date(2026, 10, 15))

    def test_outside_the_list_only_weekends_are_known(self) -> None:
        # Poza znanymi latami kalendarz zna tylko weekendy (zachowanie sprzed ALR-15).
        assert is_trading_day(date(2029, 1, 1))
        assert not is_trading_day(date(2029, 1, 6))

    def test_trading_days_between_skips_weekend_and_holiday(self) -> None:
        assert trading_days_between(date(2026, 10, 16), date(2026, 10, 19)) == 1
        assert trading_days_between(date(2026, 11, 25), date(2026, 11, 27)) == 1
        assert trading_days_between(date(2026, 12, 23), date(2027, 1, 4)) == 6

    def test_trading_days_between_same_or_reversed_days_is_zero(self) -> None:
        assert trading_days_between(date(2026, 10, 15), date(2026, 10, 15)) == 0
        assert trading_days_between(date(2026, 10, 19), date(2026, 10, 16)) == 0


class TestSessions:
    def test_regular_session_hours(self) -> None:
        session = session_on(date(2026, 10, 15))
        assert session is not None
        assert session.opens_at == _et(2026, 10, 15, 9, 30)
        assert session.closes_at == _et(2026, 10, 15, 16, 0)
        assert not session.is_early_close

    def test_day_after_thanksgiving_closes_at_1_pm(self) -> None:
        session = session_on(date(2026, 11, 27))
        assert session is not None
        assert session.is_early_close
        assert session.closes_at == _et(2026, 11, 27, 13, 0)

    def test_holiday_has_no_session(self) -> None:
        assert session_on(date(2026, 12, 25)) is None

    def test_upcoming_session_is_the_one_in_progress(self) -> None:
        session = upcoming_session(_et(2026, 10, 15, 11, 0))
        assert session.day == date(2026, 10, 15)

    def test_upcoming_session_after_close_skips_the_holiday(self) -> None:
        session = upcoming_session(_et(2026, 11, 25, 17, 0))
        assert session.day == date(2026, 11, 27)
        assert session.is_early_close


class TestNextOpen:
    def test_premarket_opens_the_same_day(self) -> None:
        assert next_open(_et(2026, 10, 15, 6, 30)) == _et(2026, 10, 15, 9, 30)

    def test_friday_after_hours_opens_on_monday(self) -> None:
        assert next_open(_et(2026, 10, 16, 17, 0)) == _et(2026, 10, 19, 9, 30)

    def test_before_thanksgiving_opens_on_friday(self) -> None:
        assert next_open(_et(2026, 11, 25, 17, 0)) == _et(2026, 11, 27, 9, 30)

    def test_weekend_before_mlk_day_opens_on_tuesday(self) -> None:
        assert next_open(_et(2026, 1, 17, 12, 0)) == _et(2026, 1, 20, 9, 30)

    def test_during_session_the_next_open_is_tomorrow(self) -> None:
        assert next_open(_et(2026, 10, 15, 9, 30)) == _et(2026, 10, 16, 9, 30)

    def test_utc_input_respects_daylight_saving_end(self) -> None:
        # 2026-11-01 kończy się czas letni: poniedziałkowe 09:30 EST to 14:30 UTC.
        opening = next_open(datetime(2026, 11, 1, 12, 0, tzinfo=UTC))
        assert opening.astimezone(UTC) == datetime(2026, 11, 2, 14, 30, tzinfo=UTC)

    def test_naive_datetime_is_read_as_utc(self) -> None:
        naive = datetime(2026, 10, 15, 10, 30)
        assert next_open(naive) == next_open(naive.replace(tzinfo=UTC))


def test_closure_reason_names_the_holiday_the_weekend_or_nothing() -> None:
    # FIN-34: Fast Loop pomija akcje w dzień zamknięcia NYSE i loguje powód.
    from src.domain.market_calendar import closure_reason

    assert closure_reason(date(2026, 12, 25)) == "Boże Narodzenie"
    assert closure_reason(date(2026, 10, 24)) == "weekend"
    assert closure_reason(date(2026, 12, 23)) is None
    # Skrócona sesja to wciąż dzień handlowy.
    assert closure_reason(date(2026, 11, 27)) is None


class TestLastStartedSession:
    """DAT-14: sesja, którą świeże notowanie musi już odzwierciedlać."""

    def test_during_a_session_it_is_today(self) -> None:
        from src.domain.market_calendar import last_started_session

        session = last_started_session(datetime(2026, 10, 16, 17, 0, tzinfo=UTC))

        assert session.day == date(2026, 10, 16)

    def test_before_the_open_it_is_the_previous_trading_day(self) -> None:
        from src.domain.market_calendar import last_started_session

        session = last_started_session(datetime(2026, 10, 19, 12, 0, tzinfo=UTC))

        assert session.day == date(2026, 10, 16)

    def test_on_a_holiday_it_is_the_day_before(self) -> None:
        from src.domain.market_calendar import last_started_session

        session = last_started_session(datetime(2026, 11, 26, 17, 0, tzinfo=UTC))

        assert session.day == date(2026, 11, 25)
