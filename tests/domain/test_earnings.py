"""Testy domeny earnings — klasyfikacja bliskości raportu wyników."""

from __future__ import annotations

from datetime import date

from src.domain.earnings import EarningsEvent, EarningsProximity


def test_proximity_imminent_within_two_days() -> None:
    # Domyślny próg IMMINENT to <= 2 dni.
    assert EarningsEvent("AAPL", days_until=0).proximity() is EarningsProximity.IMMINENT
    assert EarningsEvent("AAPL", days_until=2).proximity() is EarningsProximity.IMMINENT


def test_proximity_near_within_a_week() -> None:
    # Powyżej IMMINENT, ale w obrębie tygodnia → NEAR.
    assert EarningsEvent("AAPL", days_until=3).proximity() is EarningsProximity.NEAR
    assert EarningsEvent("AAPL", days_until=7).proximity() is EarningsProximity.NEAR


def test_proximity_far_beyond_week() -> None:
    assert EarningsEvent("AAPL", days_until=8).proximity() is EarningsProximity.FAR
    assert EarningsEvent("AAPL", days_until=90).proximity() is EarningsProximity.FAR


def test_proximity_none_is_far() -> None:
    # Brak danych o nadchodzącym raporcie → FAR.
    assert EarningsEvent("AAPL", days_until=None).proximity() is EarningsProximity.FAR


def test_proximity_negative_is_far() -> None:
    # Raport w przeszłości (ujemne dni) → FAR.
    assert EarningsEvent("AAPL", days_until=-1).proximity() is EarningsProximity.FAR


def test_proximity_respects_custom_thresholds() -> None:
    event = EarningsEvent("AAPL", days_until=5)
    assert event.proximity(imminent_days=5, near_days=10) is EarningsProximity.IMMINENT
    assert event.proximity(imminent_days=1, near_days=3) is EarningsProximity.FAR


def test_threshold_multiplier_maps_each_proximity() -> None:
    # Kontrakt jak RegimeDetector.threshold_multiplier: IMMINENT zacieśnia
    # najmocniej, NEAR słabiej, FAR neutralnie (1.0).
    from src.domain.earnings import earnings_threshold_multiplier

    assert earnings_threshold_multiplier(EarningsProximity.IMMINENT) == 1.5
    assert earnings_threshold_multiplier(EarningsProximity.NEAR) == 1.15
    assert earnings_threshold_multiplier(EarningsProximity.FAR) == 1.0


def test_threshold_multiplier_never_loosens_gate() -> None:
    # Niezmiennik FinOps: dla KAŻDEJ wartości enuma mnożnik >= 1.0 — reguła
    # earnings może tylko zacieśnić bramkę volatility, nigdy jej poluzować.
    from src.domain.earnings import earnings_threshold_multiplier

    for proximity in EarningsProximity:
        assert earnings_threshold_multiplier(proximity) >= 1.0


def test_threshold_multiplier_returns_float() -> None:
    # Spójność typu z regime_multiplier (float), żeby iloczyn w _pick_threshold
    # pozostał float.
    from src.domain.earnings import earnings_threshold_multiplier

    result = earnings_threshold_multiplier(EarningsProximity.IMMINENT)
    assert isinstance(result, float)


class TestCachedEarningsDate:
    """FIN-11: cache Slow Loopa trzyma datę raportu, nie liczbę dni — Fast Loop liczy
    `days_until` od dnia cyklu, więc wpis z niedzieli działa cały tydzień."""

    def test_event_counts_days_from_the_cycle_day(self) -> None:
        from datetime import UTC, date, datetime

        from src.domain.earnings import CachedEarningsDate

        cached = CachedEarningsDate(
            symbol="AAPL",
            report_date=date(2026, 10, 29),
            fetched_at=datetime(2026, 10, 25, 3, 0, tzinfo=UTC),
        )

        event = cached.event(today=date(2026, 10, 27))

        assert event == EarningsEvent("AAPL", days_until=2, report_date="2026-10-29")
        assert event is not None and event.proximity() is EarningsProximity.IMMINENT

    def test_report_on_the_cycle_day_is_still_upcoming(self) -> None:
        from datetime import UTC, date, datetime

        from src.domain.earnings import CachedEarningsDate

        cached = CachedEarningsDate("AAPL", date(2026, 10, 27), datetime(2026, 10, 25, tzinfo=UTC))

        assert cached.event(today=date(2026, 10, 27)) == EarningsEvent(
            "AAPL", days_until=0, report_date="2026-10-27"
        )

    def test_report_already_published_gives_no_event(self) -> None:
        from datetime import UTC, date, datetime

        from src.domain.earnings import CachedEarningsDate

        cached = CachedEarningsDate("AAPL", date(2026, 10, 22), datetime(2026, 10, 18, tzinfo=UTC))

        assert cached.event(today=date(2026, 10, 27)) is None

    def test_cache_ttl_covers_the_fast_loop_week_after_the_sunday_refresh(self) -> None:
        # Slow Loop: niedziela 03:00 UTC; Fast Loop: pon–pt. Po 6 dniach wpis wygasa,
        # więc następna niedziela odświeża go z Alpha Vantage.
        from src.domain.earnings import EARNINGS_CACHE_TTL_HOURS

        assert 5 * 24 < EARNINGS_CACHE_TTL_HOURS < 7 * 24


class TestEarningsThisWeek:
    """DAT-20: poniedziałkowa sekcja „📅 Ten tydzień” — raporty od dziś do niedzieli."""

    MONDAY = date(2026, 10, 26)

    def test_keeps_reports_until_sunday_sorted_by_day_then_symbol(self) -> None:
        from src.domain.earnings import earnings_this_week

        events = [
            EarningsEvent("MSFT", days_until=2),
            EarningsEvent("AAPL", days_until=3),
            EarningsEvent("AMD", days_until=2),
            EarningsEvent("NVDA", days_until=0),
        ]

        week = earnings_this_week(events, self.MONDAY)

        assert [event.symbol for event in week] == ["NVDA", "AMD", "MSFT", "AAPL"]

    def test_drops_next_week_past_unknown_and_missing(self) -> None:
        from src.domain.earnings import earnings_this_week

        events = [
            EarningsEvent("SUN", days_until=6),
            EarningsEvent("NEXT", days_until=7),
            EarningsEvent("PAST", days_until=-1),
            EarningsEvent("NONE", days_until=None),
            None,
        ]

        assert [e.symbol for e in earnings_this_week(events, self.MONDAY)] == ["SUN"]

    def test_midweek_day_counts_only_the_rest_of_the_week(self) -> None:
        from src.domain.earnings import earnings_this_week

        thursday = date(2026, 10, 29)
        events = [EarningsEvent("FRI", days_until=1), EarningsEvent("MON", days_until=4)]

        assert [e.symbol for e in earnings_this_week(events, thursday)] == ["FRI"]


class TestEarningsDueWithin:
    """DAT-20: kandydaci leadu „wyniki Twojej spółki” — raporty dziś i jutro."""

    def test_keeps_today_and_tomorrow_sorted_by_day_then_symbol(self) -> None:
        from src.domain.earnings import earnings_due_within

        events = [
            EarningsEvent("MSFT", days_until=1),
            EarningsEvent("AAPL", days_until=2),
            EarningsEvent("AMD", days_until=1),
            EarningsEvent("NVDA", days_until=0),
            EarningsEvent("PAST", days_until=-1),
            EarningsEvent("NONE", days_until=None),
            None,
        ]

        assert [event.symbol for event in earnings_due_within(events)] == [
            "NVDA",
            "AMD",
            "MSFT",
        ]
