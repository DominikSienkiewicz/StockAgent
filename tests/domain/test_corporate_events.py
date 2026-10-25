"""DAT-17: raporty bieżące 8-K — itemy, etykiety i flaga świeżego zdarzenia."""

from __future__ import annotations

from datetime import date

from src.domain.corporate_events import (
    FRESH_EVENT_MAX_AGE_DAYS,
    CorporateEvent,
    CorporateEventSnapshot,
)

AS_OF = date(2026, 10, 26)


def _event(filed_on: date, *items: str) -> CorporateEvent:
    return CorporateEvent(form="8-K", items=items, filed_on=filed_on)


class TestCorporateEvent:
    def test_exhibits_item_is_not_material(self) -> None:
        assert _event(AS_OF, "2.02", "9.01").material_items() == ("2.02",)

    def test_known_items_get_polish_labels_and_unknown_keep_their_number(self) -> None:
        event = _event(AS_OF, "5.02", "3.03", "9.01")

        assert event.labels_pl() == ("zmiana w zarządzie", "item 3.03")

    def test_item_2_02_is_an_earnings_release(self) -> None:
        assert _event(AS_OF, "2.02", "9.01").is_earnings()
        assert not _event(AS_OF, "5.02").is_earnings()


class TestCorporateEventSnapshot:
    def test_latest_is_the_first_event_or_none(self) -> None:
        newest = _event(AS_OF, "2.02")
        snapshot = CorporateEventSnapshot("AAPL", (newest, _event(date(2026, 10, 1))), 14)

        assert snapshot.latest() is newest
        assert CorporateEventSnapshot("AAPL", (), 14).latest() is None

    def test_fresh_events_reach_back_the_default_window_inclusive(self) -> None:
        edge = date(2026, 10, 26 - FRESH_EVENT_MAX_AGE_DAYS)
        stale = date(2026, 10, 25 - FRESH_EVENT_MAX_AGE_DAYS)
        snapshot = CorporateEventSnapshot("AAPL", (_event(edge), _event(stale)), 14)

        assert [event.filed_on for event in snapshot.fresh_events(AS_OF)] == [edge]
        assert snapshot.has_fresh_event(AS_OF)

    def test_only_stale_events_are_not_fresh(self) -> None:
        snapshot = CorporateEventSnapshot("AAPL", (_event(date(2026, 10, 1)),), 30)

        assert not snapshot.has_fresh_event(AS_OF)
        assert snapshot.has_fresh_event(AS_OF, max_age_days=30)

    def test_filing_dated_after_the_cycle_day_counts_as_fresh(self) -> None:
        snapshot = CorporateEventSnapshot("AAPL", (_event(date(2026, 10, 27)),), 14)

        assert snapshot.has_fresh_event(AS_OF)
