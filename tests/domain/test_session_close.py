"""WIA-61: oficjalne zamknięcie sesji NYSE z tego samego notowania /quote.

Okres przejściowy (decyzja właściciela, opcja „oba”): rozliczenie zapisuje obok
ceny /quote zamknięcie sesji z jej datą i źródłem, żeby weryfikator sprawdził
cenę w dowolnym publicznym serwisie. Październik 2026 to czas letni (ET = UTC−4):
sesja 13:30–20:00 UTC; 26.11 to Święto Dziękczynienia, 27.11 sesja skrócona do 13:00 ET.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from src.domain.value_objects import Money, Quote

MONDAY_CLOSE = datetime(2026, 10, 19, 20, 0, tzinfo=UTC)


def _quote(price: str, quoted_at: datetime, prev_close: str | None = None) -> Quote:
    return Quote(
        price=Money(Decimal(price)),
        quoted_at=quoted_at,
        prev_close=Money(Decimal(prev_close)) if prev_close is not None else None,
    )


class TestSessionCloseOf:
    def test_quote_after_the_close_is_that_session_close(self) -> None:
        # Fast Loop przed otwarciem we wtorek: `c` to zamknięcie poniedziałku.
        from src.domain.session_close import SessionClose, SessionCloseSource, session_close_of

        now = datetime(2026, 10, 20, 5, 30, tzinfo=UTC)

        close = session_close_of(_quote("106", MONDAY_CLOSE, prev_close="100"), now)

        assert close == SessionClose(
            price=Money(Decimal("106")),
            session_day=date(2026, 10, 19),
            source=SessionCloseSource.QUOTE_CLOSE,
        )

    def test_during_the_session_the_previous_close_is_the_official_one(self) -> None:
        from src.domain.session_close import SessionClose, SessionCloseSource, session_close_of

        now = datetime(2026, 10, 20, 15, 0, tzinfo=UTC)
        quote = _quote("107", datetime(2026, 10, 20, 14, 59, tzinfo=UTC), prev_close="106")

        close = session_close_of(quote, now)

        assert close == SessionClose(
            price=Money(Decimal("106")),
            session_day=date(2026, 10, 19),
            source=SessionCloseSource.QUOTE_PREVIOUS_CLOSE,
        )

    def test_quote_from_before_the_running_session_has_no_close(self) -> None:
        # Notowanie sprzed otwarcia trwającej sesji: `pc` byłoby zamknięciem
        # o dzień wcześniejszym, a `c` nie jest zamknięciem żadnej sesji.
        from src.domain.session_close import session_close_of

        now = datetime(2026, 10, 20, 15, 0, tzinfo=UTC)

        assert session_close_of(_quote("106", MONDAY_CLOSE, prev_close="100"), now) is None

    def test_during_the_session_without_a_previous_close_there_is_none(self) -> None:
        from src.domain.session_close import session_close_of

        now = datetime(2026, 10, 20, 15, 0, tzinfo=UTC)
        quote = _quote("107", datetime(2026, 10, 20, 14, 59, tzinfo=UTC))

        assert session_close_of(quote, now) is None

    def test_on_an_nyse_holiday_the_close_is_the_last_session(self) -> None:
        from src.domain.session_close import SessionCloseSource, session_close_of

        now = datetime(2026, 11, 26, 16, 0, tzinfo=UTC)
        quote = _quote("110", datetime(2026, 11, 25, 21, 0, tzinfo=UTC), prev_close="109")

        close = session_close_of(quote, now)

        assert close is not None
        assert (close.session_day, close.source) == (
            date(2026, 11, 25), SessionCloseSource.QUOTE_CLOSE
        )

    def test_early_close_counts_from_13_00_et(self) -> None:
        from src.domain.session_close import SessionCloseSource, session_close_of

        now = datetime(2026, 11, 27, 18, 30, tzinfo=UTC)
        quote = _quote("111", datetime(2026, 11, 27, 18, 0, tzinfo=UTC), prev_close="110")

        close = session_close_of(quote, now)

        assert close is not None
        assert (close.price, close.session_day, close.source) == (
            Money(Decimal("111")), date(2026, 11, 27), SessionCloseSource.QUOTE_CLOSE
        )


class TestSessionCloseSettles:
    def test_close_of_the_first_session_after_the_prediction_settles_it(self) -> None:
        from src.domain.session_close import SessionClose, SessionCloseSource

        close = SessionClose(
            Money(Decimal("106")), date(2026, 10, 19), SessionCloseSource.QUOTE_CLOSE
        )

        # Predykcja przed poniedziałkowym otwarciem — horyzont to sesja poniedziałku.
        assert close.settles(datetime(2026, 10, 19, 5, 30, tzinfo=UTC))

    def test_close_from_before_the_prediction_horizon_does_not_settle(self) -> None:
        from src.domain.session_close import SessionClose, SessionCloseSource

        close = SessionClose(
            Money(Decimal("106")), date(2026, 10, 19), SessionCloseSource.QUOTE_CLOSE
        )

        # Predykcja po poniedziałkowym zamknięciu — pierwsza sesja po niej to wtorek.
        assert not close.settles(datetime(2026, 10, 19, 21, 0, tzinfo=UTC))
