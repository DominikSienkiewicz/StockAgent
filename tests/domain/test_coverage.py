"""DAT-16: pokrycie danych — sesje ze snapshotem ceny, najdłuższa luka i symbole
bez rozliczeń w oknie.

Track record z dziurami (pauza od 2026-08-17, padnięte symbole) bez informacji
o pokryciu sugeruje ciągłość, której nie było. Domena liczy pokrycie z dwóch
list dat; oczekiwane sesje zależą od klasy aktywa (akcje — dni handlowe NYSE,
krypto — każdy dzień).
"""

from __future__ import annotations

from datetime import date, timedelta

from src.domain.coverage import (
    CoverageGap,
    coverage,
    expected_sessions,
    unsettled_symbols,
)
from src.domain.value_objects import AssetType


def _days(start: date, count: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(count)]


def test_reports_longest_gap_and_ratio() -> None:
    expected = _days(date(2026, 9, 1), 10)
    # Brak 09-03 (1 sesja) oraz 09-06…09-08 (3 sesje) — dłuższa luka wygrywa.
    observed = [d for d in expected if d not in {date(2026, 9, 3), *_days(date(2026, 9, 6), 3)}]

    report = coverage(expected, observed)

    assert (report.expected, report.covered) == (10, 6)
    assert report.ratio == 0.6
    assert report.longest_gap == CoverageGap(
        first=date(2026, 9, 6), last=date(2026, 9, 8), sessions=3
    )


def test_weekends_are_not_gaps_for_stocks_but_are_for_crypto() -> None:
    # Pt 2026-10-16 i pn 2026-10-19 ze snapshotem; sobota i niedziela bez.
    observed = [date(2026, 10, 16), date(2026, 10, 19)]
    start, end = date(2026, 10, 16), date(2026, 10, 19)

    stocks = coverage(expected_sessions(start, end, AssetType.STOCK), observed)
    crypto = coverage(expected_sessions(start, end, AssetType.CRYPTO), observed)

    assert (stocks.ratio, stocks.longest_gap) == (1.0, None)
    assert crypto.ratio == 0.5
    assert crypto.longest_gap == CoverageGap(
        first=date(2026, 10, 17), last=date(2026, 10, 18), sessions=2
    )


def test_nyse_holiday_is_not_an_expected_stock_session() -> None:
    # Święto Dziękczynienia 2026-11-26 — giełda zamknięta, więc to nie luka.
    sessions = expected_sessions(date(2026, 11, 25), date(2026, 11, 27), AssetType.STOCK)

    assert sessions == [date(2026, 11, 25), date(2026, 11, 27)]


def test_observed_days_outside_the_expected_sessions_do_not_count() -> None:
    report = coverage([date(2026, 10, 16)], [date(2026, 10, 16), date(2026, 10, 17)])

    assert (report.expected, report.covered) == (1, 1)


def test_no_expected_sessions_has_no_ratio() -> None:
    report = coverage([], [date(2026, 10, 16)])

    assert report.ratio is None
    assert report.longest_gap is None


def test_ties_keep_the_earliest_gap() -> None:
    expected = _days(date(2026, 9, 1), 5)
    observed = [date(2026, 9, 2), date(2026, 9, 4)]

    report = coverage(expected, observed)

    assert report.longest_gap == CoverageGap(
        first=date(2026, 9, 1), last=date(2026, 9, 1), sessions=1
    )


def test_unsettled_symbols_are_sorted_and_unique() -> None:
    assert unsettled_symbols(["NVDA", "SSNLF", "AAPL", "SSNLF"], {"NVDA"}) == ("AAPL", "SSNLF")
