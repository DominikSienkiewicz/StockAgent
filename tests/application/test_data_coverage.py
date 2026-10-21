"""DAT-16: pokrycie danych z darmowych odczytów repozytorium i jego zdanie
w karcie kondycji (z symbolami) oraz w publicznym scorecardzie (bez symboli)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from unittest.mock import Mock

from src.application.data_coverage import build_data_coverage, coverage_line_pl
from src.application.ports import RepositoryPort
from src.domain.coverage import CoverageGap, CoverageReport
from src.domain.value_objects import Money

# Wtorek po sesji poniedziałku 2026-10-19; okno 10 dni → sesje 10-12…10-19.
_NOW = datetime(2026, 10, 20, 6, 0, tzinfo=UTC)


def _snapshot(day: int) -> tuple[datetime, Money]:
    # Fast Loop o 05:35 UTC zapisuje zamknięcie POPRZEDNIEJ sesji.
    return datetime(2026, 10, day, 5, 35, tzinfo=UTC), Money(Decimal("100"))


def _repo(history: dict[str, list[tuple[datetime, Money]]], settled: list[str]) -> Mock:
    repo = Mock(spec=RepositoryPort)
    repo.get_price_history.side_effect = lambda symbol, days: history.get(symbol, [])
    repo.get_resolved_predictions.return_value = [{"symbol": s} for s in settled]
    return repo


class TestBuildDataCoverage:
    def test_union_of_symbols_maps_each_snapshot_to_the_session_it_closed(self) -> None:
        # Snapshoty wt.–pt. (13–16) i pn. 19 → sesje pn.–czw. (12–15) i pt. 16;
        # brakuje snapshotu z 17-10 (sob.) — weekend to nie luka.
        repo = _repo(
            {
                "AAPL": [_snapshot(13), _snapshot(14), _snapshot(15)],
                "MSFT": [_snapshot(16), _snapshot(19)],
            },
            settled=["AAPL"],
        )

        report = build_data_coverage(
            repo, symbols=["AAPL", "MSFT"], crypto_symbols=["BTC"], days=10, now=_NOW
        )

        # Sesje 10-12…10-19: 12, 13, 14, 15, 16, 19 → pokryte 12–16; snapshotu
        # z 20-10 (zamknięcie sesji 19) w historii brak, więc 19 to luka.
        assert (report.expected, report.covered) == (6, 5)
        assert report.longest_gap == CoverageGap(
            first=date(2026, 10, 19), last=date(2026, 10, 19), sessions=1
        )
        assert report.unsettled_symbols == ("BTC", "MSFT")
        repo.get_resolved_predictions.assert_called_once_with(10)

    def test_history_error_of_one_symbol_does_not_break_coverage(self) -> None:
        repo = _repo({}, settled=[])

        def history(symbol: str, days: int) -> list[tuple[datetime, Money]]:
            if symbol == "AAPL":
                raise RuntimeError("supabase down")
            return [_snapshot(13)]

        repo.get_price_history.side_effect = history

        report = build_data_coverage(
            repo, symbols=["AAPL", "MSFT"], crypto_symbols=[], days=10, now=_NOW
        )

        assert report.covered == 1

    def test_crypto_only_watchlist_counts_every_day(self) -> None:
        repo = _repo({"BTC": [_snapshot(18), _snapshot(19)]}, settled=["BTC"])

        report = build_data_coverage(repo, symbols=[], crypto_symbols=["BTC"], days=3, now=_NOW)

        # Dni 10-18…10-20 (UTC): 18 i 19 pokryte, dzisiejszy 20 jeszcze bez snapshotu.
        assert (report.expected, report.covered) == (3, 2)


class TestCoverageLine:
    _REPORT = CoverageReport(
        expected=30,
        covered=26,
        longest_gap=CoverageGap(first=date(2026, 8, 17), last=date(2026, 8, 20), sessions=4),
        unsettled_symbols=("SSNLF", "XYZ"),
    )

    def test_private_line_names_the_unsettled_symbols(self) -> None:
        line = coverage_line_pl(self._REPORT, list_symbols=True)

        assert line == (
            "Pokrycie danych: 87% sesji ze snapshotem ceny (26 z 30); "
            "najdłuższa luka 2026-08-17 – 2026-08-20 (4 sesje); "
            "bez rozliczeń w oknie: SSNLF, XYZ."
        )

    def test_public_line_counts_symbols_without_naming_them(self) -> None:
        line = coverage_line_pl(self._REPORT, list_symbols=False)

        assert "SSNLF" not in line
        assert line.endswith("2 symbole bez rozliczeń w oknie.")

    def test_full_coverage_says_there_is_no_gap(self) -> None:
        report = CoverageReport(expected=5, covered=5, longest_gap=None)

        assert coverage_line_pl(report, list_symbols=True) == (
            "Pokrycie danych: 100% sesji ze snapshotem ceny (5 z 5); bez luk."
        )

    def test_window_without_sessions_has_no_line(self) -> None:
        assert coverage_line_pl(CoverageReport(0, 0, None), list_symbols=True) == ""
