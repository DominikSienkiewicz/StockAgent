"""Pokrycie danych track recordu (DAT-16) z darmowych odczytów repozytorium.

Zbiera daty snapshotów cen wszystkich symboli (suma — pokazuje pauzy pętli),
porównuje je z oczekiwanymi sesjami (`domain.coverage`) i dokłada listę symboli
bez żadnego rozliczenia w oknie (padnięte albo niewspierane tickery). Zero
płatnych portów. Zdanie po polsku idzie do karty kondycji (z symbolami) i do
publicznego scorecardu (bez symboli — inwariant prywatności strony).

Snapshot akcji Fast Loop zapisuje rano UTC, po zamknięciu poprzedniej sesji, więc
datą snapshotu jest sesja, którą zamyka (`last_started_session`), a okno kończy
się na ostatniej ZAKOŃCZONEJ sesji. Watchlista bez akcji liczy się dniami UTC.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from datetime import date, datetime, timedelta

from src.application.ports import RepositoryPort
from src.application.report_formatting import plural_pl
from src.domain.coverage import CoverageReport, coverage, expected_sessions, unsettled_symbols
from src.domain.market_calendar import last_started_session
from src.domain.value_objects import AssetType

logger = logging.getLogger(__name__)


def build_data_coverage(
    repository: RepositoryPort,
    *,
    symbols: Sequence[str],
    crypto_symbols: Sequence[str],
    days: int,
    now: datetime,
) -> CoverageReport:
    """Pokrycie ostatnich `days` dni: sesje NYSE, gdy są akcje, inaczej dni UTC.

    Błąd odczytu historii jednego symbolu go pomija; błąd odczytu rozliczeń
    leci wyżej (wołający chowa wtedy zdanie)."""
    asset = AssetType.STOCK if symbols else AssetType.CRYPTO
    tracked = list(symbols) if symbols else list(crypto_symbols)
    observed: set[date] = set()
    for symbol in tracked:
        try:
            history = repository.get_price_history(symbol, days)
        except Exception:
            logger.exception("data coverage history fetch failed for %s", symbol)
            continue
        observed.update(_snapshot_session(stamp, asset) for stamp, _ in history)
    start = (now - timedelta(days=days)).date() + timedelta(days=1)
    report = coverage(expected_sessions(start, _window_end(now, asset), asset), observed)
    settled = {str(row.get("symbol")) for row in repository.get_resolved_predictions(days)}
    return dataclasses.replace(
        report, unsettled_symbols=unsettled_symbols([*symbols, *crypto_symbols], settled)
    )


def _snapshot_session(stamp: datetime, asset: AssetType) -> date:
    if asset is AssetType.CRYPTO:
        return stamp.date()
    return last_started_session(stamp).day


def _window_end(now: datetime, asset: AssetType) -> date:
    """Ostatnia sesja, której snapshot mógł już powstać: dziś dla krypto,
    ostatnia zakończona sesja NYSE dla akcji."""
    if asset is AssetType.CRYPTO:
        return now.date()
    session = last_started_session(now)
    if session.closes_at > now:
        session = last_started_session(session.opens_at - timedelta(minutes=1))
    return session.day


def coverage_line_pl(report: CoverageReport, *, list_symbols: bool) -> str:
    """Zdanie o pokryciu; `list_symbols=False` podaje samą liczbę symboli bez
    rozliczeń (publiczna strona nie ujawnia watchlisty). Okno bez sesji → ""."""
    if report.ratio is None:
        return ""
    parts = [
        f"Pokrycie danych: {round(report.ratio * 100)}% sesji ze snapshotem ceny "
        f"({report.covered} z {report.expected})"
    ]
    gap = report.longest_gap
    if gap is None:
        parts.append("bez luk")
    else:
        sessions = plural_pl(gap.sessions, "sesja", "sesje", "sesji")
        parts.append(
            f"najdłuższa luka {gap.first.isoformat()} – {gap.last.isoformat()} "
            f"({gap.sessions} {sessions})"
        )
    unsettled = report.unsettled_symbols
    if unsettled and list_symbols:
        parts.append("bez rozliczeń w oknie: " + ", ".join(unsettled))
    elif unsettled:
        noun = plural_pl(len(unsettled), "symbol", "symbole", "symboli")
        parts.append(f"{len(unsettled)} {noun} bez rozliczeń w oknie")
    return "; ".join(parts) + "."
