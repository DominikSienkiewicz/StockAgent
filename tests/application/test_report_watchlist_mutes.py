"""MAIL-45 — linia „watchlista: N z config − K wyciszone” w Zapleczu maila."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

from src.application.report_builder import ReportContext, build_html_report
from src.application.report_watchlist_mutes import WATCHLIST_MUTES_SECTION
from src.domain.watchlist import SymbolMute, effective

NOW = datetime(2026, 10, 25, 8, 0, tzinfo=UTC)
CONFIG = ("AAPL", "NVDA", "TSLA", "BTC")


def _context(*mutes: SymbolMute) -> ReportContext:
    return ReportContext(watchlist_mutes=effective(CONFIG, mutes, NOW))


def test_muted_symbols_get_one_line_with_their_end_dates() -> None:
    context = _context(SymbolMute("NVDA", NOW + timedelta(days=3)))

    html, text = build_html_report([], NOW, 1.0, context)

    for rendered in (html, text):
        assert "watchlista: 4 z config − 1 wyciszone" in rendered
        assert "NVDA do 28.10.2026" in rendered


def test_no_active_mute_and_no_flag_render_nothing() -> None:
    expired = _context(SymbolMute("NVDA", NOW - timedelta(days=1)))
    off = ReportContext()

    for context in (expired, off):
        html, text = build_html_report([], NOW, 1.0, context)
        assert "wyciszone" not in html
        assert "wyciszone" not in text


def test_section_owns_its_context_field() -> None:
    assert WATCHLIST_MUTES_SECTION.context_fields == frozenset({"watchlist_mutes"})
    assert "watchlist_mutes" in {field.name for field in dataclasses.fields(ReportContext)}
