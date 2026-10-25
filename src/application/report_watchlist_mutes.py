"""MAIL-45 — wyciszone symbole w Zapleczu maila.

Wyciszenie z bota (`/mute`, tabela `symbol_mutes`) zdejmuje symbol z płatnej analizy
i z sekcji maila; ta sekcja mówi, że go nie ma i do kiedy: „watchlista: 45 z config −
2 wyciszone” i chipy „NVDA do 01.11.2026”. Bez aktywnego wyciszenia albo z flagą
`symbol_mutes_enabled` off sekcji nie ma — mail bajt w bajt jak przed MAIL-45.
"""

from __future__ import annotations

from src.application.report_formatting import PALETTE, format_local_date
from src.application.report_formatting import escape as _html
from src.application.report_formatting import section_h2 as _section_h2
from src.application.reporting.section import ReportSection, SectionInput
from src.domain.watchlist import EffectiveWatchlist, SymbolMute, watchlist_summary

_TITLE = "Wyciszone z bota"
_NOTE = (
    "Bez płatnej analizy i bez sekcji do końca wyciszenia; snapshot ceny zapisany. "
    "Symbole dodaje wyłącznie config.toml."
)


def _active(data: SectionInput) -> EffectiveWatchlist | None:
    watchlist = data.context.watchlist_mutes
    return watchlist if watchlist is not None and watchlist.muted else None


def _until(mute: SymbolMute, data: SectionInput) -> str:
    return f"{mute.symbol} do {format_local_date(mute.muted_until, data.context.display_timezone)}"


def _html_section(data: SectionInput) -> str:
    watchlist = _active(data)
    if watchlist is None:
        return ""
    chips = "".join(
        "<span style='display: inline-block; padding: 2px 8px; margin: 2px; "
        f"background: {PALETTE.surface}; border-radius: 3px; font-size: 12px;'>"
        f"{_html(_until(mute, data))}</span>"
        for mute in watchlist.muted
    )
    summary = _html(watchlist_summary(watchlist))
    return (
        _section_h2(f"🔇 {_TITLE}")
        + f"<div style='font-size: 13px; margin: 2px 0;'>{summary}</div>"
        + f"<div>{chips}</div>"
        + f"<div style='font-size: 12px; color: {PALETTE.muted}; margin: 6px 0 0 0;'>"
        + f"{_html(_NOTE)}</div>"
    )


def _text_section(data: SectionInput) -> list[str]:
    watchlist = _active(data)
    if watchlist is None:
        return []
    return [
        _TITLE.upper(),
        "-" * 64,
        f"  {watchlist_summary(watchlist)}",
        f"  {', '.join(_until(mute, data) for mute in watchlist.muted)}",
        f"  {_NOTE}",
        "",
    ]


WATCHLIST_MUTES_SECTION = ReportSection(
    key="watchlist_mutes",
    render_html=_html_section,
    render_text=_text_section,
    context_fields=frozenset({"watchlist_mutes"}),
)
