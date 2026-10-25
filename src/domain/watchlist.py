"""Domena: efektywna watchlista cyklu (MAIL-45) — config.toml minus aktywne wyciszenia.

Decyzja 2026-10-23 (opcja b): baza trzyma TYLKO wyciszenia (`/mute` w bocie, tabela
`symbol_mutes`, migracja 080), a symbole dodaje się wyłącznie przez config.toml —
watchlista nie dostaje drugiego źródła prawdy. Wyciszenie symbolu spoza config.toml
niczego nie dodaje ani się nie liczy.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class SymbolMute:
    """Symbol wyciszony do chwili `muted_until` (wyłącznie, strefa wymagana)."""

    symbol: str
    muted_until: datetime

    def __post_init__(self) -> None:
        _require_aware(self.muted_until)


@dataclass(frozen=True)
class EffectiveWatchlist:
    """`symbols` — config.toml bez aktywnych wyciszeń, w kolejności configu;
    `muted` — aktywne wyciszenia symboli z configu, w tej samej kolejności."""

    symbols: tuple[str, ...]
    muted: tuple[SymbolMute, ...]
    config_count: int


def effective(
    config_symbols: Sequence[str], mutes: Iterable[SymbolMute], now: datetime
) -> EffectiveWatchlist:
    """Symbol jest wyciszony, dopóki `now < muted_until`; wygasłe wyciszenie wraca
    symbol na watchlistę bez żadnego zapisu. Przy kilku wyciszeniach jednego symbolu
    rozstrzyga najpóźniejsze. Wielkość liter tickera nie ma znaczenia."""
    _require_aware(now)
    latest: dict[str, SymbolMute] = {}
    for mute in mutes:
        key = mute.symbol.upper()
        current = latest.get(key)
        if current is None or mute.muted_until > current.muted_until:
            latest[key] = mute
    kept: list[str] = []
    muted: list[SymbolMute] = []
    for symbol in config_symbols:
        active = latest.get(symbol.upper())
        if active is not None and now < active.muted_until:
            muted.append(SymbolMute(symbol, active.muted_until))
        else:
            kept.append(symbol)
    return EffectiveWatchlist(
        symbols=tuple(kept), muted=tuple(muted), config_count=len(config_symbols)
    )


def watchlist_summary(watchlist: EffectiveWatchlist) -> str:
    """Jedna linia dla maila i bota: „watchlista: 45 z config − 2 wyciszone”."""
    return f"watchlista: {watchlist.config_count} z config − {len(watchlist.muted)} wyciszone"


def mute_until(now: datetime, days: int) -> datetime:
    """Koniec wyciszenia na `days` pełnych dób od `now` (co najmniej jedna)."""
    _require_aware(now)
    if days < 1:
        raise ValueError(f"wyciszenie trwa co najmniej 1 dzień (podano {days})")
    return now + timedelta(days=days)


def _require_aware(moment: datetime) -> None:
    if moment.tzinfo is None:
        raise ValueError(f"chwila bez strefy czasu: {moment.isoformat()}")
