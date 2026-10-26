"""Stan bota właściciela w Supabase (MAIL-48, MAIL-45).

- `bot_state` (migracja 079) — offset pollingu `getUpdates`, jeden wiersz na kanał;
- `symbol_mutes` (migracja 080) — wyciszenia z `/mute`, jeden wiersz na symbol.

Decyzja 2026-10-23 (opcja b): baza trzyma tylko wyciszenia — symbole dodaje
wyłącznie config.toml. Każdy błąd PostgREST albo transportu wychodzi jako klasa
taksonomii portów (ARCH-33), z zachowanym typem dostawcy.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
from postgrest.exceptions import APIError
from supabase import create_client

from src.application.errors import MalformedPayload, PortError, ProviderUnavailable
from src.application.ports import BotStatePort, SymbolMutePort
from src.domain.watchlist import SymbolMute
from src.infrastructure.error_translation import as_port_error

BOT_STATE_TABLE = "bot_state"
SYMBOL_MUTES_TABLE = "symbol_mutes"
BOT_CHANNEL = "telegram"


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SupabaseBotStore(BotStatePort, SymbolMutePort):
    """`client` to klient `supabase.create_client` z kluczem serwisowym."""

    def __init__(self, client: Any, *, clock: Callable[[], datetime] = _utc_now) -> None:
        self._client = client
        self._clock = clock

    @classmethod
    def connect(cls, url: str, key: str) -> SupabaseBotStore:
        return cls(create_client(url, key))

    def get_update_offset(self) -> int | None:
        rows = _rows(
            self._client.table(BOT_STATE_TABLE)
            .select("update_offset")
            .eq("channel", BOT_CHANNEL)
            .limit(1)
        )
        if not rows:
            return None
        offset = rows[0].get("update_offset")
        if not isinstance(offset, int):
            raise MalformedPayload(f"{BOT_STATE_TABLE}: update_offset is not an integer")
        return offset

    def save_update_offset(self, offset: int) -> None:
        payload = {
            "channel": BOT_CHANNEL,
            "update_offset": offset,
            "updated_at": self._clock().isoformat(),
        }
        _execute(self._client.table(BOT_STATE_TABLE).upsert(payload, on_conflict="channel"))

    def get_symbol_mutes(self) -> list[SymbolMute]:
        rows = _rows(self._client.table(SYMBOL_MUTES_TABLE).select("symbol, muted_until"))
        return [_symbol_mute(row) for row in rows]

    def mute_symbol(self, symbol: str, muted_until: datetime, *, muted_at: datetime) -> None:
        payload = {
            "symbol": symbol.upper(),
            "muted_until": muted_until.isoformat(),
            "created_at": muted_at.isoformat(),
        }
        _execute(self._client.table(SYMBOL_MUTES_TABLE).upsert(payload, on_conflict="symbol"))

    def unmute_symbol(self, symbol: str) -> None:
        _execute(self._client.table(SYMBOL_MUTES_TABLE).delete().eq("symbol", symbol.upper()))


def _execute(query: Any) -> Any:
    """Wykonanie zapytania PostgREST z błędem w taksonomii portów (ARCH-33)."""
    try:
        return query.execute()
    except httpx.TransportError as exc:
        raise as_port_error(exc, ProviderUnavailable) from None
    except (APIError, httpx.HTTPError) as exc:
        raise as_port_error(exc, PortError) from None


def _rows(query: Any) -> list[dict[str, Any]]:
    rows = _execute(query).data or []
    if not isinstance(rows, list):
        raise MalformedPayload("PostgREST returned a non-list body")
    return rows


def _symbol_mute(row: dict[str, Any]) -> SymbolMute:
    symbol, until = row.get("symbol"), row.get("muted_until")
    if not isinstance(symbol, str) or not symbol or not isinstance(until, str):
        raise MalformedPayload(f"{SYMBOL_MUTES_TABLE}: row without symbol or muted_until")
    try:
        moment = datetime.fromisoformat(until)
    except ValueError:
        raise MalformedPayload(f"{SYMBOL_MUTES_TABLE}: unreadable muted_until") from None
    if moment.tzinfo is None:
        raise MalformedPayload(f"{SYMBOL_MUTES_TABLE}: muted_until without a time zone")
    return SymbolMute(symbol, moment)
