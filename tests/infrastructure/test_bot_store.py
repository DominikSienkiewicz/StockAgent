"""MAIL-48 / MAIL-45: stan bota (`bot_state`, 079) i wyciszenia (`symbol_mutes`, 080)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from postgrest.exceptions import APIError

from src.application.errors import MalformedPayload, PortError, ProviderUnavailable
from src.application.ports import BotStatePort, SymbolMutePort
from src.domain.watchlist import SymbolMute
from src.infrastructure.adapters.bot_store import (
    BOT_CHANNEL,
    BOT_STATE_TABLE,
    SYMBOL_MUTES_TABLE,
    SupabaseBotStore,
)

NOW = datetime(2026, 10, 25, 8, 0, tzinfo=UTC)


def _client(rows: list[dict[str, Any]] | None = None) -> MagicMock:
    client = MagicMock()
    table = client.table.return_value
    table.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = (
        rows or []
    )
    table.select.return_value.execute.return_value.data = rows or []
    return client


def _store(client: MagicMock) -> SupabaseBotStore:
    return SupabaseBotStore(client, clock=lambda: NOW)


def test_store_implements_both_ports() -> None:
    assert issubclass(SupabaseBotStore, BotStatePort)
    assert issubclass(SupabaseBotStore, SymbolMutePort)


def test_offset_is_read_from_the_telegram_row() -> None:
    client = _client([{"update_offset": 870000004}])

    assert _store(client).get_update_offset() == 870000004
    client.table.assert_called_with(BOT_STATE_TABLE)
    client.table.return_value.select.return_value.eq.assert_called_with("channel", BOT_CHANNEL)


def test_missing_row_means_no_offset_yet() -> None:
    assert _store(_client([])).get_update_offset() is None


def test_offset_is_upserted_per_channel_with_the_moment() -> None:
    client = _client()

    _store(client).save_update_offset(870000004)

    upsert = client.table.return_value.upsert
    assert upsert.call_args.args[0] == {
        "channel": "telegram",
        "update_offset": 870000004,
        "updated_at": NOW.isoformat(),
    }
    assert upsert.call_args.kwargs == {"on_conflict": "channel"}


def test_mutes_are_read_as_domain_mutes() -> None:
    client = _client(
        [
            {"symbol": "NVDA", "muted_until": "2026-11-01T08:00:00+00:00"},
            {"symbol": "TSLA", "muted_until": "2026-10-20T08:00:00Z"},
        ]
    )

    mutes = _store(client).get_symbol_mutes()

    client.table.assert_called_with(SYMBOL_MUTES_TABLE)
    assert mutes == [
        SymbolMute("NVDA", datetime(2026, 11, 1, 8, 0, tzinfo=UTC)),
        SymbolMute("TSLA", datetime(2026, 10, 20, 8, 0, tzinfo=UTC)),
    ]


@pytest.mark.parametrize(
    "row",
    [{"symbol": "NVDA", "muted_until": "jutro"}, {"symbol": None, "muted_until": "2026-11-01"}],
    ids=["bad-moment", "no-symbol"],
)
def test_unreadable_mute_row_is_malformed(row: dict[str, Any]) -> None:
    with pytest.raises(MalformedPayload):
        _store(_client([row])).get_symbol_mutes()


def test_mute_upserts_the_upper_case_symbol_with_end_and_moment() -> None:
    client = _client()
    until = datetime(2026, 11, 8, 8, 0, tzinfo=UTC)

    _store(client).mute_symbol("nvda", until, muted_at=NOW)

    upsert = client.table.return_value.upsert
    assert upsert.call_args.args[0] == {
        "symbol": "NVDA",
        "muted_until": until.isoformat(),
        "created_at": NOW.isoformat(),
    }
    assert upsert.call_args.kwargs == {"on_conflict": "symbol"}


def test_unmute_deletes_the_symbol_row() -> None:
    client = _client()

    _store(client).unmute_symbol("tsla")

    client.table.assert_called_with(SYMBOL_MUTES_TABLE)
    client.table.return_value.delete.return_value.eq.assert_called_with("symbol", "TSLA")


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (httpx.ConnectError("connection reset"), ProviderUnavailable),
        (APIError({"message": "relation does not exist", "code": "42P01"}), PortError),
    ],
    ids=["transport", "api"],
)
def test_provider_errors_leave_as_port_errors(error: Exception, kind: type[PortError]) -> None:
    client = _client()
    client.table.return_value.select.return_value.execute.side_effect = error

    with pytest.raises(kind):
        _store(client).get_symbol_mutes()
