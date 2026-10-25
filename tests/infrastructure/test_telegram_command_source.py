"""MAIL-48 — bot właściciela w Telegramie: `getUpdates` z allowlistą i `sendMessage`."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests
from pytest_mock import MockerFixture

from src.application.errors import MalformedPayload, PortError, ProviderUnavailable
from src.application.ports import CommandReplyPort, UserCommandSourcePort, UserMessage
from src.infrastructure.adapters.telegram_command_source import (
    MAX_MESSAGE_LENGTH,
    TelegramCommandReply,
    TelegramCommandSource,
)

TOKEN = "123456789:AAH-secret_bot_token"
OWNER = "100000001"
STRANGER = 200000002


def _response(status_code: int = 200, body: Any = None) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.json.return_value = body
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(
            f"{status_code} Client Error for url: https://api.telegram.org/bot{TOKEN}/getUpdates",
            response=response,
        )
    return response


def _update(update_id: int, chat_id: int, text: str | None) -> dict[str, Any]:
    message: dict[str, Any] = {"message_id": update_id, "chat": {"id": chat_id, "type": "private"}}
    if text is not None:
        message["text"] = text
    return {"update_id": update_id, "message": message}


def _ok(*updates: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "result": list(updates)}


@pytest.fixture
def source() -> TelegramCommandSource:
    return TelegramCommandSource(bot_token=TOKEN, allowed_chat_ids=[OWNER])


def test_adapters_implement_their_ports() -> None:
    assert issubclass(TelegramCommandSource, UserCommandSourcePort)
    assert issubclass(TelegramCommandReply, CommandReplyPort)


def test_only_owner_text_messages_pass_and_offset_moves_past_every_update(
    source: TelegramCommandSource, mocker: MockerFixture
) -> None:
    mocker.patch(
        "requests.Session.get",
        return_value=_response(
            body=_ok(
                _update(41, int(OWNER), "/why nvda"),
                _update(42, STRANGER, "/pos"),
                _update(43, int(OWNER), None),
            )
        ),
    )

    batch = source.fetch_messages(41)

    assert batch.messages == (UserMessage(update_id=41, chat_id=OWNER, text="/why nvda"),)
    assert batch.next_offset == 44


def test_offset_and_message_filter_go_to_get_updates(
    source: TelegramCommandSource, mocker: MockerFixture
) -> None:
    get = mocker.patch("requests.Session.get", return_value=_response(body=_ok()))

    batch = source.fetch_messages(77)

    url = get.call_args.args[0]
    params = get.call_args.kwargs["params"]
    assert url == f"https://api.telegram.org/bot{TOKEN}/getUpdates"
    assert params["offset"] == 77
    assert json.loads(params["allowed_updates"]) == ["message"]
    assert batch.next_offset is None
    assert batch.messages == ()


def test_first_run_asks_without_offset(
    source: TelegramCommandSource, mocker: MockerFixture
) -> None:
    get = mocker.patch("requests.Session.get", return_value=_response(body=_ok()))

    source.fetch_messages(None)

    assert "offset" not in get.call_args.kwargs["params"]


def test_stranger_is_dropped_without_logging_its_chat_or_text(
    source: TelegramCommandSource, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    mocker.patch(
        "requests.Session.get",
        return_value=_response(body=_ok(_update(5, STRANGER, "/pos tajne"))),
    )

    batch = source.fetch_messages(None)

    assert batch.messages == ()
    assert batch.next_offset == 6
    assert str(STRANGER) not in caplog.text
    assert "tajne" not in caplog.text


def test_network_error_is_translated_and_does_not_carry_the_token(
    source: TelegramCommandSource, mocker: MockerFixture
) -> None:
    mocker.patch(
        "requests.Session.get",
        side_effect=requests.ConnectionError(
            f"HTTPSConnectionPool: Max retries exceeded with url: /bot{TOKEN}/getUpdates"
        ),
    )

    with pytest.raises(ProviderUnavailable) as caught:
        source.fetch_messages(None)

    assert TOKEN not in str(caught.value)
    assert "AAH-secret_bot_token" not in str(caught.value)


def test_http_error_is_translated_and_does_not_carry_the_token(
    source: TelegramCommandSource, mocker: MockerFixture
) -> None:
    mocker.patch("requests.Session.get", return_value=_response(status_code=401))

    with pytest.raises(PortError) as caught:
        source.fetch_messages(None)

    assert TOKEN not in str(caught.value)


@pytest.mark.parametrize(
    "body",
    [{"ok": False, "description": "Conflict"}, {"ok": True, "result": [{"no_id": 1}]}, ["x"]],
    ids=["not-ok", "update-without-id", "not-a-dict"],
)
def test_unexpected_payload_is_malformed(
    source: TelegramCommandSource, mocker: MockerFixture, body: Any
) -> None:
    mocker.patch("requests.Session.get", return_value=_response(body=body))

    with pytest.raises(MalformedPayload):
        source.fetch_messages(None)


def test_reply_posts_plain_text_to_the_chat(mocker: MockerFixture) -> None:
    post = mocker.patch("requests.Session.post", return_value=_response(body={"ok": True}))

    TelegramCommandReply(bot_token=TOKEN).reply(OWNER, "<b>NVDA</b> & co")

    assert post.call_args.args[0] == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = post.call_args.kwargs["json"]
    assert payload["chat_id"] == OWNER
    assert payload["text"] == "<b>NVDA</b> & co"
    assert "parse_mode" not in payload


def test_reply_is_clipped_to_the_telegram_limit(mocker: MockerFixture) -> None:
    post = mocker.patch("requests.Session.post", return_value=_response(body={"ok": True}))

    TelegramCommandReply(bot_token=TOKEN).reply(OWNER, "x" * (MAX_MESSAGE_LENGTH + 500))

    text = post.call_args.kwargs["json"]["text"]
    assert len(text) == MAX_MESSAGE_LENGTH
    assert text.endswith("(ucięte)")


def test_reply_network_error_does_not_carry_the_token(mocker: MockerFixture) -> None:
    mocker.patch(
        "requests.Session.post",
        side_effect=requests.Timeout(f"Read timed out: /bot{TOKEN}/sendMessage"),
    )

    with pytest.raises(ProviderUnavailable) as caught:
        TelegramCommandReply(bot_token=TOKEN).reply(OWNER, "ok")

    assert TOKEN not in str(caught.value)
