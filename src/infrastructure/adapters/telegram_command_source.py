"""Bot właściciela w Telegramie (MAIL-48): komendy z `getUpdates`, odpowiedzi `sendMessage`.

Polling zamiast webhooka: bot nie ma własnego serwera, cron GitHub Actions odpytuje
Bot API co 15 minut z offsetem z bazy (`bot_state`, migracja 079). Telegram trzyma
nieodebrane aktualizacje 24 h; wywołanie z `offset` potwierdza wszystkie starsze.

Allowlista czatów: do use case'u dociera wyłącznie wiadomość z czatu właściciela
(`TELEGRAM_CHAT_ID`). Bez niej każdy, kto znajdzie bota, zobaczyłby portfel. Obcy
czat przesuwa offset (inaczej wracałby co przebieg), ale jego treść ani id nie
trafiają do logu — tylko liczba odrzuconych.

Token bota siedzi w ścieżce URL-a (`/bot<token>/`), więc żądania idą przez
`translated_get` / `translated_post` i `raise_for_status_translated` z `_http.py`:
komunikat wyjątku nie niesie tokenu, a typ to klasa taksonomii portów (ARCH-33).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Collection
from typing import Any

from src.application.errors import MalformedPayload
from src.application.ports import (
    CommandReplyPort,
    UserCommandSourcePort,
    UserMessage,
    UserMessageBatch,
)
from src.infrastructure.adapters._http import (
    build_session,
    json_payload,
    raise_for_status_translated,
    translated_get,
    translated_post,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.telegram.org"
DEFAULT_TIMEOUT = 15
# Jedna paczka na przebieg: więcej komend w 15 minut właściciel nie wyśle, a reszta
# poczeka do następnego przebiegu (offset przesuwa się tylko o pobrane).
UPDATES_LIMIT = 100
# Limit długości wiadomości Bot API; dłuższa odpowiedź to 400 i zgubiona odpowiedź.
MAX_MESSAGE_LENGTH = 4096
TRUNCATION_MARK = "\n…(ucięte)"


class TelegramCommandSource(UserCommandSourcePort):
    """`getUpdates` z offsetem, filtrowany allowlistą `allowed_chat_ids`."""

    def __init__(
        self,
        bot_token: str,
        allowed_chat_ids: Collection[str],
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/bot{bot_token}/getUpdates"
        self._allowed = frozenset(chat.strip() for chat in allowed_chat_ids if chat.strip())
        self._timeout = timeout
        self._session = build_session()

    def fetch_messages(self, offset: int | None) -> UserMessageBatch:
        params: dict[str, str | int] = {
            "timeout": 0,
            "limit": UPDATES_LIMIT,
            "allowed_updates": json.dumps(["message"]),
        }
        if offset is not None:
            params["offset"] = offset
        response = translated_get(self._session, self._url, params=params, timeout=self._timeout)
        raise_for_status_translated(response)
        updates = _updates(json_payload(response))
        messages: list[UserMessage] = []
        rejected = 0
        for update in updates:
            message = _owner_message(update)
            if message is None:
                continue
            if message.chat_id not in self._allowed:
                rejected += 1
                continue
            messages.append(message)
        if rejected:
            logger.warning(
                "Telegram bot: %d message(s) from chats outside the allowlist dropped", rejected
            )
        next_offset = max(_update_id(update) for update in updates) + 1 if updates else None
        return UserMessageBatch(messages=tuple(messages), next_offset=next_offset)


class TelegramCommandReply(CommandReplyPort):
    """`sendMessage` czystym tekstem (bez `parse_mode` — odpowiedź niesie treść z bazy
    i z czatu, której nie escapujemy). POST bez ponowień: Bot API nie ma klucza
    idempotencji, a ponowiony 5xx mógłby zdublować odpowiedź."""

    def __init__(
        self, bot_token: str, base_url: str = DEFAULT_BASE_URL, timeout: int = DEFAULT_TIMEOUT
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/bot{bot_token}/sendMessage"
        self._timeout = timeout
        self._session = build_session(retry_post=False)

    def reply(self, chat_id: str, text: str) -> None:
        payload = {
            "chat_id": chat_id,
            "text": _within_limit(text),
            "disable_web_page_preview": True,
        }
        response = translated_post(self._session, self._url, json=payload, timeout=self._timeout)
        raise_for_status_translated(response)


def _updates(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise MalformedPayload("Telegram getUpdates: response is not ok")
    result = payload.get("result")
    if not isinstance(result, list) or not all(
        isinstance(update, dict) and isinstance(update.get("update_id"), int) for update in result
    ):
        raise MalformedPayload("Telegram getUpdates: result is not a list of updates")
    return result


def _update_id(update: dict[str, Any]) -> int:
    update_id: int = update["update_id"]
    return update_id


def _owner_message(update: dict[str, Any]) -> UserMessage | None:
    """Wiadomość tekstowa z aktualizacji; inna (naklejka, edycja, zdjęcie) → None."""
    message = update.get("message")
    if not isinstance(message, dict):
        return None
    chat = message.get("chat")
    text = message.get("text")
    if not isinstance(chat, dict) or not isinstance(text, str):
        return None
    if not isinstance(chat.get("id"), int):
        return None
    return UserMessage(update_id=_update_id(update), chat_id=str(chat["id"]), text=text)


def _within_limit(text: str) -> str:
    if len(text) <= MAX_MESSAGE_LENGTH:
        return text
    return text[: MAX_MESSAGE_LENGTH - len(TRUNCATION_MARK)] + TRUNCATION_MARK
