"""MAIL-48 / MAIL-45 — `main_bot.py`: DI bota właściciela, flaga, sekrety i bramka schematu."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

import main_bot
from src.application.errors import ProviderUnavailable
from src.application.ports import RepositoryPort
from src.application.use_cases.handle_user_commands import (
    BotRunOutcome,
    HandleUserCommandsUseCase,
)
from src.config import Settings
from src.infrastructure.adapters.bot_store import SupabaseBotStore
from src.infrastructure.adapters.telegram_command_source import TelegramCommandSource
from src.infrastructure.adapters.write_guard import ReadOnlyBotState, ReadOnlySymbolMutes
from tests.entrypoint_support import build_settings

TOKEN = "123456789:AAH-secret_bot_token"
CHAT = "100000001"


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "telegram_bot_enabled": True,
        "telegram_bot_token": TOKEN,
        "telegram_chat_id": CHAT,
        "symbols": ["AAPL", "NVDA"],
        "crypto_symbols": ["BTC"],
        "db_writes_enabled": True,
    }
    values.update(overrides)
    return build_settings(**values)


def _repository(schema_version: int | None = 80) -> MagicMock:
    repository = MagicMock(spec=RepositoryPort)
    repository.schema_version.return_value = schema_version
    return repository


@pytest.fixture
def use_case(mocker: Any) -> MagicMock:
    use_case = MagicMock(spec=HandleUserCommandsUseCase)
    use_case.run.return_value = BotRunOutcome(handled=1, failed_replies=0)
    mocker.patch("main_bot.build_command_use_case", return_value=use_case)
    return use_case


def test_flag_off_ends_the_run_before_any_adapter(mocker: Any, use_case: MagicMock) -> None:
    build_repository = mocker.patch("main_agent.build_repository")

    assert main_bot.main(_settings(telegram_bot_enabled=False)) == 0

    build_repository.assert_not_called()
    use_case.run.assert_not_called()


@pytest.mark.parametrize("missing", ["telegram_bot_token", "telegram_chat_id"])
def test_missing_secret_is_a_no_op(mocker: Any, use_case: MagicMock, missing: str) -> None:
    build_repository = mocker.patch("main_agent.build_repository")

    assert main_bot.main(_settings(**{missing: None})) == 0

    build_repository.assert_not_called()
    use_case.run.assert_not_called()


def test_enabled_bot_runs_the_use_case(mocker: Any, use_case: MagicMock) -> None:
    mocker.patch("main_agent.build_repository", return_value=_repository())

    assert main_bot.main(_settings()) == 0

    use_case.run.assert_called_once_with()


def test_unreadable_updates_exit_1_so_the_workflow_alert_fires(
    mocker: Any, use_case: MagicMock
) -> None:
    mocker.patch("main_agent.build_repository", return_value=_repository())
    use_case.run.side_effect = ProviderUnavailable("telegram down")

    assert main_bot.main(_settings()) == 1


def test_database_without_migration_079_switches_the_bot_off(
    mocker: Any, use_case: MagicMock
) -> None:
    mocker.patch("main_agent.build_repository", return_value=_repository(schema_version=76))

    assert main_bot.main(_settings()) == 0

    use_case.run.assert_not_called()


def test_wiring_allowlists_the_owner_chat_and_passes_the_config_watchlist(mocker: Any) -> None:
    mocker.patch.object(SupabaseBotStore, "connect", return_value=MagicMock(spec=SupabaseBotStore))
    settings = _settings(symbol_mutes_enabled=True)

    built = main_bot.build_command_use_case(settings, _repository())

    assert isinstance(built, HandleUserCommandsUseCase)
    source = built._source
    assert isinstance(source, TelegramCommandSource)
    assert source._allowed == frozenset({CHAT})
    assert built._options.config_symbols == ("AAPL", "NVDA", "BTC")
    assert built._options.mutes_enabled is True
    assert built._positions is None


def test_local_run_guards_the_offset_and_mute_writes(mocker: Any) -> None:
    mocker.patch.object(SupabaseBotStore, "connect", return_value=MagicMock(spec=SupabaseBotStore))

    built = main_bot.build_command_use_case(_settings(db_writes_enabled=False), _repository())

    assert isinstance(built._state, ReadOnlyBotState)
    assert isinstance(built._mutes, ReadOnlySymbolMutes)
