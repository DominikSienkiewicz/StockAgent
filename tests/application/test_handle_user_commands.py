"""MAIL-48 / MAIL-45 — obsługa komend bota właściciela na samych portach, bez LLM."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock

import pytest

from src.application.errors import ProviderUnavailable
from src.application.ports import (
    BotStatePort,
    CommandReplyPort,
    CycleRun,
    LatestPrediction,
    LLMPort,
    PortfolioPositionsPort,
    RepositoryPort,
    StoredReceipt,
    SymbolMutePort,
    UserCommandSourcePort,
    UserMessage,
    UserMessageBatch,
)
from src.application.use_cases.handle_user_commands import (
    BotOptions,
    HandleUserCommandsUseCase,
)
from src.domain.portfolio import Position
from src.domain.user_command import Help, Last, Mute, Note, Pause, Pos, Unknown, Unmute, Watch, Why
from src.domain.value_objects import Money
from src.domain.watchlist import SymbolMute

NOW = datetime(2026, 10, 25, 8, 0, tzinfo=UTC)
OWNER = "100000001"
CONFIG = ("AAPL", "NVDA", "TSLA", "BTC")


def _use_case(
    *,
    messages: tuple[UserMessage, ...] = (),
    next_offset: int | None = None,
    mutes_enabled: bool = True,
    positions: PortfolioPositionsPort | None = None,
) -> tuple[HandleUserCommandsUseCase, dict[str, Mock]]:
    ports = {
        "source": Mock(spec=UserCommandSourcePort),
        "reply": Mock(spec=CommandReplyPort),
        "state": Mock(spec=BotStatePort),
        "mutes": Mock(spec=SymbolMutePort),
        "repository": Mock(spec=RepositoryPort),
    }
    ports["source"].fetch_messages.return_value = UserMessageBatch(messages, next_offset)
    ports["state"].get_update_offset.return_value = 41
    ports["mutes"].get_symbol_mutes.return_value = []
    use_case = HandleUserCommandsUseCase(
        source=ports["source"],
        reply=ports["reply"],
        state=ports["state"],
        mutes=ports["mutes"],
        repository=ports["repository"],
        positions=positions,
        options=BotOptions(
            config_symbols=CONFIG, mutes_enabled=mutes_enabled, display_timezone="Europe/Warsaw"
        ),
        clock=lambda: NOW,
    )
    return use_case, ports


def _message(text: str, update_id: int = 41) -> UserMessage:
    return UserMessage(update_id=update_id, chat_id=OWNER, text=text)


# --- przebieg -------------------------------------------------------------


def test_run_answers_every_message_and_saves_the_next_offset() -> None:
    use_case, ports = _use_case(
        messages=(_message("/help", 41), _message("/watch", 42)), next_offset=43
    )

    outcome = use_case.run()

    ports["source"].fetch_messages.assert_called_once_with(41)
    assert [c.args[0] for c in ports["reply"].reply.call_args_list] == [OWNER, OWNER]
    ports["state"].save_update_offset.assert_called_once_with(43)
    assert outcome.handled == 2
    assert outcome.failed_replies == 0


def test_run_without_new_updates_keeps_the_offset() -> None:
    use_case, ports = _use_case()

    outcome = use_case.run()

    ports["state"].save_update_offset.assert_not_called()
    ports["reply"].reply.assert_not_called()
    assert outcome.handled == 0


def test_failed_reply_is_counted_and_the_offset_still_moves() -> None:
    use_case, ports = _use_case(
        messages=(_message("/help", 41), _message("/pos", 42)), next_offset=43
    )
    ports["reply"].reply.side_effect = [ProviderUnavailable("telegram down"), None]

    outcome = use_case.run()

    assert outcome.failed_replies == 1
    assert ports["reply"].reply.call_count == 2
    ports["state"].save_update_offset.assert_called_once_with(43)


def test_failing_read_answers_with_an_error_instead_of_stopping_the_run() -> None:
    use_case, ports = _use_case(messages=(_message("/last"),), next_offset=42)
    ports["repository"].get_cycle_runs_since.side_effect = ProviderUnavailable("db down")

    use_case.run()

    text = ports["reply"].reply.call_args.args[1]
    assert "Nie udało się" in text
    ports["state"].save_update_offset.assert_called_once_with(42)


# --- /watch i /mute (MAIL-45) ----------------------------------------------


def test_watch_shows_the_effective_watchlist_and_that_adding_goes_through_config() -> None:
    use_case, ports = _use_case()
    ports["mutes"].get_symbol_mutes.return_value = [
        SymbolMute("NVDA", NOW + timedelta(days=3)),
        SymbolMute("TSLA", NOW - timedelta(days=1)),
    ]

    text = use_case.answer(Watch())

    assert "watchlista: 4 z config − 1 wyciszone" in text
    assert "NVDA do 28.10.2026" in text
    assert "AAPL, TSLA, BTC" in text
    assert "config.toml" in text


def test_watch_with_mutes_disabled_reads_no_mutes() -> None:
    use_case, ports = _use_case(mutes_enabled=False)

    text = use_case.answer(Watch())

    ports["mutes"].get_symbol_mutes.assert_not_called()
    assert "watchlista: 4 z config − 0 wyciszone" in text
    assert "symbol_mutes_enabled" in text


def test_mute_stores_the_end_of_the_mute_for_a_config_symbol() -> None:
    use_case, ports = _use_case()

    text = use_case.answer(Mute("NVDA", 7))

    ports["mutes"].mute_symbol.assert_called_once_with(
        "NVDA", NOW + timedelta(days=7), muted_at=NOW
    )
    assert "NVDA" in text
    assert "01.11.2026" in text
    assert "snapshot" in text


def test_mute_matches_the_config_symbol_regardless_of_case() -> None:
    use_case, ports = _use_case()

    use_case.answer(Mute("BTC"))

    assert ports["mutes"].mute_symbol.call_args.args[0] == "BTC"


def test_mute_of_a_symbol_outside_config_is_refused_without_a_write() -> None:
    use_case, ports = _use_case()

    text = use_case.answer(Mute("AMD", 7))

    ports["mutes"].mute_symbol.assert_not_called()
    assert "config.toml" in text


@pytest.mark.parametrize("command", [Mute("NVDA", 7), Unmute("NVDA")])
def test_mute_commands_with_mutes_disabled_write_nothing(command: Mute | Unmute) -> None:
    use_case, ports = _use_case(mutes_enabled=False)

    text = use_case.answer(command)

    ports["mutes"].mute_symbol.assert_not_called()
    ports["mutes"].unmute_symbol.assert_not_called()
    assert "symbol_mutes_enabled" in text


def test_unmute_deletes_the_mute() -> None:
    use_case, ports = _use_case()

    text = use_case.answer(Unmute("NVDA"))

    ports["mutes"].unmute_symbol.assert_called_once_with("NVDA")
    assert "NVDA" in text


# --- /pos, /last, /why -----------------------------------------------------


def test_pos_lists_holdings_without_amounts() -> None:
    positions = Mock(spec=PortfolioPositionsPort)
    positions.get_positions.return_value = [
        Position("NVDA", Decimal("10"), Decimal("120.50")),
        Position("NVDA", Decimal("2.5"), Decimal("140")),
        Position("AAPL", Decimal("4"), Decimal("200")),
    ]
    positions.get_as_of.return_value = NOW - timedelta(days=1)
    use_case, _ = _use_case(positions=positions)

    text = use_case.answer(Pos())

    assert "NVDA: 12,5" in text
    assert "AAPL: 4" in text
    assert "120" not in text
    assert "zł" not in text


def test_pos_without_the_positions_port_says_it_is_off() -> None:
    use_case, _ = _use_case(positions=None)

    assert "portfolio_positions_enabled" in use_case.answer(Pos())


def test_last_without_symbol_summarises_the_latest_fast_loop_run() -> None:
    use_case, ports = _use_case()
    ports["repository"].get_cycle_runs_since.return_value = [
        CycleRun("fast", "1", 1, NOW - timedelta(days=2), 60.0, 40, 38, 2, {}, True),
        CycleRun("fast", "2", 1, NOW - timedelta(hours=2), 60.0, 42, 41, 1, {}, False),
        CycleRun("slow", "3", 1, NOW - timedelta(hours=1), 60.0, 4, 4, 0, {}, None),
    ]

    text = use_case.answer(Last())

    assert "41 z 42" in text
    assert "błędy: 1" in text
    assert "niewysłany" in text


def test_last_with_symbol_shows_the_latest_prediction() -> None:
    use_case, ports = _use_case()
    ports["repository"].get_latest_prediction.return_value = LatestPrediction(
        price=Money(Decimal("187.12")), timestamp=NOW - timedelta(hours=3), reasoning="Popyt AI."
    )

    text = use_case.answer(Last("NVDA"))

    ports["repository"].get_latest_prediction.assert_called_once_with("NVDA")
    assert "187,12" in text
    assert "Popyt AI." in text


def test_why_renders_reasoning_and_receipt_threshold_without_llm() -> None:
    llm = Mock(spec=LLMPort)
    use_case, ports = _use_case()
    ports["repository"].get_latest_prediction.return_value = LatestPrediction(
        price=Money(Decimal("187.12")), timestamp=NOW - timedelta(hours=3), reasoning="Popyt AI."
    )
    ports["repository"].find_receipts.return_value = [
        StoredReceipt(
            prediction_id="p-1",
            symbol="NVDA",
            predicted_at=NOW - timedelta(hours=3),
            predicted_trend="BULLISH",
            is_trend_correct=None,
            payload={"schema_version": 1, "effective_threshold": "0.03", "precedents": []},
        )
    ]

    text = use_case.answer(Why("NVDA"))

    ports["repository"].find_receipts.assert_called_once_with(
        "NVDA", (NOW - timedelta(hours=3)).date()
    )
    assert "Popyt AI." in text
    assert "progu 3,0%" in text
    llm.assert_not_called()


def test_why_without_a_prediction_says_so() -> None:
    use_case, ports = _use_case()
    ports["repository"].get_latest_prediction.return_value = None

    assert "Brak predykcji NVDA" in use_case.answer(Why("NVDA"))


# --- pozostałe -------------------------------------------------------------


@pytest.mark.parametrize("command", [Note("sprzedałem"), Pause("")])
def test_note_and_pause_are_not_available_yet(command: Note | Pause) -> None:
    use_case, _ = _use_case()

    assert "jeszcze niedostępna" in use_case.answer(command)


def test_help_lists_the_commands() -> None:
    use_case, _ = _use_case()

    text = use_case.answer(Help())

    for name in ("/why", "/last", "/pos", "/watch", "/mute", "/unmute"):
        assert name in text


def test_unknown_explains_and_points_to_help() -> None:
    use_case, _ = _use_case()

    text = use_case.answer(Unknown("/sell NVDA", "nieznana komenda /sell"))

    assert "nieznana komenda /sell" in text
    assert "/help" in text
