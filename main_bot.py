"""Bot entry point — komendy właściciela z Telegrama (MAIL-48, MAIL-45).

Cron `bot_poll.yml` co 15 minut: jeden przebieg pollingu Telegram Bot API
(`getUpdates` z offsetem z tabeli `bot_state`, migracja 079), odpowiedź na każdą
komendę z czatu `TELEGRAM_CHAT_ID` i zapis następnego offsetu. Polling zamiast
webhooka — bot nie ma własnego serwera (latencja: do kwadransa plus opóźnienie crona).

Allowlista: odpowiada wyłącznie czatowi z sekretu `TELEGRAM_CHAT_ID` (tego samego,
na który idzie push). Bez niej każdy, kto znajdzie bota, zobaczyłby portfel.

FinOps: **zero płatnych wywołań** — darmowy Bot API, odczyty Supabase i zapis
jednego wiersza (`bot_state`, `symbol_mutes` przy `/mute` i `/unmute`). Żadnego LLM,
Alpha Vantage ani embeddingów; bot nie zapisuje snapshotów cen ani predykcji.

Przebieg kończy się od razu (exit 0), gdy `telegram_bot_enabled` jest wyłączone, gdy
brakuje któregoś z sekretów albo gdy baza nie ma jeszcze migracji 079 (bramka schematu
NIEZ-30). Nieczytelne `getUpdates` albo offset → exit 1, a krok alertu workflowu
mailuje o porażce. Bot nie pinguje heartbeatu (NIEZ-13): przebieg co 15 minut
zasłoniłby ciszę Fast Loopa i watcha.

Użycie:
    uv run python main_bot.py
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from datetime import UTC, datetime

from src.application.errors import PortError
from src.application.ports import (
    BotStatePort,
    PortfolioPositionsPort,
    RepositoryPort,
    SymbolMutePort,
)
from src.application.quota_monitor import QuotaMonitor
from src.application.use_cases.handle_user_commands import BotOptions, HandleUserCommandsUseCase
from src.config import Settings
from src.infrastructure.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def build_bot_store(settings: Settings) -> tuple[BotStatePort, SymbolMutePort]:
    """Offset i wyciszenia w Supabase; bez `db_writes_enabled` (lokalny run, BEZ-08)
    odczyty działają, a zapisy są pomijane — lokalny przebieg nie przesuwa offsetu
    produkcji i nie zmienia wyciszeń."""
    from src.infrastructure.adapters.bot_store import SupabaseBotStore
    from src.infrastructure.adapters.write_guard import ReadOnlyBotState, ReadOnlySymbolMutes

    store = SupabaseBotStore.connect(settings.supabase_url, settings.supabase_key)
    if settings.db_writes_enabled:
        return store, store
    return ReadOnlyBotState(store), ReadOnlySymbolMutes(store)


def build_command_use_case(
    settings: Settings,
    repository: RepositoryPort,
    *,
    clock: Callable[[], datetime] = _utc_now,
) -> HandleUserCommandsUseCase:
    """DI bota: źródło i odpowiedź Telegrama z allowlistą `TELEGRAM_CHAT_ID`, stan bota,
    repozytorium i pozycje z fabryk Fast Loopa (tylko darmowe adaptery)."""
    from main_agent import build_positions_port
    from src.infrastructure.adapters.telegram_command_source import (
        TelegramCommandReply,
        TelegramCommandSource,
    )

    token = settings.telegram_bot_token or ""
    chat_id = settings.telegram_chat_id or ""
    state, mutes = build_bot_store(settings)
    positions: PortfolioPositionsPort | None = (
        build_positions_port(settings) if settings.portfolio_positions_enabled else None
    )
    return HandleUserCommandsUseCase(
        source=TelegramCommandSource(bot_token=token, allowed_chat_ids=[chat_id]),
        reply=TelegramCommandReply(bot_token=token),
        state=state,
        mutes=mutes,
        repository=repository,
        positions=positions,
        options=BotOptions(
            config_symbols=(*settings.symbols, *settings.crypto_symbols),
            mutes_enabled=settings.symbol_mutes_enabled,
            display_timezone=settings.display_timezone,
        ),
        clock=clock,
    )


def main(settings: Settings | None = None, *, clock: Callable[[], datetime] = _utc_now) -> int:
    resolved = settings or Settings.from_env()
    configure_logging("main_bot", resolved.secret_values())
    if not resolved.telegram_bot_enabled:
        logger.info("Bot wyłączony (telegram_bot_enabled = false) — przebieg bez odczytów")
        return 0
    if not (resolved.telegram_bot_token and resolved.telegram_chat_id):
        logger.warning(
            "Bot włączony, ale brakuje TELEGRAM_BOT_TOKEN albo TELEGRAM_CHAT_ID — przebieg "
            "pominięty (ustaw oba jako GitHub repo Secrets)"
        )
        return 0
    # Import lokalny — fabryki z DI Containera Fast Loopa; tylko darmowe adaptery.
    from main_agent import build_repository, gate_flags_by_schema

    repository = build_repository(resolved)
    # NIEZ-30: bez migracji 079 (offset) albo 080 (wyciszenia) flaga wyłącza się na
    # ten przebieg, jak w Fast Loopie i watchu.
    resolved, repository = gate_flags_by_schema(
        resolved, repository, QuotaMonitor(), now=clock(), rebuild=build_repository
    )
    if not resolved.telegram_bot_enabled:
        logger.warning("Bot: baza bez migracji 079 — przebieg pominięty")
        return 0
    try:
        build_command_use_case(resolved, repository, clock=clock).run()
    except PortError:
        logger.exception("Bot: nie da się odczytać komend ani offsetu — exit 1")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
