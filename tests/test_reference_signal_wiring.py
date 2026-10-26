"""WIA-24: krok sygnału referencyjnego w Slow Loopie (`main_trainer`).

Krok jest za flagą `reference_signal_enabled` (migracja 042), bierze darmowy router
cen (`main_agent.build_market_port`: Finnhub /quote, CoinGecko) i repozytorium
Slow Loopa — z flagą `db_writes_enabled` off to `ReadOnlyRepository`, który pomija
zapis. Żadna fabryka płatnego portu nie jest wołana.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

import main_trainer
from src.application.ports import (
    EmbeddingPort,
    FundamentalsPort,
    LLMPort,
    MarketDataPort,
    NewsPort,
    RepositoryPort,
    SentimentPort,
)
from src.config import Settings
from src.domain.value_objects import Money
from tests.settings_kwargs import NO_ENV_FILE


@pytest.fixture
def settings() -> Settings:
    return Settings(
        **NO_ENV_FILE,
        openai_api_key="sk-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1"],
        supabase_url="https://test.supabase.co",
        supabase_key="anon",
        symbols=["AAPL", "CSPX.L"],
        crypto_symbols=["BTC"],
        symbols_unsupported_price=["CSPX.L"],
        reference_signal_enabled=True,
    )


def _repository() -> Mock:
    repository = Mock(spec=RepositoryPort)
    repository.get_reference_reads.return_value = []
    repository.get_open_reference_calls.return_value = []
    return repository


def _free_router(mocker: Any) -> Mock:
    market = Mock(spec=MarketDataPort)
    market.get_current_price.return_value = Money(Decimal("100"))
    router: Mock = mocker.patch("main_agent.build_market_port", return_value=market)
    return router


def test_flag_off_skips_the_step_without_any_read(settings: Settings, mocker: Any) -> None:
    settings.reference_signal_enabled = False
    build_market = _free_router(mocker)
    repository = _repository()

    main_trainer._run_reference_signal(settings, repository)

    build_market.assert_not_called()
    assert repository.method_calls == []


def test_reads_the_trained_universe_from_the_free_price_router(
    settings: Settings, mocker: Any
) -> None:
    build_market = _free_router(mocker)
    repository = _repository()

    main_trainer._run_reference_signal(settings, repository)

    build_market.assert_called_once_with(settings)
    read_symbols = [c.args[0] for c in build_market.return_value.get_current_price.call_args_list]
    assert read_symbols == ["AAPL", "BTC"]
    assert [c.args[0].symbol for c in repository.save_reference_read.call_args_list] == [
        "AAPL",
        "BTC",
    ]


def test_never_calls_paid_ports(settings: Settings, mocker: Any) -> None:
    _free_router(mocker)
    # Każda droga do płatnego portu (LLM, rada, embeddingi, Alpha Vantage) zwraca atrapę
    # ze `spec` portu — krok nie może zbudować żadnej ani niczego na niej wywołać.
    paid = {
        "main_agent.build_llm_adapter": Mock(spec=LLMPort),
        "main_agent.build_council_adapter": Mock(spec=LLMPort),
        "main_agent.build_embedding_adapter": Mock(spec=EmbeddingPort),
        "main_agent.AlphaVantageSentimentAdapter": Mock(spec=SentimentPort),
        "main_agent.AlphaVantageNewsAdapter": Mock(spec=NewsPort),
        "main_trainer.AlphaVantageFundamentalsAdapter": Mock(spec=FundamentalsPort),
        "main_trainer._build_llm_or_unavailable": Mock(spec=LLMPort),
    }
    builders = [mocker.patch(target, return_value=port) for target, port in paid.items()]
    builders += [
        mocker.patch(f"main_agent.{name}") for name in ("build_use_case", "build_alpha_enrichment")
    ]

    main_trainer._run_reference_signal(settings, _repository())

    for builder in builders:
        builder.assert_not_called()
    for port in paid.values():
        port.assert_not_called()
        assert port.method_calls == []


def test_never_writes_price_snapshots_or_llm_predictions(settings: Settings, mocker: Any) -> None:
    _free_router(mocker)
    repository = _repository()

    main_trainer._run_reference_signal(settings, repository)

    repository.save_price_snapshot.assert_not_called()
    repository.save_prediction.assert_not_called()


def test_local_run_without_db_writes_skips_the_write(
    settings: Settings, mocker: Any, caplog: pytest.LogCaptureFixture
) -> None:
    # BEZ-08: Slow Loop bez `db_writes_enabled` dostaje ReadOnlyRepository.
    import logging

    from src.infrastructure.adapters.write_guard import ReadOnlyRepository

    _free_router(mocker)
    delegate = _repository()

    with caplog.at_level(logging.WARNING, logger="src.infrastructure.adapters.write_guard"):
        main_trainer._run_reference_signal(settings, ReadOnlyRepository(delegate))

    delegate.save_reference_read.assert_not_called()
    assert "save_reference_read" in caplog.text


def test_failing_step_does_not_raise(
    settings: Settings, mocker: Any, caplog: pytest.LogCaptureFixture
) -> None:
    _free_router(mocker)
    repository = _repository()
    repository.get_reference_reads.side_effect = RuntimeError("relation does not exist")

    main_trainer._run_reference_signal(settings, repository)

    assert "Reference signal" in caplog.text


def test_slow_loop_runs_the_step_once_after_training(settings: Settings, mocker: Any) -> None:
    mocker.patch("src.infrastructure.adapters.supabase_repo.build_supabase_client")
    fake_uc = MagicMock()
    fake_uc.run.return_value = {"status": "skipped"}
    fake_uc.run_pooled.return_value = {"status": "skipped"}
    mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
    step = mocker.patch("main_trainer._run_reference_signal")
    settings.paid_calls_enabled = False

    assert main_trainer.main(settings) == 0

    step.assert_called_once()
    assert step.call_args.args[0] is settings


def test_reads_carry_the_run_time_in_utc(settings: Settings, mocker: Any) -> None:
    _free_router(mocker)
    repository = _repository()
    moment = datetime(2026, 10, 25, 3, 0, tzinfo=UTC)

    main_trainer._run_reference_signal(settings, repository, now=moment)

    assert {c.args[0].read_at for c in repository.save_reference_read.call_args_list} == {moment}
