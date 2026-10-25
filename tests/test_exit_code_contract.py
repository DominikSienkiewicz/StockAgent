"""NIEZ-38: wspólny kontrakt kodów wyjścia czterech entry-pointów.

Decyzja 2026-10-23 (opcja a): exit 1 wyłącznie wtedy, gdy padły WSZYSTKIE symbole
przebiegu; częściowa porażka to exit 0 z WARNING-iem w logu i liczbami w step summary.
Każdy entry point jest tu prowadzony przez swój prawdziwy `main()` z trzema symbolami,
z których rzucają te z `failing` — reszta zachowuje się jak zdrowy przebieg.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

import main_agent
import main_backtest
import main_trainer
import main_watch
from src.application.ports import RepositoryPort
from src.config import Settings
from src.domain.backtest import BacktestSummary
from src.domain.value_objects import Money

SYMBOLS = ["AAPL", "MSFT", "NVDA"]

Driver = Callable[[Settings, Any, frozenset[str]], int]


def _raise_for(failing: frozenset[str], symbol: str) -> None:
    if symbol in failing:
        raise RuntimeError(f"{symbol} boom")


def _run_agent(settings: Settings, mocker: Any, failing: frozenset[str]) -> int:
    def run(symbol: str, **_: Any) -> dict[str, Any]:
        _raise_for(failing, symbol)
        return {"status": "ignored", "delta": Decimal("0")}

    fake_uc = MagicMock()
    fake_uc.run.side_effect = run
    mocker.patch("main_agent.build_use_case", return_value=fake_uc)
    return main_agent.main(settings)


def _run_trainer(settings: Settings, mocker: Any, failing: frozenset[str]) -> int:
    def run(symbol: str, **_: Any) -> dict[str, Any]:
        _raise_for(failing, symbol)
        return {"status": "skipped", "reason": "not enough data"}

    fake_uc = MagicMock()
    fake_uc.run.side_effect = run
    fake_uc.run_pooled.return_value = {"status": "trained_successfully"}
    mocker.patch("main_trainer.build_use_case", return_value=fake_uc)
    return main_trainer.main(settings)


def _run_watch(settings: Settings, mocker: Any, failing: frozenset[str]) -> int:
    def last_snapshot(symbol: str) -> tuple[Money, datetime]:
        _raise_for(failing, symbol)
        return Money(Decimal("100")), datetime.now(UTC)

    repo = MagicMock(spec=RepositoryPort)
    repo.get_sent_shock_alerts.return_value = set()
    repo.get_last_price_snapshot.side_effect = last_snapshot
    market = MagicMock()
    market.get_current_price.return_value = Money(Decimal("100"))
    mocker.patch("main_agent.build_repository", return_value=repo)
    mocker.patch("main_agent.build_market_port", return_value=market)
    mocker.patch("main_agent.build_push_notifier", return_value=MagicMock())
    return main_watch.main(settings)


def _run_backtest(settings: Settings, mocker: Any, failing: frozenset[str]) -> int:
    def run(symbol: str, **_: Any) -> BacktestSummary:
        _raise_for(failing, symbol)
        return BacktestSummary(
            n_folds=4,
            mean_candidate_rmse=0.03,
            mean_baseline_rmse=0.05,
            mean_hit_rate=0.6,
            improvement_pct=40.0,
            beats_baseline=True,
            verdict="model bije baseline o 40.0%",
        )

    fake_uc = MagicMock()
    fake_uc.run.side_effect = run
    mocker.patch("main_backtest.BacktestUseCase", return_value=fake_uc)
    return main_backtest.main(settings)


ENTRY_POINTS: dict[str, Driver] = {
    "main_agent": _run_agent,
    "main_trainer": _run_trainer,
    "main_watch": _run_watch,
    "main_backtest": _run_backtest,
}


@pytest.fixture
def contract_settings(settings: Settings) -> Settings:
    """Trzy akcje, bez krypto i bez pomijania sesji NYSE (wynik nie zależy od dnia)."""
    settings.symbols = list(SYMBOLS)
    settings.crypto_symbols = []
    settings.nyse_closure_skip_enabled = False
    settings.shock_alerts_enabled = True
    settings.weekly_recap_enabled = False
    settings.calibration_judge_enabled = False
    return settings


def _partial_failure_warnings(
    caplog: pytest.LogCaptureFixture, entry_point: str
) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == entry_point
        and record.levelno == logging.WARNING
        and "1/3" in record.getMessage()
    ]


@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS))
def test_partial_failure_policy_is_uniform(
    entry_point: str,
    contract_settings: Settings,
    mock_external_clients: None,
    mocker: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Jeden z trzech symboli rzuca: przebieg zrobił swoją robotę dla reszty,
    # więc job zostaje zielony, a porażkę widać w WARNING-u z liczbami.
    with caplog.at_level(logging.WARNING, logger=entry_point):
        exit_code = ENTRY_POINTS[entry_point](contract_settings, mocker, frozenset({"MSFT"}))

    assert exit_code == 0
    assert _partial_failure_warnings(caplog, entry_point), (
        f"{entry_point}: częściowa porażka bez WARNING-u „1/3”"
    )


@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS))
def test_total_failure_exits_one(
    entry_point: str,
    contract_settings: Settings,
    mock_external_clients: None,
    mocker: Any,
) -> None:
    # Padły wszystkie symbole (np. baza albo dostawca leży): zielony job ukryłby awarię.
    assert ENTRY_POINTS[entry_point](contract_settings, mocker, frozenset(SYMBOLS)) == 1


@pytest.mark.parametrize("entry_point", sorted(ENTRY_POINTS))
def test_clean_run_exits_zero_without_partial_failure_warning(
    entry_point: str,
    contract_settings: Settings,
    mock_external_clients: None,
    mocker: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=entry_point):
        exit_code = ENTRY_POINTS[entry_point](contract_settings, mocker, frozenset())

    assert exit_code == 0
    assert not [
        record
        for record in caplog.records
        if record.name == entry_point and "NIEZ-38" in record.getMessage()
    ]
