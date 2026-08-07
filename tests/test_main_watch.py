"""Testy entry pointa `main_watch.py` — darmowy watch szoku poza cyklem (#11).

Kontrakt FinOps jest tu absolutny: watch NIE WOŁA żadnego płatnego portu
(LLM / Alpha Vantage / embeddingi). Kontrakt poprawnościowy jest równie ostry:
watch NIE ZAPISUJE snapshotu ceny — nadpisywanie go co godzinę zamknęłoby
bramkę volatility Fast Loopa (delta liczyłaby się względem ceny sprzed godziny).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import main_watch
from src.application.ports import RepositoryPort
from src.config import Settings
from src.domain.value_objects import Money


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="sk-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1"],
        supabase_url="https://test.supabase.co",
        supabase_key="anon",
        symbols=["AAPL"],
        crypto_symbols=[],
        volatility_threshold=Decimal("0.02"),
        shock_alerts_enabled=True,
    )


def _repo(price: str, age_hours: float, *, now: datetime | None = None) -> MagicMock:
    """Repo ze snapshotem `age_hours` starszym od `now`.

    `now` musi być tą samą chwilą, którą dostaje `run_watch` — wiek snapshotu
    liczy się względem niej, a przekroczenie `SNAPSHOT_MAX_AGE_HOURS` tłumi
    alert. Rozjazd tych dwóch zegarów daje test, który przechodzi z niewłaściwego
    powodu: stęchłego snapshotu zamiast sprawdzanej reguły.
    """
    repo = MagicMock(spec=RepositoryPort)
    stamp = (now or datetime.now(UTC)) - timedelta(hours=age_hours)
    repo.get_last_price_snapshot.return_value = (Money(Decimal(price)), stamp)
    repo.get_sent_shock_alerts.return_value = set()
    return repo


def _market(price: str) -> MagicMock:
    market = MagicMock()
    market.get_current_price.return_value = Money(Decimal(price))
    return market


class TestWatchCycle:
    def test_big_drop_emits_one_alert(self, settings) -> None:
        repo = _repo("100", age_hours=1.0)
        push = MagicMock()

        sent = main_watch.run_watch(
            settings, repository=repo, market=_market("90"), push_notifier=push
        )

        assert sent == 1
        push.send_report.assert_called_once()
        repo.save_shock_alert.assert_called_once()

    def test_watch_never_writes_a_price_snapshot(self, settings) -> None:
        # Regresja krytyczna: nadpisanie snapshotu co godzinę zamknęłoby bramkę
        # volatility Fast Loopa — delta liczyłaby się względem ceny sprzed godziny.
        repo = _repo("100", age_hours=1.0)

        main_watch.run_watch(
            settings, repository=repo, market=_market("90"), push_notifier=MagicMock()
        )

        repo.save_price_snapshot.assert_not_called()

    def test_stale_snapshot_suppresses_the_alert(self, settings) -> None:
        repo = _repo("100", age_hours=48.0)
        push = MagicMock()

        assert main_watch.run_watch(
            settings, repository=repo, market=_market("90"), push_notifier=push
        ) == 0
        push.send_report.assert_not_called()

    def test_small_move_is_not_a_shock(self, settings) -> None:
        repo = _repo("100", age_hours=1.0)

        assert main_watch.run_watch(
            settings, repository=repo, market=_market("97"), push_notifier=MagicMock()
        ) == 0

    def test_debounce_blocks_a_second_alert_the_same_day(self, settings) -> None:
        # Zegar zamrożony przez szew `now=`: `run_watch` liczy dzień alertu
        # w UTC, więc klucz debounce'u musi pochodzić z TEJ SAMEJ chwili.
        # Wcześniej klucz brał się z `date.today()` (data LOKALNA), przez co test
        # przechodził tylko wtedy, gdy strefa maszyny akurat zgadzała się z UTC —
        # w CEST padał między 00:00 a 02:00, a w Australia/Sydney przez
        # większość doby. Godzina 23:30 UTC jest dobrana celowo: w strefach na
        # wschód od Greenwich data lokalna jest wtedy o dzień do przodu, więc
        # gdyby ktoś wrócił do daty lokalnej, test natychmiast zapali się z powrotem.
        moment = datetime(2026, 5, 14, 23, 30, tzinfo=UTC)
        repo = _repo("100", age_hours=1.0, now=moment)
        repo.get_sent_shock_alerts.return_value = {("AAPL", moment.date())}
        push = MagicMock()

        assert main_watch.run_watch(
            settings,
            repository=repo,
            market=_market("90"),
            push_notifier=push,
            now=moment,
        ) == 0
        push.send_report.assert_not_called()

    def test_per_symbol_failure_does_not_break_the_run(self, settings) -> None:
        settings.symbols = ["AAPL", "MSFT"]
        repo = _repo("100", age_hours=1.0)
        market = MagicMock()
        market.get_current_price.side_effect = [RuntimeError("429"), Money(Decimal("90"))]

        assert main_watch.run_watch(
            settings, repository=repo, market=market, push_notifier=MagicMock()
        ) == 1

    def test_flag_off_short_circuits(self, settings) -> None:
        settings.shock_alerts_enabled = False
        repo = _repo("100", age_hours=1.0)

        assert main_watch.run_watch(
            settings, repository=repo, market=_market("90"), push_notifier=MagicMock()
        ) == 0
        repo.get_last_price_snapshot.assert_not_called()
