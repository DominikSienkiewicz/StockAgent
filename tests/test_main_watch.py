"""Testy entry pointa `main_watch.py` — darmowy watch szoku poza cyklem (#11).

Kontrakt FinOps jest tu absolutny: watch NIE WOŁA żadnego płatnego portu
(LLM / Alpha Vantage / embeddingi). Kontrakt poprawnościowy jest równie ostry:
watch NIE ZAPISUJE snapshotu ceny — nadpisywanie go co godzinę zamknęłoby
bramkę volatility Fast Loopa (delta liczyłaby się względem ceny sprzed godziny).
"""

from __future__ import annotations

import logging
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

    def test_stale_snapshot_skips_the_price_fetch(self, settings) -> None:
        # Stęchła referencja i tak tłumi alert, więc pobieranie ceny to czysty
        # koszt (limit Finnhuba) bez żadnej szansy na wynik.
        repo = _repo("100", age_hours=48.0)
        market = _market("90")

        main_watch.run_watch(settings, repository=repo, market=market, push_notifier=MagicMock())

        market.get_current_price.assert_not_called()

    def test_all_stale_snapshots_log_blind_warning(self, settings, caplog) -> None:
        # Gdy Fast Loop stoi, każda referencja jest stęchła i watch nie może
        # zgłosić niczego. „0 alertów" na INFO wyglądałoby jak zdrowy przebieg.
        settings.symbols = ["AAPL", "MSFT"]
        repo = _repo("100", age_hours=48.0)

        with caplog.at_level(logging.WARNING, logger="main_watch"):
            sent = main_watch.run_watch(
                settings, repository=repo, market=_market("90"), push_notifier=MagicMock()
            )

        assert sent == 0
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("ślepy" in r.getMessage() for r in warnings)

    def test_partly_stale_snapshots_do_not_log_blind_warning(self, settings, caplog) -> None:
        settings.symbols = ["AAPL", "MSFT"]
        now = datetime.now(UTC)
        repo = _repo("100", age_hours=1.0, now=now)
        repo.get_last_price_snapshot.side_effect = [
            (Money(Decimal("100")), now - timedelta(hours=48.0)),
            (Money(Decimal("100")), now - timedelta(hours=1.0)),
        ]

        with caplog.at_level(logging.WARNING, logger="main_watch"):
            main_watch.run_watch(
                settings, repository=repo, market=_market("99"),
                push_notifier=MagicMock(), now=now,
            )

        assert not any("ślepy" in r.getMessage() for r in caplog.records)

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


def test_main_wires_free_adapters_only(settings, mocker) -> None:
    # `main()` składa adaptery fabrykami z `main_agent` (import lokalny). Rename
    # buildera przeszedłby przez testy `run_watch`, a wywaliłby sam entry-point.
    repository = mocker.patch("main_agent.build_repository")
    market = mocker.patch("main_agent.build_market_port")
    push = mocker.patch("main_agent.build_push_notifier")
    paid_builders = [
        mocker.patch(f"main_agent.{name}")
        for name in (
            "build_llm_adapter",
            "build_council_adapter",
            "build_embedding_adapter",
            "build_use_case",
            "build_alpha_enrichment",
        )
    ]
    # Z kanałem push — bez niego `main()` świadomie podaje None (ALR-01).
    settings.telegram_bot_token = "tg-token"
    settings.telegram_chat_id = "42"
    scan_watch = mocker.patch(
        "main_watch.scan_watch", return_value=main_watch.WatchOutcome(sent=0, failures=0, total=1)
    )

    assert main_watch.main(settings) == 0

    scan_watch.assert_called_once_with(
        settings,
        repository=repository.return_value,
        market=market.return_value,
        push_notifier=push.return_value,
    )
    for builder in paid_builders:
        builder.assert_not_called()


class TestWatchMain:
    def test_main_returns_1_when_every_symbol_raises(self, settings, mocker) -> None:
        # NIEZ-05: padnięta baza dawała zielony run z „0 alertów”.
        settings.symbols = ["AAPL", "MSFT"]
        settings.telegram_bot_token = "tg-token"
        settings.telegram_chat_id = "42"
        repo = _repo("100", age_hours=1.0)
        repo.get_last_price_snapshot.side_effect = RuntimeError("Supabase 503")
        mocker.patch("main_agent.build_repository", return_value=repo)
        mocker.patch("main_agent.build_market_port", return_value=_market("90"))
        mocker.patch("main_agent.build_push_notifier", return_value=MagicMock())

        assert main_watch.main(settings) == 1

    def test_main_without_push_channel_persists_no_alert(self, settings, mocker) -> None:
        # ALR-01: NullNotifier „wysyłał” alert, a zapis blokował debounce na dobę.
        moment = datetime.now(UTC)
        repo = _repo("100", age_hours=1.0, now=moment)
        mocker.patch("main_agent.build_repository", return_value=repo)
        mocker.patch("main_agent.build_market_port", return_value=_market("90"))
        build_push = mocker.patch("main_agent.build_push_notifier")

        assert main_watch.main(settings) == 0

        repo.save_shock_alert.assert_not_called()
        build_push.assert_not_called()

    def test_scan_without_push_channel_sends_and_saves_nothing(self, settings) -> None:
        moment = datetime.now(UTC)
        repo = _repo("100", age_hours=1.0, now=moment)

        outcome = main_watch.scan_watch(
            settings, repository=repo, market=_market("90"), push_notifier=None, now=moment
        )

        assert outcome.sent == 0 and outcome.failures == 0
        repo.save_shock_alert.assert_not_called()


def test_alert_carries_the_detection_time_in_the_display_timezone(settings) -> None:
    # MAIL-14: push szoku mówi, KIEDY go wykryto — w czasie lokalnym, nie UTC.
    moment = datetime(2026, 10, 16, 5, 30, tzinfo=UTC)
    settings.display_timezone = "Europe/Warsaw"
    repo = _repo("100", age_hours=1.0, now=moment)
    push = MagicMock()

    main_watch.run_watch(
        settings, repository=repo, market=_market("90"), push_notifier=push, now=moment
    )

    body = push.send_report.call_args.kwargs["plain_text"]
    assert "pt 16.10 07:30" in body
    assert "05:30" not in body


def test_alert_writes_the_move_with_a_polish_decimal_comma(settings) -> None:
    # MAIL-58 etap 1b: push szoku to temat i treść dla czytelnika — „-5,2%”, nie „-5.2%”.
    moment = datetime(2026, 10, 16, 5, 30, tzinfo=UTC)
    repo = _repo("100", age_hours=1.0, now=moment)
    push = MagicMock()

    main_watch.run_watch(
        settings, repository=repo, market=_market("94.8"), push_notifier=push, now=moment
    )

    sent = push.send_report.call_args.kwargs
    assert sent["subject"] == "📉 Szok rynkowy: AAPL -5,2%"
    assert sent["plain_text"].startswith("📉 AAPL -5,2% od ostatniego snapshotu ceny")


def test_watch_starts_with_stubbed_paid_ports_and_database_writes(settings, mocker) -> None:
    # FIN-35: odmowa startu dotyczy tylko jobów z płatnymi portami — watch żadnego
    # nie woła, więc kombinacja zapis + atrapy go nie zatrzymuje.
    settings.db_writes_enabled = True
    settings.paid_calls_enabled = False
    mocker.patch("main_agent.build_repository")
    mocker.patch("main_agent.build_market_port")
    mocker.patch("main_agent.build_push_notifier")
    scan_watch = mocker.patch(
        "main_watch.scan_watch", return_value=main_watch.WatchOutcome(sent=0, failures=0, total=1)
    )

    assert main_watch.main(settings) == 0
    scan_watch.assert_called_once()
