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
from src.application.ports import PreviousVerdict, RepositoryPort
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


def test_main_builds_settings_via_from_env(settings, mocker) -> None:
    # ARCH-23: ten sam punkt wejścia konfiguracji co main_agent (`Settings.from_env`),
    # bez gołego `Settings()` z ignorem mypy.
    settings_cls = mocker.patch("main_watch.Settings")
    settings_cls.from_env.return_value = settings
    mocker.patch("main_watch.configure_logging")
    mocker.patch("main_agent.run_with_heartbeat", return_value=0)

    assert main_watch.main() == 0

    settings_cls.from_env.assert_called_once_with()


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


def test_main_hands_the_nbp_port_to_the_scan_only_with_nbp_enabled(settings, mocker) -> None:
    # ALR-08: udział pozycji w złotych potrzebuje kursów NBP; bez `nbp_enabled` watch
    # nie buduje portu kursów, a skan idzie jak przed ALR-08.
    mocker.patch("main_agent.build_repository")
    mocker.patch("main_agent.build_market_port")
    mocker.patch("main_agent.build_push_notifier")
    fx = mocker.patch("main_agent.build_fx_port")
    settings.telegram_bot_token = "tg-token"
    settings.telegram_chat_id = "42"
    scan_watch = mocker.patch(
        "main_watch.scan_watch", return_value=main_watch.WatchOutcome(sent=0, failures=0, total=1)
    )

    settings.nbp_enabled = False
    main_watch.main(settings)
    assert "fx_port" not in scan_watch.call_args.kwargs
    fx.assert_not_called()

    settings.nbp_enabled = True
    main_watch.main(settings)
    assert scan_watch.call_args.kwargs["fx_port"] is fx.return_value


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


class TestNyseClosedDayWatch:
    """FIN-34: w dzień zamknięcia NYSE ceny akcji stoją, więc watch (za tą samą
    flagą co Fast Loop) pomija akcje z tym samym logiem; krypto sprawdza dalej."""

    _CHRISTMAS = datetime(2026, 12, 25, 5, 30, tzinfo=UTC)

    def _scan(self, settings, now: datetime, caplog) -> tuple[MagicMock, object]:
        settings.symbols = ["AAPL"]
        settings.crypto_symbols = ["BTC"]
        repo = _repo("100", age_hours=1.0, now=now)
        with caplog.at_level(logging.INFO, logger="main_watch"):
            outcome = main_watch.scan_watch(
                settings, repository=repo, market=_market("100"), push_notifier=MagicMock(),
                now=now,
            )
        return repo, outcome

    def test_holiday_checks_only_crypto(self, settings, caplog) -> None:
        settings.nyse_closure_skip_enabled = True

        repo, outcome = self._scan(settings, self._CHRISTMAS, caplog)

        checked = [c.args[0] for c in repo.get_last_price_snapshot.call_args_list]
        assert checked == ["BTC"]
        assert outcome.total == 1
        assert any(
            "święto NYSE (Boże Narodzenie)" in rec.message and "BTC" in rec.message
            for rec in caplog.records
        ), [rec.message for rec in caplog.records]

    def test_trading_day_checks_stocks_too(self, settings, caplog) -> None:
        settings.nyse_closure_skip_enabled = True

        repo, _ = self._scan(settings, datetime(2026, 12, 23, 5, 30, tzinfo=UTC), caplog)

        checked = [c.args[0] for c in repo.get_last_price_snapshot.call_args_list]
        assert checked == ["AAPL", "BTC"]

    def test_flag_off_checks_stocks_on_a_holiday(self, settings, caplog) -> None:
        repo, _ = self._scan(settings, self._CHRISTMAS, caplog)

        checked = [c.args[0] for c in repo.get_last_price_snapshot.call_args_list]
        assert checked == ["AAPL", "BTC"]


class TestWatchHeartbeat:
    """NIEZ-13: zewnętrzny watchdog dostaje ping na końcu przebiegu watcha."""

    def test_main_pings_heartbeat_with_ok_flag(self, settings, mocker) -> None:
        from src.application.ports import HeartbeatPort

        heartbeat = MagicMock(spec=HeartbeatPort)
        mocker.patch("main_agent.build_heartbeat", return_value=heartbeat)
        mocker.patch("main_agent.build_repository")
        mocker.patch("main_agent.build_market_port")
        mocker.patch(
            "main_watch.scan_watch",
            return_value=main_watch.WatchOutcome(sent=0, failures=0, total=1),
        )

        assert main_watch.main(settings) == 0

        heartbeat.ping.assert_called_once_with(ok=True)

    def test_run_where_every_symbol_failed_pings_a_failure(self, settings, mocker) -> None:
        from src.application.ports import HeartbeatPort

        heartbeat = MagicMock(spec=HeartbeatPort)
        mocker.patch("main_agent.build_heartbeat", return_value=heartbeat)
        mocker.patch("main_agent.build_repository")
        mocker.patch("main_agent.build_market_port")
        mocker.patch(
            "main_watch.scan_watch",
            return_value=main_watch.WatchOutcome(sent=0, failures=2, total=2),
        )

        assert main_watch.main(settings) == 1

        heartbeat.ping.assert_called_once_with(ok=False)

    def test_failed_ping_does_not_change_the_exit_code(self, settings, mocker, caplog) -> None:
        from src.application.ports import HeartbeatPort

        heartbeat = MagicMock(spec=HeartbeatPort)
        heartbeat.ping.side_effect = RuntimeError("heartbeat ping failed (ConnectTimeout)")
        mocker.patch("main_agent.build_heartbeat", return_value=heartbeat)
        mocker.patch("main_agent.build_repository")
        mocker.patch("main_agent.build_market_port")
        mocker.patch(
            "main_watch.scan_watch",
            return_value=main_watch.WatchOutcome(sent=0, failures=0, total=1),
        )

        with caplog.at_level(logging.WARNING):
            assert main_watch.main(settings) == 0

        assert "heartbeat" in caplog.text.lower()


def _daily(fraction: str, as_of: datetime) -> MagicMock:
    """Port dziennej zmiany (ALR-02) z jednym odczytem dla każdego symbolu."""
    from src.application.ports import DailyChangePort
    from src.domain.shock import DailyChange

    port = MagicMock(spec=DailyChangePort)
    port.get_daily_change.return_value = DailyChange(fraction=Decimal(fraction), as_of=as_of)
    return port


def _no_snapshot_repo() -> MagicMock:
    repo = MagicMock(spec=RepositoryPort)
    repo.get_last_price_snapshot.return_value = None
    repo.get_sent_shock_alerts.return_value = set()
    repo.get_shock_alert_times.return_value = {}
    return repo


class TestDailyChangeWatch:
    """ALR-02: dzienna zmiana z darmowego /quote zamiast snapshotów Fast Loopa —
    watch alarmuje 24/7 bez wznawiania płatnej pętli."""

    _FRIDAY_SESSION = datetime(2026, 10, 16, 15, 7, tzinfo=UTC)

    def test_daily_change_drives_alert_without_snapshot(self, settings) -> None:
        repo = _no_snapshot_repo()
        push = MagicMock()

        sent = main_watch.run_watch(
            settings,
            repository=repo,
            market=_market("90"),
            push_notifier=push,
            now=self._FRIDAY_SESSION,
            daily_change=_daily("-0.09", self._FRIDAY_SESSION - timedelta(minutes=5)),
        )

        assert sent == 1
        push.send_report.assert_called_once()
        repo.save_shock_alert.assert_called_once()
        repo.save_price_snapshot.assert_not_called()

    def test_saturday_reading_of_friday_move_does_not_alert(self, settings) -> None:
        # `dp` w sobotę to piątkowy ruch — bez guardu na `t` dublowałby piątkowy alert.
        saturday = datetime(2026, 10, 17, 9, 7, tzinfo=UTC)
        friday_close = datetime(2026, 10, 16, 20, 0, tzinfo=UTC)
        push = MagicMock()

        sent = main_watch.run_watch(
            settings,
            repository=_no_snapshot_repo(),
            market=_market("90"),
            push_notifier=push,
            now=saturday,
            daily_change=_daily("-0.09", friday_close),
        )

        assert sent == 0
        push.send_report.assert_not_called()

    def test_fresh_reading_wins_over_snapshot_with_a_single_alert(self, settings) -> None:
        # Oba źródła widzą szok: jeden alert, a snapshot i cena bieżąca nie są czytane.
        now = self._FRIDAY_SESSION
        repo = _repo("100", age_hours=1.0, now=now)
        market = _market("90")
        push = MagicMock()

        sent = main_watch.run_watch(
            settings,
            repository=repo,
            market=market,
            push_notifier=push,
            now=now,
            daily_change=_daily("-0.09", now),
        )

        assert sent == 1
        push.send_report.assert_called_once()
        repo.get_last_price_snapshot.assert_not_called()
        market.get_current_price.assert_not_called()

    def test_fresh_calm_reading_is_final(self, settings) -> None:
        # Świeży odczyt rozstrzyga — snapshot nie dokłada drugiej opinii o tym samym symbolu.
        now = self._FRIDAY_SESSION
        repo = _repo("100", age_hours=1.0, now=now)

        sent = main_watch.run_watch(
            settings,
            repository=repo,
            market=_market("90"),
            push_notifier=MagicMock(),
            now=now,
            daily_change=_daily("-0.01", now),
        )

        assert sent == 0
        repo.get_last_price_snapshot.assert_not_called()

    def test_stale_reading_falls_back_to_the_snapshot(self, settings) -> None:
        saturday = datetime(2026, 10, 17, 9, 7, tzinfo=UTC)
        repo = _repo("100", age_hours=1.0, now=saturday)
        push = MagicMock()

        sent = main_watch.run_watch(
            settings,
            repository=repo,
            market=_market("90"),
            push_notifier=push,
            now=saturday,
            daily_change=_daily("-0.09", datetime(2026, 10, 16, 20, 0, tzinfo=UTC)),
        )

        assert sent == 1
        assert push.send_report.call_args.kwargs["plain_text"].startswith(
            "📉 AAPL -10,0% od ostatniego snapshotu ceny"
        )

    def test_failing_reading_falls_back_to_the_snapshot(self, settings, caplog) -> None:
        now = self._FRIDAY_SESSION
        repo = _repo("100", age_hours=1.0, now=now)
        daily = _daily("-0.09", now)
        daily.get_daily_change.side_effect = RuntimeError("429")

        with caplog.at_level(logging.WARNING, logger="main_watch"):
            outcome = main_watch.scan_watch(
                settings,
                repository=repo,
                market=_market("90"),
                push_notifier=MagicMock(),
                now=now,
                daily_change=daily,
            )

        assert outcome.sent == 1 and outcome.failures == 0
        assert any("AAPL" in r.getMessage() for r in caplog.records)

    def test_stock_alert_names_the_previous_close(self, settings) -> None:
        now = self._FRIDAY_SESSION
        push = MagicMock()

        main_watch.run_watch(
            settings,
            repository=_no_snapshot_repo(),
            market=_market("90"),
            push_notifier=push,
            now=now,
            daily_change=_daily("-0.09", now),
        )

        assert push.send_report.call_args.kwargs["plain_text"].startswith(
            "📉 AAPL -9,0% od poprzedniego zamknięcia"
        )

    def test_crypto_alert_names_the_24h_window(self, settings) -> None:
        settings.symbols = []
        settings.crypto_symbols = ["BTC"]
        now = self._FRIDAY_SESSION
        push = MagicMock()

        main_watch.run_watch(
            settings,
            repository=_no_snapshot_repo(),
            market=_market("90"),
            push_notifier=push,
            now=now,
            daily_change=_daily("-0.08", now),
        )

        assert push.send_report.call_args.kwargs["plain_text"].startswith(
            "📉 BTC -8,0% w ciągu ostatnich 24 h"
        )

    def test_fresh_readings_keep_the_watch_from_being_blind(self, settings, caplog) -> None:
        # Fast Loop stoi (wszystkie snapshoty stęchłe), ale dzienna zmiana jest świeża.
        now = self._FRIDAY_SESSION
        repo = _repo("100", age_hours=48.0, now=now)

        with caplog.at_level(logging.WARNING, logger="main_watch"):
            outcome = main_watch.scan_watch(
                settings,
                repository=repo,
                market=_market("90"),
                push_notifier=MagicMock(),
                now=now,
                daily_change=_daily("-0.01", now),
            )

        assert outcome.checked == 1 and outcome.stale == 0
        assert not any("ślepy" in r.getMessage() for r in caplog.records)

    def test_debounce_also_covers_the_daily_change(self, settings) -> None:
        now = self._FRIDAY_SESSION
        repo = _no_snapshot_repo()
        repo.get_sent_shock_alerts.return_value = {("AAPL", now.date())}
        daily = _daily("-0.09", now)

        assert main_watch.run_watch(
            settings,
            repository=repo,
            market=_market("90"),
            push_notifier=MagicMock(),
            now=now,
            daily_change=daily,
        ) == 0
        daily.get_daily_change.assert_not_called()


def test_main_feeds_the_free_router_as_the_daily_change_source(settings, mocker) -> None:
    # ALR-02: router darmowych cen (Finnhub /quote, CoinGecko) jest też źródłem
    # dziennej zmiany; płatne fabryki dalej nietknięte.
    from src.infrastructure.adapters.routing_market_data import RoutingMarketDataPort

    router = MagicMock(spec=RoutingMarketDataPort)
    repository = mocker.patch("main_agent.build_repository")
    mocker.patch("main_agent.build_market_port", return_value=router)
    push = mocker.patch("main_agent.build_push_notifier")
    paid_builders = [
        mocker.patch(f"main_agent.{name}")
        for name in ("build_llm_adapter", "build_council_adapter", "build_embedding_adapter")
    ]
    settings.telegram_bot_token = "tg-token"
    settings.telegram_chat_id = "42"
    scan_watch = mocker.patch(
        "main_watch.scan_watch", return_value=main_watch.WatchOutcome(sent=0, failures=0, total=1)
    )

    assert main_watch.main(settings) == 0

    scan_watch.assert_called_once_with(
        settings,
        repository=repository.return_value,
        market=router,
        push_notifier=push.return_value,
        daily_change=router,
    )
    for builder in paid_builders:
        builder.assert_not_called()


class TestCryptoRollingWindowCooldown:
    """ALR-02, decyzja 2: zmiana 24 h CoinGecko zawiera ruch jeszcze dobę po alercie,
    więc krypto czeka, aż poprzedni alert symbolu będzie starszy niż 24 h."""

    _AFTER_MIDNIGHT = datetime(2026, 10, 17, 0, 7, tzinfo=UTC)

    def _scan(self, settings, alerted_hours_ago: float) -> tuple[int, MagicMock]:
        now = self._AFTER_MIDNIGHT
        alerted_at = now - timedelta(hours=alerted_hours_ago)
        repo = _no_snapshot_repo()
        repo.get_sent_shock_alerts.return_value = {
            (symbol, alerted_at.date()) for symbol in ("AAPL", "BTC")
        }
        repo.get_shock_alert_times.return_value = {"AAPL": alerted_at, "BTC": alerted_at}
        push = MagicMock()
        sent = main_watch.run_watch(
            settings,
            repository=repo,
            market=_market("90"),
            push_notifier=push,
            now=now,
            daily_change=_daily("-0.09", now - timedelta(minutes=5)),
        )
        return sent, push

    def test_crypto_alert_from_23_hours_ago_suppresses_the_daily_change(self, settings) -> None:
        settings.symbols = []
        settings.crypto_symbols = ["BTC"]

        sent, push = self._scan(settings, alerted_hours_ago=23)

        assert sent == 0
        push.send_report.assert_not_called()

    def test_crypto_alert_from_25_hours_ago_does_not_suppress(self, settings) -> None:
        settings.symbols = []
        settings.crypto_symbols = ["BTC"]

        sent, push = self._scan(settings, alerted_hours_ago=25)

        assert sent == 1
        assert "BTC" in push.send_report.call_args.kwargs["subject"]

    def test_stocks_ignore_the_rolling_window(self, settings) -> None:
        # `dp` akcji liczy się od poprzedniego zamknięcia — świeżość dnia UTC wystarcza.
        settings.symbols = ["AAPL"]
        settings.crypto_symbols = ["BTC"]

        sent, push = self._scan(settings, alerted_hours_ago=23)

        assert sent == 1
        assert "AAPL" in push.send_report.call_args.kwargs["subject"]


class TestQuietHoursWatch:
    """ALR-09: szok w nocy (czas odbiorcy) dociera bez dźwięku — informacja zostaje,
    pobudka o 3:00 nie. Poza oknem ciszy i z flagą off push brzmi jak dotąd."""

    _NIGHT = datetime(2026, 10, 16, 1, 30, tzinfo=UTC)  # 03:30 w Warszawie
    _DAY = datetime(2026, 10, 16, 8, 30, tzinfo=UTC)  # 10:30 w Warszawie

    def _push(self, settings, now: datetime) -> MagicMock:
        settings.display_timezone = "Europe/Warsaw"
        push = MagicMock()
        main_watch.run_watch(
            settings,
            repository=_repo("100", age_hours=1.0, now=now),
            market=_market("90"),
            push_notifier=push,
            now=now,
        )
        push.send_report.assert_called_once()
        return push

    def test_night_alert_is_delivered_silently(self, settings) -> None:
        settings.quiet_hours_enabled = True

        push = self._push(settings, self._NIGHT)

        assert push.send_report.call_args.kwargs["silent"] is True

    def test_daytime_alert_keeps_the_sound(self, settings) -> None:
        settings.quiet_hours_enabled = True

        push = self._push(settings, self._DAY)

        assert push.send_report.call_args.kwargs["silent"] is False

    def test_flag_off_never_silences(self, settings) -> None:
        settings.quiet_hours_enabled = False

        push = self._push(settings, self._NIGHT)

        assert push.send_report.call_args.kwargs["silent"] is False

    def test_window_comes_from_settings(self, settings) -> None:
        from datetime import time

        settings.quiet_hours_enabled = True
        settings.quiet_hours_start = time(1, 0)
        settings.quiet_hours_end = time(2, 0)

        push = self._push(settings, self._NIGHT)

        assert push.send_report.call_args.kwargs["silent"] is False


def _previous_verdict(recommendation: str, strength: float, at: datetime) -> PreviousVerdict:
    from src.domain.council import CouncilVerdict

    verdict = CouncilVerdict(
        final_recommendation=recommendation,  # type: ignore[arg-type]
        consensus_strength=strength,
        summary="Rada widzi popyt na AI.",
        dissenting_views=(),
        investor_opinions=(),
    )
    return PreviousVerdict(verdict=verdict, sentiment_score=0.3, timestamp=at)


class TestAlertContext:
    """ALR-08: push szoku mówi, czy ruch dotyczy Twojej pozycji i co rada
    sądziła ostatnio — same darmowe odczyty, a każdy z nich jest best-effort."""

    _NOW = datetime(2026, 10, 16, 15, 7, tzinfo=UTC)  # pt 17:07 w Warszawie

    def _repo(self) -> MagicMock:
        repo = _repo("100", age_hours=1.0, now=self._NOW)
        repo.get_previous_verdict.return_value = None
        repo.get_latest_prediction.return_value = None
        return repo

    @staticmethod
    def _market(prices: dict[str, str]) -> MagicMock:
        market = MagicMock()
        market.get_current_price.side_effect = lambda symbol: Money(Decimal(prices[symbol]))
        return market

    def _body(
        self,
        settings,
        repo: MagicMock,
        *,
        portfolio: object = None,
        prices: dict[str, str] | None = None,
    ) -> str:
        settings.display_timezone = "Europe/Warsaw"
        push = MagicMock()
        sent = main_watch.run_watch(
            settings,
            repository=repo,
            market=self._market(prices or {"AAPL": "90"}),
            push_notifier=push,
            now=self._NOW,
            portfolio_loader=lambda: portfolio,
        )
        assert sent == 1
        body: str = push.send_report.call_args.kwargs["plain_text"]
        return body

    def _portfolio(self) -> object:
        from src.domain.portfolio import Portfolio, Position

        return Portfolio(
            positions=(
                Position(symbol="AAPL", quantity=Decimal("10"), purchase_price=Decimal("80")),
                Position(symbol="MSFT", quantity=Decimal("1"), purchase_price=Decimal("300")),
            ),
            as_of=self._NOW - timedelta(days=1),
        )

    def test_alert_body_includes_previous_verdict_and_position(self, settings) -> None:
        repo = self._repo()
        repo.get_previous_verdict.return_value = _previous_verdict(
            "BUY", 0.72, self._NOW - timedelta(days=1)
        )

        body = self._body(
            settings, repo, portfolio=self._portfolio(), prices={"AAPL": "90", "MSFT": "300"}
        )

        assert "Rada wczoraj: KUP, zgoda 72%" in body
        # AAPL 10 × 90 = 900 z 1200 wartości portfela; 90 / 80 − 1 = +12,5%.
        assert "Twoja pozycja: 75% portfela, +12,5% od zakupu" in body
        assert body.startswith("📉 AAPL -10,0% od ostatniego snapshotu ceny")
        assert body.rstrip().endswith("pełna analiza w najbliższym dziennym digeście.")

    def test_week_old_verdict_carries_its_age(self, settings) -> None:
        repo = self._repo()
        repo.get_previous_verdict.return_value = _previous_verdict(
            "SELL", 0.55, self._NOW - timedelta(days=8)
        )

        body = self._body(settings, repo)

        assert "Rada 8 dni temu: SPRZEDAJ, zgoda 55%" in body

    def test_alert_quotes_the_last_prediction_reasoning(self, settings) -> None:
        from src.application.ports import LatestPrediction

        repo = self._repo()
        repo.get_latest_prediction.return_value = LatestPrediction(
            price=Money(Decimal("101")),
            timestamp=self._NOW,
            reasoning="Słabsze prognozy marż. " + "x" * 400,
        )

        body = self._body(settings, repo)

        line = next(row for row in body.splitlines() if "Ostatnia predykcja" in row)
        assert line.startswith("🧠 Ostatnia predykcja (dziś): Słabsze prognozy marż.")
        assert line.endswith("…")
        assert len(line) < 320

    def test_symbol_outside_the_portfolio_gets_no_position_line(self, settings) -> None:
        from src.domain.portfolio import Portfolio, Position

        portfolio = Portfolio(
            positions=(
                Position(symbol="MSFT", quantity=Decimal("1"), purchase_price=Decimal("1")),
            ),
            as_of=self._NOW,
        )

        body = self._body(settings, self._repo(), portfolio=portfolio)

        assert "Twoja pozycja" not in body

    def test_stale_portfolio_is_marked(self, settings) -> None:
        from src.domain.portfolio import Portfolio, Position

        portfolio = Portfolio(
            positions=(
                Position(symbol="AAPL", quantity=Decimal("1"), purchase_price=Decimal("90")),
            ),
            as_of=self._NOW - timedelta(days=40),
        )

        body = self._body(settings, self._repo(), portfolio=portfolio)

        assert "Twoja pozycja: 100% portfela, +0,0% od zakupu" in body
        assert "stan portfela z 06.09.2026 — może być nieaktualny" in body

    def test_failing_context_reads_keep_the_alert(self, settings) -> None:
        repo = self._repo()
        repo.get_previous_verdict.side_effect = RuntimeError("Supabase 503")
        repo.get_latest_prediction.side_effect = RuntimeError("Supabase 503")

        def broken_portfolio() -> object:
            raise RuntimeError("positions down")

        settings.display_timezone = "Europe/Warsaw"
        push = MagicMock()
        outcome = main_watch.scan_watch(
            settings,
            repository=repo,
            market=self._market({"AAPL": "90"}),
            push_notifier=push,
            now=self._NOW,
            portfolio_loader=broken_portfolio,
        )

        assert outcome.sent == 1 and outcome.failures == 0
        body = push.send_report.call_args.kwargs["plain_text"]
        assert "Rada" not in body and "Twoja pozycja" not in body

    def test_context_never_writes_a_price_snapshot(self, settings) -> None:
        repo = self._repo()

        self._body(settings, repo, portfolio=self._portfolio(), prices={"AAPL": "90", "MSFT": "1"})

        repo.save_price_snapshot.assert_not_called()

    def test_portfolio_flag_off_reads_no_positions(self, settings, mocker) -> None:
        # Domyślny loader to fabryka Fast Loopa: bez `portfolio_positions_enabled`
        # nie dotyka tabeli `positions`.
        positions_port = mocker.patch("main_agent.build_positions_port")
        settings.portfolio_positions_enabled = False
        push = MagicMock()

        main_watch.run_watch(
            settings,
            repository=self._repo(),
            market=self._market({"AAPL": "90"}),
            push_notifier=push,
            now=self._NOW,
        )

        push.send_report.assert_called_once()
        positions_port.assert_not_called()


class TestPositionShareInZlotys:
    """ALR-08: udział pozycji w alercie szoku liczony w złotych (kursy NBP). Portfel
    z pozycją w USD i w PLN dawał wagę z kwot w różnych walutach; brak kursu
    pomija linię udziału, a odczyt kursu nigdy nie zatrzymuje alertu."""

    _NOW = datetime(2026, 10, 16, 15, 7, tzinfo=UTC)

    @staticmethod
    def _fx(**rates: str) -> MagicMock:
        from src.application.ports import FxRatePort
        from src.domain.fx import FxRate

        port = MagicMock(spec=FxRatePort)
        port.get_rate.side_effect = lambda currency, day: (
            FxRate(currency, day, day, Decimal(rates[currency])) if currency in rates else None
        )
        return port

    @staticmethod
    def _mixed_portfolio(as_of: datetime):
        from src.domain.portfolio import Portfolio, Position

        return Portfolio(
            positions=(
                Position(symbol="AAPL", quantity=Decimal("10"), purchase_price=Decimal("80")),
                Position(
                    symbol="PKN", quantity=Decimal("100"), purchase_price=Decimal("50"),
                    currency="PLN",
                ),
            ),
            as_of=as_of,
        )

    def _body(self, settings, fx_port: MagicMock | None) -> str:
        settings.display_timezone = "Europe/Warsaw"
        repo = _repo("100", age_hours=1.0, now=self._NOW)
        repo.get_previous_verdict.return_value = None
        repo.get_latest_prediction.return_value = None
        market = MagicMock()
        market.get_current_price.side_effect = lambda symbol: Money(
            Decimal({"AAPL": "90", "PKN": "50"}[symbol])
        )
        push = MagicMock()

        main_watch.scan_watch(
            settings,
            repository=repo,
            market=market,
            push_notifier=push,
            now=self._NOW,
            portfolio_loader=lambda: self._mixed_portfolio(self._NOW),
            fx_port=fx_port,
        )

        body: str = push.send_report.call_args.kwargs["plain_text"]
        return body

    def test_share_of_the_usd_position_is_converted_with_the_nbp_rate(self, settings) -> None:
        body = self._body(settings, self._fx(USD="4.00"))

        # 10 × 90 USD × 4,00 = 3 600 zł z 8 600 zł (41,9%); z kwot natywnych: 900 / 5 900.
        assert "Twoja pozycja: 42% portfela, +12,5% od zakupu" in body

    def test_without_a_nbp_rate_the_share_is_left_out_but_the_alert_goes(self, settings) -> None:
        body = self._body(settings, self._fx())

        assert "Twoja pozycja: +12,5% od zakupu" in body
        assert "% portfela" not in body

    def test_without_an_fx_port_the_share_is_left_out(self, settings) -> None:
        body = self._body(settings, None)

        assert "Twoja pozycja: +12,5% od zakupu" in body

    def test_a_failing_rate_read_never_stops_the_alert(self, settings) -> None:
        fx = self._fx(USD="4.00")
        fx.get_rate.side_effect = RuntimeError("NBP 503")

        body = self._body(settings, fx)

        assert "Twoja pozycja: +12,5% od zakupu" in body

    def test_only_todays_rates_are_requested(self, settings) -> None:
        fx = self._fx(USD="4.00")

        self._body(settings, fx)

        assert {call.args[0] for call in fx.get_rate.call_args_list} == {"USD"}
        assert {call.args[1] for call in fx.get_rate.call_args_list} == {self._NOW.date()}


# ---------------------------------------------------------------------------
# ALR-11: alerty poziomów cenowych ustawiane przez użytkownika
# ---------------------------------------------------------------------------


class TestPriceLevelAlerts:
    """Push przy PRZEKROCZENIU poziomu od `last_seen_price`, wyłączenie po trafieniu,
    lista startowa z config.toml dopisywana przez watch. Kontrakty watcha bez zmian:
    zero płatnych portów, zero `save_price_snapshot`."""

    _NOW = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)

    @staticmethod
    def _settings(settings: Settings, **overrides: object) -> Settings:
        return settings.model_copy(
            update={"price_level_alerts_enabled": True, "quiet_hours_enabled": False, **overrides}
        )

    @staticmethod
    def _alert(last_seen: str | None, *, direction: str = "ABOVE", note: str | None = None):
        from src.domain.price_levels import LevelDirection, PriceLevel, PriceLevelAlert

        return PriceLevelAlert(
            id="7",
            level=PriceLevel(
                symbol="NVDA",
                level=Decimal("150"),
                direction=LevelDirection(direction),
                note=note,
            ),
            last_seen_price=Decimal(last_seen) if last_seen is not None else None,
        )

    def _scan(self, settings: Settings, repo: MagicMock, price: str, push: MagicMock | None):
        return main_watch.scan_price_levels(
            settings, repository=repo, market=_market(price), push_notifier=push, now=self._NOW
        )

    @staticmethod
    def _repo(*alerts: object) -> MagicMock:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_active_price_levels.return_value = list(alerts)
        return repo

    def test_crossing_sends_the_push_with_the_note_and_disables_the_alert(
        self, settings
    ) -> None:
        repo = self._repo(self._alert("148", note="realizacja części zysku"))
        push = MagicMock()

        sent = self._scan(self._settings(settings), repo, "151.2", push)

        assert sent == 1
        kwargs = push.send_report.call_args.kwargs
        assert kwargs["subject"] == "🎯 NVDA przebiła 150"
        body = kwargs["plain_text"]
        assert body.startswith("🎯 NVDA przebiła 150 (Twoja notatka: realizacja części zysku).")
        assert "151,20" in body
        repo.mark_price_level_triggered.assert_called_once_with("7", Decimal("151.2"), self._NOW)
        repo.record_price_level_reading.assert_not_called()

    def test_falling_through_a_level_names_the_drop(self, settings) -> None:
        repo = self._repo(self._alert("152", direction="BELOW"))
        push = MagicMock()

        self._scan(self._settings(settings), repo, "149.5", push)

        assert push.send_report.call_args.kwargs["subject"] == "🎯 NVDA spadła do 150"

    def test_price_on_the_same_side_only_records_the_reading(self, settings) -> None:
        repo = self._repo(self._alert("152"))
        push = MagicMock()

        assert self._scan(self._settings(settings), repo, "155", push) == 0

        push.send_report.assert_not_called()
        repo.record_price_level_reading.assert_called_once_with("7", Decimal("155"))
        repo.mark_price_level_triggered.assert_not_called()

    def test_first_reading_is_the_reference_not_a_hit(self, settings) -> None:
        repo = self._repo(self._alert(None))
        push = MagicMock()

        assert self._scan(self._settings(settings), repo, "151", push) == 0

        repo.record_price_level_reading.assert_called_once_with("7", Decimal("151"))

    def test_config_list_is_added_to_the_table(self, settings) -> None:
        from src.config import PriceAlertSettings
        from src.domain.price_levels import LevelDirection, PriceLevel

        seeds = [PriceAlertSettings(symbol="nvda", level=Decimal("150"), direction="ABOVE")]
        repo = self._repo()

        self._scan(self._settings(settings, price_alerts=seeds), repo, "151", MagicMock())

        repo.add_price_levels.assert_called_once_with(
            (PriceLevel(symbol="NVDA", level=Decimal("150"), direction=LevelDirection.ABOVE),)
        )

    def test_flag_off_touches_nothing(self, settings) -> None:
        repo = self._repo(self._alert("148"))
        market = _market("151")

        sent = main_watch.scan_price_levels(
            settings.model_copy(update={"price_level_alerts_enabled": False}),
            repository=repo,
            market=market,
            push_notifier=MagicMock(),
            now=self._NOW,
        )

        assert sent == 0
        assert repo.method_calls == []
        market.get_current_price.assert_not_called()

    def test_without_a_push_channel_the_hit_waits_untouched(self, settings) -> None:
        repo = self._repo(self._alert("148"))

        assert self._scan(self._settings(settings), repo, "151", None) == 0

        repo.mark_price_level_triggered.assert_not_called()
        repo.record_price_level_reading.assert_not_called()

    def test_missing_table_skips_the_scan(self, settings, caplog) -> None:
        repo = self._repo()
        repo.get_active_price_levels.side_effect = RuntimeError("PGRST205")

        with caplog.at_level(logging.ERROR):
            assert self._scan(self._settings(settings), repo, "151", MagicMock()) == 0

        assert "057" in caplog.text

    def test_level_scan_never_writes_a_price_snapshot_or_calls_paid_ports(
        self, settings, mocker
    ) -> None:
        paid_builders = [
            mocker.patch(f"main_agent.{name}")
            for name in ("build_llm_adapter", "build_council_adapter", "build_embedding_adapter")
        ]
        repo = self._repo(self._alert("148"))

        self._scan(self._settings(settings), repo, "151", MagicMock())

        repo.save_price_snapshot.assert_not_called()
        for builder in paid_builders:
            builder.assert_not_called()

    def test_main_runs_the_level_scan_on_the_watch_adapters(self, settings, mocker) -> None:
        repository = mocker.patch("main_agent.build_repository")
        market = mocker.patch("main_agent.build_market_port")
        push = mocker.patch("main_agent.build_push_notifier")
        settings.telegram_bot_token = "tg-token"
        settings.telegram_chat_id = "42"
        mocker.patch(
            "main_watch.scan_watch",
            return_value=main_watch.WatchOutcome(sent=0, failures=0, total=1),
        )
        levels = mocker.patch("main_watch.scan_price_levels", return_value=0)

        assert main_watch.main(settings) == 0

        levels.assert_called_once_with(
            settings,
            repository=repository.return_value,
            market=market.return_value,
            push_notifier=push.return_value,
        )


class TestPortfolioShockWatch:
    """ALR-07: skoordynowany spadek wielu pozycji to szok portfela, choć żaden symbol
    nie przekracza własnego progu. Same darmowe odczyty; watch nadal nie zapisuje
    snapshotu ceny i nie woła płatnych portów."""

    _NOW = datetime(2026, 10, 27, 14, 0, tzinfo=UTC)
    _SYMBOLS = ("NVDA", "AMD", "AVGO", "TSM")

    def _settings(self, settings: Settings, **overrides: object) -> Settings:
        return settings.model_copy(
            update={
                "symbols": list(self._SYMBOLS),
                "portfolio_shock_enabled": True,
                "portfolio_shock_threshold": Decimal("0.03"),
                "quiet_hours_enabled": False,
                **overrides,
            }
        )

    def _portfolio(self, *symbols: str):
        from src.domain.portfolio import Portfolio, Position

        return Portfolio(
            positions=tuple(
                Position(symbol=symbol, quantity=Decimal("10"), purchase_price=Decimal("90"))
                for symbol in (symbols or self._SYMBOLS)
            ),
            as_of=self._NOW - timedelta(days=1),
        )

    def _repo(self) -> MagicMock:
        repo = MagicMock(spec=RepositoryPort)
        stamp = self._NOW - timedelta(hours=1)
        repo.get_last_price_snapshot.return_value = (Money(Decimal("100")), stamp)
        repo.get_sent_shock_alerts.return_value = set()
        repo.get_previous_verdict.return_value = None
        repo.get_latest_prediction.return_value = None
        return repo

    @staticmethod
    def _market(price: str = "96.5", **overrides: str) -> MagicMock:
        prices = overrides
        market = MagicMock()
        market.get_current_price.side_effect = lambda symbol: Money(
            Decimal(prices.get(symbol, price))
        )
        return market

    def _run(
        self,
        settings: Settings,
        repo: MagicMock,
        *,
        push: MagicMock | None,
        portfolio: object = "default",
        market: MagicMock | None = None,
    ) -> int:
        loaded = self._portfolio() if portfolio == "default" else portfolio
        return main_watch.scan_watch(
            settings,
            repository=repo,
            market=market or self._market(),
            push_notifier=push,
            now=self._NOW,
            portfolio_loader=lambda: loaded,  # type: ignore[arg-type, return-value]
        ).sent

    def test_correlated_small_moves_send_one_portfolio_alert(self, settings) -> None:
        repo = self._repo()
        push = MagicMock()

        sent = self._run(self._settings(settings), repo, push=push)

        assert sent == 1
        push.send_report.assert_called_once()
        kwargs = push.send_report.call_args.kwargs
        assert "💼 Portfel -3,5%" in kwargs["plain_text"]
        assert kwargs["subject"] == "💼 Szok portfela: -3,5%"
        repo.save_shock_alert.assert_called_once_with(
            "__PORTFOLIO__", self._NOW.date(), pytest.approx(-0.035), "DOWN"
        )
        repo.save_price_snapshot.assert_not_called()

    def test_the_alert_names_the_biggest_movers(self, settings) -> None:
        repo = self._repo()
        push = MagicMock()
        market = self._market("97", NVDA="94")  # NVDA −6%, reszta −3%: Σ = −3,75%

        self._run(self._settings(settings), repo, push=push, market=market)

        body = push.send_report.call_args.kwargs["plain_text"]
        assert "💼 Portfel -3,8%" in body
        assert "NVDA -6,0%" in body

    def test_flag_off_leaves_the_watch_as_it_was(self, settings) -> None:
        repo = self._repo()
        push = MagicMock()
        loader = MagicMock(return_value=self._portfolio())

        sent = main_watch.scan_watch(
            self._settings(settings, portfolio_shock_enabled=False),
            repository=repo,
            market=self._market(),
            push_notifier=push,
            now=self._NOW,
            portfolio_loader=loader,
        ).sent

        assert sent == 0
        push.send_report.assert_not_called()
        loader.assert_not_called()
        repo.save_shock_alert.assert_not_called()

    def test_below_the_portfolio_threshold_stays_silent(self, settings) -> None:
        push = MagicMock()

        sent = self._run(
            self._settings(settings), self._repo(), push=push, market=self._market("98.5")
        )

        assert sent == 0
        push.send_report.assert_not_called()

    def test_debounce_one_portfolio_alert_a_day(self, settings) -> None:
        repo = self._repo()
        repo.get_sent_shock_alerts.return_value = {("__PORTFOLIO__", self._NOW.date())}
        push = MagicMock()

        assert self._run(self._settings(settings), repo, push=push) == 0
        push.send_report.assert_not_called()

    def test_a_symbol_alerted_earlier_today_still_counts_toward_the_portfolio(
        self, settings
    ) -> None:
        repo = self._repo()
        repo.get_sent_shock_alerts.return_value = {("NVDA", self._NOW.date())}
        push = MagicMock()
        market = self._market("97", NVDA="94")

        sent = self._run(self._settings(settings), repo, push=push, market=market)

        # Alert NVDA poszedł wcześniej; portfel (Σ = −3,75%) dostaje własny.
        assert sent == 1
        assert "NVDA -6,0%" in push.send_report.call_args.kwargs["plain_text"]

    def test_a_symbol_shock_and_the_portfolio_shock_are_independent(self, settings) -> None:
        repo = self._repo()
        push = MagicMock()
        market = self._market("97", NVDA="90")  # NVDA −10% (szok symbolu), Σ ≈ −4,75%

        sent = self._run(self._settings(settings), repo, push=push, market=market)

        assert sent == 2
        saved = {call.args[0] for call in repo.save_shock_alert.call_args_list}
        assert saved == {"NVDA", "__PORTFOLIO__"}

    def test_without_a_push_channel_it_logs_and_saves_nothing(self, settings, caplog) -> None:
        repo = self._repo()

        with caplog.at_level(logging.ERROR):
            sent = self._run(self._settings(settings), repo, push=None)

        assert sent == 0
        assert "portfel" in caplog.text.lower()
        repo.save_shock_alert.assert_not_called()

    def test_no_positions_no_portfolio_alert(self, settings) -> None:
        push = MagicMock()

        sent = self._run(self._settings(settings), self._repo(), push=push, portfolio=None)

        assert sent == 0
        push.send_report.assert_not_called()

    def test_positions_without_a_move_are_reported_not_guessed(self, settings) -> None:
        repo = self._repo()
        push = MagicMock()
        portfolio = self._portfolio("NVDA", "AMD", "AVGO", "OFFLIST")
        market = self._market("94", OFFLIST="100")  # watchlista −6%; OFFLIST poza watchlistą

        sent = self._run(
            self._settings(settings), repo, push=push, portfolio=portfolio, market=market
        )

        assert sent >= 1
        bodies = [call.kwargs["plain_text"] for call in push.send_report.call_args_list]
        portfolio_body = next(body for body in bodies if "💼 Portfel" in body)
        assert "75% portfela" in portfolio_body

    def test_daily_change_readings_drive_the_portfolio_too(self, settings) -> None:
        repo = _no_snapshot_repo()
        repo.get_previous_verdict.return_value = None
        repo.get_latest_prediction.return_value = None
        push = MagicMock()
        daily = _daily("-0.035", self._NOW - timedelta(minutes=5))

        sent = main_watch.scan_watch(
            self._settings(settings),
            repository=repo,
            market=self._market(),
            push_notifier=push,
            now=self._NOW,
            daily_change=daily,
            portfolio_loader=lambda: self._portfolio(),
        ).sent

        assert sent == 1
        assert "💼 Portfel -3,5%" in push.send_report.call_args.kwargs["plain_text"]


class TestPortfolioShockInZlotys:
    """ALR-07 + ALR-08: wagi szoku portfela w złotych, a bez kursów NBP portfela
    w kilku walutach nie oceniamy (nie zgadujemy)."""

    _NOW = datetime(2026, 10, 27, 14, 0, tzinfo=UTC)

    def _run(self, settings: Settings, fx_port: MagicMock | None, caplog=None) -> MagicMock:
        from src.domain.portfolio import Portfolio, Position

        portfolio = Portfolio(
            positions=(
                Position(symbol="AAPL", quantity=Decimal("10"), purchase_price=Decimal("80")),
                Position(
                    symbol="PKN", quantity=Decimal("100"), purchase_price=Decimal("50"),
                    currency="PLN",
                ),
            ),
            as_of=self._NOW,
        )
        repo = MagicMock(spec=RepositoryPort)
        repo.get_last_price_snapshot.side_effect = lambda symbol: (
            Money(Decimal({"AAPL": "100", "PKN": "50"}[symbol])),
            self._NOW - timedelta(hours=1),
        )
        repo.get_sent_shock_alerts.return_value = set()
        repo.get_previous_verdict.return_value = None
        repo.get_latest_prediction.return_value = None
        # AAPL −8% (USD), PKN bez zmiany: w złotych AAPL waży ~42%, więc portfel −3,4%.
        market = MagicMock()
        market.get_current_price.side_effect = lambda symbol: Money(
            Decimal({"AAPL": "92", "PKN": "50"}[symbol])
        )
        push = MagicMock()
        cfg = settings.model_copy(
            update={
                "symbols": ["AAPL", "PKN"],
                "portfolio_shock_enabled": True,
                "portfolio_shock_threshold": Decimal("0.03"),
                "quiet_hours_enabled": False,
                "volatility_threshold": Decimal("0.05"),  # symbol: 10% progu, −8% nie alarmuje
            }
        )

        main_watch.scan_watch(
            cfg,
            repository=repo,
            market=market,
            push_notifier=push,
            now=self._NOW,
            portfolio_loader=lambda: portfolio,
            fx_port=fx_port,
        )
        return push

    def test_weights_are_in_zlotys(self, settings) -> None:
        from src.application.ports import FxRatePort
        from src.domain.fx import FxRate

        fx = MagicMock(spec=FxRatePort)
        fx.get_rate.side_effect = lambda currency, day: FxRate(
            currency, day, day, Decimal("4.00")
        )

        push = self._run(settings, fx)

        push.send_report.assert_called_once()
        # Waga AAPL przed ruchem: 10 × 100 × 4 = 4 000 zł z 9 000 zł → −8% × 44,4% = −3,6%.
        assert "💼 Portfel -3,6%" in push.send_report.call_args.kwargs["plain_text"]

    def test_mixed_currencies_without_rates_are_not_assessed(self, settings, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            push = self._run(settings, None)

        push.send_report.assert_not_called()
        assert "kurs" in caplog.text.lower()


class TestThesisInvalidationAlerts:
    """ML-27: push „Teza X unieważniona”, gdy darmowa cena przekroczy poziom
    unieważnienia otwartej predykcji — raz na predykcję (stempel w bazie po wysyłce).
    Kontrakty watcha bez zmian: zero płatnych portów, zero `save_price_snapshot`."""

    _NOW = datetime(2026, 10, 22, 15, 0, tzinfo=UTC)

    @staticmethod
    def _settings(settings: Settings, **overrides: object) -> Settings:
        return settings.model_copy(
            update={"invalidation_level_enabled": True, "quiet_hours_enabled": False, **overrides}
        )

    @classmethod
    def _open(cls, trend: str = "BULLISH", level: str = "118"):
        from src.application.ports import OpenInvalidation

        return OpenInvalidation(
            prediction_id="pred-1",
            symbol="NVDA",
            predicted_trend=trend,
            price_at_prediction=Decimal("120"),
            invalidation_price=Decimal(level),
            predicted_at=datetime(2026, 10, 22, 5, 30, tzinfo=UTC),
        )

    @staticmethod
    def _repo(*open_predictions: object) -> MagicMock:
        repo = MagicMock(spec=RepositoryPort)
        repo.get_open_invalidations.return_value = list(open_predictions)
        return repo

    def _scan(self, settings: Settings, repo: MagicMock, price: str, push: MagicMock | None):
        return main_watch.scan_invalidations(
            settings, repository=repo, market=_market(price), push_notifier=push, now=self._NOW
        )

    def test_crossed_level_sends_the_push_and_stamps_the_prediction(self, settings) -> None:
        repo = self._repo(self._open())
        push = MagicMock()

        sent = self._scan(self._settings(settings), repo, "117.5", push)

        assert sent == 1
        kwargs = push.send_report.call_args.kwargs
        assert kwargs["subject"] == "🛑 Teza NVDA unieważniona"
        assert "117,50" in kwargs["plain_text"]
        assert "118,00" in kwargs["plain_text"]
        assert "(prognoza wzrostu," in kwargs["plain_text"]
        repo.mark_invalidation_alerted.assert_called_once_with("pred-1", self._NOW)
        repo.get_open_invalidations.assert_called_once_with(
            self._NOW - timedelta(hours=main_watch.INVALIDATION_LOOKBACK_HOURS)
        )

    def test_level_above_breaks_a_bearish_thesis_on_a_rise(self, settings) -> None:
        repo = self._repo(self._open(trend="BEARISH", level="125"))
        push = MagicMock()

        assert self._scan(self._settings(settings), repo, "126", push) == 1

    def test_level_not_crossed_sends_nothing(self, settings) -> None:
        repo = self._repo(self._open())
        push = MagicMock()

        assert self._scan(self._settings(settings), repo, "119", push) == 0

        push.send_report.assert_not_called()
        repo.mark_invalidation_alerted.assert_not_called()

    def test_flag_off_touches_nothing(self, settings) -> None:
        repo = self._repo(self._open())
        market = _market("100")

        sent = main_watch.scan_invalidations(
            settings.model_copy(update={"invalidation_level_enabled": False}),
            repository=repo,
            market=market,
            push_notifier=MagicMock(),
            now=self._NOW,
        )

        assert sent == 0
        assert repo.method_calls == []
        market.get_current_price.assert_not_called()

    def test_without_a_push_channel_the_alert_waits(self, settings) -> None:
        repo = self._repo(self._open())

        assert self._scan(self._settings(settings), repo, "100", None) == 0

        repo.mark_invalidation_alerted.assert_not_called()

    def test_unreadable_predictions_skip_the_scan(self, settings, caplog) -> None:
        from src.application.errors import ProviderUnavailable

        repo = self._repo()
        repo.get_open_invalidations.side_effect = ProviderUnavailable("PGRST")

        with caplog.at_level(logging.ERROR):
            assert self._scan(self._settings(settings), repo, "100", MagicMock()) == 0

        assert "074" in caplog.text

    def test_scan_never_writes_a_price_snapshot_or_calls_paid_ports(
        self, settings, mocker
    ) -> None:
        paid_builders = [
            mocker.patch(f"main_agent.{name}")
            for name in ("build_llm_adapter", "build_council_adapter", "build_embedding_adapter")
        ]
        repo = self._repo(self._open())

        self._scan(self._settings(settings), repo, "100", MagicMock())

        repo.save_price_snapshot.assert_not_called()
        for builder in paid_builders:
            builder.assert_not_called()

    def test_main_runs_the_invalidation_scan_on_the_watch_adapters(
        self, settings, mocker
    ) -> None:
        repository = mocker.patch("main_agent.build_repository")
        market = mocker.patch("main_agent.build_market_port")
        push = mocker.patch("main_agent.build_push_notifier")
        settings.telegram_bot_token = "tg-token"
        settings.telegram_chat_id = "42"
        mocker.patch(
            "main_watch.scan_watch",
            return_value=main_watch.WatchOutcome(sent=0, failures=0, total=1),
        )
        mocker.patch("main_watch.scan_price_levels", return_value=0)
        theses = mocker.patch("main_watch.scan_invalidations", return_value=0)

        assert main_watch.main(settings) == 0

        theses.assert_called_once_with(
            settings,
            repository=repository.return_value,
            market=market.return_value,
            push_notifier=push.return_value,
        )
