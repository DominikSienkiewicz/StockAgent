"""MAIL-35: polityka cichego dnia w composition root (`deliver_report`).

Z `digest_quiet_policy = "compact_when_quiet"` cykl bez niczego ponad stan
(pusty lead, zero rozliczeń, szoków, symboli za bramką i błędów) wysyła zwarty
mail; każdy inny dzień i polityka „always” — pełny. Archiwum (MAIL-24) dostaje
zawsze pełny mail, a zwarty linkuje do niego.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

import main_agent
from src.application.ports import DigestArchivePort, RepositoryPort
from src.application.quota_monitor import QuotaMonitor
from src.application.report_builder import ReportContext, ResolvedPrediction, SymbolResult
from src.application.report_delivery import DeliveryOutcome
from src.application.report_evidence import PaperCosts
from src.application.report_formatting import DATA_TERMS_NOTICE
from src.application.run_context import RunContext
from src.application.track_record_window import TrackRecordReads
from src.application.use_cases.assemble_report import AssembledReport
from src.config import Settings
from src.domain.cycle_maturity import SkipReason
from tests.settings_kwargs import NO_ENV_FILE

STARTED = datetime(2026, 10, 15, 5, 30, tzinfo=UTC)
URL = "https://proj.supabase.co/storage/v1/object/sign/digests/c.html?token=t"
QUIET_BLOCK = "Cichy dzień"
FULL_ONLY = "⏸ Pominięte"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        **NO_ENV_FILE,
        openai_api_key="sk-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1"],
        supabase_url="https://test.supabase.co",
        supabase_key="sb_secret_test",
        symbols=["AAPL", "MSFT"],
        messenger_digest_enabled=False,
        attestation_enabled=False,
        public_scorecard_enabled=False,
        digest_quiet_policy="compact_when_quiet",
    )


def _ignored(symbol: str, delta: str) -> SymbolResult:
    return SymbolResult(
        symbol=symbol,
        status="ignored",
        delta=Decimal(delta),
        current_price=Decimal("100"),
        skip_reason=SkipReason.BELOW_THRESHOLD,
        effective_threshold=Decimal("0.02"),
    )


QUIET = [_ignored("AAPL", "0.004"), _ignored("MSFT", "-0.006")]


def _assemble(**context: Any) -> Callable[[], AssembledReport]:
    def assemble() -> AssembledReport:
        return AssembledReport(
            context=ReportContext(
                **{
                    "run_context": RunContext(run_id="18234567890"),
                    "resolved_predictions": [],
                    **context,
                }
            ),
            reads=TrackRecordReads(MagicMock(spec=RepositoryPort), 30),
            costs=PaperCosts(),
        )

    return assemble


def _deliver(
    settings: Settings,
    notifier: MagicMock,
    results: list[SymbolResult] | None = None,
    **context: Any,
) -> DeliveryOutcome:
    return main_agent.deliver_report(
        _assemble(**context),
        results=QUIET if results is None else results,
        started_at=STARTED,
        duration=1.0,
        failures=0,
        settings=settings,
        repository=MagicMock(spec=RepositoryPort),
        quota_monitor=QuotaMonitor(),
        notifier=notifier,
        subscribers=[],
        all_symbols=["AAPL", "MSFT"],
    )


def _sent(notifier: MagicMock) -> dict[str, str]:
    kwargs: dict[str, str] = notifier.send_report.call_args.kwargs
    return kwargs


def test_quiet_day_with_the_compact_policy_sends_the_compact_mail(settings: Settings) -> None:
    notifier = MagicMock()

    outcome = _deliver(settings, notifier)

    sent = _sent(notifier)
    assert QUIET_BLOCK in sent["html_body"]
    assert FULL_ONLY not in sent["html_body"]
    assert DATA_TERMS_NOTICE in sent["plain_text"]
    assert outcome.delivered is True


def test_always_policy_sends_the_full_mail_on_a_quiet_day(settings: Settings) -> None:
    settings.digest_quiet_policy = "always"
    notifier = MagicMock()

    _deliver(settings, notifier)

    html = _sent(notifier)["html_body"]
    assert QUIET_BLOCK not in html
    assert FULL_ONLY in html


def test_a_settlement_keeps_the_full_mail(settings: Settings) -> None:
    notifier = MagicMock()
    settled = ResolvedPrediction(symbol="AAPL", predicted_trend="BULLISH", is_correct=True)

    _deliver(settings, notifier, resolved_predictions=[settled])

    assert QUIET_BLOCK not in _sent(notifier)["html_body"]


def test_a_failed_symbol_keeps_the_full_mail(settings: Settings) -> None:
    notifier = MagicMock()
    failed = SymbolResult(symbol="MSFT", status="error", error_message="HTTPError: 429")

    _deliver(settings, notifier, results=[QUIET[0], failed])

    assert QUIET_BLOCK not in _sent(notifier)["html_body"]


def test_the_archive_keeps_the_full_mail_and_the_compact_one_links_it(
    settings: Settings, mocker: Any
) -> None:
    settings.db_writes_enabled = True
    settings.digest_archive_enabled = True
    archive = MagicMock(spec=DigestArchivePort)
    archive.store.return_value = URL
    mocker.patch("main_agent.build_digest_archive", return_value=archive)
    notifier = MagicMock()

    _deliver(settings, notifier)

    archived_html = archive.store.call_args.args[1]
    assert FULL_ONLY in archived_html
    assert QUIET_BLOCK not in archived_html
    sent = _sent(notifier)
    assert QUIET_BLOCK in sent["html_body"]
    assert "Otwórz w przeglądarce" in sent["html_body"]
