"""MAIL-24 / BEZ-27: prywatne archiwum digestu w composition root.

Za flagą `digest_archive_enabled` (i tylko z `db_writes_enabled`) pełny HTML
maila właściciela trafia do prywatnego bucketu, mail i push dostają podpisany
URL, a indeks `digests` — wiersz cyklu. Błąd archiwum nie zatrzymuje maila.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

import main_agent
from src.application.errors import ProviderUnavailable
from src.application.ports import DigestArchivePort, DigestRecord, RepositoryPort
from src.application.quota_monitor import QuotaMonitor
from src.application.report_builder import ReportContext, SymbolResult
from src.application.report_delivery import (
    ChannelDelivery,
    DeliveryOutcome,
    DeliveryStatus,
    ReportChannel,
)
from src.application.report_evidence import PaperCosts
from src.application.run_context import RunContext
from src.application.track_record_window import TrackRecordReads
from src.application.use_cases.assemble_report import AssembledReport
from src.config import Settings
from src.domain.subscriber import Subscriber
from tests.settings_kwargs import NO_ENV_FILE

STARTED = datetime(2026, 10, 15, 5, 30, tzinfo=UTC)
URL = "https://proj.supabase.co/storage/v1/object/sign/digests/c.html?token=t"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        **NO_ENV_FILE,
        openai_api_key="sk-test",
        finnhub_api_key="fh",
        alpha_vantage_api_keys=["av1"],
        supabase_url="https://test.supabase.co",
        supabase_key="sb_secret_test",
        symbols=["AAPL"],
        db_writes_enabled=True,
        paid_calls_enabled=True,
        messenger_digest_enabled=False,
        attestation_enabled=False,
        public_scorecard_enabled=False,
        digest_archive_enabled=True,
    )


def _assembled() -> AssembledReport:
    return AssembledReport(
        context=ReportContext(run_context=RunContext(run_id="18234567890")),
        reads=TrackRecordReads(MagicMock(spec=RepositoryPort), 30),
        costs=PaperCosts(),
    )


def _deliver(settings: Settings, notifier: MagicMock, **overrides: Any) -> DeliveryOutcome:
    kwargs: dict[str, Any] = {
        "results": [SymbolResult(symbol="AAPL", status="ignored", delta=Decimal("0.004"))],
        "started_at": STARTED,
        "duration": 1.0,
        "failures": 0,
        "settings": settings,
        "repository": MagicMock(spec=RepositoryPort),
        "quota_monitor": QuotaMonitor(),
        "notifier": notifier,
        "subscribers": [],
        "all_symbols": ["AAPL"],
    }
    kwargs.update(overrides)
    return main_agent.deliver_report(_assembled, **kwargs)


def _archive(mocker: Any, store: object = URL) -> MagicMock:
    archive = MagicMock(spec=DigestArchivePort)
    if isinstance(store, Exception):
        archive.store.side_effect = store
    else:
        archive.store.return_value = store
    mocker.patch("main_agent.build_digest_archive", return_value=archive)
    return archive


class TestBuildDigestArchive:
    def test_flag_off_builds_no_archive(self, settings: Settings) -> None:
        settings.digest_archive_enabled = False

        assert main_agent.build_digest_archive(settings) is None

    def test_read_only_run_builds_no_archive(self, settings: Settings) -> None:
        settings.db_writes_enabled = False

        assert main_agent.build_digest_archive(settings) is None

    def test_flag_on_builds_the_supabase_storage_archive(
        self, settings: Settings, mocker: Any
    ) -> None:
        from src.infrastructure.adapters.supabase_storage_archive import (
            SupabaseStorageDigestArchive,
        )

        create = mocker.patch("src.infrastructure.adapters.supabase_storage_archive.create_client")

        archive = main_agent.build_digest_archive(settings)

        assert isinstance(archive, SupabaseStorageDigestArchive)
        create.assert_called_once_with("https://test.supabase.co", "sb_secret_test")


def test_cycle_id_is_the_utc_start_and_run_without_personal_data() -> None:
    assert (
        main_agent.digest_cycle_id(STARTED, RunContext(run_id="18234567890"))
        == "20261015T0530Z-18234567890"
    )
    assert main_agent.digest_cycle_id(STARTED, None) == "20261015T0530Z-local"


def test_owner_mail_links_the_archived_digest(settings: Settings, mocker: Any) -> None:
    archive = _archive(mocker)
    notifier = MagicMock()

    outcome = _deliver(settings, notifier)

    cycle_id, archived_html = archive.store.call_args.args
    assert cycle_id == "20261015T0530Z-18234567890"
    assert "Otwórz w przeglądarce" not in archived_html
    sent = notifier.send_report.call_args.kwargs
    assert "Otwórz w przeglądarce" in sent["html_body"]
    assert URL in sent["plain_text"]
    record = archive.record.call_args.args[0]
    assert isinstance(record, DigestRecord)
    assert record.cycle_id == cycle_id
    assert record.subject == sent["subject"]
    assert ChannelDelivery(ReportChannel.ARCHIVE, DeliveryStatus.SENT) in outcome.channels


def test_archive_failure_still_sends_the_mail_without_a_link(
    settings: Settings, mocker: Any
) -> None:
    archive = _archive(mocker, ProviderUnavailable("bucket missing"))
    notifier = MagicMock()

    outcome = _deliver(settings, notifier)

    sent = notifier.send_report.call_args.kwargs
    assert "Otwórz w przeglądarce" not in sent["html_body"]
    archive.record.assert_not_called()
    assert ChannelDelivery(ReportChannel.ARCHIVE, DeliveryStatus.FAILED) in outcome.channels
    assert outcome.delivered is True


def test_flag_off_mail_has_no_archive_channel(settings: Settings, mocker: Any) -> None:
    settings.digest_archive_enabled = False
    notifier = MagicMock()

    outcome = _deliver(settings, notifier)

    assert "Otwórz w przeglądarce" not in notifier.send_report.call_args.kwargs["html_body"]
    assert all(c.channel is not ReportChannel.ARCHIVE for c in outcome.channels)


def test_only_the_owner_subscriber_gets_the_link(settings: Settings, mocker: Any) -> None:
    # BEZ-01: podpisany URL otwiera mail właściciela z portfelem — obcy go nie dostaje.
    _archive(mocker)
    settings.resend_api_key = "re_test"
    settings.notifications_enabled = True
    settings.digest_to_email = "owner@example.com"
    sent: dict[str, str] = {}

    def resend(*, recipient: str, **_: object) -> MagicMock:
        channel = MagicMock()
        channel.send_report.side_effect = lambda **kw: sent.__setitem__(recipient, kw["html_body"])
        return channel

    mocker.patch("main_agent.ResendNotifier", side_effect=resend)

    _deliver(
        settings,
        MagicMock(),
        subscribers=[
            Subscriber(email="owner@example.com", symbols=frozenset()),
            Subscriber(email="reader@example.com", symbols=frozenset()),
        ],
    )

    assert "Otwórz w przeglądarce" in sent["owner@example.com"]
    assert "Otwórz w przeglądarce" not in sent["reader@example.com"]


def test_push_digest_carries_the_link(settings: Settings, mocker: Any) -> None:
    _archive(mocker)
    settings.messenger_digest_enabled = True
    push = MagicMock()
    mocker.patch("main_agent.build_push_notifier", return_value=push)

    _deliver(settings, MagicMock())

    assert URL in push.send_report.call_args.kwargs["plain_text"]
