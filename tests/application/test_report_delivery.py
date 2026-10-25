"""ARCH-08: `DeliveryOutcome.delivered` — mail cyklu wyszedł bez błędu."""

from __future__ import annotations

from src.application.report_delivery import (
    ChannelDelivery,
    DeliveryOutcome,
    DeliveryStatus,
    ReportChannel,
)

_SENT = ChannelDelivery(ReportChannel.EMAIL, DeliveryStatus.SENT)
_FAILED_PUSH = ChannelDelivery(ReportChannel.PUSH_DIGEST, DeliveryStatus.FAILED)


def test_sent_report_with_every_email_out_is_delivered() -> None:
    assert DeliveryOutcome(True, (_SENT, _FAILED_PUSH)).delivered is True


def test_one_failed_subscriber_email_means_not_delivered() -> None:
    failed = ChannelDelivery(ReportChannel.EMAIL, DeliveryStatus.FAILED, "f***@example.com")

    assert DeliveryOutcome(True, (_SENT, failed)).delivered is False


def test_skipped_email_channel_is_not_a_failure() -> None:
    skipped = ChannelDelivery(ReportChannel.EMAIL, DeliveryStatus.SKIPPED)

    assert DeliveryOutcome(True, (skipped,)).delivered is True


def test_report_that_never_reached_its_channels_is_not_delivered() -> None:
    assert DeliveryOutcome(False).delivered is False
