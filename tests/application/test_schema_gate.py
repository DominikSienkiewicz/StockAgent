"""NIEZ-30: bramka schematu — flagi bez swojej migracji wyłączają się na start.

Decyzja właściciela (2026-10-24): gdy `schema_version()` jest niedostępne, flagi
ZOSTAJĄ (fail-open), a ostrzeżenie trafia do logu i summary — bez alertu w mailu.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.application.errors import NotFound, ProviderUnavailable
from src.application.schema_gate import SCHEMA_ALERT_SOURCE, evaluate_schema_gate
from src.domain.quota import QuotaSeverity

NOW = datetime(2026, 10, 26, 5, 30, tzinfo=UTC)


def _source(result: object) -> MagicMock:
    source = MagicMock()
    source.schema_version.return_value = result
    return source


def test_a_flag_ahead_of_the_schema_is_disabled_with_a_warning_alert() -> None:
    outcome = evaluate_schema_gate(
        _source(19), {"receipts_enabled": True, "debate_mode_enabled": True}, now=NOW
    )

    assert outcome.schema_version == 19
    assert outcome.disabled == frozenset({"receipts_enabled"})
    assert outcome.alert is not None
    assert outcome.alert.source == SCHEMA_ALERT_SOURCE
    assert outcome.alert.severity is QuotaSeverity.WARNING
    assert outcome.alert.occurred_at == NOW
    assert "receipts_enabled" in outcome.alert.message
    assert "19" in outcome.alert.message
    assert "DB Migrate" in outcome.alert.action


def test_every_disabled_flag_is_named_once_in_the_alert() -> None:
    outcome = evaluate_schema_gate(
        _source(19),
        {"shock_alerts_enabled": True, "receipts_enabled": True, "model_scorecard_enabled": True},
        now=NOW,
    )

    assert outcome.disabled == frozenset({"shock_alerts_enabled", "receipts_enabled"})
    assert outcome.alert is not None
    for flag in outcome.disabled:
        assert outcome.alert.message.count(flag) == 1


def test_a_migrated_database_changes_nothing() -> None:
    outcome = evaluate_schema_gate(_source(71), {"receipts_enabled": True}, now=NOW)

    assert outcome.schema_version == 71
    assert outcome.disabled == frozenset()
    assert outcome.alert is None


@pytest.mark.parametrize(
    "unavailable",
    [NotFound("PGRST202"), ProviderUnavailable("down")],
    ids=["function-missing", "database-down"],
)
def test_an_unavailable_probe_keeps_the_flags_and_only_logs(
    unavailable: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    source = MagicMock()
    source.schema_version.side_effect = unavailable

    with caplog.at_level(logging.WARNING):
        outcome = evaluate_schema_gate(source, {"receipts_enabled": True}, now=NOW)

    assert outcome.schema_version is None
    assert outcome.disabled == frozenset()
    assert outcome.alert is None
    assert type(unavailable).__name__ in caplog.text
    assert "left on" in caplog.text


@pytest.mark.parametrize("unknown", [None, "70", True, MagicMock()], ids=repr)
def test_an_unknown_version_is_never_read_as_zero(
    unknown: object, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        outcome = evaluate_schema_gate(_source(unknown), {"receipts_enabled": True}, now=NOW)

    assert outcome.schema_version is None
    assert outcome.disabled == frozenset()
    assert outcome.alert is None
    assert "left on" in caplog.text
