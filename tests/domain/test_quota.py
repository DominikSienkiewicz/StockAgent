"""Testy domeny QuotaAlert — value object dla wyczerpań limitów."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.domain.quota import QuotaAlert, QuotaSeverity


def _alert(
    severity: QuotaSeverity = QuotaSeverity.CRITICAL,
    source: str = "Alpha Vantage",
) -> QuotaAlert:
    return QuotaAlert(
        source=source,
        severity=severity,
        message="All 5 keys exhausted at 14:32",
        action="Add new API key or wait until daily reset",
        occurred_at=datetime(2026, 6, 1, 14, 32, tzinfo=UTC),
    )


class TestImmutability:
    def test_is_frozen(self) -> None:
        alert = _alert()
        # FrozenInstanceError jest dataclass-specific.
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            alert.source = "Other"  # type: ignore[misc]

    def test_equality_by_value(self) -> None:
        a = _alert()
        b = _alert()
        assert a == b
        assert hash(a) == hash(b)


class TestSeverityOrdering:
    """CRITICAL musi być „wyższe” niż WARNING — banner używa max() do\
 ustalenia kolorów i prefiksu subjectu."""

    def test_critical_ranks_above_warning(self) -> None:
        assert QuotaSeverity.CRITICAL.rank > QuotaSeverity.WARNING.rank

    def test_max_severity_picks_critical(self) -> None:
        levels = [QuotaSeverity.WARNING, QuotaSeverity.CRITICAL, QuotaSeverity.WARNING]
        assert max(levels, key=lambda s: s.rank) is QuotaSeverity.CRITICAL


class TestRendering:
    def test_label_is_human_readable(self) -> None:
        alert = _alert()
        assert "Alpha Vantage" in str(alert.source)
        assert alert.message.startswith("All 5 keys")
        assert alert.action.startswith("Add new")
