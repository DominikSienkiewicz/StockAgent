"""NIEZ-22: podsumowanie przebiegu w GITHUB_STEP_SUMMARY — czyste renderery markdown.

Repo jest publiczne, więc strona runu w Actions jest publiczna: summary niesie
WYŁĄCZNIE agregaty. Inwariant jak w publicznym scorecardzie — zero symboli
watchlisty, zero adresów e-mail, zero treści błędów, reasoning, kwot i pozycji.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from src.application.report_models import SymbolResult
from src.application.report_ops import (
    render_cycle_summary_markdown,
    render_trainer_summary_markdown,
    render_watch_summary_markdown,
)
from src.domain.cycle_maturity import SkipReason
from src.domain.finops import estimate_cycle_cost
from src.domain.quota import QuotaAlert, QuotaSeverity

_LEAKY_ERROR = "connection reset by peer at https://api.example.test/quote?token=abc"
_LEAKY_REASONING = "Strong buy, owner holds 120 shares worth 14 000 zł"


def _results() -> list[SymbolResult]:
    return [
        SymbolResult(
            symbol="AAPLX",
            status="saved",
            delta=Decimal("0.031"),
            current_price=Decimal("231.40"),
            trend="BULLISH",
            target_price=Decimal("240.00"),
            reasoning=_LEAKY_REASONING,
        ),
        SymbolResult(
            symbol="NVDAX",
            status="error",
            error_message=_LEAKY_ERROR,
            error_type="TimeoutError",
        ),
        SymbolResult(
            symbol="MSFTX",
            status="ignored",
            delta=Decimal("0.004"),
            skip_reason=SkipReason.BELOW_THRESHOLD,
        ),
        SymbolResult(
            symbol="CSPXX.L",
            status="ignored",
            skip_reason=SkipReason.UNSUPPORTED_PRICE,
            error_message="unsupported by current price adapter (Finnhub free)",
        ),
    ]


def _alerts() -> list[QuotaAlert]:
    return [
        QuotaAlert(
            source="Resend",
            severity=QuotaSeverity.CRITICAL,
            message="Mail do owner@example.com nie wyszedł",
            action="Sprawdź klucz",
            occurred_at=datetime(2026, 10, 20, 5, 30, tzinfo=UTC),
        ),
        QuotaAlert(
            source="Alpha Vantage",
            severity=QuotaSeverity.WARNING,
            message="Rotacja klucza",
            action="Brak",
            occurred_at=datetime(2026, 10, 20, 5, 31, tzinfo=UTC),
        ),
    ]


def _render(**overrides: object) -> str:
    kwargs: dict[str, object] = {
        "paid_calls": {"llm": 3, "council_llm": 16},
        "cycle_cost": estimate_cycle_cost({"llm": 3, "council_llm": 16}),
        "failures": 1,
        "duration_s": 83.4,
        "quota_alerts": _alerts(),
        "delivered": True,
    }
    kwargs.update(overrides)
    return render_cycle_summary_markdown(_results(), **kwargs)  # type: ignore[arg-type]


def test_cycle_summary_counts_failures_and_paid_calls() -> None:
    markdown = _render()

    assert "| error | 1 |" in markdown
    assert "| saved | 1 |" in markdown
    assert "| ignored | 2 |" in markdown
    assert "| TimeoutError | 1 |" in markdown
    assert "| llm | 3 |" in markdown
    assert "| council_llm | 16 |" in markdown
    assert "| Failures | 1 |" in markdown


def test_cycle_summary_counts_skip_reasons_and_quota_alerts_per_level() -> None:
    markdown = _render()

    assert "| BELOW_THRESHOLD | 1 |" in markdown
    assert "| UNSUPPORTED_PRICE | 1 |" in markdown
    assert "| CRITICAL | 1 |" in markdown
    assert "| WARNING | 1 |" in markdown


def test_cycle_summary_reports_duration_cost_and_delivery() -> None:
    markdown = _render()

    assert "83.4 s" in markdown
    assert "| Report email | delivered |" in markdown
    assert "estimate" in markdown
    assert "$0.0700" in markdown


def test_cycle_summary_names_a_failed_and_an_unconfigured_delivery() -> None:
    assert "| Report email | not delivered |" in _render(delivered=False)
    assert "| Report email | not configured |" in _render(delivered=None)


def test_cycle_summary_without_cost_says_so() -> None:
    # FIN-35: lokalny run z atrapami nie ma kosztu (None) — summary tego nie zgaduje.
    assert "| Cycle cost | n/a |" in _render(cycle_cost=None)


def test_summary_contains_no_symbol_email_or_error_text() -> None:
    markdown = _render()

    for result in _results():
        assert result.symbol not in markdown
    assert "@" not in markdown
    assert _LEAKY_ERROR not in markdown
    assert "Finnhub" not in markdown
    assert "reset by peer" not in markdown
    assert "120 shares" not in markdown
    assert "zł" not in markdown
    assert "231" not in markdown


def test_error_without_type_is_counted_as_unknown() -> None:
    markdown = render_cycle_summary_markdown(
        [SymbolResult(symbol="ZZZQ", status="error", error_message="No result")],
        paid_calls={},
        cycle_cost=None,
        failures=1,
        duration_s=1.0,
        quota_alerts=[],
        delivered=None,
    )

    assert "| unknown | 1 |" in markdown
    assert "ZZZQ" not in markdown


def test_free_cycle_lists_no_paid_call_rows() -> None:
    markdown = _render(paid_calls={}, cycle_cost=estimate_cycle_cost({}))

    assert "| (none) | 0 |" in markdown


def test_watch_summary_is_three_numbers() -> None:
    markdown = render_watch_summary_markdown(checked=5, stale=2, sent=1, failures=0, total=6)

    assert "| Checked | 5 |" in markdown
    assert "| Stale reference | 2 |" in markdown
    assert "| Alerts sent | 1 |" in markdown
    assert "| Failures | 0 / 6 |" in markdown


def test_trainer_summary_counts_statuses_without_symbols() -> None:
    markdown = render_trainer_summary_markdown(
        symbol_statuses={"evaluated_passed_gate": 2, "skipped": 3},
        pooled_status="trained_successfully",
        pooled_samples=812,
        failures=1,
        total=6,
    )

    assert "| evaluated_passed_gate | 2 |" in markdown
    assert "| skipped | 3 |" in markdown
    assert "| Pooled model | trained_successfully |" in markdown
    assert "| Pooled samples | 812 |" in markdown
    assert "| Failures | 1 / 6 |" in markdown


def test_trainer_summary_without_a_pooled_result() -> None:
    markdown = render_trainer_summary_markdown(
        symbol_statuses={}, pooled_status=None, pooled_samples=None, failures=1, total=1
    )

    assert "| Pooled model | failed |" in markdown
    assert "| Pooled samples | n/a |" in markdown


def test_trainer_summary_names_a_failed_feature_store_refresh() -> None:
    markdown = render_trainer_summary_markdown(
        symbol_statuses={},
        pooled_status=None,
        pooled_samples=None,
        failures=1,
        total=1,
        feature_store_refreshed=False,
    )

    assert "| Feature store refresh | failed |" in markdown
