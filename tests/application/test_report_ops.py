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
    render_direction_variant_markdown,
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


def test_trainer_summary_shows_the_label_overlap_of_the_pooled_sample() -> None:
    # ML-25: pomiar nakładania etykiet w summary przebiegu, nie tylko w logu.
    markdown = render_trainer_summary_markdown(
        symbol_statuses={},
        pooled_status="trained_successfully",
        pooled_samples=812,
        failures=0,
        total=1,
        pooled_label_overlap=(10, 49),
    )

    assert "| Label overlaps (settle after the next row) | 10 / 49 |" in markdown


def test_trainer_summary_without_a_label_overlap_has_no_row() -> None:
    markdown = render_trainer_summary_markdown(
        symbol_statuses={}, pooled_status=None, pooled_samples=None, failures=1, total=1
    )

    assert "Label overlaps" not in markdown


def test_direction_variant_summary_compares_the_classifier_with_the_base_rate() -> None:
    from src.domain.direction_variant import DirectionReport, DirectionSummary
    from src.domain.model_scorecard import TrainingOutcome

    report = DirectionReport(
        TrainingOutcome.REJECTED,
        n_samples=236,
        n_in_dead_band=64,
        dead_band=0.001,
        summary=DirectionSummary(
            n_folds=4,
            n_oos_points=189,
            brier=0.2461,
            base_rate_brier=0.2499,
            log_loss=0.6851,
            base_rate_log_loss=0.6930,
            beats_base_rate=False,
            verdict="klasyfikator nie bije stopy bazowej o 2%",
        ),
        reason="klasyfikator nie bije stopy bazowej o 2%",
    )

    markdown = render_direction_variant_markdown(report)

    assert "### Direction classifier (ML-30)" in markdown
    assert "| Status | rejected |" in markdown
    assert "| Rows outside the dead band (±0.001) | 236 |" in markdown
    assert "| Rows inside the dead band | 64 |" in markdown
    assert "| Brier (classifier / base rate) | 0.2461 / 0.2499 |" in markdown
    assert "| Log loss (classifier / base rate) | 0.6851 / 0.6930 |" in markdown
    assert "served regression model" in markdown


def test_skipped_direction_variant_summary_has_no_scores() -> None:
    from src.domain.direction_variant import DirectionReport
    from src.domain.model_scorecard import TrainingOutcome

    markdown = render_direction_variant_markdown(
        DirectionReport(
            TrainingOutcome.SKIPPED, n_samples=12, n_in_dead_band=3, dead_band=0.0, reason="few"
        )
    )

    assert "| Status | skipped |" in markdown
    assert "| Brier (classifier / base rate) | n/a / n/a |" in markdown
    assert "| Verdict | few |" in markdown


def test_cycle_summary_lists_http_requests_per_host() -> None:
    # NIEZ-24: żądania per host i klasy statusów — sam host, bez ścieżki i query.
    markdown = _render(
        http_calls={
            "finnhub.io": {"requests": 40, "2xx": 38, "429": 2, "retries": 2},
            "www.alphavantage.co": {"requests": 3, "2xx": 3},
        }
    )

    assert "### HTTP requests by host" in markdown
    assert "| Host | Requests | 2xx | 4xx | 429 | 5xx | Retries |" in markdown
    assert "| finnhub.io | 40 | 38 | 0 | 2 | 0 | 2 |" in markdown
    assert "| www.alphavantage.co | 3 | 3 | 0 | 0 | 0 | 0 |" in markdown


def test_cycle_summary_without_http_meter_has_no_http_section() -> None:
    assert "HTTP requests by host" not in _render()


def test_challenger_summary_shows_the_paired_verdict_of_each_challenger() -> None:
    # ML-33: werdykt promocji w summary Slow Loopa — liczby par, straty, z i werdykt;
    # sam raport, podmiana wag zostaje decyzją właściciela.
    from src.application.report_ops import render_challenger_verdicts_markdown
    from src.domain.promotion import (
        ChallengerVerdict,
        ForecastKind,
        PromotionStatus,
        PromotionVerdict,
    )

    markdown = render_challenger_verdicts_markdown(
        (
            ChallengerVerdict(
                "alt",
                ForecastKind.RETURN,
                2,
                PromotionVerdict(PromotionStatus.INSUFFICIENT_DATA, 12, 0.0004, 0.0003, 1.2),
            ),
            ChallengerVerdict(
                "direction",
                ForecastKind.DIRECTION,
                1,
                PromotionVerdict(PromotionStatus.PROMOTE, 80, 0.5, 0.2, 6.25),
            ),
        )
    )

    assert "### Champion/challenger (ML-33)" in markdown
    assert "| alt | squared return error | 2 | 12 | 0.0004 / 0.0003 | 25.0% | 1.20 |" in markdown
    assert "insufficient_data |" in markdown
    assert "| direction | direction Brier | 1 | 80 | 0.5000 / 0.2000 | 60.0% | 6.25 |" in markdown
    assert "promote |" in markdown
    assert "owner" in markdown


def test_challenger_summary_without_pairs_says_so() -> None:
    from src.application.report_ops import render_challenger_verdicts_markdown

    markdown = render_challenger_verdicts_markdown(())

    assert "### Champion/challenger (ML-33)" in markdown
    assert "No challenger forecast" in markdown
