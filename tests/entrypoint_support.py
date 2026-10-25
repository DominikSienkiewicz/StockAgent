"""Pomocniki wspólne dla testów entry pointów: test_main_agent, test_main_trainer,
test_main_backtest."""

from datetime import UTC, datetime

from src.domain.quota import QuotaAlert, QuotaSeverity


def build_quota_alert(source: str = "Resend") -> QuotaAlert:
    return QuotaAlert(
        source=source,
        severity=QuotaSeverity.CRITICAL,
        message=f"{source} returned 503",
        action="sprawdź dostawcę",
        occurred_at=datetime(2026, 10, 18, 5, 30, tzinfo=UTC),
    )


def build_usage(model: str = "claude-sonnet-4-6") -> "object":
    from src.domain.finops import LLMUsage

    return LLMUsage(provider="anthropic", model=model, input_tokens=1000, output_tokens=200)


UNDELIVERED_ANNOTATION = "::error title=StockAgent::Raport nie został dostarczony"
