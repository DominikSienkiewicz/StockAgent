"""Testy renderera bannera QuotaAlert."""

from __future__ import annotations

from datetime import UTC, datetime

from src.application.report_quota_banner import (
    render_quota_banner_html,
    render_quota_banner_text,
)
from src.domain.quota import QuotaAlert, QuotaSeverity


def _a(
    source: str,
    severity: QuotaSeverity = QuotaSeverity.WARNING,
    msg: str = "msg",
    when: datetime | None = None,
) -> QuotaAlert:
    return QuotaAlert(
        source=source,
        severity=severity,
        message=msg,
        action="act",
        occurred_at=when or datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
    )


class TestEmptyBanner:
    def test_empty_alerts_returns_empty_string(self) -> None:
        assert render_quota_banner_html([]) == ""
        assert render_quota_banner_text([]) == ""


class TestHtmlBanner:
    def test_banner_lists_all_alerts(self) -> None:
        html = render_quota_banner_html(
            [
                _a("Alpha Vantage", QuotaSeverity.CRITICAL, msg="exhausted"),
                _a("OpenAI", QuotaSeverity.WARNING, msg="TPM"),
            ]
        )
        assert "Alpha Vantage" in html
        assert "OpenAI" in html
        assert "exhausted" in html
        assert "TPM" in html
        assert "ALERTY SUBSKRYPCJI" in html

    def test_dedupes_identical_alerts(self) -> None:
        a = _a("OpenAI", QuotaSeverity.WARNING, msg="TPM")
        html = render_quota_banner_html([a, a, a])
        assert html.count("TPM") == 1

    def test_critical_takes_main_color(self) -> None:
        html = render_quota_banner_html(
            [
                _a("X", QuotaSeverity.WARNING, msg="m1"),
                _a("Y", QuotaSeverity.CRITICAL, msg="m2"),
            ]
        )
        # Czerwony = #dc2626 dla bordera
        assert "#dc2626" in html

    def test_critical_before_warning_in_output(self) -> None:
        html = render_quota_banner_html(
            [
                _a("AA", QuotaSeverity.WARNING, msg="w1"),
                _a("CC", QuotaSeverity.CRITICAL, msg="c1"),
            ]
        )
        assert html.index("c1") < html.index("w1")


class TestLocalTime:
    """MAIL-59: czas alertu w strefie wyświetlania maila, jak nagłówek i temat —
    nie „… UTC” obok godzin lokalnych."""

    WHEN = datetime(2026, 10, 16, 5, 30, tzinfo=UTC)

    def test_html_shows_the_alert_in_the_display_timezone(self) -> None:
        html = render_quota_banner_html([_a("OpenAI", when=self.WHEN)], "Europe/Warsaw")

        assert "pt 16.10 07:30" in html
        assert "05:30" not in html
        assert "UTC" not in html

    def test_text_shows_the_alert_in_the_display_timezone(self) -> None:
        text = render_quota_banner_text([_a("OpenAI", when=self.WHEN)], "Europe/Warsaw")

        assert "(pt 16.10 07:30)" in text
        assert "05:30" not in text
        assert "UTC" not in text

    def test_winter_offset_differs_from_summer(self) -> None:
        winter = datetime(2026, 12, 4, 5, 30, tzinfo=UTC)

        html = render_quota_banner_html([_a("OpenAI", when=winter)], "Europe/Warsaw")

        assert "pt 04.12 06:30" in html

    def test_default_timezone_is_the_mail_display_timezone(self) -> None:
        html = render_quota_banner_html([_a("OpenAI", when=self.WHEN)])

        assert "pt 16.10 07:30" in html

    def test_naive_timestamp_is_read_as_utc(self) -> None:
        naive = datetime(2026, 10, 16, 5, 30)

        html = render_quota_banner_html([_a("OpenAI", when=naive)], "Europe/Warsaw")

        assert "pt 16.10 07:30" in html


class TestPlainText:
    def test_text_includes_action(self) -> None:
        text = render_quota_banner_text([_a("OpenAI", msg="TPM")])
        assert "OpenAI" in text
        assert "TPM" in text
        assert "act" in text or "→" in text

    def test_dedupes_in_text(self) -> None:
        a = _a("OpenAI", msg="TPM")
        text = render_quota_banner_text([a, a, a])
        assert text.count("TPM") == 1


class TestLlmBudgetBanner:
    """FIN-25: wyczerpany limit USD doby albo miesiąca — banner limitów na górze maila
    mówi, który limit, ile wydano i co to znaczy dla cyklu."""

    STARTED = datetime(2026, 10, 25, 5, 30, tzinfo=UTC)

    def _mail(self, breach: object) -> tuple[str, str]:
        from src.application.report_builder import ReportContext, build_html_report
        from src.domain.finops import BudgetBreach

        assert breach is None or isinstance(breach, BudgetBreach)
        return build_html_report([], self.STARTED, 1.0, ReportContext(llm_budget_breach=breach))

    def test_daily_breach_names_the_cap_and_the_spend(self) -> None:
        from src.domain.finops import BudgetBreach, BudgetWindow

        html, text = self._mail(BudgetBreach(BudgetWindow.DAY, spent_usd=2.134, cap_usd=2.0))

        for body in (html, text):
            assert "LIMIT LLM WYCZERPANY" in body
            assert "Dzienny limit" in body
            assert "$2,13 z $2,00" in body
            assert "bez płatnych wywołań" in body

    def test_monthly_breach_names_the_month(self) -> None:
        from src.domain.finops import BudgetBreach, BudgetWindow

        html, text = self._mail(BudgetBreach(BudgetWindow.MONTH, spent_usd=30.5, cap_usd=30.0))

        assert "Miesięczny limit" in html
        assert "$30,50 z $30,00" in text

    def test_banner_comes_before_the_session_block(self) -> None:
        from src.domain.finops import BudgetBreach, BudgetWindow

        _, text = self._mail(BudgetBreach(BudgetWindow.DAY, spent_usd=2.5, cap_usd=2.0))

        assert text.index("LIMIT LLM WYCZERPANY") < text.index("STOCKAGENT — RAPORT CYKLU")

    def test_no_breach_no_banner(self) -> None:
        html, text = self._mail(None)

        assert "LIMIT LLM" not in html
        assert "LIMIT LLM" not in text
