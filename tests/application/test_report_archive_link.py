"""MAIL-24: link „Otwórz w przeglądarce” do prywatnego archiwum digestu."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from src.application.report_builder import ReportContext, SymbolResult, build_html_report
from src.application.report_messenger import MAX_DIGEST_CHARS, build_messenger_digest

AT = datetime(2026, 10, 15, 5, 30, tzinfo=UTC)
URL = (
    "https://proj.supabase.co/storage/v1/object/sign/digests/20261015T0530Z-1.html"
    "?token=eyJhbGciOiJIUzI1NiJ9.abc&x=1"
)
RESULTS = [SymbolResult(symbol="MSFT", status="ignored", delta=Decimal("0.005"))]


def test_header_links_the_archived_digest() -> None:
    html, text = build_html_report(RESULTS, AT, 1.0, ReportContext(archive_url=URL))

    masthead = html.split("Czas rozpoczęcia")[1].split("</div>")[0]
    assert 'href="https://proj.supabase.co/storage/v1/object/sign/digests/' in masthead
    assert "&amp;x=1" in masthead
    assert "Otwórz w przeglądarce" in masthead
    assert f"Wersja w przeglądarce (link ważny 7 dni): {URL}" in text


def test_no_link_without_an_archive_or_with_an_unsafe_url() -> None:
    for archive_url in (None, "javascript:alert(1)"):
        html, text = build_html_report(
            RESULTS, AT, 1.0, ReportContext(archive_url=archive_url)
        )

        assert "Otwórz w przeglądarce" not in html
        assert "Wersja w przeglądarce" not in text


def test_push_digest_carries_the_archive_link_within_its_limit() -> None:
    digest = build_messenger_digest(RESULTS, AT, archive_url=URL)

    assert f"🔗 Pełny raport (link ważny 7 dni): {URL}" in digest.splitlines()
    assert len(digest) <= MAX_DIGEST_CHARS


def test_push_digest_without_archive_has_no_link_line() -> None:
    assert "🔗" not in build_messenger_digest(RESULTS, AT)
