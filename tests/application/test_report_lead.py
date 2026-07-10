"""Testy renderera sekcji leadu — „Lead 5 sekund" (pierwszy ekran maila)."""

from __future__ import annotations

from src.application.report_lead import render_lead_html, render_lead_text
from src.domain.digest_lead import LeadItem


def _item(icon: str, headline: str, detail: str) -> LeadItem:
    return LeadItem(icon=icon, headline=headline, detail=detail)


class TestRenderLeadHtml:
    def test_empty_list_suppresses_section(self) -> None:
        # Sekcja samosupresująca: pusty lead = brak sekcji, nie pusty nagłówek.
        assert render_lead_html([]) == ""

    def test_renders_headline_and_detail(self) -> None:
        html = render_lead_html(
            [_item("🔥", "AAPL +4.2%, rada zgodna: BUY", "Konsensus 6/7")]
        )

        assert "AAPL +4.2%, rada zgodna: BUY" in html
        assert "Konsensus 6/7" in html
        assert "🔥" in html

    def test_escapes_headline_icon_and_detail(self) -> None:
        # Treść pochodzi pośrednio z newsów — musi być escapowana.
        html = render_lead_html(
            [
                _item(
                    "<b>",
                    "<script>alert(1)</script>",
                    "detail & <img>",
                )
            ]
        )

        assert "<script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
        assert "&lt;b&gt;" in html
        assert "detail &amp; &lt;img&gt;" in html

    def test_empty_detail_is_omitted_gracefully(self) -> None:
        # `detail` bywa puste (np. quota alert bez akcji) — render nie może paść.
        html = render_lead_html([_item("🚨", "Coś się pali", "")])

        assert "Coś się pali" in html
        assert html != ""

    def test_preserves_domain_order(self) -> None:
        # KONTRAKT: renderer NIE zmienia kolejności — priorytety ustalił
        # `build_lead` (CRITICAL quota > split rady > silny konsensus > werdykt).
        html = render_lead_html(
            [
                _item("🚨", "QUOTA: limit LLM wyczerpany", "Odśwież klucz"),
                _item("⚖️", "TSLA -6.1%, rada PODZIELONA", "3 BUY / 4 SELL"),
                _item("🔥", "AAPL +4.2%, rada zgodna: BUY", "Konsensus 6/7"),
            ]
        )

        assert (
            html.index("QUOTA: limit LLM wyczerpany")
            < html.index("TSLA -6.1%, rada PODZIELONA")
            < html.index("AAPL +4.2%, rada zgodna: BUY")
        )


class TestRenderLeadText:
    def test_empty_list_suppresses_section(self) -> None:
        assert render_lead_text([]) == ""

    def test_renders_headline_and_detail(self) -> None:
        text = render_lead_text(
            [_item("🔥", "AAPL +4.2%, rada zgodna: BUY", "Konsensus 6/7")]
        )

        assert "AAPL +4.2%, rada zgodna: BUY" in text
        assert "Konsensus 6/7" in text
        assert "🔥" in text

    def test_does_not_escape_plain_text(self) -> None:
        # Plain-text nie jest HTML — brak encji, treść czytelna wprost.
        text = render_lead_text([_item("⚖️", "TSLA & SPY", "3 BUY / 4 SELL")])

        assert "TSLA & SPY" in text
        assert "&amp;" not in text

    def test_empty_detail_is_omitted_gracefully(self) -> None:
        text = render_lead_text([_item("🚨", "Coś się pali", "")])

        assert "Coś się pali" in text
        assert text != ""

    def test_preserves_domain_order(self) -> None:
        # KONTRAKT (jak w HTML): kolejność pozycji = kolejność z domeny.
        text = render_lead_text(
            [
                _item("🚨", "QUOTA: limit LLM wyczerpany", "Odśwież klucz"),
                _item("⚖️", "TSLA -6.1%, rada PODZIELONA", "3 BUY / 4 SELL"),
                _item("🔥", "AAPL +4.2%, rada zgodna: BUY", "Konsensus 6/7"),
            ]
        )

        assert (
            text.index("QUOTA: limit LLM wyczerpany")
            < text.index("TSLA -6.1%, rada PODZIELONA")
            < text.index("AAPL +4.2%, rada zgodna: BUY")
        )
