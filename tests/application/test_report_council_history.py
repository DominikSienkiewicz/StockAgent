"""Testy renderera sekcji "Rada — historia głosów" (HTML + plain text).

Feature U5 — council-vote drill-down. Czysta logika prezentacji, bez I/O:
bierze już-zmaterializowany audit trail z `council_votes` (jako lista
`InvestorHistory`) i produkuje fragment maila "jak Soros głosował na NVDA
w ostatnich 30 dniach". Zero płatnych wywołań — render-only.
"""

from __future__ import annotations

from src.application.report_council_history import (
    InvestorHistory,
    render_council_history_html,
    render_council_history_text,
)


def _history(
    investor: str,
    symbol: str,
    recommendations: list[str],
    *,
    window_days: int = 30,
) -> InvestorHistory:
    """Buduje rekord historii: rekomendacje od najnowszej do najstarszej."""
    return InvestorHistory(
        investor_name=investor,
        symbol=symbol,
        recommendations=tuple(recommendations),  # type: ignore[arg-type]
        window_days=window_days,
    )


class TestEmptyHistory:
    def test_empty_list_returns_empty_html(self) -> None:
        assert render_council_history_html([]) == ""

    def test_empty_list_returns_empty_text(self) -> None:
        assert render_council_history_text([]) == ""


class TestHtmlRendering:
    def test_renders_section_header(self) -> None:
        html = render_council_history_html(
            [_history("Soros", "NVDA", ["SELL", "SELL", "HOLD"])]
        )
        assert "NVDA" in html
        assert "Soros" in html

    def test_renders_vote_sequence_in_order(self) -> None:
        html = render_council_history_html(
            [_history("Soros", "NVDA", ["SELL", "SELL", "HOLD"])]
        )
        # Sekwencja głosów (od najnowszego) widoczna w renderze.
        seq_pos = html.index("SELL")
        assert "SELL" in html
        assert "HOLD" in html
        # SELL pojawia się przed HOLD (kolejność od najnowszego).
        assert seq_pos < html.index("HOLD")

    def test_renders_buy_sell_hold_tally(self) -> None:
        # 2x SELL, 1x HOLD, 0x BUY.
        html = render_council_history_html(
            [_history("Soros", "NVDA", ["SELL", "SELL", "HOLD"])]
        )
        # Tally musi pokazać liczby dla każdej rekomendacji.
        assert "2" in html  # SELL count
        assert "1" in html  # HOLD count

    def test_renders_window_days(self) -> None:
        html = render_council_history_html(
            [_history("Soros", "NVDA", ["SELL", "SELL"], window_days=30)]
        )
        assert "30" in html

    def test_renders_multiple_investors(self) -> None:
        html = render_council_history_html(
            [
                _history("Soros", "NVDA", ["SELL", "SELL", "HOLD"]),
                _history("Wood", "NVDA", ["BUY", "BUY", "BUY"]),
            ]
        )
        assert "Soros" in html
        assert "Wood" in html

    def test_escapes_malicious_investor_name(self) -> None:
        html = render_council_history_html(
            [_history("<script>alert(1)</script>", "NVDA", ["BUY"])]
        )
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_escapes_malicious_symbol(self) -> None:
        html = render_council_history_html(
            [_history("Soros", "<img src=x onerror=1>", ["BUY"])]
        )
        assert "<img src=x onerror=1>" not in html
        assert "&lt;img" in html

    def test_history_with_no_recommendations_is_skipped(self) -> None:
        # Inwestor bez głosów w oknie → nie pojawia się jako pusty wiersz.
        html = render_council_history_html(
            [
                _history("Soros", "NVDA", []),
                _history("Wood", "NVDA", ["BUY"]),
            ]
        )
        assert "Soros" not in html
        assert "Wood" in html

    def test_all_histories_empty_returns_empty_string(self) -> None:
        # Self-suppressing: same puste historie też dają "".
        assert render_council_history_html([_history("Soros", "NVDA", [])]) == ""


class TestMostBearishMostBullish:
    def test_labels_most_bearish_voter(self) -> None:
        # Soros sam SELL, Wood sam BUY — Soros najbardziej bearish.
        html = render_council_history_html(
            [
                _history("Soros", "NVDA", ["SELL", "SELL", "SELL"]),
                _history("Wood", "NVDA", ["BUY", "BUY", "BUY"]),
            ]
        )
        assert "bearish" in html.lower()


class TestPlainTextRendering:
    def test_text_lists_investor_and_sequence(self) -> None:
        text = render_council_history_text(
            [_history("Soros", "NVDA", ["SELL", "SELL", "HOLD"])]
        )
        assert "Soros" in text
        assert "NVDA" in text
        assert "SELL" in text
        assert "HOLD" in text

    def test_text_shows_window_days(self) -> None:
        text = render_council_history_text(
            [_history("Soros", "NVDA", ["SELL"], window_days=30)]
        )
        assert "30" in text

    def test_text_skips_empty_history(self) -> None:
        text = render_council_history_text(
            [
                _history("Soros", "NVDA", []),
                _history("Wood", "NVDA", ["BUY"]),
            ]
        )
        assert "Soros" not in text
        assert "Wood" in text

    def test_text_empty_when_all_empty(self) -> None:
        assert render_council_history_text([_history("Soros", "NVDA", [])]) == ""
