"""Testy renderera sekcji "🔄 Zmiany nastawienia" (HTML + plain text).

Roadmap #8, część renderująca. Czysta logika prezentacji, bez I/O: bierze
już-sklasyfikowane obiekty domenowe `ConsensusShift` (para `(symbol, shift)`)
i produkuje fragment maila opisujący realny ruch narracji rady między cyklami.

Największe ryzyko feature'u to "fałszywa dramaturgia ze stęchłych danych":
cron Fast Loopa bywa zakomentowany, więc "wczoraj" potrafi być sprzed tygodnia.
Testy poniżej zamrażają, że:
  (a) same STABLE → sekcja znika (samosupresja),
  (b) STALE_COMPARISON renderuje się z JAWNYM wiekiem porównania w dniach,
  (c) FLIP pokazuje liczbę zmienionych głosów.
"""

from __future__ import annotations

from src.application.report_consensus_shift import (
    render_consensus_shift_html,
    render_consensus_shift_text,
)
from src.domain.consensus_shift import ConsensusShift, ShiftKind


def _shift(
    kind: ShiftKind,
    *,
    previous: str = "BUY",
    current: str = "SELL",
    votes_changed: int = 0,
    sentiment_delta: float = 0.0,
    consensus_delta: float = 0.0,
    gap_hours: float = 1.0,
) -> ConsensusShift:
    """Buduje obiekt zmiany nastawienia o zadanych parametrach."""
    return ConsensusShift(
        kind=kind,
        previous_recommendation=previous,
        current_recommendation=current,
        votes_changed=votes_changed,
        sentiment_delta=sentiment_delta,
        consensus_delta=consensus_delta,
        gap_hours=gap_hours,
    )


class TestEmptyAndSelfSuppression:
    def test_empty_list_returns_empty_html(self) -> None:
        assert render_consensus_shift_html([]) == ""

    def test_empty_list_returns_empty_text(self) -> None:
        assert render_consensus_shift_text([]) == ""

    def test_only_stable_returns_empty_html(self) -> None:
        # (a) Same STABLE = brak ruchu narracji → sekcja znika.
        shifts = [
            ("NVDA", _shift(ShiftKind.STABLE, previous="BUY", current="BUY")),
            ("TSLA", _shift(ShiftKind.STABLE, previous="HOLD", current="HOLD")),
        ]
        assert render_consensus_shift_html(shifts) == ""

    def test_only_stable_returns_empty_text(self) -> None:
        shifts = [("NVDA", _shift(ShiftKind.STABLE, previous="BUY", current="BUY"))]
        assert render_consensus_shift_text(shifts) == ""

    def test_stable_rows_filtered_out_when_mixed(self) -> None:
        # STABLE odfiltrowane, FLIP zostaje.
        shifts = [
            ("STABLESYM", _shift(ShiftKind.STABLE, previous="BUY", current="BUY")),
            (
                "FLIPSYM",
                _shift(ShiftKind.FLIP, previous="BUY", current="SELL", votes_changed=3),
            ),
        ]
        html = render_consensus_shift_html(shifts)
        assert "STABLESYM" not in html
        assert "FLIPSYM" in html


class TestStaleComparison:
    def test_stale_html_shows_age_in_days(self) -> None:
        # (b) 168h = 7 dni. Wiek MUSI być jawnie widoczny.
        shifts = [
            (
                "NVDA",
                _shift(
                    ShiftKind.STALE_COMPARISON,
                    previous="BUY",
                    current="SELL",
                    gap_hours=168.0,
                ),
            )
        ]
        html = render_consensus_shift_html(shifts)
        assert "7" in html
        assert "dni" in html.lower()
        assert "sprzed" in html.lower()

    def test_stale_text_shows_age_in_days(self) -> None:
        shifts = [
            (
                "NVDA",
                _shift(
                    ShiftKind.STALE_COMPARISON,
                    previous="BUY",
                    current="SELL",
                    gap_hours=168.0,
                ),
            )
        ]
        text = render_consensus_shift_text(shifts)
        assert "7" in text
        assert "dni" in text.lower()
        assert "sprzed" in text.lower()


class TestFlip:
    def test_flip_html_shows_votes_changed(self) -> None:
        # (c) FLIP musi pokazać liczbę zmienionych głosów.
        shifts = [
            (
                "NVDA",
                _shift(ShiftKind.FLIP, previous="BUY", current="SELL", votes_changed=3),
            )
        ]
        html = render_consensus_shift_html(shifts)
        assert "3" in html

    def test_flip_html_shows_direction(self) -> None:
        shifts = [
            (
                "NVDA",
                _shift(ShiftKind.FLIP, previous="BUY", current="SELL", votes_changed=2),
            )
        ]
        html = render_consensus_shift_html(shifts)
        assert "BUY" in html
        assert "SELL" in html

    def test_flip_text_shows_votes_and_direction(self) -> None:
        shifts = [
            (
                "NVDA",
                _shift(ShiftKind.FLIP, previous="SELL", current="BUY", votes_changed=4),
            )
        ]
        text = render_consensus_shift_text(shifts)
        assert "4" in text
        assert "SELL" in text
        assert "BUY" in text


class TestSoftening:
    def test_softening_renders_symbol(self) -> None:
        shifts = [
            (
                "TSLA",
                _shift(
                    ShiftKind.SOFTENING,
                    previous="BUY",
                    current="HOLD",
                    votes_changed=1,
                    consensus_delta=-0.3,
                ),
            )
        ]
        html = render_consensus_shift_html(shifts)
        assert "TSLA" in html
        assert "BUY" in html
        assert "HOLD" in html


class TestEscaping:
    def test_html_escapes_symbol(self) -> None:
        shifts = [
            (
                "<script>alert(1)</script>",
                _shift(ShiftKind.FLIP, previous="BUY", current="SELL", votes_changed=1),
            )
        ]
        html = render_consensus_shift_html(shifts)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_html_escapes_recommendation(self) -> None:
        shifts = [
            (
                "NVDA",
                _shift(
                    ShiftKind.FLIP,
                    previous="<b>BUY</b>",
                    current="SELL",
                    votes_changed=1,
                ),
            )
        ]
        html = render_consensus_shift_html(shifts)
        assert "<b>BUY</b>" not in html
        assert "&lt;b&gt;" in html
