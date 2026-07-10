"""Testy renderu sekcji „Tydzień StockAgenta" (#17) — HTML i plain-text.

Zamrażają: samosupresję (recap None → ""), obecność czterech sekcji (strzał,
wtopa z lekcją, wybronieni dysydenci, delta ECE), escaping HTML i polską treść.
"""

from __future__ import annotations

from decimal import Decimal

from src.application.report_weekly_recap import (
    render_weekly_recap_html,
    render_weekly_recap_text,
)
from src.domain.council import InvestorOpinion
from src.domain.weekly_recap import RecapHighlight, WeeklyRecap


def _recap(
    *,
    best_insight: str = "",
    worst_insight: str = "Zignorowano ostrzeżenie",
    vindicated: tuple[InvestorOpinion, ...] = (),
    previous_ece: float | None = 0.20,
    ece_delta: float | None = -0.05,
) -> WeeklyRecap:
    best = RecapHighlight(
        symbol="NVDA",
        accuracy=Decimal("0.95"),
        correction_insight=best_insight,
    )
    worst = RecapHighlight(
        symbol="TSLA",
        accuracy=Decimal("0.10"),
        correction_insight=worst_insight,
        vindicated_dissenters=vindicated,
    )
    return WeeklyRecap(
        sample_size=7,
        best=best,
        worst=worst,
        current_ece=0.15,
        previous_ece=previous_ece,
        ece_delta=ece_delta,
    )


class TestWeeklyRecapHtml:
    def test_none_recap_is_self_suppressing(self) -> None:
        assert render_weekly_recap_html(None) == ""

    def test_contains_all_sections(self) -> None:
        soros = InvestorOpinion(
            investor_name="Soros",
            recommendation="SELL",
            confidence=0.9,
            reasoning="rynek przegrzany",
            key_factors=(),
        )
        html = render_weekly_recap_html(
            _recap(vindicated=(soros,)),
            narrative="Tydzień pełen zwrotów.",
        )
        assert "NVDA" in html
        assert "TSLA" in html
        assert "Zignorowano ostrzeżenie" in html
        assert "Soros" in html
        assert "Tydzień pełen zwrotów." in html
        # Delta ECE obecna (kalibracja).
        assert "ECE" in html

    def test_escapes_html_in_insight_and_narrative(self) -> None:
        html = render_weekly_recap_html(
            _recap(worst_insight="<script>alert('x')</script>"),
            narrative="<b>hej</b>",
        )
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "<b>hej</b>" not in html

    def test_no_previous_ece_still_renders(self) -> None:
        html = render_weekly_recap_html(
            _recap(previous_ece=None, ece_delta=None)
        )
        assert html != ""
        assert "TSLA" in html


class TestWeeklyRecapText:
    def test_none_recap_is_self_suppressing(self) -> None:
        assert render_weekly_recap_text(None) == ""

    def test_contains_heroes_and_narrative(self) -> None:
        text = render_weekly_recap_text(
            _recap(), narrative="Podsumowanie tygodnia."
        )
        assert "NVDA" in text
        assert "TSLA" in text
        assert "Podsumowanie tygodnia." in text
