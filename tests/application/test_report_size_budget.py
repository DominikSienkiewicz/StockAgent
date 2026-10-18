"""Budżet rozmiaru maila (MAIL-06): Gmail przycina wiadomości powyżej ~102 KB i chowa
resztę pod „[Wiadomość przycięta]” — razem ze stopką i zastrzeżeniem."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.application.report_builder import SymbolResult, TopNewsItem, build_html_report
from src.domain.council import CouncilVerdict, InvestorOpinion

PERSONAS = ("Buffett", "Graham", "Lynch", "Dalio", "Soros", "Wood", "Marks")
STARTED = datetime(2026, 10, 18, 5, 30, tzinfo=UTC)


def _saved(i: int) -> SymbolResult:
    council = CouncilVerdict(
        final_recommendation="BUY",
        consensus_strength=0.71,
        summary="Większość rady widzi przewagę konkurencyjną i rozsądną wycenę. " * 2,
        dissenting_views=("Soros: rynek przegrzany, możliwa korekta po wynikach.",),
        investor_opinions=tuple(
            InvestorOpinion(
                investor_name=persona,
                recommendation="BUY",
                confidence=0.7,
                reasoning="Silne przepływy pieniężne, stabilne marże i przewaga skali. " * 2,
                key_factors=("moat", "FCF", "wycena"),
            )
            for persona in PERSONAS
        ),
    )
    return SymbolResult(
        symbol=f"SYM{i}",
        status="saved",
        delta=Decimal("0.031"),
        current_price=Decimal("123.45"),
        trend="BULLISH",
        target_price=Decimal("125.10"),
        confidence_score=0.72,
        reasoning="Wyniki kwartalne powyżej oczekiwań i pozytywny sentyment newsów. " * 4,
        sentiment_score=0.35,
        sentiment_label="Somewhat-Bullish",
        news_volume=12,
        top_news=[
            TopNewsItem(
                title=f"Headline {k} about company {i} beating expectations",
                url=f"https://news.example.com/{i}/{k}",
                source="Reuters",
                relevance=0.8,
                sentiment=0.3,
            )
            for k in range(5)
        ],
        council_verdict=council,
    )


def _size(results: list[SymbolResult]) -> int:
    html, _ = build_html_report(results, STARTED, 60.0)
    return len(html.encode("utf-8"))


def test_markup_whitespace_between_tags_is_collapsed() -> None:
    # Wcięcia szablonów to kilka procent maila, a w HTML bez <pre> jedna spacja
    # renderuje się tak samo jak cały ciąg białych znaków.
    html, _ = build_html_report([_saved(0)], STARTED, 60.0)
    assert not re.search(r">\s{2,}<", html)


def test_typical_cycle_with_council_stays_under_the_budget() -> None:
    assert _size([_saved(i) for i in range(8)]) <= 95_000


@pytest.mark.xfail(
    strict=True,
    reason="MAIL-06 (b)/(d): zwinięta rada i degradacja top-N czekają na snapshot TEST-02",
)
def test_big_move_day_stays_under_gmail_clipping() -> None:
    assert _size([_saved(i) for i in range(45)]) < 100_000
