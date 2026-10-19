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


# ---------------------------------------------------------------------------
# MAIL-06 (b): rada spoza top 10 sygnałów (i spoza leadu) zwinięta do jednej linii
# ---------------------------------------------------------------------------


def _ranked(i: int, *, split: bool = False, summary: str | None = None) -> SymbolResult:
    """Symbol o sile sygnału malejącej z `i` (prognoza +(20 - i)%, ta sama pewność).

    Rada bez silnego konsensusu i bez sprzecznych głosów nie trafia do leadu;
    `split=True` dokłada głos SELL, więc rada jest podzielona i symbol wchodzi do leadu."""
    symbol = f"R{i:02d}"
    opinions = [
        InvestorOpinion(
            investor_name=persona,
            recommendation="BUY",
            confidence=0.6,
            reasoning="Przewaga skali.",
            key_factors=(f"czynnik-{symbol}",),
        )
        for persona in ("Buffett", "Lynch")
    ]
    if split:
        opinions.append(
            InvestorOpinion(
                investor_name="Soros",
                recommendation="SELL",
                confidence=0.6,
                reasoning="Przegrzanie.",
                key_factors=(f"czynnik-{symbol}",),
            )
        )
    council = CouncilVerdict(
        final_recommendation="BUY",
        consensus_strength=0.6,
        summary=summary or f"Podsumowanie rady {symbol}.",
        dissenting_views=(f"Soros o {symbol}: za drogo.",),
        investor_opinions=tuple(opinions),
    )
    return SymbolResult(
        symbol=symbol,
        status="saved",
        delta=Decimal("0.03"),
        current_price=Decimal("100"),
        trend="BULLISH",
        target_price=Decimal(120 - i),
        confidence_score=0.7,
        reasoning=f"Teza modelu dla {symbol}.",
        council_verdict=council,
    )


def _html_of(results: list[SymbolResult]) -> str:
    html, _ = build_html_report(results, STARTED, 60.0)
    return html


def test_council_outside_the_top_signals_collapses_to_one_line() -> None:
    html = _html_of([_ranked(i) for i in range(12)])

    # #10 w rankingu ma pełną tabelę person, #11 tylko jedną linię rady.
    assert "czynnik-R09" in html
    assert "czynnik-R10" not in html
    assert "🟢 2 KUP · 🔴 0 SPRZEDAJ · 🟡 0 TRZYMAJ — „Podsumowanie rady R10.”" in html


def test_collapsed_council_keeps_the_dissenting_views_visible() -> None:
    html = _html_of([_ranked(i) for i in range(12)])

    assert "⚠️ Głosy niezgodne: Soros o R11: za drogo." in html


def test_lead_symbol_keeps_the_full_council_outside_the_top_signals() -> None:
    html = _html_of([*(_ranked(i) for i in range(11)), _ranked(11, split=True)])

    assert "czynnik-R11" in html
    assert "czynnik-R10" not in html


def test_cycle_within_the_top_signals_keeps_every_full_council() -> None:
    html = _html_of([_ranked(i) for i in range(10)])

    assert all(f"czynnik-R{i:02d}" in html for i in range(10))


def test_collapsed_council_escapes_the_llm_summary() -> None:
    payload = "<img src=x onerror=alert(1)>"
    results = [*(_ranked(i) for i in range(11)), _ranked(11, summary=payload)]

    html = _html_of(results)

    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "<img" not in html


# ---------------------------------------------------------------------------
# MAIL-06 (d): pełne bloki uzasadnień tylko dla top 10 (i leadu), reszta w skrócie
# ---------------------------------------------------------------------------

COMPACT_HEADING = "📋 Pozostałe predykcje (skrót)"


def _compact_section(html: str) -> str:
    """Skrót od nagłówka do końca jego tabeli — dalej Zaplecze wymienia każdy symbol."""
    start = html.index(COMPACT_HEADING)
    return html[start : html.index("</table>", start)]


def test_reasoning_outside_the_top_signals_becomes_a_compact_row() -> None:
    html = _html_of([_ranked(i) for i in range(12)])

    compact = _compact_section(html)
    assert "Teza modelu dla R09." in html
    assert "Teza modelu dla R10." not in html
    assert "R10" in compact
    assert "Podsumowanie rady R10." in compact
    assert "⚠️ Głosy niezgodne: Soros o R10: za drogo." in compact
    assert "R09" not in compact


def test_lead_symbol_outside_the_top_signals_keeps_its_reasoning_block() -> None:
    html = _html_of([*(_ranked(i) for i in range(11)), _ranked(11, split=True)])

    assert "Teza modelu dla R11." in html
    assert "R11" not in _compact_section(html)


def test_cycle_within_the_top_signals_has_no_compact_rows() -> None:
    html = _html_of([_ranked(i) for i in range(10)])

    assert COMPACT_HEADING not in html
    assert all(f"Teza modelu dla R{i:02d}." in html for i in range(10))
