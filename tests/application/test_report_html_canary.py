"""Kanarek HTML w pełnym renderze maila (TEST-26).

Reguła „tekst z zewnątrz idzie do HTML-a wyłącznie escapowany” żyła dotąd tylko w
komentarzach modułów sekcji. Ten test wstrzykuje kanarka — `<script>` i próbę
wyjścia z atrybutu z handlerem zdarzenia — w każde pole tekstowe, które pochodzi
od LLM-a, dostawcy danych albo z wyjątku, i renderuje CAŁY mail (`build_html_report`
z pełnym kontekstem scenariusza snapshotu). Parser HTML nie może znaleźć ani
elementu `script`, ani atrybutu `on*`, a escapowany kanarek musi być w treści —
inaczej pole w ogóle nie dotarło do maila i przypadek niczego nie sprawdza.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from datetime import timedelta
from decimal import Decimal
from html.parser import HTMLParser
from typing import Any

import pytest

from src.application.report_builder import (
    ReportContext,
    ResolvedPrediction,
    SymbolResult,
    build_html_report,
    to_symbol_result,
)
from src.application.report_council_history import InvestorHistory
from src.application.report_lessons import LessonItem, LessonsReport
from src.domain.council import CouncilVerdict, InvestorOpinion
from src.domain.persona_track_record import PersonaTrackRecord
from src.domain.quota import QuotaAlert, QuotaSeverity
from src.domain.value_objects import Ratio
from tests.application.test_report_builder_snapshot import (
    DURATION_SECONDS,
    STARTED,
    _full_context,
    _mapped,
    _saved_stock_raw,
)

CANARY = "\"'><script>alert('TEST-26')</script><img src=x onerror=alert(26)>"
ESCAPED_MARKER = "&lt;script&gt;alert("

Cycle = tuple[list[SymbolResult], ReportContext]


class _ActiveMarkup(HTMLParser):
    """Zbiera to, co przeglądarka wykonałaby z kanarka: elementy `script`
    i atrybuty handlerów zdarzeń (`onerror`, `onload`, …)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.found: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            self.found.append("<script>")
        self.found.extend(f"<{tag} {name}=…>" for name, _ in attrs if name.startswith("on"))


def _active_markup(html: str) -> list[str]:
    parser = _ActiveMarkup()
    parser.feed(html)
    parser.close()
    return parser.found


def _raw_case(*path: str | int) -> Callable[[], Cycle]:
    """Doklejenie kanarka do pola surowego wyjścia grafu (przez mapper wyników)."""

    def build() -> Cycle:
        raw = _saved_stock_raw()
        target: Any = raw
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = f"{target[path[-1]]}{CANARY}"
        return [_mapped("NVDA", raw)], _full_context()

    return build


def _council_case() -> Cycle:
    raw = _saved_stock_raw()
    raw["council_verdict"] = CouncilVerdict(
        final_recommendation="BUY",
        consensus_strength=0.7,
        summary=f"Rada radzi kupować {CANARY}",
        dissenting_views=(f"Soros: {CANARY}",),
        investor_opinions=(
            InvestorOpinion(
                investor_name=f"Buffett {CANARY}",
                recommendation="BUY",
                confidence=0.7,
                reasoning=f"Fosa {CANARY}",
                key_factors=(f"FCF {CANARY}",),
            ),
        ),
    )
    return [_mapped("NVDA", raw)], _full_context()


def _error_case() -> Cycle:
    failed = to_symbol_result("ASML", None, error=f"HTTPError: 429 {CANARY}", now=STARTED)
    return [_mapped("NVDA", _saved_stock_raw()), failed], _full_context()


def _context_case(**fields: Any) -> Callable[[], Cycle]:
    def build() -> Cycle:
        return [_mapped("NVDA", _saved_stock_raw())], dataclasses.replace(_full_context(), **fields)

    return build


CASES: dict[str, Callable[[], Cycle]] = {
    "llm_reasoning": _raw_case("llm_analysis", "reasoning"),
    "sentiment_label": _raw_case("sentiment", "av_sentiment_label"),
    "news_title": _raw_case("news", 0, "title"),
    "news_source": _raw_case("news", 0, "source"),
    "news_url": _raw_case("news", 0, "url"),
    "precedent_summary": _raw_case("similar_precedents", 0, "summary"),
    "reflection_context": _raw_case("reflection_context"),
    "council_verdict": _council_case,
    "resolved_prediction": _context_case(
        resolved_predictions=[
            ResolvedPrediction(
                symbol="AMD",
                predicted_trend="BULLISH",
                is_correct=False,
                reasoning=f"Odbicie {CANARY}",
                insight=f"Słabszy popyt {CANARY}",
                price_at_prediction=Decimal("170.00"),
                actual_price=Decimal("165.80"),
                predicted_at=STARTED - timedelta(days=1),
            )
        ]
    ),
    "quota_alert": _context_case(
        quota_alerts=[
            QuotaAlert(
                source=f"Alpha Vantage {CANARY}",
                severity=QuotaSeverity.WARNING,
                message=f"Klucz wyczerpany {CANARY}",
                action=f"Sprawdź limit {CANARY}",
                occurred_at=STARTED - timedelta(minutes=50),
            )
        ]
    ),
    "lessons": _context_case(
        lessons=LessonsReport(
            items=(
                LessonItem(
                    symbol="AMD",
                    predicted_trend="BULLISH",
                    insight=f"Słabszy popyt {CANARY}",
                    actual_change_ratio=Ratio(Decimal("-0.0247")),
                ),
            ),
            total_mistakes=1,
            recurring_theme=f"popyt {CANARY}",
        )
    ),
    "council_history": _context_case(
        council_history=[
            InvestorHistory(
                investor_name=f"Soros {CANARY}",
                symbol="NVDA",
                recommendations=("HOLD", "SELL", "SELL"),
            )
        ]
    ),
    "persona_track_record": _context_case(
        persona_track_record=[
            PersonaTrackRecord(investor_name=f"Soros {CANARY}", hit_rate=0.52, votes=21)
        ]
    ),
}


def test_mail_without_canary_has_no_active_markup() -> None:
    results = [_mapped("NVDA", _saved_stock_raw())]

    html, _ = build_html_report(results, STARTED, DURATION_SECONDS, _full_context())

    assert _active_markup(html) == []


def test_symbol_error_text_never_reaches_the_mail() -> None:
    """MAIL-32: surowy wyjątek symbolu zostaje w logach — mail pokazuje tylko kategorię,
    więc kanarek z treści błędu nie dociera do HTML-a w żadnej postaci."""
    results, context = _error_case()

    html, _ = build_html_report(results, STARTED, DURATION_SECONDS, context)

    assert _active_markup(html) == []
    assert ESCAPED_MARKER not in html
    assert "TEST-26" not in html


@pytest.mark.parametrize("case", sorted(CASES))
def test_injected_markup_reaches_the_mail_only_escaped(case: str) -> None:
    results, context = CASES[case]()

    html, _ = build_html_report(results, STARTED, DURATION_SECONDS, context)

    assert _active_markup(html) == [], f"{case}: nieescapowany kanarek w HTML-u maila"
    assert ESCAPED_MARKER in html, f"{case}: pole nie dotarło do HTML-a maila"
