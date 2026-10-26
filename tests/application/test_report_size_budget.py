"""Budżet rozmiaru maila (MAIL-06): Gmail przycina wiadomości powyżej ~102 KB i chowa
resztę pod „[Wiadomość przycięta]” — razem ze stopką i zastrzeżeniem."""

from __future__ import annotations

import dataclasses
import re
from datetime import UTC, datetime
from decimal import Decimal

from src.application.report_builder import (
    ReportContext,
    SymbolResult,
    TopNewsItem,
    build_html_report,
)
from src.application.report_evidence import PaperCosts
from src.application.report_track_record import TrackRecordStats
from src.domain.benchmark import compare_with_benchmarks
from src.domain.council import CouncilVerdict, InvestorOpinion
from src.domain.equity_curve import PaperTrade
from src.domain.value_objects import Ratio

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


def test_big_move_day_stays_under_gmail_clipping() -> None:
    assert _size([_saved(i) for i in range(45)]) < 100_000


def _track_record_context() -> ReportContext:
    """WIA-21: pełny Track Record — 30 cykli, koszty, koszyk i IVV w sparkline."""
    cycles = [
        [
            PaperTrade(
                f"SYM{i}",
                "BULLISH" if (day + i) % 3 else "BEARISH",
                0.004 * ((day * 7 + i) % 5 - 2),
                10,
                benchmark_change=0.001 * (day % 4 - 1),
            )
            for i in range(8)
        ]
        for day in range(30)
    ]
    comparison = compare_with_benchmarks(cycles, index_symbol="IVV", index_cost_bps=10)
    return ReportContext(
        equity_curve=comparison.agent,
        track_record_stats=TrackRecordStats(costs=PaperCosts(cost_bps=10), benchmark=comparison),
    )


def test_typical_cycle_with_the_benchmark_track_record_stays_under_the_budget() -> None:
    html, _ = build_html_report(
        [_saved(i) for i in range(8)], STARTED, 60.0, _track_record_context()
    )

    assert len(html.encode("utf-8")) <= 95_000


def test_benchmark_line_and_second_sparkline_row_stay_within_4_kb() -> None:
    # Druga linia sparklini i linia „na tle rynku” nie mogą przewrócić budżetu
    # dużego dnia — ich koszt mierzymy wprost, niezależnie od reszty maila.
    with_benchmark = _track_record_context()
    stats = with_benchmark.track_record_stats
    assert stats is not None
    without = dataclasses.replace(
        with_benchmark, track_record_stats=dataclasses.replace(stats, benchmark=None)
    )
    results = [_saved(i) for i in range(45)]

    html, _ = build_html_report(results, STARTED, 60.0, with_benchmark)
    baseline, _ = build_html_report(results, STARTED, 60.0, without)

    assert len(html.encode("utf-8")) - len(baseline.encode("utf-8")) <= 4_000


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

    # Dziesiąty sygnał rankingu ma pełną tabelę person, jedenasty tylko jedną linię rady.
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


# ---------------------------------------------------------------------------
# MAIL-06 (c): powtarzalne style w klasach arkusza, kluczowe elementy inline
# ---------------------------------------------------------------------------


def test_footer_and_disclaimer_stay_within_the_first_100_kb_on_a_big_day() -> None:
    html = _html_of([_saved(i) for i in range(45)])

    for marker in ("nie stanowi rekomendacji", "raport automatyczny"):
        assert len(html[: html.index(marker)].encode("utf-8")) < 100_000


def _class_names(html: str) -> set[str]:
    attributes = re.findall(r"class=[\"']([^\"']+)[\"']", html)
    return {name for attr in attributes for name in attr.split()}


def _stylesheet_classes(html: str) -> set[str]:
    sheet = html[html.index("<style>") : html.index("</style>")]
    return set(re.findall(r"\.([a-z][a-z0-9]*)\{", sheet))


def test_every_class_in_the_mail_has_a_rule_in_the_stylesheet() -> None:
    html = _html_of([*(_saved(i) for i in range(45)), _ranked(0, split=True)])

    assert _class_names(html)
    assert _class_names(html) <= _stylesheet_classes(html)


def test_first_screen_and_signals_keep_inline_styles() -> None:
    # Gmail w aplikacji dla kont spoza Google pomija <style>: pierwszy ekran,
    # lead i karty sygnałów muszą wyglądać tak samo bez arkusza.
    html = _html_of([_saved(i) for i in range(12)])

    first_screen = html[html.index("</style>") : html.index("🔮 Wygenerowane predykcje")]
    assert "Najsilniejsze sygnały" in first_screen
    assert "class=" not in first_screen
    footer = html[html.index("nie stanowi rekomendacji") - 300 :]
    assert "class=" not in footer


def _full_track_record_context() -> ReportContext:
    """Pełny Track Record: krzywa z kosztem, benchmark, dowód trafności, Brier,
    pokrycie przedziałów, kalibracja i lekcje."""
    from decimal import Decimal

    from src.application.report_lessons import LessonItem, LessonsReport
    from src.domain.calibration_curve import CalibrationBucket, probability_scores
    from src.domain.conformal import IntervalCoverage
    from src.domain.track_record import ScoredCall, assess_directional_hit_rate

    base = _track_record_context()
    assert base.track_record_stats is not None
    calls = [
        ScoredCall(
            cycle=f"2026-09-{day + 1:02d}",
            predicted_trend="BULLISH" if (day + i) % 3 else "BEARISH",
            is_correct=(day + i) % 2 == 0,
            realized_change=0.01 if (day + i) % 2 == 0 else -0.01,
        )
        for day in range(30)
        for i in range(8)
    ]
    stats = dataclasses.replace(
        base.track_record_stats,
        hit_rate=assess_directional_hit_rate(calls),
        scores=probability_scores([(0.6, True)] * 120 + [(0.6, False)] * 120),
        coverage=IntervalCoverage(hits=150, n=200),
    )
    return dataclasses.replace(
        base,
        track_record_stats=stats,
        calibration_buckets=[
            CalibrationBucket(
                lower=k / 10,
                upper=(k + 1) / 10,
                count=24,
                mean_confidence=k / 10 + 0.05,
                hit_rate=0.5,
            )
            for k in range(5, 10)
        ],
        lessons=LessonsReport(
            items=tuple(
                LessonItem(
                    symbol=f"SYM{i}",
                    predicted_trend="BULLISH",
                    insight="Rynek wycenił słabszy popyt w segmencie klienckim. " * 2,
                    actual_change_ratio=Ratio(Decimal("-0.0247")),
                )
                for i in range(5)
            ),
            total_mistakes=5,
            recurring_theme="popyt",
        ),
    )


def test_big_move_day_with_the_full_track_record_stays_under_gmail_clipping() -> None:
    # Decyzja 2026-10-19: w duży dzień Track Record jest zwarty, a zastrzeżenie
    # inwestycyjne mieści się w pierwszych 100 KB, przed przycięciem Gmaila.
    html, _ = build_html_report(
        [_saved(i) for i in range(45)], STARTED, 60.0, _full_track_record_context()
    )

    assert len(html.encode("utf-8")) < 100_000
    assert len(html[: html.index("nie stanowi rekomendacji")].encode("utf-8")) < 100_000


def test_big_move_day_renders_the_compact_track_record() -> None:
    html, _ = build_html_report(
        [_saved(i) for i in range(11)], STARTED, 60.0, _full_track_record_context()
    )

    assert "wersja zwarta" in html
    assert "paper-PnL (koszt 10 bps)" in html
    assert "Na tle rynku: <strong>agent " in html
    assert "Trafność kierunkowa" in html
    # Bez sparklinii (druga linia koszyka), kalibracji i lekcji.
    assert "background:#9ca3af" not in html
    assert "Krzywa kalibracji" not in html
    assert "Lessons learned" not in html


def test_typical_day_keeps_the_full_track_record() -> None:
    # Próg jak w MAIL-06: do 10 zapisanych predykcji mail ma pełną wersję.
    html, _ = build_html_report(
        [_saved(i) for i in range(10)], STARTED, 60.0, _full_track_record_context()
    )

    assert "wersja zwarta" not in html
    assert "background:#9ca3af" in html
    assert "Lessons learned" in html
    assert "Krzywa kalibracji" in html
    assert len(html.encode("utf-8")) <= 95_000
