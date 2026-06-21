"""Testy panelu 'lessons learned' (T4 — Mistake-Diagnosis Audit Trail).

build_lessons destyluje z zamkniętych predykcji wyłącznie BŁĘDY z diagnozą
(insight), zachowuje kolejność wejścia (most-recent-first), liczy całkowitą
liczbę błędów przed limitem i wykrywa powracający motyw (keyword)."""

from __future__ import annotations

from decimal import Decimal

from src.application.report_lessons import (
    LessonItem,
    LessonsReport,
    build_lessons,
)
from src.application.report_models import ResolvedPrediction


def _resolved(
    symbol: str,
    *,
    is_correct: bool,
    insight: str | None,
    price_at_prediction: Decimal | None = None,
    actual_price: Decimal | None = None,
) -> ResolvedPrediction:
    return ResolvedPrediction(
        symbol=symbol,
        predicted_trend="up",
        is_correct=is_correct,
        insight=insight,
        price_at_prediction=price_at_prediction,
        actual_price=actual_price,
    )


def test_keeps_only_mistakes_with_insight() -> None:
    """2 trafne + 3 błędne(z insightem) + 1 błędny(pusty insight) → 3 błędy,
    items=3 (pusty insight wykluczony)."""
    resolved = [
        _resolved("AAA", is_correct=True, insight="trafny przypadek"),
        _resolved("BBB", is_correct=True, insight=None),
        _resolved("CCC", is_correct=False, insight="rynek zaskoczył reakcją"),
        _resolved("DDD", is_correct=False, insight="rynek zignorował dane"),
        _resolved("EEE", is_correct=False, insight="rynek wycenił inaczej"),
        _resolved("FFF", is_correct=False, insight="   "),  # pusty po stripie
    ]

    report = build_lessons(resolved)

    assert report.total_mistakes == 3
    assert len(report.items) == 3
    assert {item.symbol for item in report.items} == {"CCC", "DDD", "EEE"}
    assert all(isinstance(item, LessonItem) for item in report.items)


def test_preserves_most_recent_first_order() -> None:
    """Kolejność wejścia (most-recent-first) jest zachowana w items."""
    resolved = [
        _resolved("CCC", is_correct=False, insight="powód trzeci"),
        _resolved("DDD", is_correct=False, insight="powód drugi"),
        _resolved("EEE", is_correct=False, insight="powód pierwszy"),
    ]

    report = build_lessons(resolved)

    assert [item.symbol for item in report.items] == ["CCC", "DDD", "EEE"]


def test_limit_caps_items_but_total_counts_all() -> None:
    """limit przycina items, ale total_mistakes liczy wszystkie błędy."""
    resolved = [
        _resolved(f"S{i}", is_correct=False, insight=f"diagnoza numer {i}")
        for i in range(10)
    ]

    report = build_lessons(resolved, limit=3)

    assert report.total_mistakes == 10
    assert len(report.items) == 3
    assert [item.symbol for item in report.items] == ["S0", "S1", "S2"]


def test_recurring_theme_picks_repeated_keyword() -> None:
    """recurring_theme wybiera najczęstszy nietrywialny keyword z insightów."""
    resolved = [
        _resolved("AAA", is_correct=False, insight="inflacja zaskoczyła rynek"),
        _resolved("BBB", is_correct=False, insight="inflacja przebiła prognozy"),
        _resolved("CCC", is_correct=False, insight="inflacja ponownie wyższa"),
    ]

    report = build_lessons(resolved)

    assert report.recurring_theme == "inflacja"


def test_recurring_theme_ignores_polish_stopwords() -> None:
    """Stopwordy (i/w/na/się/...) nie mogą wygrać jako motyw."""
    resolved = [
        _resolved("AAA", is_correct=False, insight="dane i prognozy na rynku"),
        _resolved("BBB", is_correct=False, insight="dane na sesji się rozjechały"),
    ]

    report = build_lessons(resolved)

    # "na" i "się" to stopwordy / za krótkie → musi wygrać "dane".
    assert report.recurring_theme == "dane"


def test_all_correct_yields_empty_report() -> None:
    """Brak błędów → pusty raport (krotka, 0, None)."""
    resolved = [
        _resolved("AAA", is_correct=True, insight="trafione"),
        _resolved("BBB", is_correct=True, insight="trafione drugie"),
    ]

    report = build_lessons(resolved)

    assert report == LessonsReport((), 0, None)


def test_empty_input_yields_empty_report() -> None:
    """Pusta lista → pusty raport."""
    assert build_lessons([]) == LessonsReport((), 0, None)


def test_lesson_item_carries_actual_change_pct() -> None:
    """LessonItem niesie predicted_trend, insight i actual_change_pct z predykcji."""
    resolved = [
        _resolved(
            "AAA",
            is_correct=False,
            insight="źle oszacowany kierunek",
            price_at_prediction=Decimal("100"),
            actual_price=Decimal("90"),
        ),
    ]

    report = build_lessons(resolved)

    item = report.items[0]
    assert item.symbol == "AAA"
    assert item.predicted_trend == "up"
    assert item.insight == "źle oszacowany kierunek"
    assert item.actual_change_pct == Decimal("-0.1")


def test_recurring_theme_ignores_pure_digit_runs() -> None:
    """Powtarzający się token czysto cyfrowy (np. rok '2024') NIE może wygrać
    jako motyw — szum numeryczny w diagnozach nie jest 'lekcją'. Prawdziwe
    powtarzające się słowo wygrywa mimo częstszych cyfr."""
    resolved = [
        _resolved("AAA", is_correct=False, insight="raport 2024 pokazał spadek"),
        _resolved("BBB", is_correct=False, insight="dane 2024 zaskoczyły spadek"),
    ]

    report = build_lessons(resolved)

    # "2024" pojawia się 2× (jak "spadek"), ale czyste cyfry są odrzucane →
    # motywem musi być realne słowo, nie rok.
    assert report.recurring_theme == "spadek"


def test_no_clear_recurring_winner_returns_none() -> None:
    """Brak powtórzeń keywordów → recurring_theme None."""
    resolved = [
        _resolved("AAA", is_correct=False, insight="alfa beta gamma"),
        _resolved("BBB", is_correct=False, insight="delta epsilon zeta"),
    ]

    report = build_lessons(resolved)

    assert report.recurring_theme is None
