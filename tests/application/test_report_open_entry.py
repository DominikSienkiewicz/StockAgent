"""WIA-54: druga krzywa kapitału „od otwarcia” — cena wykonalna zamiast ceny
z chwili predykcji. Fast Loop chodzi przed sesją, więc cena predykcji to ostatnie
zamknięcie, a wynik od niej obejmuje lukę na otwarciu, której nikt przed sesją nie
kupi. Sekcja pokazuje obie krzywe na tych samych transakcjach."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from src.application.report_evidence import PaperCosts
from src.application.report_models import ResolvedPrediction
from src.application.report_open_entry import (
    OPEN_ENTRY_SECTION,
    build_open_entry_curve,
    render_open_entry_html,
    render_open_entry_text,
)


def _resolved(
    symbol: str,
    trend: str,
    *,
    entry: str | None,
    day: int = 19,
    price: str = "100",
    actual: str = "106",
) -> ResolvedPrediction:
    return ResolvedPrediction(
        symbol=symbol,
        predicted_trend=trend,
        is_correct=True,
        price_at_prediction=Decimal(price),
        actual_price=Decimal(actual),
        predicted_at=datetime(2026, 10, day, 5, 30, tzinfo=UTC),
        entry_price=Decimal(entry) if entry is not None else None,
    )


def test_resolved_prediction_executable_change_excludes_opening_gap() -> None:
    # Zamknięcie 100, luka na otwarciu do 105, rozliczenie 106: od predykcji +6%,
    # a wykonalnie — od otwarcia — tylko ~+0,95%.
    prediction = _resolved("NVDA", "BULLISH", entry="105")

    assert prediction.actual_change_ratio == pytest.approx(Decimal("0.06"))
    assert prediction.executable_change == pytest.approx(Decimal("1") / Decimal("105"))


def test_prediction_without_an_executable_entry_has_no_executable_change() -> None:
    assert _resolved("BTC", "BULLISH", entry=None).executable_change is None


def test_curve_compares_both_entries_on_the_same_trades() -> None:
    curve = build_open_entry_curve(
        [
            _resolved("NVDA", "BULLISH", entry="105"),
            _resolved("AAPL", "BEARISH", entry="104", actual="102"),
            _resolved("BTC", "BULLISH", entry=None),
        ]
    )

    assert curve is not None
    assert curve.with_entry == 2
    assert curve.settled == 3
    # Jeden cykl, równa waga: od otwarcia (1/105 + 2/104) / 2, od predykcji (6% − 2%) / 2.
    assert curve.from_open.total_return == pytest.approx((1 / 105 + 2 / 104) / 2)
    assert curve.from_prediction.total_return == pytest.approx((0.06 - 0.02) / 2)


def test_no_executable_entry_no_curve() -> None:
    assert build_open_entry_curve([_resolved("BTC", "BULLISH", entry=None)]) is None


def test_costs_are_charged_on_both_curves() -> None:
    costs = PaperCosts(cost_bps=10.0)

    curve = build_open_entry_curve([_resolved("NVDA", "BULLISH", entry="105")], costs)

    assert curve is not None
    assert curve.from_open.total_return == pytest.approx(1 / 105 - 0.001)
    assert curve.from_prediction.total_return == pytest.approx(0.06 - 0.001)
    assert curve.costs == costs


def test_section_renders_both_numbers_and_the_coverage() -> None:
    curve = build_open_entry_curve(
        [_resolved("NVDA", "BULLISH", entry="105"), _resolved("BTC", "BULLISH", entry=None)]
    )
    assert curve is not None

    html = render_open_entry_html(curve)
    text = "\n".join(render_open_entry_text(curve))

    for rendered in (html, text):
        assert "Od otwarcia sesji" in rendered
        assert "+1,0%" in rendered
        assert "+6,0%" in rendered
        assert "1 z 2" in rendered


def test_section_hides_without_a_curve() -> None:
    from types import SimpleNamespace

    data = SimpleNamespace(context=SimpleNamespace(open_entry_curve=None))

    assert OPEN_ENTRY_SECTION.render_html(data) == ""  # type: ignore[arg-type]
    assert OPEN_ENTRY_SECTION.render_text(data) == []  # type: ignore[arg-type]
