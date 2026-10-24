"""NIEZ-19: monitor rozkładu wyjść LLM — udział klas, średnia pewność, bias.

Cicha podmiana modelu u dostawcy albo edycja promptu zmienia zachowanie, a bez
monitora widać to dopiero w trafności po tygodniach. Rynek sam zmienia udział
wzrostów, więc flaga patrzy na bias (udział BULLISH − udział realnych wzrostów),
nie na surowy udział BULLISH.
"""

from __future__ import annotations

import math

import pytest

from src.domain.output_drift import (
    MIN_WINDOW_SIZE,
    OutputSample,
    chi_square_p_value,
    output_distribution_shift,
    summarize_window,
)


def _window(
    bullish: int,
    bearish: int,
    sideways: int,
    *,
    ups: int,
    confidence: float = 0.6,
    recipe: str | None = None,
) -> list[OutputSample]:
    trends = ["BULLISH"] * bullish + ["BEARISH"] * bearish + ["SIDEWAYS"] * sideways
    return [
        OutputSample(
            trend=trend,
            confidence=confidence,
            realized_up=index < ups,
            recipe_version=recipe,
        )
        for index, trend in enumerate(trends)
    ]


def test_shift_flagged_when_bullish_share_jumps_from_30_to_80_percent() -> None:
    previous = _window(30, 35, 35, ups=50)
    current = _window(80, 10, 10, ups=50)

    report = output_distribution_shift(previous, current)

    assert report.previous.n == 100 and report.current.n == 100
    assert report.p_value < 0.001
    assert report.bias_shift == pytest.approx(0.50)
    assert report.flagged
    assert "30% → 80%" in report.describe_pl()


def test_no_flag_for_identical_distributions() -> None:
    window = _window(40, 30, 30, ups=50)

    report = output_distribution_shift(window, list(window))

    assert report.chi_square == pytest.approx(0.0)
    assert report.p_value == pytest.approx(1.0)
    assert report.bias_shift == pytest.approx(0.0)
    assert not report.flagged


def test_bullish_share_following_the_market_is_not_a_shift() -> None:
    # Hossa: realne wzrosty 30% → 80% i BULLISH 30% → 80% — bias się nie rusza.
    previous = _window(30, 35, 35, ups=30)
    current = _window(80, 10, 10, ups=80)

    report = output_distribution_shift(previous, current)

    assert report.p_value < 0.001
    assert report.bias_shift == pytest.approx(0.0)
    assert not report.flagged


def test_confidence_jump_is_flagged_without_a_class_shift() -> None:
    previous = _window(40, 30, 30, ups=50, confidence=0.55)
    current = _window(40, 30, 30, ups=50, confidence=0.75)

    report = output_distribution_shift(previous, current)

    assert report.confidence_shift == pytest.approx(0.20)
    assert report.flagged


def test_small_windows_are_never_flagged() -> None:
    previous = _window(3, 4, 3, ups=5)
    current = _window(9, 1, 0, ups=5, confidence=0.95)

    report = output_distribution_shift(previous, current)

    assert report.current.n < MIN_WINDOW_SIZE
    assert not report.flagged
    assert "za mało" in report.describe_pl()


def test_flag_after_a_recipe_change_names_the_new_recipe() -> None:
    previous = _window(30, 35, 35, ups=50, recipe="r-old")
    current = _window(80, 10, 10, ups=50, recipe="r-new")

    report = output_distribution_shift(previous, current)

    assert report.recipe_changed
    assert report.new_recipes == ("r-new",)
    assert "zmiana zachowania po zmianie przepisu r-new" in report.describe_pl()


def test_flag_without_a_recipe_change_says_so() -> None:
    report = output_distribution_shift(
        _window(30, 35, 35, ups=50, recipe="r1"), _window(80, 10, 10, ups=50, recipe="r1")
    )

    assert not report.recipe_changed
    assert "bez zmiany przepisu" in report.describe_pl()


def test_recipe_change_is_not_claimed_when_the_previous_recipe_is_unknown() -> None:
    report = output_distribution_shift(
        _window(30, 35, 35, ups=50), _window(80, 10, 10, ups=50, recipe="r-new")
    )

    assert report.flagged
    assert not report.recipe_changed


def test_bias_is_unknown_without_settled_outcomes() -> None:
    samples = [OutputSample(trend="BULLISH", confidence=0.7, realized_up=None)] * 30

    summary = summarize_window(samples)

    assert summary.bullish_bias is None
    assert summary.mean_confidence == pytest.approx(0.7)


def test_unknown_trend_labels_are_ignored() -> None:
    summary = summarize_window(
        [OutputSample(trend="bullish"), OutputSample(trend="STRONG_BUY"), OutputSample("BEARISH")]
    )

    assert summary.n == 2
    assert dict(summary.counts) == {"BULLISH": 1, "BEARISH": 1, "SIDEWAYS": 0}


@pytest.mark.parametrize(
    ("statistic", "df", "expected"),
    [
        (5.991, 2, 0.05),  # tablicowa wartość krytyczna chi²(2) dla α = 0.05
        (3.841, 1, 0.05),  # chi²(1)
        (0.0, 2, 1.0),
    ],
)
def test_chi_square_p_value_matches_tables(statistic: float, df: int, expected: float) -> None:
    assert chi_square_p_value(statistic, df) == pytest.approx(expected, abs=1e-3)


def test_chi_square_without_degrees_of_freedom_is_not_significant() -> None:
    assert chi_square_p_value(12.0, 0) == 1.0
    assert math.isclose(chi_square_p_value(math.inf, 2), 0.0)
