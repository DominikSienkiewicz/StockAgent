"""ML-29 — rozkład P(BULLISH/BEARISH/SIDEWAYS) z odpowiedzi LLM: normalizacja,
argmax jako trend, max p jako pewność i fallback na stare pola odpowiedzi."""

from __future__ import annotations

import pytest

from src.domain.forecast import ProbabilityForecast, normalize_distribution
from src.domain.prediction import TrendDirection


def test_probabilities_are_renormalized_and_argmax_is_trend() -> None:
    forecast = ProbabilityForecast.from_llm(
        {"bullish": 0.6, " BEARISH ": 0.3, "Sideways": 0.3},
        fallback_trend=TrendDirection.SIDEWAYS,
        fallback_confidence=0.9,
    )

    assert forecast.trend is TrendDirection.BULLISH
    assert forecast.confidence == pytest.approx(0.5)
    assert forecast.as_dict() == pytest.approx(
        {"BULLISH": 0.5, "BEARISH": 0.25, "SIDEWAYS": 0.25}
    )
    assert sum(p for _, p in forecast.probabilities or ()) == pytest.approx(1.0)


def test_missing_probabilities_fall_back_to_trend_and_confidence() -> None:
    forecast = ProbabilityForecast.from_llm(
        None, fallback_trend=TrendDirection.BEARISH, fallback_confidence=0.72
    )

    assert forecast.trend is TrendDirection.BEARISH
    assert forecast.confidence == 0.72
    assert forecast.probabilities is None
    assert forecast.as_dict() is None


def test_negative_probabilities_are_clamped_to_zero() -> None:
    distribution = normalize_distribution({"BULLISH": 0.7, "BEARISH": -0.2, "SIDEWAYS": 0.3})

    assert distribution == pytest.approx(
        {TrendDirection.BULLISH: 0.7, TrendDirection.BEARISH: 0.0, TrendDirection.SIDEWAYS: 0.3}
    )


def test_percentages_and_a_missing_class_normalize_like_fractions() -> None:
    distribution = normalize_distribution({"BULLISH": 20, "BEARISH": 60})

    assert distribution == pytest.approx(
        {TrendDirection.BULLISH: 0.25, TrendDirection.BEARISH: 0.75, TrendDirection.SIDEWAYS: 0.0}
    )


@pytest.mark.parametrize(
    "raw",
    [
        "BULLISH",
        {},
        {"BULLISH": "high", "BEARISH": 0.2},
        {"UP": 0.6, "DOWN": 0.4},
        {"BULLISH": 0.0, "BEARISH": -1.0},
        {"BULLISH": float("nan"), "BEARISH": 0.5},
        {"BULLISH": True, "BEARISH": 0.5},
    ],
)
def test_unusable_distributions_fall_back(raw: object) -> None:
    forecast = ProbabilityForecast.from_llm(
        raw, fallback_trend=TrendDirection.BULLISH, fallback_confidence=0.6
    )

    assert (forecast.trend, forecast.confidence, forecast.probabilities) == (
        TrendDirection.BULLISH,
        0.6,
        None,
    )


def test_a_tie_resolves_to_sideways_first() -> None:
    forecast = ProbabilityForecast.from_llm(
        {"BULLISH": 0.4, "BEARISH": 0.2, "SIDEWAYS": 0.4},
        fallback_trend=TrendDirection.BULLISH,
        fallback_confidence=0.9,
    )

    assert forecast.trend is TrendDirection.SIDEWAYS
    assert forecast.confidence == pytest.approx(0.4)
