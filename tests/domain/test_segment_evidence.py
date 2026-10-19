"""Testy domeny segmentacji dowodu trafności (WIA-55).

Każdy segment dostaje ten sam dowód co Track Record (n, Wilson na n_eff, naiwne
baseline'y na tych samych wierszach, p-wartość), a rodzina segmentów jednej
segmentacji jest korygowana BH-FDR — „najlepszy segment” to inaczej selekcja szumu.
"""

from __future__ import annotations

import pytest

from src.domain.segment_evidence import (
    COMPONENTS,
    MIN_SEGMENT_N,
    UNKNOWN_SEGMENT,
    assess_segments,
    confidence_bucket,
    council_trend,
    model_trend,
    sign_segment,
    trend_hit,
)
from src.domain.track_record import ScoredCall


def _call(cycle: int, trend: str, correct: bool, change: float) -> ScoredCall:
    return ScoredCall(
        cycle=f"2026-08-{cycle + 1:02d}",
        predicted_trend=trend,
        is_correct=correct,
        realized_change=change,
    )


class TestComponentDirections:
    @pytest.mark.parametrize(
        ("recommendation", "trend"),
        [("BUY", "BULLISH"), ("SELL", "BEARISH"), ("HOLD", "SIDEWAYS"), (" buy ", "BULLISH")],
    )
    def test_council_recommendation_maps_to_a_trend(self, recommendation, trend) -> None:
        assert council_trend(recommendation) == trend

    @pytest.mark.parametrize("recommendation", [None, "", "STRONG_BUY"])
    def test_missing_or_unknown_recommendation_is_no_call(self, recommendation) -> None:
        assert council_trend(recommendation) is None

    def test_model_trend_follows_the_target_price(self) -> None:
        assert model_trend(100.0, 101.0) == "BULLISH"
        assert model_trend(100.0, 99.0) == "BEARISH"

    def test_target_equal_to_price_is_the_cold_start_baseline_without_a_bet(self) -> None:
        assert model_trend(100.0, 100.0) == "SIDEWAYS"

    def test_missing_target_or_price_is_no_call(self) -> None:
        assert model_trend(100.0, None) is None
        assert model_trend(None, 101.0) is None

    def test_trend_hit_uses_the_prediction_rules(self) -> None:
        assert trend_hit("BULLISH", 0.01) is True
        assert trend_hit("BULLISH", 0.0) is False
        assert trend_hit("BEARISH", -0.01) is True
        assert trend_hit("SIDEWAYS", 0.004) is True
        assert trend_hit("SIDEWAYS", 0.006) is False

    def test_components_are_llm_council_and_model(self) -> None:
        assert COMPONENTS == ("llm", "council", "xgboost")


class TestSegmentKeys:
    @pytest.mark.parametrize(
        ("score", "bucket"),
        [
            (0.0, "0.00-0.60"),
            (0.59, "0.00-0.60"),
            (0.6, "0.60-0.70"),
            (0.7, "0.70-0.80"),
            (0.8, "0.80-1.00"),
            (1.0, "0.80-1.00"),
        ],
    )
    def test_confidence_bucket_boundaries(self, score, bucket) -> None:
        assert confidence_bucket(score) == bucket

    @pytest.mark.parametrize("score", [None, -0.1, 1.1, float("nan")])
    def test_invalid_confidence_is_unknown(self, score) -> None:
        assert confidence_bucket(score) == UNKNOWN_SEGMENT

    def test_sign_segment(self) -> None:
        assert sign_segment(0.3) == "positive"
        assert sign_segment(-0.3) == "negative"
        assert sign_segment(0.0) == "zero"
        assert sign_segment(None) == UNKNOWN_SEGMENT


class TestAssessSegments:
    def test_each_segment_carries_its_own_evidence(self) -> None:
        segments = assess_segments(
            [
                ("a", _call(0, "BULLISH", True, 0.01)),
                ("a", _call(1, "BEARISH", False, 0.01)),
                ("b", _call(0, "BULLISH", True, 0.02)),
            ]
        )

        by_key = {s.key: s for s in segments}
        assert (by_key["a"].n, by_key["a"].rate) == (2, pytest.approx(0.5))
        assert (by_key["b"].n, by_key["b"].rate) == (1, pytest.approx(1.0))
        assert by_key["a"].evidence.interval is not None

    def test_expected_keys_come_first_even_without_calls(self) -> None:
        segments = assess_segments(
            [("z", _call(0, "BULLISH", True, 0.01)), ("b", _call(0, "BULLISH", True, 0.01))],
            expected_keys=("x", "b"),
        )

        assert [s.key for s in segments] == ["x", "b", "z"]
        assert segments[0].n == 0
        assert segments[0].rate is None
        assert segments[0].fdr_significant is False

    def test_insufficient_below_the_minimum_number_of_bets(self) -> None:
        calls = [("a", _call(i, "BULLISH", True, 0.01)) for i in range(MIN_SEGMENT_N - 1)]
        calls += [("b", _call(i, "BULLISH", True, 0.01)) for i in range(MIN_SEGMENT_N)]
        # SIDEWAYS nie jest zakładem, więc nie podnosi n segmentu.
        calls += [("a", _call(20, "SIDEWAYS", True, 0.0))]

        by_key = {s.key: s for s in assess_segments(calls)}

        assert by_key["a"].insufficient is True
        assert by_key["b"].insufficient is False

    def test_benjamini_hochberg_runs_over_the_segment_family(self) -> None:
        # Segment „a”: 6 trafień na zmianę BULLISH/BEARISH (baseline 50%) → p ≈ 0.016.
        # Sam przechodzi q=0.05; w rodzinie z pięcioma słabymi segmentami próg
        # rangi 1 to 0.05/6 ≈ 0.008, więc ten sam wynik już nie przechodzi.
        strong = [
            ("a", _call(i, trend, True, 0.01 if trend == "BULLISH" else -0.01))
            for i, trend in enumerate(["BULLISH", "BEARISH"] * 3)
        ]
        weak = [
            (key, _call(i, "BULLISH", i == 0, 0.01 if i == 0 else -0.01))
            for key in ("b", "c", "d", "e", "f")
            for i in range(2)
        ]

        alone = assess_segments(strong, q=0.05)
        in_family = {s.key: s for s in assess_segments(strong + weak, q=0.05)}

        assert alone[0].evidence.p_value == pytest.approx(0.5**6)
        assert alone[0].fdr_significant is True
        assert in_family["a"].fdr_significant is False
        assert not any(s.fdr_significant for s in in_family.values())
