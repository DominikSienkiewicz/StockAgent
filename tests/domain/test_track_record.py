"""Testy domeny track recordu: naiwne baseline'y (WIA-31) i zebranie dowodu
trafności kierunkowej (WIA-22/23/32) w jeden obiekt.

Hit-rate bez punktu odniesienia nic nie dowodzi: w hossie strategia
„zawsze BULLISH” trafia w ponad połowie dni bez żadnej wiedzy.
"""

from __future__ import annotations

from typing import cast

import pytest

from src.domain.number_format import DecimalMark
from src.domain.significance import MIN_BOOTSTRAP_CYCLES
from src.domain.track_record import (
    NaiveBaselines,
    ScoredCall,
    assess_directional_hit_rate,
    group_by_cycle,
    naive_baselines,
)


class TestNaiveBaselines:
    def test_always_bullish_baseline_counts_up_moves(self) -> None:
        baselines = naive_baselines([0.02, -0.01, 0.03])

        assert baselines.always_bullish == pytest.approx(2 / 3)

    def test_always_bearish_counts_down_moves(self) -> None:
        assert naive_baselines([0.02, -0.01, 0.03]).always_bearish == pytest.approx(1 / 3)

    def test_always_sideways_counts_moves_inside_the_band(self) -> None:
        # Pasmo ±0.5% (domyślna tolerancja SIDEWAYS dla akcji) — włącznie.
        baselines = naive_baselines([0.004, -0.005, 0.02, -0.03])

        assert baselines.always_sideways == pytest.approx(0.5)

    def test_zero_move_is_neither_up_nor_down(self) -> None:
        # Ta sama reguła co `Prediction.is_trend_correct`: BULLISH wymaga delta > 0.
        baselines = naive_baselines([0.0])

        assert baselines.always_bullish == pytest.approx(0.0)
        assert baselines.always_bearish == pytest.approx(0.0)

    def test_best_directional_baseline_is_the_stronger_of_long_and_short(self) -> None:
        best = naive_baselines([0.02, -0.01, 0.03]).best_directional

        assert best == ("BULLISH", cast(float, pytest.approx(2 / 3)))

    def test_empty_changes_have_no_baseline(self) -> None:
        assert naive_baselines([]) == NaiveBaselines(
            always_bullish=None, always_bearish=None, always_sideways=None, n=0
        )
        assert naive_baselines([]).best_directional is None


class TestGroupByCycle:
    def test_groups_in_chronological_key_order(self) -> None:
        grouped = group_by_cycle([("2026-10-02", "b"), ("2026-10-01", "a"), ("2026-10-02", "c")])

        assert grouped == [["a"], ["b", "c"]]


def _call(cycle: str, trend: str, correct: bool, change: float | None) -> ScoredCall:
    return ScoredCall(
        cycle=cycle, predicted_trend=trend, is_correct=correct, realized_change=change
    )


class TestAssessDirectionalHitRate:
    def test_baseline_is_computed_on_the_directional_calls_only(self) -> None:
        # Baseline na tym samym zbiorze co hit-rate: SIDEWAYS (+5%) nie wchodzi.
        calls = [
            _call("2026-10-01", "BULLISH", True, 0.02),
            _call("2026-10-01", "BEARISH", False, 0.01),
            _call("2026-10-02", "SIDEWAYS", False, 0.05),
        ]

        evidence = assess_directional_hit_rate(calls)

        assert evidence.hit_rate.rate == pytest.approx(0.5)
        assert evidence.baselines.n == 2
        assert evidence.baselines.always_bullish == pytest.approx(1.0)

    def test_interval_and_p_value_use_effective_n(self) -> None:
        # 4 cykle × 10 identycznych wyników w cyklu → n_eff = 4, nie 40.
        calls = [
            _call(f"2026-10-0{day}", "BULLISH", day % 2 == 1, 0.01 if day % 2 == 1 else -0.01)
            for day in range(1, 5)
            for _ in range(10)
        ]

        evidence = assess_directional_hit_rate(calls)

        assert evidence.hit_rate.directional == 40
        assert evidence.cycles == 4
        assert evidence.n_eff == pytest.approx(4.0)
        assert evidence.interval is not None
        # Wilson na n_eff=4 jest dużo szerszy niż na surowym n=40.
        assert evidence.interval[1] - evidence.interval[0] > 0.5

    def test_bootstrap_is_skipped_below_the_cycle_minimum(self) -> None:
        calls = [_call("2026-10-01", "BULLISH", True, 0.01)]

        evidence = assess_directional_hit_rate(calls)

        assert evidence.cycles < MIN_BOOTSTRAP_CYCLES
        assert evidence.bootstrap is None

    def test_verdict_names_the_coin_flip_when_not_significant(self) -> None:
        calls = [
            _call("2026-10-01", "BULLISH", True, 0.01),
            _call("2026-10-02", "BULLISH", False, -0.01),
        ]

        evidence = assess_directional_hit_rate(calls)

        assert evidence.p_value is not None
        assert evidence.verdict is not None
        assert "rzutu monetą" in evidence.verdict

    def test_verdict_can_write_the_p_value_with_a_comma(self) -> None:
        # MAIL-58: `verdict` zostaje z kropką (publiczny scorecard), mail bierze przecinek.
        calls = [
            _call("2026-10-01", "BULLISH", True, 0.01),
            _call("2026-10-02", "BULLISH", False, -0.01),
        ]

        evidence = assess_directional_hit_rate(calls)
        polish = evidence.verdict_in(DecimalMark.COMMA)

        assert evidence.verdict is not None and polish is not None
        assert "p=0." in evidence.verdict
        assert "p=0," in polish
        assert polish == evidence.verdict.replace("p=0.", "p=0,")

    def test_no_directional_calls_means_no_interval_and_no_verdict(self) -> None:
        evidence = assess_directional_hit_rate([_call("2026-10-01", "SIDEWAYS", True, 0.0)])

        assert evidence.hit_rate.rate is None
        assert evidence.interval is None
        assert evidence.p_value is None
        assert evidence.verdict is None


class TestBestAnyBaseline:
    """WIA-22: persona rady głosuje KUP / SPRZEDAJ / TRZYMAJ, więc jej naiwnym
    odpowiednikiem jest najlepsza z trzech strategii, także „zawsze bez ruchu”."""

    def test_best_any_includes_the_sideways_strategy(self) -> None:
        best = naive_baselines([0.001, -0.002, -0.003, 0.02]).best_any

        assert best is not None
        assert best[0] == "SIDEWAYS"
        assert best[1] == cast(float, pytest.approx(0.75))

    def test_best_any_falls_back_to_the_stronger_direction(self) -> None:
        best = naive_baselines([0.02, -0.01, 0.03]).best_any

        assert best is not None
        assert best[0] == "BULLISH"
        assert best[1] == cast(float, pytest.approx(2 / 3))

    def test_empty_changes_have_no_best_any(self) -> None:
        assert naive_baselines([]).best_any is None
