"""Testy domeny testu echa momentum (EXP-01).

Bramka przepuszcza tylko |Δ| ≥ próg, a LLM widzi tę zmianę w prompcie. Jeśli
trend powtarza jej znak, darmowa reguła „znak ruchu” daje to samo za 0 zł.
Porównanie z baseline'em jest sparowane: o przewadze decydują wyłącznie wiersze,
na których LLM i baseline stawiają przeciwnie (test znaków na n_eff, BH-FDR).
"""

from __future__ import annotations

import pytest

from src.domain.momentum_echo import EchoCall, assess_momentum_echo


def _call(day: int, trend: str, gate_delta: float, change: float | None) -> EchoCall:
    return EchoCall(
        cycle=f"2026-08-{day + 1:02d}",
        predicted_trend=trend,
        gate_delta=gate_delta,
        realized_change=change,
    )


class TestEchoAndReversal:
    def test_echo_and_reversal_shares_of_calls_with_a_gate_delta(self) -> None:
        report = assess_momentum_echo(
            [
                _call(0, "BULLISH", 0.03, 0.01),
                _call(1, "BEARISH", 0.03, 0.01),
                _call(2, "SIDEWAYS", -0.03, 0.0),
                _call(3, "bearish", -0.02, -0.01),
            ]
        )

        assert report.with_delta == 4
        assert (report.echo, report.reversal) == (2, 1)
        assert report.echo_rate == pytest.approx(0.5)
        assert report.reversal_rate == pytest.approx(0.25)

    def test_empty_input_has_no_rates(self) -> None:
        report = assess_momentum_echo([])

        assert report.with_delta == 0
        assert report.echo_rate is None
        assert report.llm_rate is None
        assert report.vs_momentum.p_value is None
        assert report.vs_momentum.fdr_significant is False


class TestBaselinesOnTheSameRows:
    def test_unsettled_rows_count_for_echo_but_not_for_hit_rates(self) -> None:
        report = assess_momentum_echo(
            [_call(0, "BULLISH", 0.03, None), _call(1, "BULLISH", 0.03, 0.01)]
        )

        assert report.echo == 2
        assert report.compared == 1
        assert report.momentum_rate == pytest.approx(1.0)
        assert report.contrarian_rate == pytest.approx(0.0)

    def test_zero_gate_delta_has_no_momentum_sign(self) -> None:
        report = assess_momentum_echo([_call(0, "BULLISH", 0.0, 0.01)])

        assert (report.with_delta, report.echo, report.reversal) == (1, 0, 0)
        assert report.compared == 0


class TestHeadToHead:
    def _reversals(self, wins: int, losses: int) -> list[EchoCall]:
        # LLM odwraca momentum: wygrywa, gdy ruch poszedł przeciw delcie bramki.
        won = [_call(i, "BEARISH", 0.03, -0.01) for i in range(wins)]
        lost = [_call(wins + i, "BEARISH", 0.03, 0.01) for i in range(losses)]
        return won + lost

    def test_llm_beats_momentum_only_on_the_rows_where_it_disagrees(self) -> None:
        # 8 ech (wspólne trafienia nic nie rozstrzygają) + 10 odwróceń, 10/10 wygranych.
        echoes = [_call(20 + i, "BULLISH", 0.03, 0.01) for i in range(8)]
        report = assess_momentum_echo(echoes + self._reversals(10, 0))

        assert report.vs_momentum.n == 10
        assert report.vs_momentum.wins == 10
        assert report.vs_momentum.p_value == pytest.approx(0.5**10)
        assert report.vs_momentum.fdr_significant is True
        # Na echach LLM = znak ruchu, więc vs „przeciwny znak” wygrywa 8/8.
        assert report.vs_contrarian.n == 8
        assert report.vs_contrarian.wins == 8

    def test_a_coin_flip_on_disagreements_is_not_significant(self) -> None:
        report = assess_momentum_echo(self._reversals(5, 5))

        assert report.disagreement_rate == pytest.approx(0.5)
        assert report.vs_momentum.fdr_significant is False

    def test_ties_on_a_flat_move_are_left_out_of_the_sign_test(self) -> None:
        report = assess_momentum_echo([*self._reversals(1, 0), _call(5, "BEARISH", 0.03, 0.0)])

        assert report.disagreement == 2
        assert report.disagreement_rate == pytest.approx(0.5)
        assert report.vs_momentum.n == 1

    def test_n_eff_counts_one_bet_per_correlated_cycle(self) -> None:
        # Cztery odwrócenia jednego dnia, wszystkie trafione: jeden zakład, nie cztery.
        same_day = [_call(0, "BEARISH", 0.03, -0.01) for _ in range(4)]

        report = assess_momentum_echo(same_day)

        assert report.vs_momentum.n == 4
        assert report.vs_momentum.n_eff == pytest.approx(1.0)
        assert report.vs_momentum.p_value == pytest.approx(0.5)
