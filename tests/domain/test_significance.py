"""Testy domenowego modułu istotności (WIA-22, WIA-23) — czysty stdlib.

Każda trafność była gołym procentem. Test dwumianowy mówi, czy wynik bije
naiwny baseline, BH-FDR koryguje wielokrotne porównania (7 person, kilkanaście
segmentów), a n_eff i bootstrap blokowy uwzględniają, że ~40 spółek w jednym
cyklu rusza się razem.
"""

from __future__ import annotations

import pytest

from src.domain.number_format import DecimalMark
from src.domain.significance import (
    benjamini_hochberg,
    binomial_test_greater,
    cluster_bootstrap_ci,
    effective_n,
    verdict_pl,
)


class TestBinomialTestGreater:
    def test_binomial_tail_matches_known_value(self) -> None:
        # P(X >= 60 | n=100, p=0.5) ≈ 0.0284 (tablice rozkładu dwumianowego).
        assert binomial_test_greater(60, 100, 0.5) == pytest.approx(0.0284, abs=1e-4)

    def test_tail_at_or_below_zero_successes_is_certain(self) -> None:
        assert binomial_test_greater(0, 10, 0.3) == pytest.approx(1.0)

    def test_more_successes_than_trials_is_impossible(self) -> None:
        assert binomial_test_greater(11, 10, 0.3) == pytest.approx(0.0)

    def test_large_n_uses_normal_approximation_close_to_exact(self) -> None:
        # n > 1000: przybliżenie normalne z korektą ciągłości; dla p0=0.5,
        # k = n/2 + 1.96·σ ogon ≈ 0.025.
        n = 4000
        k = 2000 + round(1.96 * (n * 0.25) ** 0.5)

        assert binomial_test_greater(k, n, 0.5) == pytest.approx(0.025, abs=3e-3)

    def test_degenerate_baseline_probabilities(self) -> None:
        assert binomial_test_greater(5, 10, 0.0) == pytest.approx(0.0)
        assert binomial_test_greater(5, 10, 1.0) == pytest.approx(1.0)


class TestBenjaminiHochberg:
    def test_bh_rejects_only_smallest_pvalue_when_others_large(self) -> None:
        assert benjamini_hochberg([0.001, 0.6, 0.8, 0.9]) == [True, False, False, False]

    def test_step_up_rejects_every_hypothesis_below_the_largest_passing_rank(self) -> None:
        # Progi q·k/m dla q=0.10, m=4: 0.025, 0.05, 0.075, 0.10. Największa ranga
        # spełniająca p_(k) <= próg to k=3 → odrzucamy trzy najmniejsze.
        assert benjamini_hochberg([0.07, 0.04, 0.03, 0.5]) == [True, True, True, False]

    def test_preserves_input_order(self) -> None:
        assert benjamini_hochberg([0.9, 0.001]) == [False, True]

    def test_empty_input(self) -> None:
        assert benjamini_hochberg([]) == []


class TestVerdictPl:
    def test_verdict_says_coin_flip_when_p_above_alpha(self) -> None:
        verdict = verdict_pl(0.41, 37)

        assert verdict.startswith("nieodróżnialne od rzutu monetą")
        assert "p=0.41" in verdict
        assert "n=37" in verdict

    def test_verdict_without_n_eff_warns_about_independence(self) -> None:
        # Bez n_eff p-wartość zakłada niezależne próby — werdykt musi to mówić.
        assert "niezależne" in verdict_pl(0.41, 37)

    def test_verdict_with_n_eff_reports_it(self) -> None:
        verdict = verdict_pl(0.003, 412, 61.4)

        assert verdict.startswith("istotnie lepsze od naiwnego baseline'u")
        assert "n_eff≈61" in verdict
        assert "niezależne" not in verdict

    def test_tiny_p_value_is_not_printed_as_zero(self) -> None:
        assert "p<0.001" in verdict_pl(1e-9, 500, 120.0)


class TestVerdictPlDecimalMark:
    """MAIL-58: mail pisze p-wartość po polsku, publiczny scorecard zostaje przy kropce."""

    def test_default_keeps_the_decimal_point(self) -> None:
        assert "p=0.41" in verdict_pl(0.41, 37)
        assert "p=0.003" in verdict_pl(0.003, 412, 61.4)

    def test_comma_mark_writes_the_p_value_the_polish_way(self) -> None:
        assert "p=0,41" in verdict_pl(0.41, 37, mark=DecimalMark.COMMA)
        assert "p=0,003" in verdict_pl(0.003, 412, 61.4, mark=DecimalMark.COMMA)
        assert "p<0,001" in verdict_pl(1e-9, 500, 120.0, mark=DecimalMark.COMMA)

    def test_comma_mark_rounds_half_up(self) -> None:
        assert "p=0,13" in verdict_pl(0.125, 37, mark=DecimalMark.COMMA)


class TestEffectiveN:
    def test_effective_n_equals_n_when_outcomes_independent_across_cycles(self) -> None:
        # Każdy cykl ma ten sam odsetek trafień — zero korelacji wewnątrz cyklu.
        cycles = [[True, False, True, False] for _ in range(10)]

        assert effective_n(cycles) == pytest.approx(40.0)

    def test_effective_n_collapses_to_cycle_count_when_cycle_outcomes_identical(
        self,
    ) -> None:
        # 10 cykli × 40 identycznych wyników: dzień spadku NASDAQ to jeden zakład.
        cycles = [[day % 2 == 0] * 40 for day in range(10)]

        assert effective_n(cycles) == pytest.approx(10.0)

    def test_single_observation_cycles_are_independent(self) -> None:
        assert effective_n([[True], [False], [True]]) == pytest.approx(3.0)

    def test_no_variance_assumes_the_worst_case(self) -> None:
        # Same trafienia: korelacji nie da się oszacować — przyjmujemy najgorszy
        # przypadek (jeden zakład na cykl), zamiast udawać niezależność.
        assert effective_n([[True] * 5, [True] * 5]) == pytest.approx(2.0)

    def test_empty_input(self) -> None:
        assert effective_n([]) == pytest.approx(0.0)
        assert effective_n([[], []]) == pytest.approx(0.0)


class TestClusterBootstrap:
    @staticmethod
    def _cycles(count: int) -> list[list[bool]]:
        return [[(day + i) % 3 != 0 for i in range(5)] for day in range(count)]

    def test_cluster_bootstrap_is_deterministic_for_seed(self) -> None:
        cycles = self._cycles(30)

        first = cluster_bootstrap_ci(cycles, n_boot=300, seed=7)
        second = cluster_bootstrap_ci(cycles, n_boot=300, seed=7)

        assert first is not None
        assert first == second

    def test_interval_brackets_the_pooled_hit_rate(self) -> None:
        cycles = self._cycles(30)
        pooled = sum(sum(c) for c in cycles) / sum(len(c) for c in cycles)

        interval = cluster_bootstrap_ci(cycles, n_boot=500, seed=1)

        assert interval is not None
        assert interval[0] <= pooled <= interval[1]

    def test_too_few_cycles_give_no_interval(self) -> None:
        # < 20 cykli: bootstrap niestabilny — zamiast przedziału zostaje n_eff.
        assert cluster_bootstrap_ci(self._cycles(19), n_boot=100, seed=0) is None


class TestEffectiveNValues:
    """EXP-05: n_eff dla wartości ciągłych (kwadraty błędu Briera) — ta sama
    reguła co `effective_n`, a dla wyników 0/1 ten sam wynik."""

    @pytest.mark.parametrize(
        "cycles",
        [
            [[True, False, True, False] for _ in range(10)],
            [[day % 2 == 0] * 40 for day in range(10)],
            [[True], [False], [True]],
            [[True] * 5, [True] * 5],
            [[True, True, False], [False, False], [True, False, True, True]],
            [],
        ],
    )
    def test_matches_effective_n_on_binary_outcomes(self, cycles: list[list[bool]]) -> None:
        from src.domain.significance import effective_n_values

        values = [[1.0 if hit else 0.0 for hit in cycle] for cycle in cycles]

        assert effective_n_values(values) == pytest.approx(effective_n(cycles))

    def test_identical_values_within_cycles_collapse_to_cycle_count(self) -> None:
        from src.domain.significance import effective_n_values

        cycles = [[0.04 * (day % 3)] * 6 for day in range(8)]

        assert effective_n_values(cycles) == pytest.approx(8.0)


class TestClusterBootstrapDifference:
    """EXP-05: przedział różnicy średnich dwóch ramion z bootstrapu po CAŁYCH
    cyklach — ramiona jednego dnia losują się razem."""

    @staticmethod
    def _cycles(count: int) -> list[tuple[list[float], list[float]]]:
        return [
            ([1.0 if (day + i) % 4 else 0.0 for i in range(3)],
             [1.0 if (day + i) % 2 else 0.0 for i in range(3)])
            for day in range(count)
        ]

    def test_is_deterministic_for_seed_and_brackets_the_difference(self) -> None:
        from src.domain.significance import cluster_bootstrap_difference_ci

        cycles = self._cycles(30)
        control = [v for a, _ in cycles for v in a]
        treatment = [v for _, b in cycles for v in b]
        difference = sum(treatment) / len(treatment) - sum(control) / len(control)

        first = cluster_bootstrap_difference_ci(cycles, n_boot=400, seed=5)
        second = cluster_bootstrap_difference_ci(cycles, n_boot=400, seed=5)

        assert first is not None
        assert first == second
        assert first[0] <= difference <= first[1]

    def test_too_few_cycles_give_no_interval(self) -> None:
        from src.domain.significance import cluster_bootstrap_difference_ci

        assert cluster_bootstrap_difference_ci(self._cycles(19), n_boot=100) is None

    def test_an_arm_without_observations_gives_no_interval(self) -> None:
        from src.domain.significance import cluster_bootstrap_difference_ci

        cycles: list[tuple[list[float], list[float]]] = [([1.0, 0.0], []) for _ in range(25)]

        assert cluster_bootstrap_difference_ci(cycles, n_boot=100) is None


class TestClusterBootstrapStatistic:
    """WIA-58 / WIA-38: ten sam bootstrap po cyklach dla dowolnej statystyki
    próbki (ECE, AUC), nie tylko dla hit-rate."""

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    def test_matches_the_hit_rate_bootstrap_for_the_mean(self) -> None:
        from src.domain.significance import cluster_bootstrap_statistic_ci

        cycles = [[True, False, True], [False], [True, True]] * 8

        generic = cluster_bootstrap_statistic_ci(
            cycles, lambda xs: self._mean([1.0 if x else 0.0 for x in xs]), n_boot=200, seed=4
        )

        assert generic == cluster_bootstrap_ci(cycles, n_boot=200, seed=4)

    def test_too_few_cycles_give_no_interval(self) -> None:
        from src.domain.significance import (
            MIN_BOOTSTRAP_CYCLES,
            cluster_bootstrap_statistic_ci,
        )

        cycles = [[1.0, 2.0]] * (MIN_BOOTSTRAP_CYCLES - 1)

        assert cluster_bootstrap_statistic_ci(cycles, self._mean, n_boot=50) is None

    def test_undefined_statistic_on_every_resample_gives_no_interval(self) -> None:
        from src.domain.significance import (
            MIN_BOOTSTRAP_CYCLES,
            cluster_bootstrap_statistic_ci,
        )

        cycles = [[1.0]] * MIN_BOOTSTRAP_CYCLES

        assert cluster_bootstrap_statistic_ci(cycles, lambda _: None, n_boot=50) is None

