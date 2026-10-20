"""EXP-05: eksperymenty w pętli — przydział ramienia, aktywność, walidacja
i porównanie ramion z przedziałem z bootstrapu blokowego po cyklach."""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta

import pytest

from src.domain.experiment import (
    ARM_BLIND_COUNCIL,
    ARM_NO_MEMORY,
    ArmComparison,
    Experiment,
    ExperimentCall,
    ExperimentMetric,
    PromptPart,
    active_arms,
    assign_arm,
    compare_experiment,
    has_treatment,
    metric_value,
    validate_experiments,
)

_DAY = date(2026, 11, 2)


def _experiment(
    experiment_id: str = "EXP-02",
    *,
    arms: tuple[str, ...] = ("sighted", ARM_BLIND_COUNCIL),
    start: date = date(2026, 11, 1),
    end: date = date(2026, 12, 31),
    metric: ExperimentMetric = ExperimentMetric.COUNCIL_HIT_RATE,
    enabled: bool = True,
) -> Experiment:
    return Experiment(
        id=experiment_id,
        hypothesis="Rada bez werdyktu LLM jest trafniejsza.",
        arms=arms,
        start=start,
        end=end,
        primary_metric=metric,
        enabled=enabled,
    )


def _memory_experiment(**overrides: object) -> Experiment:
    params: dict[str, object] = {
        "arms": ("memory", ARM_NO_MEMORY),
        "metric": ExperimentMetric.HIT_RATE,
    }
    params.update(overrides)
    return _experiment("EXP-03", **params)  # type: ignore[arg-type]


class TestAssignArm:
    def test_assignment_is_deterministic(self) -> None:
        arms = ("sighted", ARM_BLIND_COUNCIL)

        first = assign_arm("EXP-02", "AAPL", _DAY, arms)

        assert all(assign_arm("EXP-02", "AAPL", _DAY, arms) == first for _ in range(5))
        assert first in arms

    def test_assignment_is_balanced_within_tolerance_over_1000_keys(self) -> None:
        # 40 symboli × 25 dni = 1000 kluczy; ±50 to ~3 odchylenia standardowe.
        arms = ("sighted", ARM_BLIND_COUNCIL)
        counts = Counter(
            assign_arm("EXP-02", f"SYM{index}", _DAY + timedelta(days=day), arms)
            for index in range(40)
            for day in range(25)
        )

        assert sum(counts.values()) == 1000
        assert all(abs(counts[arm] - 500) <= 50 for arm in arms)

    def test_assignment_depends_on_the_experiment_id(self) -> None:
        # Różne id → niezależne (ortogonalne) przydziały na tych samych kluczach.
        arms = ("a", "b")
        keys = [(f"SYM{index}", _DAY) for index in range(200)]

        first = [assign_arm("EXP-02", symbol, day, arms) for symbol, day in keys]
        second = [assign_arm("EXP-03", symbol, day, arms) for symbol, day in keys]

        assert first != second

    def test_assignment_changes_with_the_day(self) -> None:
        arms = ("a", "b")

        per_day = {assign_arm("EXP-02", "AAPL", _DAY + timedelta(days=d), arms) for d in range(30)}

        assert per_day == set(arms)

    def test_no_arms_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            assign_arm("EXP-02", "AAPL", _DAY, ())


class TestActivity:
    def test_inactive_experiment_returns_control(self) -> None:
        disabled = _experiment(enabled=False)
        before_window = _experiment(start=_DAY + timedelta(days=1))

        for experiment in (disabled, before_window):
            arms = {experiment.arm_for(f"SYM{i}", _DAY) for i in range(50)}
            assert arms == {experiment.control_arm}

    def test_control_arm_is_the_first_arm(self) -> None:
        assert _experiment().control_arm == "sighted"

    def test_window_is_inclusive(self) -> None:
        experiment = _experiment(start=_DAY, end=_DAY)

        assert experiment.is_active(_DAY)
        assert not experiment.is_active(_DAY - timedelta(days=1))
        assert not experiment.is_active(_DAY + timedelta(days=1))

    def test_active_arms_only_carry_active_experiments(self) -> None:
        running = _experiment()
        stopped = _memory_experiment(enabled=False)

        arms = active_arms((running, stopped), "AAPL", _DAY)

        assert set(arms) == {"EXP-02"}
        assert arms["EXP-02"] == assign_arm("EXP-02", "AAPL", _DAY, running.arms)

    def test_has_treatment_looks_at_assigned_arms(self) -> None:
        assert has_treatment({"EXP-02": ARM_BLIND_COUNCIL}, ARM_BLIND_COUNCIL)
        assert not has_treatment({"EXP-02": "sighted"}, ARM_BLIND_COUNCIL)
        assert not has_treatment({}, ARM_NO_MEMORY)


class TestDefinitionValidation:
    def test_part_comes_from_the_treatment_arm(self) -> None:
        assert _experiment().part is PromptPart.COUNCIL
        assert _memory_experiment().part is PromptPart.PREDICTION

    @pytest.mark.parametrize(
        "arms",
        [
            ("sighted",),
            ("sighted", "sighted"),
            ("sighted", "blnd"),
            (ARM_BLIND_COUNCIL, "sighted"),
            ("sighted", ARM_BLIND_COUNCIL, ARM_NO_MEMORY),
        ],
    )
    def test_invalid_arms_are_rejected(self, arms: tuple[str, ...]) -> None:
        # Literówka w ramieniu dałaby test A/A: ramię bez żadnego efektu.
        with pytest.raises(ValueError):
            _experiment(arms=arms)

    def test_window_must_not_end_before_it_starts(self) -> None:
        with pytest.raises(ValueError):
            _experiment(start=_DAY, end=_DAY - timedelta(days=1))

    def test_primary_metric_must_fit_the_prompt_part(self) -> None:
        with pytest.raises(ValueError):
            _experiment(metric=ExperimentMetric.HIT_RATE)

    def test_two_enabled_experiments_on_one_prompt_part_cannot_overlap(self) -> None:
        first = _experiment("EXP-02")
        second = _experiment("EXP-09", start=date(2026, 12, 1), end=date(2027, 1, 31))

        with pytest.raises(ValueError, match="council_prompt"):
            validate_experiments((first, second))

    def test_disjoint_windows_or_a_disabled_twin_are_fine(self) -> None:
        first = _experiment("EXP-02", end=date(2026, 11, 30))
        later = _experiment("EXP-09", start=date(2026, 12, 1))
        disabled = _experiment("EXP-10", enabled=False)

        validate_experiments((first, later, disabled))

    def test_experiments_on_different_parts_may_run_together(self) -> None:
        validate_experiments((_experiment(), _memory_experiment()))

    def test_duplicate_ids_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_experiments((_experiment(enabled=False), _experiment(enabled=False)))


def _call(arm: str, day: int, **fields: object) -> ExperimentCall:
    params: dict[str, object] = {
        "cycle": (_DAY + timedelta(days=day)).isoformat(),
        "arm": arm,
        "predicted_trend": "BULLISH",
        "is_correct": True,
    }
    params.update(fields)
    return ExperimentCall(**params)  # type: ignore[arg-type]


class TestMetricValue:
    def test_hit_rate_counts_only_directional_llm_calls(self) -> None:
        assert metric_value(ExperimentMetric.HIT_RATE, _call("a", 0)) == 1.0
        assert metric_value(ExperimentMetric.HIT_RATE, _call("a", 0, is_correct=False)) == 0.0
        sideways = _call("a", 0, predicted_trend="SIDEWAYS")
        assert metric_value(ExperimentMetric.HIT_RATE, sideways) is None

    def test_brier_scores_confidence_against_the_hit(self) -> None:
        call = _call("a", 0, is_correct=False, confidence=0.8)

        assert metric_value(ExperimentMetric.BRIER, call) == pytest.approx(0.64)
        assert metric_value(ExperimentMetric.BRIER, _call("a", 0)) is None

    def test_council_agreement_compares_recommendation_with_llm_trend(self) -> None:
        agree = _call("a", 0, council_recommendation="BUY")
        disagree = _call("a", 0, council_recommendation="HOLD")

        assert metric_value(ExperimentMetric.COUNCIL_LLM_AGREEMENT, agree) == 1.0
        assert metric_value(ExperimentMetric.COUNCIL_LLM_AGREEMENT, disagree) == 0.0
        assert metric_value(ExperimentMetric.COUNCIL_LLM_AGREEMENT, _call("a", 0)) is None

    def test_council_hit_rate_scores_buy_and_sell_on_the_realized_move(self) -> None:
        sell_down = _call("a", 0, council_recommendation="SELL", realized_change=-0.02)
        hold = _call("a", 0, council_recommendation="HOLD", realized_change=-0.02)

        assert metric_value(ExperimentMetric.COUNCIL_HIT_RATE, sell_down) == 1.0
        assert metric_value(ExperimentMetric.COUNCIL_HIT_RATE, hold) is None

    def test_council_brier_scores_consensus_against_the_council_hit(self) -> None:
        call = _call(
            "a", 0, council_recommendation="BUY", consensus_strength=0.9, realized_change=-0.01
        )

        assert metric_value(ExperimentMetric.COUNCIL_BRIER, call) == pytest.approx(0.81)

    def test_brier_metrics_are_lower_is_better(self) -> None:
        assert ExperimentMetric.BRIER.lower_is_better
        assert ExperimentMetric.COUNCIL_BRIER.lower_is_better
        assert not ExperimentMetric.HIT_RATE.lower_is_better


class TestCompareExperiment:
    def _comparison(
        self, calls: list[ExperimentCall], metric: ExperimentMetric = ExperimentMetric.HIT_RATE
    ) -> ArmComparison:
        experiment = _memory_experiment()
        comparisons = compare_experiment(experiment, calls, n_boot=400, seed=3)
        return next(c for c in comparisons if c.metric is metric)

    def test_difference_is_treatment_minus_control_with_n_per_arm(self) -> None:
        calls = [
            *(_call("memory", day, is_correct=day % 4 != 0) for day in range(30)),
            *(_call(ARM_NO_MEMORY, day, is_correct=day % 2 == 0) for day in range(30)),
        ]

        comparison = self._comparison(calls)

        assert comparison.control.arm == "memory"
        assert comparison.treatment.arm == ARM_NO_MEMORY
        assert comparison.control.n == 30
        assert comparison.treatment.n == 30
        assert comparison.control.mean == pytest.approx(22 / 30)
        assert comparison.treatment.mean == pytest.approx(15 / 30)
        assert comparison.difference == pytest.approx(15 / 30 - 22 / 30)
        assert comparison.cycles == 30

    def test_block_interval_brackets_the_difference_and_is_deterministic(self) -> None:
        calls = [
            *(_call("memory", day, is_correct=day % 4 != 0) for day in range(30)),
            *(_call(ARM_NO_MEMORY, day, is_correct=day % 2 == 0) for day in range(30)),
        ]

        first = self._comparison(calls)
        second = self._comparison(calls)

        assert first.interval is not None
        assert first.interval == second.interval
        low, high = first.interval
        assert first.difference is not None
        assert low <= first.difference <= high

    def test_correlated_cycles_shrink_n_eff(self) -> None:
        # Całe cykle trafione albo chybione: 5 predykcji jednego dnia to jeden zakład.
        calls = [
            _call(arm, day, is_correct=day % 2 == 0)
            for day in range(10)
            for arm in ("memory", ARM_NO_MEMORY)
            for _ in range(5)
        ]

        comparison = self._comparison(calls)

        assert comparison.control.n == 50
        assert comparison.control.n_eff == pytest.approx(10.0)

    def test_too_few_cycles_give_no_interval(self) -> None:
        calls = [_call(arm, day) for day in range(5) for arm in ("memory", ARM_NO_MEMORY)]

        assert self._comparison(calls).interval is None

    def test_empty_arm_has_no_mean_and_no_difference(self) -> None:
        comparison = self._comparison([_call("memory", day) for day in range(25)])

        assert comparison.treatment.n == 0
        assert comparison.treatment.mean is None
        assert comparison.difference is None
        assert comparison.interval is None

    def test_rows_of_other_arms_are_ignored(self) -> None:
        comparison = self._comparison([_call("legacy", 0), _call("memory", 0)])

        assert comparison.control.n == 1
        assert comparison.treatment.n == 0

    def test_comparisons_follow_the_metrics_of_the_prompt_part(self) -> None:
        comparisons = compare_experiment(_experiment(), [], n_boot=10)

        assert [c.metric for c in comparisons] == [
            ExperimentMetric.COUNCIL_LLM_AGREEMENT,
            ExperimentMetric.COUNCIL_HIT_RATE,
            ExperimentMetric.COUNCIL_BRIER,
        ]
