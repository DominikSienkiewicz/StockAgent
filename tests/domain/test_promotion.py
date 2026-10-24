"""ML-33: pre-rejestrowana reguła promocji challengera nogi ML na predykcjach live."""

from __future__ import annotations

import math

import pytest

from src.domain.backtest import MIN_OOS_POINTS, MIN_REL_IMPROVEMENT, PAIRED_Z_CRIT
from src.domain.promotion import (
    CHAMPION_MODEL_NAME,
    MIN_LIVE_N,
    PROMOTION_MARGIN,
    PROMOTION_Z_CRIT,
    ForecastKind,
    PromotionStatus,
    SettledShadowForecast,
    ShadowForecast,
    challenger_verdicts,
    paired_loss,
    should_promote,
)


def _champion(predicted_return: float, version: str = "a" * 64) -> ShadowForecast:
    return ShadowForecast(CHAMPION_MODEL_NAME, version, predicted_return=predicted_return)


def _alternating(base: float, n: int, wobble: float = 0.01) -> list[float]:
    """Straty z rozrzutem — zerowa wariancja różnic dałaby z = ±inf."""
    return [base + (wobble if index % 2 else -wobble) for index in range(n)]


class TestShouldPromote:
    def test_rule_is_the_training_gate_applied_live(self) -> None:
        # Pre-rejestracja: te same progi co bramka walk-forward (marża, minimum
        # punktów, test sparowany EXP-04) — nie stroimy ich pod wyniki live.
        assert (MIN_LIVE_N, PROMOTION_MARGIN, PROMOTION_Z_CRIT) == (
            MIN_OOS_POINTS,
            MIN_REL_IMPROVEMENT,
            PAIRED_Z_CRIT,
        )

    def test_no_promotion_below_min_live_n(self) -> None:
        champion = _alternating(1.0, 10)
        challenger = [loss / 2 for loss in champion]

        verdict = should_promote(champion, challenger, min_live_n=11, margin=0.02)

        assert verdict.status is PromotionStatus.INSUFFICIENT_DATA
        assert verdict.promote is False
        assert verdict.n_pairs == 10
        assert verdict.challenger_loss == pytest.approx(0.5)

    def test_promotes_when_paired_loss_better_by_margin(self) -> None:
        champion = _alternating(1.0, 80)
        challenger = _alternating(0.9, 80, wobble=0.012)

        verdict = should_promote(champion, challenger, min_live_n=60, margin=0.02)

        assert verdict.status is PromotionStatus.PROMOTE
        assert verdict.promote is True
        assert verdict.n_pairs == 80
        assert verdict.champion_loss == pytest.approx(1.0)
        assert verdict.challenger_loss == pytest.approx(0.9)
        assert verdict.improvement == pytest.approx(0.1)
        assert verdict.paired_z is not None and verdict.paired_z > PAIRED_Z_CRIT

    def test_keeps_champion_when_better_by_less_than_margin(self) -> None:
        champion = _alternating(1.0, 80)
        challenger = _alternating(0.99, 80, wobble=0.0101)

        verdict = should_promote(champion, challenger, min_live_n=60, margin=0.02)

        assert verdict.status is PromotionStatus.KEEP_CHAMPION
        assert verdict.improvement == pytest.approx(0.01)

    def test_keeps_champion_when_the_paired_test_is_not_significant(self) -> None:
        # Średnio 10% lepiej, ale różnice skaczą ±1 — szum, nie umiejętność.
        champion = [1.0] * 80
        challenger = [0.9 + (1.0 if index % 2 else -1.0) for index in range(80)]

        verdict = should_promote(champion, challenger, min_live_n=60, margin=0.02)

        assert verdict.status is PromotionStatus.KEEP_CHAMPION
        assert verdict.paired_z is not None and verdict.paired_z < PAIRED_Z_CRIT

    def test_perfect_champion_cannot_be_beaten(self) -> None:
        verdict = should_promote([0.0] * 70, [0.0] * 70, min_live_n=60, margin=0.02)

        assert verdict.status is PromotionStatus.KEEP_CHAMPION
        assert verdict.improvement is None

    def test_no_pairs_is_insufficient_data_without_losses(self) -> None:
        verdict = should_promote([], [])

        assert verdict.status is PromotionStatus.INSUFFICIENT_DATA
        assert (verdict.champion_loss, verdict.challenger_loss, verdict.paired_z) == (
            None,
            None,
            None,
        )

    @pytest.mark.parametrize(
        ("champion", "challenger", "kwargs"),
        [
            ([1.0, 1.0], [1.0], {}),
            ([1.0], [1.0], {"min_live_n": 1}),
            ([1.0], [1.0], {"margin": -0.01}),
            ([1.0], [1.0], {"margin": 1.0}),
        ],
    )
    def test_rejects_unpaired_losses_and_rules_that_cannot_hold(
        self, champion: list[float], challenger: list[float], kwargs: dict[str, float]
    ) -> None:
        with pytest.raises(ValueError):
            should_promote(champion, challenger, **kwargs)  # type: ignore[arg-type]


class TestShadowForecast:
    def test_kind_follows_the_value_the_model_gives(self) -> None:
        regressor = ShadowForecast("alt", "b" * 64, predicted_return=0.01)
        classifier = ShadowForecast("direction", "c" * 64, proba=0.7)

        assert (regressor.kind, classifier.kind) == (ForecastKind.RETURN, ForecastKind.DIRECTION)
        assert _champion(0.0).is_champion and not classifier.is_champion

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"model_name": "", "model_version": "v"},
            {"model_name": "alt", "model_version": ""},
            {"model_name": "alt", "model_version": "v"},
            {"model_name": "alt", "model_version": "v", "predicted_return": math.nan},
            {"model_name": "alt", "model_version": "v", "proba": 1.5},
            {"model_name": "alt", "model_version": "v", "proba": math.nan},
        ],
    )
    def test_rejects_forecasts_that_cannot_be_scored(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            ShadowForecast(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("predicted_return", "probability"), [(0.02, 1.0), (-0.01, 0.0), (0.0, 0.5)]
    )
    def test_a_return_forecast_is_a_hard_direction_call(
        self, predicted_return: float, probability: float
    ) -> None:
        assert _champion(predicted_return).up_probability == probability

    def test_a_classifier_keeps_its_probability(self) -> None:
        assert ShadowForecast("direction", "c" * 64, proba=0.3).up_probability == 0.3


class TestPairedLoss:
    def test_regressors_are_scored_by_squared_return_error(self) -> None:
        challenger = ShadowForecast("alt", "b" * 64, predicted_return=0.01)

        losses = paired_loss(_champion(0.03), challenger, 0.02, dead_band=0.0)

        assert losses == pytest.approx((0.0001, 0.0001))

    def test_a_classifier_and_the_champion_are_scored_by_the_direction_brier(self) -> None:
        # Champion przewiduje wzrost (twarde p = 1), klasyfikator p = 0.8; rynek spadł.
        challenger = ShadowForecast("direction", "c" * 64, proba=0.8)

        losses = paired_loss(_champion(0.01), challenger, -0.02, dead_band=0.0)

        assert losses == pytest.approx((1.0, 0.64))

    def test_moves_inside_the_dead_band_are_not_scored_for_direction(self) -> None:
        challenger = ShadowForecast("direction", "c" * 64, proba=0.8)

        assert paired_loss(_champion(0.01), challenger, 0.004, dead_band=0.005) is None

    def test_a_regressor_needs_a_champion_return(self) -> None:
        classifier_champion = ShadowForecast(CHAMPION_MODEL_NAME, "a" * 64, proba=0.6)
        challenger = ShadowForecast("alt", "b" * 64, predicted_return=0.01)

        assert paired_loss(classifier_champion, challenger, 0.02, dead_band=0.0) is None

    def test_an_unknown_outcome_is_not_scored(self) -> None:
        challenger = ShadowForecast("alt", "b" * 64, predicted_return=0.01)

        assert paired_loss(_champion(0.0), challenger, math.nan, dead_band=0.0) is None


def _settled(
    prediction_id: str, forecast: ShadowForecast, realized: float
) -> SettledShadowForecast:
    return SettledShadowForecast(prediction_id, forecast, realized)


class TestChallengerVerdicts:
    def test_pairs_each_challenger_with_the_champion_of_the_same_prediction(self) -> None:
        rows: list[SettledShadowForecast] = []
        for index in range(70):
            realized = 0.01 if index % 2 else -0.01
            rows.append(_settled(f"p{index}", _champion(-realized), realized))
            rows.append(
                _settled(
                    f"p{index}",
                    ShadowForecast("direction", "c" * 64, proba=0.9 if realized > 0 else 0.1),
                    realized,
                )
            )
        # Challenger bez championa (cold start) nie ma pary.
        rows.append(_settled("orphan", ShadowForecast("direction", "c" * 64, proba=0.5), 0.01))

        (verdict,) = challenger_verdicts(rows, dead_band=0.0)

        assert (verdict.model_name, verdict.kind, verdict.n_versions) == (
            "direction",
            ForecastKind.DIRECTION,
            1,
        )
        assert verdict.verdict.n_pairs == 70
        assert verdict.verdict.status is PromotionStatus.PROMOTE

    def test_reports_each_challenger_and_kind_separately(self) -> None:
        rows = [
            _settled("p1", _champion(0.01), 0.02),
            _settled("p1", ShadowForecast("alt", "b" * 64, predicted_return=0.02), 0.02),
            _settled("p1", ShadowForecast("direction", "c" * 64, proba=0.6), 0.02),
            _settled("p2", _champion(0.01), 0.004),
            _settled("p2", ShadowForecast("alt", "d" * 64, predicted_return=0.0), 0.004),
            _settled("p2", ShadowForecast("direction", "c" * 64, proba=0.6), 0.004),
        ]

        verdicts = challenger_verdicts(rows, dead_band=0.005)

        assert [(v.model_name, v.kind, v.n_versions, v.verdict.n_pairs) for v in verdicts] == [
            ("alt", ForecastKind.RETURN, 2, 2),
            ("direction", ForecastKind.DIRECTION, 1, 1),
        ]
        assert all(v.verdict.status is PromotionStatus.INSUFFICIENT_DATA for v in verdicts)

    def test_no_challenger_rows_means_no_verdict(self) -> None:
        assert challenger_verdicts([_settled("p1", _champion(0.01), 0.02)], dead_band=0.0) == ()
