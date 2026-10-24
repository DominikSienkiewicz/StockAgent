"""ML-33: werdykt promocji challengerów nogi ML w Slow Loopie — odczyt rozliczonych
prognoz cienia z okna i reguła domenowa; bez zapisu i bez podmiany wag."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

from src.application.ports import RepositoryPort
from src.application.use_cases.review_challengers import ReviewChallengersUseCase
from src.domain.promotion import (
    CHAMPION_MODEL_NAME,
    LIVE_WINDOW_DAYS,
    ForecastKind,
    PromotionStatus,
    SettledShadowForecast,
    ShadowForecast,
)

NOW = datetime(2026, 10, 25, 7, 30, tzinfo=UTC)


def _rows(n: int) -> list[SettledShadowForecast]:
    rows: list[SettledShadowForecast] = []
    for index in range(n):
        realized = 0.01 if index % 2 else -0.01
        rows.append(
            SettledShadowForecast(
                f"p{index}",
                ShadowForecast(CHAMPION_MODEL_NAME, "a" * 64, predicted_return=-realized),
                realized,
            )
        )
        rows.append(
            SettledShadowForecast(
                f"p{index}",
                ShadowForecast("direction", "c" * 64, proba=0.9 if realized > 0 else 0.1),
                realized,
            )
        )
    return rows


def test_reads_the_live_window_and_returns_the_domain_verdict() -> None:
    repository = Mock(spec=RepositoryPort)
    repository.get_settled_shadow_predictions.return_value = _rows(70)

    verdicts = ReviewChallengersUseCase(repository, clock=lambda: NOW).run(dead_band=0.0)

    repository.get_settled_shadow_predictions.assert_called_once_with(
        NOW - timedelta(days=LIVE_WINDOW_DAYS)
    )
    assert len(verdicts) == 1
    (verdict,) = verdicts
    assert (verdict.model_name, verdict.kind) == ("direction", ForecastKind.DIRECTION)
    assert verdict.verdict.status is PromotionStatus.PROMOTE
    assert verdict.verdict.n_pairs == 70


def test_the_dead_band_of_the_protocol_reaches_the_direction_loss() -> None:
    repository = Mock(spec=RepositoryPort)
    repository.get_settled_shadow_predictions.return_value = _rows(70)

    verdicts = ReviewChallengersUseCase(repository, clock=lambda: NOW).run(dead_band=0.02)

    assert len(verdicts) == 1
    (verdict,) = verdicts
    # Każdy ruch ±1% leży w martwej strefie ±2% — kierunku nie ma czym ocenić.
    assert verdict.verdict.n_pairs == 0
    assert verdict.verdict.status is PromotionStatus.INSUFFICIENT_DATA
