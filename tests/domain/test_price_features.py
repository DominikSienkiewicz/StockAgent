"""ML-31 — cechy zmienności i momentum z darmowych snapshotów cen (point-in-time)."""

from __future__ import annotations

import math
import statistics
from datetime import UTC, datetime, timedelta

import pytest

from src.domain.price_features import (
    PRICE_FEATURE_NAMES,
    SIGMA_FLOOR,
    SIGMA_MIN_RETURNS,
    SIGMA_WINDOW_CYCLES,
    PriceFeatures,
    price_features,
    return_sigma,
)

_START = datetime(2026, 9, 1, 21, 0, tzinfo=UTC)


def _series(prices: list[float], start: datetime = _START) -> list[tuple[datetime, float]]:
    """Jedno zamknięcie dziennie, dzień po dniu."""
    return [(start + timedelta(days=i), price) for i, price in enumerate(prices)]


def _zigzag(n: int, base: float = 100.0) -> list[float]:
    """Szereg z niezerową zmiennością i lekkim trendem."""
    return [base * (1.0 + 0.002 * i) * (1.0 + (0.01 if i % 3 == 0 else -0.004)) for i in range(n)]


def test_features_use_only_prices_before_as_of() -> None:
    history = _series(_zigzag(30))
    as_of = history[24][0]
    future = [(as_of + timedelta(hours=h), 1_000.0 * h) for h in range(1, 6)]

    with_future = price_features(history + future, as_of)
    without_future = price_features(history[:25], as_of)

    assert with_future == without_future
    assert with_future.ret_1 == pytest.approx(history[24][1] / history[23][1] - 1.0)


def test_vol_normalized_return_is_scale_invariant() -> None:
    prices = _zigzag(30)
    as_of = _START + timedelta(days=29)

    base = price_features(_series(prices), as_of).as_features()
    scaled = price_features(_series([p * 100.0 for p in prices]), as_of).as_features()

    assert base["ret_1_over_vol_20"] is not None
    for name in PRICE_FEATURE_NAMES:
        assert scaled[name] == pytest.approx(base[name], rel=1e-9), name


def test_long_window_features_need_twenty_one_daily_closes() -> None:
    prices = _zigzag(10)

    features = price_features(_series(prices), _START + timedelta(days=9))

    assert features.ret_1 is not None
    assert features.ret_5 is not None
    assert features.vol_ratio_5_20 is None
    assert features.ret_20 is None
    assert features.ret_1_over_vol_20 is None
    assert features.log_realized_vol_20 is None
    assert features.dist_from_20d_high is None


def test_a_gap_longer_than_five_days_masks_the_window() -> None:
    # 25 zamknięć, potem 10 dni pauzy pętli i 3 zamknięcia po wznowieniu.
    before = _series(_zigzag(25))
    resumed_at = before[-1][0] + timedelta(days=10)
    after = _series([110.0, 111.0, 112.0], start=resumed_at)

    features = price_features(before + after, after[-1][0])

    assert features.ret_1 == pytest.approx(112.0 / 111.0 - 1.0)
    assert features.ret_5 is None
    assert features.ret_20 is None
    assert features.ret_1_over_vol_20 is None


def test_last_snapshot_of_a_day_is_its_close_and_the_gate_uses_raw_snapshots() -> None:
    history = _series(_zigzag(25))
    as_of = history[-1][0] + timedelta(hours=1)
    # Drugi cykl tego samego dnia: zamknięcie dnia to on, a ruch bramki liczy się
    # od poprzedniego snapshotu (z tego samego dnia), nie od wczorajszego zamknięcia.
    second = (as_of, history[-1][1] * 1.02)

    features = price_features([*history, second], as_of)

    assert features.ret_1 == pytest.approx(second[1] / history[-2][1] - 1.0)
    assert features.log_realized_vol_20 is not None
    assert features.gate_move_z == pytest.approx(
        0.02 / math.exp(features.log_realized_vol_20)
    )


def test_distance_from_the_high_is_non_positive() -> None:
    features = price_features(_series(_zigzag(25)[::-1]), _START + timedelta(days=24))

    assert features.dist_from_20d_high is not None
    assert features.dist_from_20d_high < 0.0


def test_empty_or_invalid_history_gives_no_features() -> None:
    history = [(_START, 0.0), (_START + timedelta(days=1), float("nan"))]

    assert price_features(history, _START + timedelta(days=2)) == PriceFeatures()
    assert set(PriceFeatures().as_features()) == set(PRICE_FEATURE_NAMES)


# -- ML-11: σ zwrotów z poprzednich cykli (normalizacja targetu modelu wspólnego) --


def _cycle_returns(prices: list[float]) -> list[float]:
    return [b / a - 1.0 for a, b in zip(prices, prices[1:], strict=False)]


def test_sigma_uses_only_returns_before_as_of() -> None:
    # Jedna funkcja w treningu i servingu: ruch po `as_of` (etykieta) nie może zmienić σ.
    prices = _zigzag(30)
    history = _series(prices)
    as_of = history[24][0] + timedelta(hours=12)
    crash = [(as_of + timedelta(hours=1), prices[24] * 0.5), *history[25:]]

    sigma = return_sigma(history[:25], as_of)

    assert sigma == return_sigma([*history[:25], *crash], as_of)
    assert sigma == pytest.approx(statistics.stdev(_cycle_returns(prices[4:25])))


def test_sigma_takes_the_last_window_of_cycle_returns() -> None:
    prices = _zigzag(40)
    history = _series(prices)
    as_of = history[-1][0] + timedelta(days=1)

    expected = statistics.stdev(_cycle_returns(prices)[-SIGMA_WINDOW_CYCLES:])

    assert SIGMA_WINDOW_CYCLES == 20
    assert return_sigma(history, as_of) == pytest.approx(expected)


def test_price_of_the_current_cycle_is_not_a_previous_cycle() -> None:
    # Snapshot ceny cyklu zapisuje węzeł terminalny tuż PRZED wektorem cech — w treningu
    # ma chwilę sprzed `as_of` wektora, a w servingu (predict) jeszcze go nie ma.
    prices = _zigzag(30)
    history = _series(prices)
    as_of = history[-1][0] + timedelta(days=1)
    same_cycle = (as_of - timedelta(seconds=2), prices[-1] * 1.2)

    assert return_sigma([*history, same_cycle], as_of) == return_sigma(history, as_of)


def test_flat_history_is_floored() -> None:
    history = _series([100.0] * 25)

    assert return_sigma(history, history[-1][0] + timedelta(days=1)) == SIGMA_FLOOR


def test_short_history_has_no_sigma() -> None:
    as_of = _START + timedelta(days=30)

    assert return_sigma(_series(_zigzag(SIGMA_MIN_RETURNS)), as_of) is None
    assert return_sigma(_series(_zigzag(SIGMA_MIN_RETURNS + 1)), as_of) is not None


def test_sigma_ignores_broken_prices() -> None:
    prices = _zigzag(25)
    history = _series(prices)
    broken = [
        (history[3][0] + timedelta(hours=1), 0.0),
        (history[5][0] + timedelta(hours=1), math.nan),
    ]
    as_of = history[-1][0] + timedelta(days=1)

    assert return_sigma([*history, *broken], as_of) == return_sigma(history, as_of)
