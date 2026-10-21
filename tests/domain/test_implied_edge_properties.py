"""TEST-12 (G65): edge modelu vs rynek opcji przy NaN i danych bez sensu.

`edge_sigma` trafia do kolumny `prediction_logs.edge_sigma` i do etykiety
w mailu. NaN porównany z progami nie jest ani ≥ 1.5, ani ≤ 0.5, więc bez
strażnika wychodził jako ALIGNED — „model zgodny z rynkiem” z niczego.
"""

from __future__ import annotations

import math

from hypothesis import assume, given
from hypothesis import strategies as st

from src.domain.implied_edge import EdgeLabel, evaluate_edge

_any_float = st.floats(allow_nan=True, allow_infinity=True)
_finite_move = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)
_finite_iv = st.floats(min_value=0.01, max_value=3.0, allow_nan=False)
_horizon = st.floats(min_value=1.0, max_value=24.0 * 7, allow_nan=False)


@given(
    model_move=st.one_of(st.just(math.nan), _finite_move),
    iv=st.one_of(st.just(math.nan), _finite_iv),
    horizon_hours=st.one_of(st.just(math.nan), _horizon),
)
def test_implied_edge_with_nan_input_is_no_signal(
    model_move: float, iv: float, horizon_hours: float
) -> None:
    assume(any(math.isnan(value) for value in (model_move, iv, horizon_hours)))

    edge = evaluate_edge(model_move, iv, horizon_hours)

    assert edge.label is EdgeLabel.NO_SIGNAL
    assert edge.edge_sigma is None
    assert edge.implied_move is None


@given(
    model_move=_any_float,
    iv=st.one_of(st.none(), _any_float),
    horizon_hours=_any_float,
)
def test_implied_edge_sigma_is_finite_or_absent(
    model_move: float, iv: float | None, horizon_hours: float
) -> None:
    edge = evaluate_edge(model_move, iv, horizon_hours)

    if edge.label is EdgeLabel.NO_SIGNAL:
        assert edge.edge_sigma is None
    else:
        assert edge.edge_sigma is not None and math.isfinite(edge.edge_sigma)
        assert edge.implied_move is not None and math.isfinite(edge.implied_move)
