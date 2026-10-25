"""TEST-12 (G65): własności matematyki ryzyka sprawdzane Hypothesis.

Niezmienniki były nazwane w testach przykładowych, ale sprawdzane na 1–6
próbkach. Tu każdy dostaje 200 losowych wejść (profil `ci` w conftest).
Strategie podają jawnie `allow_nan`: domyślne `st.floats` generuje NaN.
"""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from src.domain.correlation import pearson
from src.domain.equity_curve import PaperTrade, equity_curve, paper_portfolio_curve
from src.domain.sizing import suggest_band
from src.domain.var import conditional_var, historical_var

# Błąd zaokrąglenia interpolacji kwantyla i średniej ogona na zwrotach rzędu 1.
_TOLERANCE = 1e-12

_returns = st.lists(
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_subnormal=False),
    min_size=2,
    max_size=60,
)
_confidence = st.floats(min_value=0.5, max_value=0.999, allow_nan=False)
_series = st.lists(
    st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_subnormal=False),
    max_size=40,
)
_trend = st.sampled_from(["BULLISH", "BEARISH", "SIDEWAYS"])
# Cena nie spada poniżej zera, więc zmiana ≥ -100%; w górę bez praktycznej granicy.
_change = st.floats(min_value=-1.0, max_value=5.0, allow_nan=False)
_cost_bps = st.floats(min_value=0.0, max_value=100.0, allow_nan=False)
_unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@given(returns=_returns, first=_confidence, second=_confidence)
def test_var_is_monotonic_in_confidence(returns: list[float], first: float, second: float) -> None:
    lower, higher = sorted((first, second))

    assert historical_var(returns, lower) <= historical_var(returns, higher) + _TOLERANCE


@given(returns=_returns, confidence=_confidence)
def test_cvar_is_at_least_var(returns: list[float], confidence: float) -> None:
    assert conditional_var(returns, confidence) >= historical_var(returns, confidence) - _TOLERANCE


@given(first=_series, second=_series)
def test_pearson_is_symmetric_and_bounded(first: list[float], second: list[float]) -> None:
    forward = pearson(first, second)

    assert forward == pearson(second, first)
    assert -1.0 <= forward <= 1.0


@given(trades=st.lists(st.tuples(_trend, _change), max_size=30), cost_bps=_cost_bps)
def test_equity_max_drawdown_within_unit_interval(
    trades: list[tuple[str, float]], cost_bps: float
) -> None:
    """Obsunięcie to ułamek szczytu: kapitał nie spada poniżej zera (ruina),
    więc nawet short przy +150% nie daje obsunięcia ponad 100%."""
    curve = equity_curve(trades, cost_bps=cost_bps)

    assert 0.0 <= curve.max_drawdown <= 1.0
    assert all(point.equity >= 0.0 for point in curve.points)


@given(
    cycles=st.lists(
        st.lists(
            st.builds(
                PaperTrade,
                symbol=st.sampled_from(["AAA", "BBB", "CCC"]),
                predicted_trend=_trend,
                change=_change,
                cost_bps=_cost_bps,
            ),
            max_size=4,
        ),
        max_size=12,
    )
)
def test_paper_portfolio_max_drawdown_within_unit_interval(
    cycles: list[list[PaperTrade]],
) -> None:
    curve = paper_portfolio_curve(cycles)

    assert 0.0 <= curve.max_drawdown <= 1.0
    assert all(point.equity >= 0.0 for point in curve.points)


@given(
    consensus=_unit,
    dissent=_unit,
    hit_rate=st.one_of(st.none(), st.sampled_from([math.nan, math.inf, -math.inf])),
    hit_sample=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1e4, allow_nan=False)),
)
def test_sizing_is_never_full_without_a_finite_hit_rate(
    consensus: float, dissent: float, hit_rate: float | None, hit_sample: float | None
) -> None:
    """„Pozycja pełna” wymaga znanego, przyzwoitego track recordu — NaN albo
    nieskończoność to brak track recordu, nie hit-rate 100%."""
    band = suggest_band(consensus, dissent, hit_rate, hit_sample=hit_sample)

    assert band.tier != "full"
