"""ML-32 — cechy cross-sectional: ruch symbolu na tle uniwersum cyklu."""

from __future__ import annotations

import pytest

from src.domain.cross_section import (
    BENCHMARK_SYMBOL,
    CROSS_SECTION_FEATURE_NAMES,
    CrossSectionFeatures,
    SymbolMove,
    cross_section_features,
    percentile_ranks,
)


def _move(ret_1: float, asset_class: str = "STOCK", threshold: float = 0.02) -> SymbolMove:
    return SymbolMove(ret_1=ret_1, asset_class=asset_class, gate_threshold=threshold)


def test_rank_is_uniform_in_unit_interval() -> None:
    ranks = percentile_ranks({"A": 0.03, "B": -0.01, "C": 0.0, "D": 0.05, "E": -0.02})

    assert sorted(ranks.values()) == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert ranks["E"] == 0.0
    assert ranks["D"] == 1.0


def test_excess_vs_benchmark_subtracts_ivv_return() -> None:
    features = cross_section_features(
        {"AAPL": _move(-0.03), "MSFT": _move(-0.025), BENCHMARK_SYMBOL: _move(-0.03, "ETF")}
    )

    assert features["AAPL"].ret_1_excess_vs_benchmark == pytest.approx(0.0)
    assert features["MSFT"].ret_1_excess_vs_benchmark == pytest.approx(0.005)
    assert features[BENCHMARK_SYMBOL].ret_1_excess_vs_benchmark == pytest.approx(0.0)


def test_without_the_benchmark_the_excess_is_unknown() -> None:
    features = cross_section_features({"AAPL": _move(0.01), "MSFT": _move(0.02)})

    assert features["AAPL"].ret_1_excess_vs_benchmark is None


def test_ties_share_the_average_rank_and_a_single_symbol_sits_in_the_middle() -> None:
    assert percentile_ranks({"A": 0.01, "B": 0.01, "C": 0.02}) == pytest.approx(
        {"A": 0.25, "B": 0.25, "C": 1.0}
    )
    assert percentile_ranks({"A": 0.01}) == {"A": 0.5}
    assert percentile_ranks({}) == {}


def test_excess_vs_class_median_uses_the_symbol_asset_class() -> None:
    features = cross_section_features(
        {
            "AAPL": _move(0.01),
            "MSFT": _move(0.03),
            "NVDA": _move(0.05),
            "BTC": _move(-0.04, "CRYPTO", 0.05),
            "ETH": _move(-0.02, "CRYPTO", 0.05),
        }
    )

    assert features["AAPL"].ret_1_excess_vs_class_median == pytest.approx(-0.02)
    assert features["NVDA"].ret_1_excess_vs_class_median == pytest.approx(0.02)
    assert features["BTC"].ret_1_excess_vs_class_median == pytest.approx(-0.01)


def test_breadth_is_the_share_of_moves_at_or_above_their_gate() -> None:
    features = cross_section_features(
        {
            "AAPL": _move(0.02),  # dokładnie próg — liczy się
            "MSFT": _move(-0.03),
            "NVDA": _move(0.001),
            "BTC": _move(0.04, "CRYPTO", 0.05),  # pod progiem krypto
        }
    )

    assert [f.breadth for f in features.values()] == pytest.approx([0.5] * 4)


def test_empty_universe_has_no_features() -> None:
    assert cross_section_features({}) == {}
    assert set(CrossSectionFeatures().as_features()) == set(CROSS_SECTION_FEATURE_NAMES)
