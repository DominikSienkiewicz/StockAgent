import datetime
from typing import cast
from unittest.mock import Mock

import pytest

from src.application.agent_graph import AgentState, _build_fetch_fundamentals_node
from src.application.ports import FundamentalsPort
from src.domain.asset import Asset
from src.domain.value_objects import (
    AssetType,
    Fundamentals,
    ValuationVerdict,
)


def test_node_skips_for_etf() -> None:
    port = Mock(spec=FundamentalsPort)
    node = _build_fetch_fundamentals_node(port)

    state = cast(AgentState, {"asset": Asset(symbol="VOO", asset_type=AssetType.ETF)})
    result = node(state)

    assert result["fundamentals"] is None
    assert result["valuation_verdict"] is ValuationVerdict.UNKNOWN
    port.get_fundamentals.assert_not_called()


def test_node_fetches_and_evaluates_for_stock() -> None:
    port = Mock(spec=FundamentalsPort)
    port.get_fundamentals.return_value = Fundamentals(
        trailing_pe=25.0, forward_pe=20.0, peg_ratio=0.8,
        eps_growth_yoy=0.20,
        fetched_at=datetime.datetime(2026, 5, 17, tzinfo=datetime.UTC),
    )
    node = _build_fetch_fundamentals_node(port)
    state = cast(AgentState, {"asset": Asset(symbol="AAPL", asset_type=AssetType.STOCK)})
    result = node(state)

    assert result["valuation_verdict"] is ValuationVerdict.UNDERVALUED
    assert result["fundamentals"] is not None
    assert result["fundamentals"].peg_ratio == pytest.approx(0.8)


def test_node_returns_unknown_when_port_raises() -> None:
    port = Mock(spec=FundamentalsPort)
    port.get_fundamentals.side_effect = RuntimeError("boom")
    node = _build_fetch_fundamentals_node(port)
    result = node(cast(AgentState, {"asset": Asset(symbol="AAPL", asset_type=AssetType.STOCK)}))

    assert result["fundamentals"] is None
    assert result["valuation_verdict"] is ValuationVerdict.UNKNOWN
